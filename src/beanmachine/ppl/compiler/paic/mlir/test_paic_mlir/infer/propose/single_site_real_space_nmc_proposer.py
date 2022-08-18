import logging
from typing import NamedTuple, Optional, Tuple, Union

import torch
import torch.distributions as dist

import beanmachine.ppl.inference.proposer.nmc.single_site_real_space_nmc_proposer
from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.infer.meta_world import MetaWorld
from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.infer.propose.BaseSingleSiteMHProposer_MetaWorld import \
    BaseSingleSiteMHProposer_MetaWorld
from beanmachine.ppl.legacy.inference.proposer.newtonian_monte_carlo_utils import (
    is_scalar,
    is_valid,
)


class _ProposalArgs(NamedTuple):
    alpha: torch.Tensor
    beta: torch.Tensor

LOGGER = logging.getLogger("beanmachine")
from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.infer.meta_ad import AD

class SingleSiteRealSpaceNMCProposer_MetaWorld(BaseSingleSiteMHProposer_MetaWorld):
    """
    Single-Site Real Space Newtonian Monte Carlo Proposer
    See sec. 3.1 of [1]

    [1] Arora, Nim, et al. `Newtonian Monte Carlo: single-site MCMC meets second-order gradient methods`
    """

    def __init__(self, node: beanmachine.ppl.RVIdentifier, ad:AD, alpha: float = 10.0, beta: float = 1.0):
        super().__init__(node)
        self.ad = ad
        self.alpha_: Union[float, torch.Tensor] = alpha
        self.beta_: Union[float, torch.Tensor] = beta
        self.learning_rate_ = torch.tensor(0.0)
        self.running_mean_, self.running_var_ = torch.tensor(0.0), torch.tensor(0.0)
        self.accepted_samples_ = 0
        # cached proposal args
        self._proposal_args: Optional[_ProposalArgs] = None

    def _sample_frac_dist(self, world: MetaWorld) -> torch.Tensor:
        node_val_flatten = world.value_of(self.node).flatten()
        # If any of alpha or beta are scalar, we have to reshape them
        # random variable shape to allow for per-index learning rate.
        if is_scalar(self.alpha_) or is_scalar(self.beta_):
            self.alpha_ = self.alpha_ * torch.ones_like(node_val_flatten)
            self.beta_ = self.beta_ * torch.ones_like(node_val_flatten)
        beta_ = dist.Beta(self.alpha_, self.beta_)
        return beta_.sample()

    def _get_proposal_distribution_from_args(
        self, args: _ProposalArgs
    ) -> dist.Distribution:
        mean = args.alpha.squeeze(0)
        if args.beta is not None:
            # why is this multivariate normal and not normal? is this in the event that the node is a tensor and not a scalar?
            proposal_dist = dist.MultivariateNormal(mean, args.beta)
        else:
            raise NotImplementedError("We did not compute the eigenvalues of the Hessian because we instead computed the jvp of the vjp.")
        # todo: reshape to match the original sample shape. Can we delegate reshaping to the compiler?
        #  answer: maybe. But then when you switch over to Python it will break because there is not implicit reshaping in PyTorch
        return proposal_dist

    def get_proposal_distribution(self, world: MetaWorld) -> dist.Distribution:
        """
        Returns the proposal distribution of the node.

        Args:
            world: the world in which we're proposing a new value for node
                required to find a proposal distribution which in this case is the
                fraction of distance between the current value and NMC mean that we're
                going to pick as our proposer mean.
        Returns:
            The proposal distribution.
        """
        frac_dist = self._sample_frac_dist(world)
        self.learning_rate_ = frac_dist

        if self._proposal_args is not None and world.latent_variable_count() == 1:
            return self._get_proposal_distribution_from_args(self._proposal_args)

        node_val = world.value_of(self.node)
        # can I separate this into a differential function over a function expressed over queries over a graph?
        def log_prob_children(x:torch.Tensor) -> torch.Tensor:
            world_with_grad = world.replace(self.node, x)
            children = world_with_grad.children_of(self.node)
            score = world_with_grad.log_prob(children.__or__({self.node}))
            return score

        first_order_derivative, second_order_derivative = self.ad.jacobian_and_hessian(x=node_val, func=log_prob_children)
        if not is_valid(first_order_derivative) or not is_valid(second_order_derivative):
            LOGGER.warning(
                "Gradient or Hessian is invalid at node {nv}.\n".format(
                    nv=str(self.node)
                )
                + "Node {n} has invalid proposal solution. ".format(n=self.node)
                + "Proposer falls back to SingleSiteAncestralProposer.\n"
            )
            return super().get_proposal_distribution(world)
        proposal_args = _ProposalArgs(
            alpha=node_val - frac_dist * (first_order_derivative/second_order_derivative),
            beta=torch.sqrt(-1/second_order_derivative)
        )
        self._proposal_args = proposal_args
        return self._get_proposal_distribution_from_args(proposal_args)

    def compute_beta_priors_from_accepted_lr(
        self, max_lr_num: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Alpha and Beta using Method of Moments.
        """
        # Running mean and variance are computed following the link below:
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        old_mu = self.running_mean_
        old_var = self.running_var_

        n = self.accepted_samples_
        xn = self.learning_rate_

        new_mu = old_mu + (xn - old_mu) / n
        new_var = old_var + ((xn - old_mu) * (xn - new_mu) - old_var) / n
        self.running_var_ = new_var
        self.running_mean_ = new_mu
        if n < max_lr_num:
            return (
                torch.tensor(1.0, dtype=self.learning_rate_.dtype),
                torch.tensor(1.0, dtype=self.learning_rate_.dtype),
            )
        # alpha and beta are calculated following the link below.
        # https://stats.stackexchange.com/questions/12232/calculating-the-
        # parameters-of-a-beta-distribution-using-the-mean-and-variance
        alpha = ((1.0 - new_mu) / new_var - (1.0 / new_mu)) * (new_mu**2)
        beta = alpha * (1.0 - new_mu) / new_mu
        alpha = torch.where(alpha <= 0, torch.ones_like(alpha), alpha)
        beta = torch.where(beta <= 0, torch.ones_like(beta), beta)
        return alpha, beta

    def do_adaptation(
        self,
        world: MetaWorld,
        accept_log_prob: torch.Tensor,
        is_accepted: bool = False,
        *args,
        **kwargs
    ) -> None:
        """
        Do adaption based on the learning rates.

        Args:
            world: the world in which we're operating in.
            accept_log_prob: Current accepted log prob (Not used in this particular proposer).
            is_accepted: bool representing whether the new value was accepted.
        """
        if not is_accepted:
            if self.accepted_samples_ == 0:
                self.alpha_ = 1.0
                self.beta_ = 1.0
        else:
            self.accepted_samples_ += 1
            self.alpha_, self.beta_ = self.compute_beta_priors_from_accepted_lr()