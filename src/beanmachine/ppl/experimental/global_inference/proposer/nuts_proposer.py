from typing import NamedTuple, Optional

import torch
from beanmachine.ppl.experimental.global_inference.proposer.hmc_proposer import (
    HMCProposer,
)
from beanmachine.ppl.experimental.global_inference.simple_world import (
    RVDict,
    SimpleWorld,
)


class _TreeNode(NamedTuple):
    world: SimpleWorld
    momentums: RVDict
    pe_grad: RVDict


class _Tree(NamedTuple):
    left: _TreeNode
    right: _TreeNode
    proposal: SimpleWorld
    pe: torch.Tensor
    pe_grad: RVDict
    log_weight: torch.Tensor
    sum_momentums: RVDict
    sum_accept_prob: torch.Tensor
    num_proposals: int
    turned_or_diverged: bool


class _TreeArgs(NamedTuple):
    log_slice: torch.Tensor
    direction: int
    step_size: float
    initial_energy: torch.Tensor
    mass_inv: RVDict


class NUTSProposer(HMCProposer):
    """
    The No-U-Turn Sampler (NUTS) as described in [1]. Unlike vanilla HMC, it does not
    require users to specify a trajectory length. The current implementation roughly
    follows Algorithm 6 of [1]. If multinomial_sampling is True, then the next state
    will be drawn from a multinomial distribution (weighted by acceptance probability,
    as introduced in Appendix 2 of [2]) instead of drawn uniformly.

    Reference:
    [1] Matthew Hoffman and Andrew Gelman. "The No-U-Turn Sampler: Adaptively
        Setting Path Lengths in Hamiltonian Monte Carlo" (2014).
        https://arxiv.org/abs/1111.4246

    [2] Michael Betancourt. "A Conceptual Introduction to Hamiltonian Monte Carlo"
        (2017). https://arxiv.org/abs/1701.02434
    """

    def __init__(
        self,
        initial_world: SimpleWorld,
        max_tree_depth: int = 10,
        max_delta_energy: float = 1000.0,
        initial_step_size: float = 1.0,
        adapt_step_size: bool = True,
        adapt_mass_matrix: bool = True,
        multinomial_sampling: bool = True,
    ):
        # note that trajectory_length is not used in NUTS
        super().__init__(
            initial_world,
            trajectory_length=0.0,
            initial_step_size=initial_step_size,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
        )
        self._max_tree_depth = max_tree_depth
        self._max_delta_energy = max_delta_energy
        self._multinomial_sampling = multinomial_sampling

    def _is_u_turning(
        self,
        mass_inv: RVDict,
        left_momentums: RVDict,
        right_momentums: RVDict,
        sum_momentums: RVDict,
    ) -> bool:
        """The generalized U-turn condition, as described in [2] Appendix 4.2"""
        left_r = torch.cat([left_momentums[node] for node in mass_inv])
        right_r = torch.cat([right_momentums[node] for node in mass_inv])
        rho = torch.cat([mass_inv[node] * sum_momentums[node] for node in mass_inv])
        return bool((torch.dot(left_r, rho) <= 0) or (torch.dot(right_r, rho) <= 0))

    def _build_tree_base_case(self, root: _TreeNode, args: _TreeArgs) -> _Tree:
        """Base case of the recursive tree building algorithm: take a single leapfrog
        step in the specified direction and return a subtree."""
        world, momentums, pe, pe_grad = self._leapfrog_step(
            root.world,
            root.momentums,
            args.step_size * args.direction,
            args.mass_inv,
            root.pe_grad,
        )
        new_energy = self._hamiltonian(world, momentums, args.mass_inv, pe)
        # initial_energy == -L(\theta^{m-1}) + 1/2 r_0^2 in Algorithm 6 of [1]
        delta_energy = torch.nan_to_num(new_energy - args.initial_energy, float("inf"))
        if self._multinomial_sampling:
            log_weight = -delta_energy
        else:
            # slice sampling as introduced in the original NUTS paper [1]
            log_weight = (args.log_slice <= -new_energy).log()

        tree_node = _TreeNode(world=world, momentums=momentums, pe_grad=pe_grad)
        return _Tree(
            left=tree_node,
            right=tree_node,
            proposal=world,
            pe=pe,
            pe_grad=pe_grad,
            log_weight=log_weight,
            sum_momentums=momentums,
            sum_accept_prob=torch.clamp(torch.exp(-delta_energy), max=1.0),
            num_proposals=1,
            turned_or_diverged=bool(
                args.log_slice >= self._max_delta_energy - new_energy
            ),
        )

    def _build_tree(self, root: _TreeNode, tree_depth: int, args: _TreeArgs) -> _Tree:
        """Build the binary tree by recursively build the left and right subtrees and
        combine the two."""
        if tree_depth == 0:
            return self._build_tree_base_case(root, args)

        # build the first half of the tree
        sub_tree = self._build_tree(root, tree_depth - 1, args)
        if sub_tree.turned_or_diverged:
            return sub_tree

        # build the other half of the tree
        other_sub_tree = self._build_tree(
            root=sub_tree.left if args.direction == -1 else sub_tree.right,
            tree_depth=tree_depth - 1,
            args=args,
        )

        return self._combine_tree(
            sub_tree, other_sub_tree, args.direction, args.mass_inv, biased=False
        )

    def _combine_tree(
        self,
        old_tree: _Tree,
        new_tree: _Tree,
        direction: int,
        mass_inv: RVDict,
        biased: bool,
    ) -> _Tree:
        """Combine the old tree and the new tree into a single (large) tree. The new
        tree will be add to the left of the old tree if direction is -1, otherwise it
        will be add to the right. If biased is True, then we will prefer choosing from
        new tree (which is away from the starting location) than old tree when sampling
        the next state from the trajectory. This function assumes old_tree is not
        turned or diverged."""
        # if old tree hsa turned or diverged, then we shouldn't build the new tree in
        # the first place
        assert not old_tree.turned_or_diverged
        # log of the sum of the weights from both trees
        log_weight = torch.logaddexp(old_tree.log_weight, new_tree.log_weight)

        if new_tree.turned_or_diverged:
            selected_subtree = old_tree
        else:
            # progressively sample from the trajectory
            if biased:
                # biased progressive sampling (Appendix 3.2 of [2])
                log_tree_prob = new_tree.log_weight - old_tree.log_weight
            else:
                # uniform progressive sampling (Appendix 3.1 of [2])
                log_tree_prob = new_tree.log_weight - log_weight

            if torch.rand_like(log_tree_prob).log() < log_tree_prob:
                selected_subtree = new_tree
            else:
                selected_subtree = old_tree

        if direction == -1:
            left_tree, right_tree = new_tree, old_tree
        else:
            left_tree, right_tree = old_tree, new_tree

        sum_momentums = {
            node: left_tree.sum_momentums[node] + right_tree.sum_momentums[node]
            for node in left_tree.sum_momentums
        }
        turned_or_diverged = new_tree.turned_or_diverged or self._is_u_turning(
            mass_inv,
            left_tree.left.momentums,
            right_tree.right.momentums,
            sum_momentums,
        )
        # More robust U-turn condition
        # https://discourse.mc-stan.org/t/nuts-misses-u-turns-runs-in-circles-until-max-treedepth/9727
        if not turned_or_diverged and right_tree.num_proposals > 1:
            extended_sum_momentums = {
                node: left_tree.sum_momentums[node] + right_tree.left.momentums[node]
                for node in sum_momentums
            }
            turned_or_diverged = self._is_u_turning(
                mass_inv,
                left_tree.left.momentums,
                right_tree.left.momentums,
                extended_sum_momentums,
            )
        if not turned_or_diverged and left_tree.num_proposals > 1:
            extended_sum_momentums = {
                node: right_tree.sum_momentums[node] + left_tree.right.momentums[node]
                for node in sum_momentums
            }
            turned_or_diverged = self._is_u_turning(
                mass_inv,
                left_tree.right.momentums,
                right_tree.right.momentums,
                extended_sum_momentums,
            )

        return _Tree(
            left=left_tree.left,
            right=right_tree.right,
            proposal=selected_subtree.proposal,
            pe=selected_subtree.pe,
            pe_grad=selected_subtree.pe_grad,
            log_weight=log_weight,
            sum_momentums=sum_momentums,
            sum_accept_prob=old_tree.sum_accept_prob + new_tree.sum_accept_prob,
            num_proposals=old_tree.num_proposals + new_tree.num_proposals,
            turned_or_diverged=turned_or_diverged,
        )

    def propose(self, world: Optional[SimpleWorld] = None) -> SimpleWorld:
        if world is not None and world is not self.world:
            # re-compute cached values in case world was modified by other sources
            self._pe, self._pe_grad = self._potential_grads(self.world)
            self.world = world

        momentums = self._initialize_momentums(self.world)
        current_energy = self._hamiltonian(
            self.world, momentums, self._mass_inv, self._pe
        )
        if self._multinomial_sampling:
            # log slice is only used to check the divergence
            log_slice = -current_energy
        else:
            # this is a more stable way to sample from log(Uniform(0, exp(-current_energy)))
            log_slice = torch.log1p(-torch.rand(())) - current_energy
        tree_node = _TreeNode(self.world, momentums, self._pe_grad)
        tree = _Tree(
            left=tree_node,
            right=tree_node,
            proposal=self.world,
            pe=self._pe,
            pe_grad=self._pe_grad,
            log_weight=torch.tensor(0.0),  # log accept prob of staying at current state
            sum_momentums=momentums,
            sum_accept_prob=torch.tensor(0.0),
            num_proposals=0,
            turned_or_diverged=False,
        )

        for j in range(self._max_tree_depth):
            direction = 1 if torch.rand(()) > 0.5 else -1
            tree_args = _TreeArgs(
                log_slice, direction, self.step_size, current_energy, self._mass_inv
            )
            if direction == -1:
                new_tree = self._build_tree(tree.left, j, tree_args)
            else:
                new_tree = self._build_tree(tree.right, j, tree_args)

            tree = self._combine_tree(
                tree, new_tree, direction, self._mass_inv, biased=True
            )
            if tree.turned_or_diverged:
                break

        self.world, self._pe, self._pe_grad = tree.proposal, tree.pe, tree.pe_grad
        self._alpha = tree.sum_accept_prob / tree.num_proposals
        return self.world