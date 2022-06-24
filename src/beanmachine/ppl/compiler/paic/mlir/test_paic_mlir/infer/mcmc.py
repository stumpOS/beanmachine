import logging
import typing
from typing import Callable
from typing import List, Set

import torch
from torch import Tensor
from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.infer.functional_sample import new_world
from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.infer.meta_world import MetaWorld, RealWorld
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import init_to_uniform, InitializeFn, RVDict
from beanmachine.ppl.world.utils import is_constraint_eq
from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.infer.import_inference import import_inference
from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.infer.propose.single_site_real_space_nmc_proposer import \
    SingleSiteRealSpaceNMCProposer_MetaWorld
from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.infer.propose.BaseProposer_MetaWorld import \
    BaseProposer_MetaWorld
from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.infer.meta_ad import AD
LOGGER = logging.getLogger("beanmachine")


class Sliced_NMC:
    def __init__(
            self,
            lower: bool,
            ad:AD,
            real_space_alpha: float = 10.0,
            real_space_beta: float = 1.0,
    ):
        self._proposers = {}
        self.ad = ad
        self.lower = lower
        self.alpha = real_space_alpha
        self.beta = real_space_beta

    # maximum value of a seed
    _MAX_SEED_VAL: int = 2 ** 32 - 1

    def create_world(self, queries, observations) -> MetaWorld:
        return RealWorld(queries, observations)

    def infer(
            self,
            queries: List[RVIdentifier],
            observations: RVDict,
            num_samples: int
    ) -> MonteCarloSamples:
        num_adaptive_samples = 0
        if self.lower:
            chain_results: tuple[list[Tensor], list[Tensor]] = import_inference(_single_chain_infer)(queries, lambda w,
                                                                                                                     rv: self.get_proposers(
                w, rv), observations, num_samples)
        else:
            chain_results: tuple[list[Tensor], list[Tensor]] = _single_chain_infer(queries,
                                                                                   lambda w, rv: self.get_proposers(w,
                                                                                                                    rv),
                                                                                   observations, num_samples)

        all_samples = chain_results[0]
        all_log_liklihoods = chain_results[1]
        # the hash of RVIdentifier can change when it is being sent to another process,
        # so we have to rely on the order of the returned list to determine which samples
        # correspond to which RVIdentifier
        i = 0
        # should be length Q, where Q is the number of queries
        sample_dict_list = []

        # all_samples has an element for each query
        # if we had multiple chains, we would have multiple lists with Q entries
        q = 0
        dict = {}
        for sample_list in all_samples:
            dict[queries[q]] = sample_list
            q = q + 1
        sample_dict_list.append(dict)

        i = 0
        log_liklihood_dict = {}
        obsversation_keys = list(observations.keys())
        for log_liklihood in all_log_liklihoods:
            obs = obsversation_keys[i]
            log_liklihood_dict[obs] = log_liklihood
            i = i + 1

        return MonteCarloSamples(
            sample_dict_list,
            num_adaptive_samples,
            [log_liklihood_dict],
            observations,
        )

    def get_proposers(
            self,
            world: MetaWorld,
            target_rvs: Set[RVIdentifier],
    ) -> List[BaseProposer_MetaWorld]:
        proposers = []
        for node in target_rvs:
            if node not in self._proposers:
                rv = world.rv_metadata(node)
                support = rv.support()
                if is_constraint_eq(support, torch.distributions.constraints.real):
                    self._proposers[node] = SingleSiteRealSpaceNMCProposer_MetaWorld(node, self.ad, self.alpha, self.beta)
                else:
                    raise NotImplementedError("Not implemented yet")
            proposers.append(self._proposers[node])
        return proposers


def _single_chain_infer(queries: List[RVIdentifier],
                        get_proposers: Callable[[MetaWorld, Set[RVIdentifier]], List[BaseProposer_MetaWorld]],
                        observations: RVDict,
                        num_samples: int
                        ) -> typing.Tuple[List[torch.Tensor], List[torch.Tensor]]:
    current_world = RealWorld(queries, observations)
    samples = [[] for _ in queries]
    log_likelihoods = [[] for _ in observations]

    # Main inference loop
    for _ in range(num_samples):
        world = new_world(current_world, get_proposers, 0)
        for idx, obs in enumerate(observations):
            log_likelihoods[idx].append(world.log_prob([obs]))
        # Extract samples
        for idx, query in enumerate(queries):
            raw_val = world.value_of(query)
            if not isinstance(raw_val, torch.Tensor):
                raise TypeError(
                    "The value returned by a queried function must be a tensor."
                )
            samples[idx].append(raw_val)
        # if rejected, current_world should be the same as world
        current_world = world

    samples = [torch.stack(val) for val in samples]
    log_likelihoods = [torch.stack(val) for val in log_likelihoods]
    return samples, log_likelihoods
