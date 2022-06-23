# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from abc import abstractmethod

import torch
import torch.distributions as dist

from beanmachine.ppl.inference.utils import safe_log_prob_sum
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.infer.meta_world import MetaWorld
from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.infer.propose.BaseProposer_MetaWorld import BaseProposer_MetaWorld

class BaseSingleSiteMHProposer_MetaWorld(BaseProposer_MetaWorld):
    def __init__(self, target_rv: RVIdentifier):
        self.node = target_rv

    def propose(self, world: MetaWorld):
        """
        Propose a new value for self.node with `Metropolis-Hasting algorithm
        <https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm#Formal_derivation>`_
        Classes that inherit this proposer should override `get_proposal_distribution`
        to define the algorithm-specific way to sample the next state.

        Args:
            world: World to calculate proposal for.
        """
        forward_dist = self.get_proposal_distribution(world)
        proposal_dist = forward_dist
        old_value = world.value_of(self.node)
        proposed_value = proposal_dist.sample()
        new_world = world.replace(self.node, proposed_value)
        backward_dist = self.get_proposal_distribution(new_world)

        # calculate MH acceptance probability
        # log P(x, y)
        old_log_prob = world.log_prob()
        # log P(x', y)
        new_log_prob = new_world.log_prob()
        # log g(x'|x)
        forward_log_prob = forward_dist.log_prob(proposed_value).sum()
        # log g(x|x')
        # because proposed_value is sampled from forward_dist, it is guaranteed to be
        # within the valid range. However, there's no guarantee that the old value
        # is in the support of backward_dist
        backward_log_prob = safe_log_prob_sum(backward_dist, old_value)

        # log [(P(x', y) * g(x|x')) / (P(x, y) * g(x'|x))]
        accept_log_prob = (
            new_log_prob + backward_log_prob - old_log_prob - forward_log_prob
        )
        # model size adjustment log (n/n')
        nodeCount = len(world.observations()) + world.latent_variable_count()
        accept_log_prob += math.log(nodeCount) - math.log(len(new_world.observations()) + new_world.latent_variable_count())

        if torch.isnan(accept_log_prob):
            accept_log_prob = torch.tensor(
                float("-inf"),
                device=accept_log_prob.device,
                dtype=accept_log_prob.dtype,
            )

        return new_world, accept_log_prob

    @abstractmethod
    def get_proposal_distribution(self, world: MetaWorld) -> dist.Distribution:
        """Return a probability distribution of moving self.node to a new value
        conditioned on its current value in world.
        """
        raise NotImplementedError


BaseProposer_MetaWorld.register(BaseSingleSiteMHProposer_MetaWorld)