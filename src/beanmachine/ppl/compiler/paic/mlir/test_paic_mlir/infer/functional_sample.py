from typing import List, Set, Callable

from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.infer.meta_world import MetaWorld
import random
import torch
from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.infer.propose.BaseProposer_MetaWorld import BaseProposer_MetaWorld


# instead of giving a sampler to call when we enter a new iteration of the loop,
# let's call the proposer in the loop and ask for a new world
def new_world(current_world: MetaWorld, get_proposers: Callable[[MetaWorld, Set[RVIdentifier]], List[
    BaseProposer_MetaWorld]], num_adaptive_sample_remaining: int) -> MetaWorld:
    proposers = get_proposers(current_world, set(current_world.queries()))
    random.shuffle(proposers)
    next_world = current_world
    for proposer in proposers:
        try:
            new_world, accept_log_prob = proposer.propose(current_world)
            accept_log_prob = accept_log_prob.clamp(max=0.0)
            accepted = torch.rand_like(accept_log_prob).log() < accept_log_prob
            if accepted:
                next_world = new_world
        except RuntimeError as e:
            if "singular U" in str(e) or "input is not positive-definite" in str(e):
                # since it's normal to run into cholesky error during GP, instead of
                # throwing an error, we simply skip current proposer (which is
                # equivalent to a rejection) and will retry in the next iteration
                continue
            else:
                raise e

        if num_adaptive_sample_remaining > 0:
            proposer.do_adaptation(
                world=next_world, accept_log_prob=accept_log_prob, is_accepted=accepted
            )
            if num_adaptive_sample_remaining == 1:
                # we just reach the end of adaptation period
                proposer.finish_adaptation()
    return next_world
