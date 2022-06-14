import inspect
import typing
import unittest

import beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.meta_world
import beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.to_metal

from beanmachine.ppl.model.rv_identifier import RVIdentifier
import torch
import torch.distributions as dist
from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.to_metal import to_metal


# given a function over queries and observations
# assumes most inference algorithm do the following:
# (1) extract values from world
# (2) query log probs: P(X=x|Y=y)
# (3) set new values into world (it is a stateful entity)
# here I wanted to demonstrate a minimal example, so the code is nonsense but the operations are meant to demonstrate what
# a real inference algorithm might use
@to_metal
def foo_infer(queries:[RVIdentifier], observations:typing.Dict[RVIdentifier, float]) -> typing.Dict[RVIdentifier,float]:
    world = beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.meta_world.World(observations=observations, queries=queries)
    fake_parameters = {}
    for rv in queries:
        new_val = 1.0
        prob = world.log_prob_of(rv, new_val)
        world.set_value(rv, prob)
        fake_parameters[rv] = prob
    return fake_parameters

class NormalNormal:

    @beanmachine.ppl.random_variable
    def mu(self):
        return dist.Normal(
            torch.zeros(1).to(self.device), 10 * torch.ones(1).to(self.device)
        )

    @beanmachine.ppl.random_variable
    def x(self, i):
        return dist.Normal(self.mu(), torch.ones(1).to(self.device))

class PaicMLIRTest(unittest.TestCase):
    def test_foo_infer(self):
        model = NormalNormal()
        call = foo_infer(queries=[model.mu()],
                         observations={model.x(1): torch.tensor(9.0), model.x(2): torch.tensor(10.0)})


if __name__ == "__main__":
    model = NormalNormal()
    call = foo_infer(queries=[model.mu()],
                     observations={model.x(1): torch.tensor(9.0),model.x(2): torch.tensor(10.0)})
