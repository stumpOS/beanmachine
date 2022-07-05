import torch

from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.infer.meta_world import MetaWorld, RealWorld
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import RVDict
from typing import List, Callable
import torch.distributions as dist
from torch import tensor
import beanmachine.ppl as bm
from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.infer.import_inference import import_inference

def fake_inference(world: MetaWorld):
    world.print()

@import_inference
def entry_point_of_fake_inference(queries: List[RVIdentifier],
              observations: RVDict,
              world_creator: Callable[[List[RVIdentifier], RVDict], MetaWorld],
              inference: Callable[[MetaWorld], None]
            ):
    inference(world_creator(queries, observations))

class SampleNormalModel:
    @bm.random_variable
    def foo(self):
        return dist.Normal(tensor(2.0), tensor(2.0))

    @bm.random_variable
    def bar(self, i):
        return dist.Normal(self.foo(), torch.tensor(1.0))

def test_create_type():
    # create model
    model = SampleNormalModel()
    foo_value = dist.Normal(tensor(2.0), tensor(2.0)).sample(torch.Size((1, 1)))
    observations = {}
    bar_parent = dist.Normal(foo_value, torch.tensor(1.0))
    for i in range(0, 20):
        observations[model.bar(i)] = bar_parent.sample(torch.Size((1, 1)))

    inf: Callable[[MetaWorld], None] = fake_inference
    entry_point_of_fake_inference(queries=[model.foo()],
                                                      observations=observations,
                                                      world_creator=lambda q,o:RealWorld(q,o),
                                                      inference=fake_inference)


    # TODO: assert that the samples are as expected
    print("\ncompiles")

if __name__ == "__main__":
    test_create_type()