import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.infer.mcmc import Sliced_NMC
import math
from torch import tensor, Tensor


class SampleNormalModel:
    @bm.random_variable
    def foo(self):
        return dist.Normal(tensor(2.0), tensor(2.0))

    @bm.random_variable
    def bar(self, i):
        return dist.Normal(self.foo(), torch.tensor(1.0))

def test_infer():
    # arrange: create model and inference
    model = SampleNormalModel()
    foo_value = dist.Normal(tensor(2.0), tensor(2.0)).sample(torch.Size((1, 1)))
    observations = {}
    bar_parent = dist.Normal(foo_value, torch.tensor(1.0))
    for i in range(0,20):
        observations[model.bar(i)] = bar_parent.sample(torch.Size((1, 1)))

    # ensure it works with regular NMC first
    not_sliced_nnc = bm.SingleSiteNewtonianMonteCarlo()
    samples = not_sliced_nnc.infer(queries=[model.foo()], observations=observations, num_samples=100, num_chains=1)
    foo_samples: Tensor = samples.get_variable(model.foo())
    approx_foo = torch.mean(foo_samples)
    assert abs(approx_foo - foo_value) < 1.0


    # now try with Sliced version
    sliced_nnc = Sliced_NMC()
    sliced_samples = sliced_nnc.infer(queries=[model.foo()], observations=observations, num_samples=100)
    sliced_foo_samples = sliced_samples.get_variable(model.foo())
    sliced_approx_foo = torch.mean(sliced_foo_samples)
    assert abs(sliced_approx_foo - foo_value) < 1.0
