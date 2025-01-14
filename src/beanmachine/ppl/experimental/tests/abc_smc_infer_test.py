# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.abc.abc_smc_infer import (
    ApproximateBayesianComputationSequentialMonteCarlo,
)


class ApproximateBayesianComputationTest(unittest.TestCase):
    torch.manual_seed(42)

    class CoinTossModel:
        def __init__(self, observation_shape):
            self.observation_shape = observation_shape

        @bm.random_variable
        def bias(self):
            return dist.Beta(0.5, 0.5)

        @bm.random_variable
        def coin_toss(self):
            return dist.Bernoulli(self.bias().repeat(self.observation_shape))

        def toss_head_count(self, toss_vals):
            return torch.sum(toss_vals)

        def toss_mean(self, toss_vals):
            return torch.mean(toss_vals)

        @bm.functional
        def num_heads(self):
            return self.toss_head_count(self.coin_toss())

        @bm.functional
        def mean_value(self):
            return self.toss_mean(self.coin_toss())

    def test_abc_smc_inference(self):
        model = self.CoinTossModel(observation_shape=10)
        COIN_TOSS_DATA = dist.Bernoulli(0.77).sample([10])
        num_heads_key = model.num_heads()
        mean_value_key = model.mean_value()
        abc_smc = ApproximateBayesianComputationSequentialMonteCarlo(
            tolerance_schedule={
                num_heads_key: [4.0, 2.0, 1.0],
                mean_value_key: [0.4, 0.2, 0.1],
            }
        )
        observations = {
            num_heads_key: model.toss_head_count(COIN_TOSS_DATA),
            mean_value_key: model.toss_mean(COIN_TOSS_DATA),
        }
        queries = [model.bias()]
        samples = abc_smc.infer(queries, observations, num_samples=100, num_chains=1)
        self.assertAlmostEqual(
            torch.mean(samples[model.bias()][0]).item(), 0.77, delta=0.3
        )
        abc_smc.reset()

    def test_max_attempts(self):
        model = self.CoinTossModel(observation_shape=100)
        COIN_TOSS_DATA = dist.Bernoulli(0.9).sample([100])
        abc_smc = ApproximateBayesianComputationSequentialMonteCarlo(
            tolerance_schedule={model.num_heads(): [20.0, 10.0]},
            max_attempts_per_sample=1,
        )
        observations = {model.num_heads(): model.toss_head_count(COIN_TOSS_DATA)}
        queries = [model.bias()]
        with self.assertRaises(RuntimeError):
            abc_smc.infer(
                queries, observations, num_samples=100, num_chains=1, verbose=None
            )
        abc_smc.reset()

    def test_shape_mismatch(self):
        model = self.CoinTossModel(observation_shape=100)
        abc_smc = ApproximateBayesianComputationSequentialMonteCarlo(
            tolerance_schedule={model.num_heads(): [20, 10]}
        )
        observations = {model.num_heads(): torch.tensor([3, 4])}
        queries = [model.bias()]
        with self.assertRaises(ValueError):
            abc_smc.infer(
                queries, observations, num_samples=100, num_chains=1, verbose=None
            )
        abc_smc.reset()
