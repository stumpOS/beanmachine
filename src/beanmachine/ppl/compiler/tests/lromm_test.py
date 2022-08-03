# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compilation test of Todd's Linear Regression Outliers Marginalized model"""
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.distributions.unit import Unit
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import logaddexp, ones, tensor
from torch.distributions import Bernoulli, Beta, Gamma, Normal


_x_obs = tensor([0, 3, 9])
_y_obs = tensor([33, 68, 34])
_err_obs = tensor([3.6, 3.9, 2.6])


@bm.random_variable
def beta_0():
    return Normal(0, 10)


@bm.random_variable
def beta_1():
    return Normal(0, 10)


@bm.random_variable
def sigma_out():
    return Gamma(1, 1)


@bm.random_variable
def theta():
    return Beta(2, 5)


@bm.functional
def f():
    mu = beta_0() + beta_1() * _x_obs
    ns = Normal(mu, sigma_out())
    ne = Normal(mu, _err_obs)
    log_likelihood_outlier = theta().log() + ns.log_prob(_y_obs)
    log_likelihood = (1 - theta()).log() + ne.log_prob(_y_obs)
    return logaddexp(log_likelihood_outlier, log_likelihood)


@bm.random_variable
def y():
    return Unit(f())


# Same model, but with the "Bernoulli trick" instead of a Unit:


@bm.random_variable
def d():
    return Bernoulli(f().exp())


class LROMMTest(unittest.TestCase):
    def test_lromm_unit_to_dot(self) -> None:
        self.maxDiff = None
        queries = [beta_0(), beta_1(), sigma_out(), theta()]
        observations = {y(): _y_obs}
        with self.assertRaises(ValueError) as ex:
            BMGInference().to_dot(queries, observations)
        expected = """
Function Unit is not supported by Bean Machine Graph.
        """
        observed = str(ex.exception)
        self.assertEqual(observed.strip(), expected.strip())

    def test_lromm_bern_to_dot(self) -> None:
        self.maxDiff = None
        queries = [beta_0(), beta_1(), sigma_out(), theta()]
        observations = {d(): ones(len(_y_obs))}
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N00[label=2.0];
  N01[label=5.0];
  N02[label=Beta];
  N03[label=Sample];
  N04[label=0.0];
  N05[label=10.0];
  N06[label=Normal];
  N07[label=Sample];
  N08[label=Sample];
  N09[label=1.0];
  N10[label=Gamma];
  N11[label=Sample];
  N12[label=Log];
  N13[label=ToReal];
  N14[label=Normal];
  N15[label=33.0];
  N16[label=LogProb];
  N17[label="+"];
  N18[label=complement];
  N19[label=Log];
  N20[label=ToReal];
  N21[label=3.5999999046325684];
  N22[label=Normal];
  N23[label=LogProb];
  N24[label="+"];
  N25[label=LogSumExp];
  N26[label=Exp];
  N27[label=ToProb];
  N28[label=Bernoulli];
  N29[label=Sample];
  N30[label=3.0];
  N31[label="*"];
  N32[label="+"];
  N33[label=Normal];
  N34[label=68.0];
  N35[label=LogProb];
  N36[label="+"];
  N37[label=3.9000000953674316];
  N38[label=Normal];
  N39[label=LogProb];
  N40[label="+"];
  N41[label=LogSumExp];
  N42[label=Exp];
  N43[label=ToProb];
  N44[label=Bernoulli];
  N45[label=Sample];
  N46[label=9.0];
  N47[label="*"];
  N48[label="+"];
  N49[label=Normal];
  N50[label=34.0];
  N51[label=LogProb];
  N52[label="+"];
  N53[label=2.5999999046325684];
  N54[label=Normal];
  N55[label=LogProb];
  N56[label="+"];
  N57[label=LogSumExp];
  N58[label=Exp];
  N59[label=ToProb];
  N60[label=Bernoulli];
  N61[label=Sample];
  N62[label="Observation True"];
  N63[label="Observation True"];
  N64[label="Observation True"];
  N65[label=Query];
  N66[label=Query];
  N67[label=Query];
  N68[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N03 -> N12;
  N03 -> N18;
  N03 -> N68;
  N04 -> N06;
  N05 -> N06;
  N06 -> N07;
  N06 -> N08;
  N07 -> N14;
  N07 -> N22;
  N07 -> N32;
  N07 -> N48;
  N07 -> N65;
  N08 -> N31;
  N08 -> N47;
  N08 -> N66;
  N09 -> N10;
  N09 -> N10;
  N10 -> N11;
  N11 -> N14;
  N11 -> N33;
  N11 -> N49;
  N11 -> N67;
  N12 -> N13;
  N13 -> N17;
  N13 -> N36;
  N13 -> N52;
  N14 -> N16;
  N15 -> N16;
  N15 -> N23;
  N16 -> N17;
  N17 -> N25;
  N18 -> N19;
  N19 -> N20;
  N20 -> N24;
  N20 -> N40;
  N20 -> N56;
  N21 -> N22;
  N22 -> N23;
  N23 -> N24;
  N24 -> N25;
  N25 -> N26;
  N26 -> N27;
  N27 -> N28;
  N28 -> N29;
  N29 -> N62;
  N30 -> N31;
  N31 -> N32;
  N32 -> N33;
  N32 -> N38;
  N33 -> N35;
  N34 -> N35;
  N34 -> N39;
  N35 -> N36;
  N36 -> N41;
  N37 -> N38;
  N38 -> N39;
  N39 -> N40;
  N40 -> N41;
  N41 -> N42;
  N42 -> N43;
  N43 -> N44;
  N44 -> N45;
  N45 -> N63;
  N46 -> N47;
  N47 -> N48;
  N48 -> N49;
  N48 -> N54;
  N49 -> N51;
  N50 -> N51;
  N50 -> N55;
  N51 -> N52;
  N52 -> N57;
  N53 -> N54;
  N54 -> N55;
  N55 -> N56;
  N56 -> N57;
  N57 -> N58;
  N58 -> N59;
  N59 -> N60;
  N60 -> N61;
  N61 -> N64;
}
        """
        self.assertEqual(observed.strip(), expected.strip())
