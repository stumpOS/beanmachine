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
  N14[label="[0,3,9]"];
  N15[label=MatrixScale];
  N16[label=0];
  N17[label=index];
  N18[label="+"];
  N19[label=Normal];
  N20[label=33.0];
  N21[label=LogProb];
  N22[label="+"];
  N23[label=complement];
  N24[label=Log];
  N25[label=ToReal];
  N26[label=3.5999999046325684];
  N27[label=Normal];
  N28[label=LogProb];
  N29[label="+"];
  N30[label=LogSumExp];
  N31[label=Exp];
  N32[label=ToProb];
  N33[label=Bernoulli];
  N34[label=Sample];
  N35[label=1];
  N36[label=index];
  N37[label="+"];
  N38[label=Normal];
  N39[label=68.0];
  N40[label=LogProb];
  N41[label="+"];
  N42[label=3.9000000953674316];
  N43[label=Normal];
  N44[label=LogProb];
  N45[label="+"];
  N46[label=LogSumExp];
  N47[label=Exp];
  N48[label=ToProb];
  N49[label=Bernoulli];
  N50[label=Sample];
  N51[label=2];
  N52[label=index];
  N53[label="+"];
  N54[label=Normal];
  N55[label=34.0];
  N56[label=LogProb];
  N57[label="+"];
  N58[label=2.5999999046325684];
  N59[label=Normal];
  N60[label=LogProb];
  N61[label="+"];
  N62[label=LogSumExp];
  N63[label=Exp];
  N64[label=ToProb];
  N65[label=Bernoulli];
  N66[label=Sample];
  N67[label=3];
  N68[label=ToMatrix];
  N69[label="Observation tensor([1., 1., 1.])"];
  N70[label=Query];
  N71[label=Query];
  N72[label=Query];
  N73[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N03 -> N12;
  N03 -> N23;
  N03 -> N73;
  N04 -> N06;
  N05 -> N06;
  N06 -> N07;
  N06 -> N08;
  N07 -> N18;
  N07 -> N37;
  N07 -> N53;
  N07 -> N70;
  N08 -> N15;
  N08 -> N71;
  N09 -> N10;
  N09 -> N10;
  N10 -> N11;
  N11 -> N19;
  N11 -> N38;
  N11 -> N54;
  N11 -> N72;
  N12 -> N13;
  N13 -> N22;
  N13 -> N41;
  N13 -> N57;
  N14 -> N15;
  N15 -> N17;
  N15 -> N36;
  N15 -> N52;
  N16 -> N17;
  N17 -> N18;
  N18 -> N19;
  N18 -> N27;
  N19 -> N21;
  N20 -> N21;
  N20 -> N28;
  N21 -> N22;
  N22 -> N30;
  N23 -> N24;
  N24 -> N25;
  N25 -> N29;
  N25 -> N45;
  N25 -> N61;
  N26 -> N27;
  N27 -> N28;
  N28 -> N29;
  N29 -> N30;
  N30 -> N31;
  N31 -> N32;
  N32 -> N33;
  N33 -> N34;
  N34 -> N68;
  N35 -> N36;
  N35 -> N68;
  N36 -> N37;
  N37 -> N38;
  N37 -> N43;
  N38 -> N40;
  N39 -> N40;
  N39 -> N44;
  N40 -> N41;
  N41 -> N46;
  N42 -> N43;
  N43 -> N44;
  N44 -> N45;
  N45 -> N46;
  N46 -> N47;
  N47 -> N48;
  N48 -> N49;
  N49 -> N50;
  N50 -> N68;
  N51 -> N52;
  N52 -> N53;
  N53 -> N54;
  N53 -> N59;
  N54 -> N56;
  N55 -> N56;
  N55 -> N60;
  N56 -> N57;
  N57 -> N62;
  N58 -> N59;
  N59 -> N60;
  N60 -> N61;
  N61 -> N62;
  N62 -> N63;
  N63 -> N64;
  N64 -> N65;
  N65 -> N66;
  N66 -> N68;
  N67 -> N68;
  N68 -> N69;
}
        """
        self.assertEqual(observed.strip(), expected.strip())
