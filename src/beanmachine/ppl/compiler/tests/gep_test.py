# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.inference import BMGInference

trials = torch.tensor([29854.0, 2016.0])
pos = torch.tensor([4.0, 0.0])
buck_rep = torch.tensor([0.0006, 0.01])
n_buckets = len(trials)


def log1mexp(x):
    return (1 - x.exp()).log()


@bm.random_variable
def eta():  # k reals
    return dist.Normal(0.0, 1.0).expand((n_buckets,))


@bm.random_variable
def alpha():  # atomic R+
    return dist.half_normal.HalfNormal(5.0)


@bm.random_variable
def sigma():  # atomic R+
    return dist.half_normal.HalfNormal(1.0)


@bm.random_variable
def length_scale():  # R+
    return dist.half_normal.HalfNormal(0.1)


@bm.functional
def cholesky():  # k by k reals
    delta = 1e-3
    alpha_sq = alpha() * alpha()
    rho_sq = length_scale() * length_scale()
    cov = (buck_rep - buck_rep.unsqueeze(-1)) ** 2
    cov = alpha_sq * torch.exp(-cov / (2 * rho_sq))
    cov += torch.eye(buck_rep.size(0)) * delta
    return torch.linalg.cholesky(cov)


@bm.random_variable
def prev():  # k reals
    return dist.Normal(torch.matmul(cholesky(), eta()), sigma())


@bm.random_variable
def bucket_prob():  # atomic bool
    phi_prev = dist.Normal(0, 1).cdf(prev())  # k probs
    log_prob = pos * torch.log(phi_prev)
    log_prob += (trials - pos) * torch.log1p(-phi_prev)
    joint_log_prob = log_prob.sum()
    # Convert the joint log prob to a log-odds.
    logit_prob = joint_log_prob - log1mexp(joint_log_prob)
    return dist.Bernoulli(logits=logit_prob)


class GEPTest(unittest.TestCase):
    def test_gep_model_compilation(self) -> None:
        self.maxDiff = None
        queries = [prev()]
        observations = {bucket_prob(): torch.tensor([1.0])}
        observed = BMGInference().to_dot(queries, observations)

        expected = """
digraph "graph" {
  N00[label=5.0];
  N01[label=HalfNormal];
  N02[label=Sample];
  N03[label=0.10000000149011612];
  N04[label=HalfNormal];
  N05[label=Sample];
  N06[label=0.0];
  N07[label=1.0];
  N08[label=Normal];
  N09[label=Sample];
  N10[label=Normal];
  N11[label=Sample];
  N12[label=HalfNormal];
  N13[label=Sample];
  N14[label=2];
  N15[label="*"];
  N16[label=0.0];
  N17[label=Exp];
  N18[label="8.836000051815063e-05"];
  N19[label=2.0];
  N20[label="*"];
  N21[label=-1.0];
  N22[label="**"];
  N23[label="*"];
  N24[label="-"];
  N25[label=Exp];
  N26[label=Exp];
  N27[label=Exp];
  N28[label=ToMatrix];
  N29[label=ToPosRealMatrix];
  N30[label=MatrixScale];
  N31[label=0];
  N32[label=ColumnIndex];
  N33[label=index];
  N34[label=0.0010000000474974513];
  N35[label="+"];
  N36[label=1];
  N37[label=index];
  N38[label=ColumnIndex];
  N39[label=index];
  N40[label=index];
  N41[label="+"];
  N42[label=ToMatrix];
  N43[label=Cholesky];
  N44[label=ToMatrix];
  N45[label="@"];
  N46[label=index];
  N47[label=Normal];
  N48[label=Sample];
  N49[label=index];
  N50[label=Normal];
  N51[label=Sample];
  N52[label=4.0];
  N53[label=Phi];
  N54[label=Log];
  N55[label="-"];
  N56[label="*"];
  N57[label="-"];
  N58[label=29850.0];
  N59[label=complement];
  N60[label=Log];
  N61[label="-"];
  N62[label="*"];
  N63[label="-"];
  N64[label="+"];
  N65[label=2016.0];
  N66[label=Phi];
  N67[label=complement];
  N68[label=Log];
  N69[label="-"];
  N70[label="*"];
  N71[label="-"];
  N72[label="+"];
  N73[label=ToReal];
  N74[label=Log1mexp];
  N75[label="-"];
  N76[label=ToReal];
  N77[label="+"];
  N78[label="Bernoulli(logits)"];
  N79[label=Sample];
  N80[label="Observation True"];
  N81[label=ToMatrix];
  N82[label=Query];
  N00 -> N01;
  N01 -> N02;
  N02 -> N15;
  N02 -> N15;
  N03 -> N04;
  N04 -> N05;
  N05 -> N20;
  N05 -> N20;
  N06 -> N08;
  N06 -> N10;
  N07 -> N08;
  N07 -> N10;
  N07 -> N12;
  N08 -> N09;
  N09 -> N44;
  N10 -> N11;
  N11 -> N44;
  N12 -> N13;
  N13 -> N47;
  N13 -> N50;
  N14 -> N28;
  N14 -> N28;
  N14 -> N42;
  N14 -> N42;
  N14 -> N44;
  N14 -> N81;
  N15 -> N30;
  N16 -> N17;
  N16 -> N27;
  N17 -> N28;
  N18 -> N23;
  N19 -> N20;
  N20 -> N22;
  N21 -> N22;
  N22 -> N23;
  N23 -> N24;
  N24 -> N25;
  N24 -> N26;
  N25 -> N28;
  N26 -> N28;
  N27 -> N28;
  N28 -> N29;
  N29 -> N30;
  N30 -> N32;
  N30 -> N38;
  N31 -> N32;
  N31 -> N33;
  N31 -> N39;
  N31 -> N46;
  N32 -> N33;
  N32 -> N37;
  N33 -> N35;
  N34 -> N35;
  N34 -> N41;
  N35 -> N42;
  N36 -> N37;
  N36 -> N38;
  N36 -> N40;
  N36 -> N44;
  N36 -> N49;
  N36 -> N81;
  N37 -> N42;
  N38 -> N39;
  N38 -> N40;
  N39 -> N42;
  N40 -> N41;
  N41 -> N42;
  N42 -> N43;
  N43 -> N45;
  N44 -> N45;
  N45 -> N46;
  N45 -> N49;
  N46 -> N47;
  N47 -> N48;
  N48 -> N53;
  N48 -> N81;
  N49 -> N50;
  N50 -> N51;
  N51 -> N66;
  N51 -> N81;
  N52 -> N56;
  N53 -> N54;
  N53 -> N59;
  N54 -> N55;
  N55 -> N56;
  N56 -> N57;
  N57 -> N64;
  N58 -> N62;
  N59 -> N60;
  N60 -> N61;
  N61 -> N62;
  N62 -> N63;
  N63 -> N64;
  N64 -> N72;
  N65 -> N70;
  N66 -> N67;
  N67 -> N68;
  N68 -> N69;
  N69 -> N70;
  N70 -> N71;
  N71 -> N72;
  N72 -> N73;
  N72 -> N74;
  N73 -> N77;
  N74 -> N75;
  N75 -> N76;
  N76 -> N77;
  N77 -> N78;
  N78 -> N79;
  N79 -> N80;
  N81 -> N82;
}
"""
        self.assertEqual(expected.strip(), observed.strip())