#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

# Dirichlet compiler tests

import unittest

import beanmachine.ppl as bm
from beanmachine.graph import (
    AtomicType,
    DistributionType,
    Graph,
    InferenceType,
    OperatorType,
    ValueType,
    VariableType,
)
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from torch import tensor
from torch.distributions import Dirichlet


def tidy(s: str) -> str:
    return "\n".join(c.strip() for c in s.strip().split("\n")).strip()


# Support for Dirichlet distributions has recently been added to BMG;
# this is the first time that the compiler will have to deal with
# tensor-valued quantities directly so we anticipate having a number
# of problems to solve in type analysis and code generation that have
# been put off until now.
#
# We'll start by just taking the BMG code for a spin directly and see
# what gives errors and what gives results.

dirichlet = DistributionType.DIRICHLET
simplex = VariableType.COL_SIMPLEX_MATRIX
broadcast = VariableType.BROADCAST_MATRIX
real = AtomicType.REAL
prob = AtomicType.PROBABILITY
sample = OperatorType.SAMPLE
s3x1 = ValueType(simplex, prob, 3, 1)
r3x1 = ValueType(broadcast, real, 3, 1)
nmc = InferenceType.NMC
rejection = InferenceType.REJECTION

# Here are some simple models we'll use to test the compiler.


@bm.random_variable
def d0():
    return Dirichlet(tensor([]))


# Torch rejects this one.
# @bm.random_variable
# def d1a():
#     return Dirichlet(tensor(0.5))


@bm.random_variable
def d1b():
    return Dirichlet(tensor([1.0]))


@bm.random_variable
def d1c():
    return Dirichlet(tensor([[1.5]]))


@bm.random_variable
def d1d():
    return Dirichlet(tensor([[[2.0]]]))


# Torch rejects this one
# @bm.random_variable
# def d1e():
#     return Dirichlet(tensor([[[-2.0]]]))


@bm.random_variable
def d2a():
    return Dirichlet(tensor([2.5, 3.0]))


@bm.random_variable
def d2b():
    return Dirichlet(tensor([[3.5, 4.0]]))


@bm.random_variable
def d2c():
    return Dirichlet(tensor([[[4.5, 5.0]]]))


@bm.random_variable
def d23():
    return Dirichlet(tensor([[5.5, 6.0, 6.5], [7.0, 7.5, 8.0]]))


class DirichletTest(unittest.TestCase):
    def test_dirichlet_negative(self) -> None:
        self.maxDiff = None
        g = Graph()
        m1 = tensor([1.5, 1.0, 2.0])
        cm1 = g.add_constant_pos_matrix(m1)
        m2 = tensor([[1.5, 1.0], [2.0, 1.5]])
        cm2 = g.add_constant_pos_matrix(m2)
        two = g.add_constant(2)
        # Input must be a positive real matrix with one column.
        with self.assertRaises(ValueError):
            g.add_distribution(dirichlet, s3x1, [two])
        with self.assertRaises(ValueError):
            g.add_distribution(dirichlet, s3x1, [cm2])
        # Must be only one input
        with self.assertRaises(ValueError):
            g.add_distribution(dirichlet, s3x1, [cm1, two])
        # Output type must be simplex
        with self.assertRaises(ValueError):
            g.add_distribution(dirichlet, r3x1, [cm1])

    def test_dirichlet_sample(self) -> None:
        self.maxDiff = None
        g = Graph()
        m1 = tensor([1.5, 1.0, 2.0])
        cm1 = g.add_constant_pos_matrix(m1)
        d = g.add_distribution(dirichlet, s3x1, [cm1])
        ds = g.add_operator(sample, [d])
        g.query(ds)
        samples = g.infer(1, rejection)
        # samples has form [[array([[a1],[a2],[a3]])]]
        result = tensor(samples[0][0]).reshape([3])
        # We get a three-element simplex, so it should sum to 1.0.
        self.assertAlmostEqual(1.0, float(sum(result)))

    def test_constant_pos_real_matrix(self) -> None:

        # To make a BMG graph with a Dirichlet distribution the first thing
        # we'll need to do is make a positive real matrix as its input.
        # Demonstrate that we can add such a matrix to a graph builder,
        # do a type analysis, and generate C++ and Python code that builds
        # the graph.  Finally, actually build the graph.

        self.maxDiff = None

        bmg = BMGraphBuilder()
        c1 = bmg.add_pos_real_matrix(tensor(1.0))
        c2 = bmg.add_pos_real_matrix(tensor([1.0, 1.5]))
        c3 = bmg.add_pos_real_matrix(tensor([[1.0, 1.5], [2.0, 2.5]]))
        c4 = bmg.add_pos_real_matrix(tensor([1.0, 1.5]))

        # These should be deduplicated
        self.assertTrue(c4 is c2)

        # Verify that we can add these nodes to the graph, do a type analysis,
        # and survive the problem-fixing pass without generating an exception.
        bmg.add_query(c1)
        bmg.add_query(c2)
        bmg.add_query(c3)
        expected = """
digraph "graph" {
  N0[label="1.0:R+>=OH"];
  N1[label="Query:R+>=OH"];
  N2[label="[1.0,1.5]:MR+[1,2]>=MR+[1,2]"];
  N3[label="Query:MR+[1,2]>=MR+[1,2]"];
  N4[label="[[1.0,1.5],\\\\n[2.0,2.5]]:MR+[2,2]>=MR+[2,2]"];
  N5[label="Query:MR+[2,2]>=MR+[2,2]"];
  N0 -> N1;
  N2 -> N3;
  N4 -> N5;
}"""
        observed = bmg.to_dot(
            graph_types=True,
            inf_types=True,
            point_at_input=True,
            label_edges=False,
            after_transform=True,
        )
        self.assertEqual(expected.strip(), observed.strip())

        # We should be able to generate correct C++ and Python code to build
        # a graph that contains only positive constant matrices. Note that the
        # queries are not emitted into the graph because BMG does not allow
        # a query on a constant.
        expected = """
graph::Graph g;
Eigen::MatrixXd m0(1, 1)
m0 << 1.0;
uint n0 = g.add_constant_pos_matrix(m0);

Eigen::MatrixXd m2(2, 1)
m2 << 1.0, 1.5;
uint n2 = g.add_constant_pos_matrix(m2);

Eigen::MatrixXd m4(2, 2)
m4 << 1.0, 1.5, 2.0, 2.5;
uint n4 = g.add_constant_pos_matrix(m4);
        """
        observed = bmg.to_cpp()
        self.assertEqual(expected.strip(), observed.strip())

        expected = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_pos_matrix(tensor(1.0))

n2 = g.add_constant_pos_matrix(tensor([1.0,1.5]))

n4 = g.add_constant_pos_matrix(tensor([[1.0,1.5],[2.0,2.5]]))
        """
        observed = bmg.to_python()
        self.assertEqual(expected.strip(), observed.strip())

        # Let's actually get the graph
        g = bmg.to_bmg()
        expected = """
Node 0 type 1 parents [ ] children [ ] matrix<positive real> 1
Node 1 type 1 parents [ ] children [ ] matrix<positive real>   1 1.5
Node 2 type 1 parents [ ] children [ ] matrix<positive real>   1 1.5
 2 2.5"""
        observed = g.to_string()
        self.assertEqual(tidy(expected), tidy(observed))

    def test_dirichlet_type_analysis(self) -> None:
        self.maxDiff = None
        bmg = BMGraphBuilder()
        queries = [d0(), d1b(), d1c(), d1d(), d2a(), d2b(), d2c(), d23()]
        bmg.accumulate_graph(queries, {})
        observed = bmg.to_dot(
            graph_types=True,
            inf_types=True,
            edge_requirements=True,
            point_at_input=True,
            after_transform=False,
            label_edges=False,
        )
        expected = """
digraph "graph" {
  N00[label="[]:T>=T"];
  N01[label="Dirichlet:S[1,1]>=S[1,1]"];
  N02[label="Sample:S[1,1]>=S[1,1]"];
  N03[label="Query:S[1,1]>=S[1,1]"];
  N04[label="[1.0]:T>=OH"];
  N05[label="Dirichlet:S[1,1]>=S[1,1]"];
  N06[label="Sample:S[1,1]>=S[1,1]"];
  N07[label="Query:S[1,1]>=S[1,1]"];
  N08[label="[[1.5]]:T>=R+"];
  N09[label="Dirichlet:S[1,1]>=S[1,1]"];
  N10[label="Sample:S[1,1]>=S[1,1]"];
  N11[label="Query:S[1,1]>=S[1,1]"];
  N12[label="[[[2.0]]]:T>=N"];
  N13[label="Dirichlet:S[1,1]>=S[1,1]"];
  N14[label="Sample:S[1,1]>=S[1,1]"];
  N15[label="Query:S[1,1]>=S[1,1]"];
  N16[label="[2.5,3.0]:T>=MR+[1,2]"];
  N17[label="Dirichlet:S[1,2]>=S[1,2]"];
  N18[label="Sample:S[1,2]>=S[1,2]"];
  N19[label="Query:S[1,2]>=S[1,2]"];
  N20[label="[[3.5,4.0]]:T>=MR+[1,2]"];
  N21[label="Dirichlet:S[1,2]>=S[1,2]"];
  N22[label="Sample:S[1,2]>=S[1,2]"];
  N23[label="Query:S[1,2]>=S[1,2]"];
  N24[label="[[[4.5,5.0]]]:T>=T"];
  N25[label="Dirichlet:S[1,2]>=S[1,2]"];
  N26[label="Sample:S[1,2]>=S[1,2]"];
  N27[label="Query:S[1,2]>=S[1,2]"];
  N28[label="[[5.5,6.0,6.5],\\\\n[7.0,7.5,8.0]]:T>=MR+[2,3]"];
  N29[label="Dirichlet:S[1,3]>=S[1,3]"];
  N30[label="Sample:S[1,3]>=S[1,3]"];
  N31[label="Query:S[1,3]>=S[1,3]"];
  N00 -> N01[label="R+"];
  N01 -> N02[label="S[1,1]"];
  N02 -> N03[label="S[1,1]"];
  N04 -> N05[label="R+"];
  N05 -> N06[label="S[1,1]"];
  N06 -> N07[label="S[1,1]"];
  N08 -> N09[label="R+"];
  N09 -> N10[label="S[1,1]"];
  N10 -> N11[label="S[1,1]"];
  N12 -> N13[label="R+"];
  N13 -> N14[label="S[1,1]"];
  N14 -> N15[label="S[1,1]"];
  N16 -> N17[label="MR+[1,2]"];
  N17 -> N18[label="S[1,2]"];
  N18 -> N19[label="S[1,2]"];
  N20 -> N21[label="MR+[1,2]"];
  N21 -> N22[label="S[1,2]"];
  N22 -> N23[label="S[1,2]"];
  N24 -> N25[label="MR+[1,2]"];
  N25 -> N26[label="S[1,2]"];
  N26 -> N27[label="S[1,2]"];
  N28 -> N29[label="MR+[1,3]"];
  N29 -> N30[label="S[1,3]"];
  N30 -> N31[label="S[1,3]"];
}
        """
        self.assertEqual(expected.strip(), observed.strip())
