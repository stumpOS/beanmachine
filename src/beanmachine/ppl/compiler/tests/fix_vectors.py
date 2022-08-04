# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import typing
from typing import Callable, Dict, List, Type

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.error_report import ErrorReport
from beanmachine.ppl.compiler.fix_matrix_scale import matrix_scale_fixer
from beanmachine.ppl.compiler.fix_problem import (
    ancestors_first_graph_fixer,
    fixpoint_graph_fixer,
    GraphFixer,
    GraphFixerResult,
    Inapplicable,
    node_fixer_first_match,
    NodeFixer,
    NodeFixerResult,
    sequential_graph_fixer,
)
from beanmachine.ppl.compiler.sizer import is_scalar, Sizer

# TODO Move this to a utils module
from beanmachine.ppl.compiler.support import _prod
from torch import Size, tensor

_consumes_tensor_types = [
    bn.SumNode,
    bn.Query,
    bn.CholeskyNode,
    bn.ColumnIndexNode,
    bn.IndexNode,
    bn.LogSumExpNode,
    bn.MatrixMultiplicationNode,
    bn.ToRealMatrixNode,
    bn.TransposeNode,
]

_indexable_node_types = [
    bn.ColumnIndexNode,
    bn.ConstantTensorNode,
    bn.IndexNode,
    bn.MatrixScaleNode,
    bn.SampleNode,
    bn.TensorNode,
    bn.ToMatrixNode,
    bn.UntypedConstantNode,
]

# nodes in this category accept a single value argument
def _constant_factories(bmg: BMGraphBuilder) -> Dict[Type, Callable]:
    return {
        bn.NaturalNode: bmg.add_natural,
        bn.ConstantNode: bmg.add_constant,
        bn.RealNode: bmg.add_real,
        bn.PositiveRealNode: bmg.add_pos_real,
        bn.ConstantTensorNode: bmg.add_constant_tensor,
        bn.UntypedConstantNode: bmg.add_constant,
    }


def _node_factories(bmg: BMGraphBuilder) -> Dict[Type, Callable]:
    return {
        # Note that we expect devectorization to run *before* multiary
        # addition/multiplication rewriting, so we can assume that
        # all additions and multiplications are binary.
        bn.AdditionNode: bmg.add_addition,
        bn.BitAndNode: bmg.add_bitand,
        bn.BitOrNode: bmg.add_bitor,
        bn.BitXorNode: bmg.add_bitxor,
        bn.CholeskyNode: bmg.add_cholesky,
        bn.ColumnIndexNode: bmg.add_column_index,
        bn.DivisionNode: bmg.add_division,
        bn.EqualNode: bmg.add_equal,
        bn.Exp2Node: bmg.add_exp2,
        bn.ExpNode: bmg.add_exp,
        bn.ExpM1Node: bmg.add_expm1,
        bn.ItemNode: bmg.add_item,
        bn.IndexNode: bmg.add_index,
        bn.LogAddExpNode: bmg.add_logaddexp,
        bn.LogisticNode: bmg.add_logistic,
        bn.Log10Node: bmg.add_log10,
        bn.Log1pNode: bmg.add_log1p,
        bn.Log2Node: bmg.add_log2,
        bn.Log1mexpNode: bmg.add_log1mexp,
        bn.LogSumExpVectorNode: bmg.add_logsumexp_vector,
        bn.LogProbNode: bmg.add_log_prob,
        bn.LogNode: bmg.add_log,
        bn.LogSumExpTorchNode: bmg.add_logsumexp_torch,
        bn.MatrixMultiplicationNode: bmg.add_matrix_multiplication,
        bn.MultiplicationNode: bmg.add_multiplication,
        bn.NegateNode: bmg.add_negate,
        bn.PhiNode: bmg.add_phi,
        bn.PowerNode: bmg.add_power,
        bn.SquareRootNode: bmg.add_squareroot,
        bn.ToMatrixNode: bmg.add_to_matrix,
        bn.ToPositiveRealMatrixNode: bmg.add_to_positive_real_matrix,
        bn.ToRealMatrixNode: bmg.add_to_real_matrix,
        bn.TransposeNode: bmg.add_transpose,
        bn.ToRealNode: bmg.add_to_real,
    }


def _distribution_factories(bmg: BMGraphBuilder) -> Dict[Type, Callable]:
    # These are all the distributions that we know how to devectorize,
    # and the factory methods we need to use to generate a new node
    # of the appropriate type.

    # TODO: categorical
    # TODO: categorical logit
    # TODO: dirichlet
    return {
        bn.BernoulliLogitNode: bmg.add_bernoulli_logit,
        bn.BernoulliNode: bmg.add_bernoulli,
        bn.BetaNode: bmg.add_beta,
        bn.BinomialNode: bmg.add_binomial,
        bn.BinomialLogitNode: bmg.add_binomial_logit,
        bn.Chi2Node: bmg.add_chi2,
        bn.DirichletNode: bmg.add_dirichlet,
        bn.GammaNode: bmg.add_gamma,
        bn.HalfCauchyNode: bmg.add_halfcauchy,
        bn.HalfNormalNode: bmg.add_halfnormal,
        bn.NormalNode: bmg.add_normal,
        bn.PoissonNode: bmg.add_poisson,
        bn.StudentTNode: bmg.add_studentt,
        bn.UniformNode: bmg.add_uniform,
    }


_distribution_types = list(_distribution_factories(BMGraphBuilder()).keys())


class BuilderContext:
    def __init__(self, bmg: BMGraphBuilder):
        self.bmg = bmg
        self.dist_factories = _distribution_factories(bmg)
        self.op_factories = _node_factories(bmg)
        self.value_factories = _constant_factories(bmg)
        self.devectorized_nodes: Dict[bn.BMGNode, DevectorizedNode] = {}
        self.clones: Dict[bn.BMGNode, bn.BMGNode] = {}


def _is_fixable_size(s: Size) -> bool:
    dim = len(s)
    if dim == 1:
        return s[0] > 1
    if dim == 2:
        return s[0] > 1 or s[1] > 1
    return False


def _needs_devectorize(node: bn.BMGNode, size: Size) -> bool:
    return _is_fixable_size(size) and not _consumes_tensor_types.__contains__(
        type(node)
    )


def _node_to_index_list(
    cxt: BuilderContext, size: Size, node: bn.BMGNode
) -> List[bn.BMGNode]:
    dim = len(size)
    index_list = []
    # This code is a little confusing because BMG uses column-major matrices
    # and torch uses row-major tensors.  The Sizer always gives the size
    # that a graph node would be in *torch*, so if we have a Size([2, 3])
    # matrix node, that has two rows and three columns in torch, and would
    # be indexed first by row and then by column. But in BMG, that would
    # be two columns, three rows, and indexed by column first, then row.
    #
    # The practical upshot is: if we have, say, Size([3]) OR Size([1, 3])
    # then either way, we will have a one-column, three row BMG node, and
    # therefore we only need a single level of indexing.
    n = _clone(node, size, cxt)
    if dim == 0:
        # If we have just a single value then there's no indexing required.
        index_list.append(n)
    elif dim == 1:

        for i in range(0, size[0]):
            ci = cxt.bmg.add_constant(i)
            ni = cxt.bmg.add_index(n, ci)
            index_list.append(ni)
    elif size[0] == 1:
        assert dim == 2
        for i in range(0, size[1]):
            ci = cxt.bmg.add_constant(i)
            ni = cxt.bmg.add_index(n, ci)
            index_list.append(ni)
    else:
        # We need two levels of indexing.
        assert dim == 2
        for i in range(0, size[0]):
            ci = cxt.bmg.add_constant(i)
            ni = cxt.bmg.add_index(n, ci)
            for j in range(0, size[1]):
                cj = cxt.bmg.add_constant(j)
                nij = cxt.bmg.add_index(ni, cj)
                index_list.append(nij)
    return index_list


class DevectorizedNode:
    def __init__(self, elements: List[bn.BMGNode], shape: Size):
        self.elements: List[bn.BMGNode] = elements
        self.size = shape


def list_from_parents(
    item_count: int, parents: [], creator: Callable
) -> List[bn.BMGNode]:
    return list_from_parents_with_index(item_count, parents, lambda i, s: creator(*s))


def list_from_parents_with_index(
    item_count: int, parents: [], creator: Callable
) -> List[bn.BMGNode]:
    elements: List[bn.BMGNode] = []
    for i in range(0, item_count):
        reduced_parents = []
        for parent in parents:
            if isinstance(parent, DevectorizedNode):
                reduced_parents.append(parent.elements[i])
            else:
                reduced_parents.append(parent)
        new_node = creator(i, reduced_parents)
        elements.append(new_node)
    return elements


def _clone_parents(node: bn.BMGNode, cxt: BuilderContext) -> List[bn.BMGNode]:
    parents = []
    for p in node.inputs.inputs:
        if cxt.devectorized_nodes.__contains__(p):
            elements = cxt.devectorized_nodes[p].elements
            parent = cxt.bmg.add_tensor(cxt.devectorized_nodes[p].size, *elements)
            parents.append(parent)
        elif cxt.clones.__contains__(p):
            parent = cxt.clones[p]
            parents.append(parent)
        else:
            raise ValueError("encountered a value not in the clone context")
    return parents


def _clone(node: bn.BMGNode, size: Size, cxt: BuilderContext) -> bn.BMGNode:
    parents = _clone_parents(node, cxt)
    if isinstance(node, bn.SampleNode):
        dist = parents[0]
        assert isinstance(dist, bn.DistributionNode)
        return cxt.bmg.add_sample(operand=dist)
    if isinstance(node, bn.DistributionNode):
        return cxt.dist_factories[type(node)](*parents)
    if isinstance(node, bn.Query):
        return cxt.bmg.add_query(parents[0])
    if isinstance(node, bn.OperatorNode):
        return cxt.op_factories[type(node)](*parents)
    if isinstance(node, bn.TensorNode):
        return cxt.bmg.add_tensor(size, *parents)
    if isinstance(node, bn.Observation):
        return cxt.bmg.add_observation(parents[0], node.value)
    if cxt.value_factories.__contains__(type(node)):
        return cxt.value_factories[type(node)](node.value)
    # if isinstance(node, bn.UntypedConstantNode):
    #     return cxt.bmg.add_constant(node.value)
    # if isinstance(node, bn.ConstantTensorNode):
    #     return cxt.bmg.add_constant_tensor(node.value)
    # if isinstance(node, bn.NaturalNode):
    #     return cxt.bmg.add_natural(node.value)
    else:
        raise NotImplementedError(type(node))


def split(node: bn.BMGNode, cxt: BuilderContext, size: Size) -> DevectorizedNode:
    item_count = 0
    for i in range(0, len(size)):
        item_count += size[i]

    # identify parents
    parents = []
    for input in node.inputs.inputs:
        if cxt.devectorized_nodes.__contains__(input):
            devectorized_parent = cxt.devectorized_nodes[input]
            if devectorized_parent.size == size:
                parents.append(devectorized_parent)
            else:
                raise NotImplementedError("broadcasting not supported yet")
        elif cxt.clones.__contains__(input):
            parents.append(cxt.clones[input])
        else:
            raise ValueError("value not found in clone context")

    # create devectorized node from parents
    if isinstance(node, bn.SampleNode):
        new_nodes = list_from_parents(item_count, parents, cxt.bmg.add_sample)
        return DevectorizedNode(new_nodes, size)
    if _indexable_node_types.__contains__(type(node)):
        return DevectorizedNode(_node_to_index_list(cxt, size, node), size)
    if isinstance(node, bn.DistributionNode):
        return DevectorizedNode(
            list_from_parents(item_count, parents, cxt.dist_factories[type(node)]), size
        )
    if isinstance(node, bn.Query):
        return DevectorizedNode(
            list_from_parents(item_count, parents, cxt.bmg.add_query), size
        )
    if isinstance(node, bn.OperatorNode):
        return DevectorizedNode(
            list_from_parents(item_count, parents, cxt.op_factories[type(node)]), size
        )
    if isinstance(node, bn.Observation):
        # TODO: What if the observation is of a different size than the
        # tensor node we've just generated? That should be an error, but instead
        # we just crash here. Figure out where to put an error detection pass
        # which prevents this crash and reports the error.
        dim = len(node.value.size())
        values = []
        if dim == 1:
            for i in range(0, node.value.size()[0]):
                values.append(node.value[i])
        else:
            assert dim == 2
            for i in range(0, node.value.size()[0]):
                for j in range(0, node.value.size()[1]):
                    values.append(node.value[i][j])
        return DevectorizedNode(
            list_from_parents_with_index(
                item_count, parents, lambda i, s: cxt.bmg.add_observation(*s, values[i])
            ),
            size,
        )
    else:
        raise NotImplementedError()


def vectorized_node_fixer(sizer: Sizer) -> GraphFixer:
    def vobs_fixer(bmg_old: BMGraphBuilder) -> GraphFixerResult:
        bmg = BMGraphBuilder()
        # clone with splits
        cxt = BuilderContext(bmg)
        for node in bmg_old.all_nodes():
            size: Size = sizer[node]
            if _needs_devectorize(node, size):
                cxt.devectorized_nodes[node] = split(node, cxt, size)
            else:
                cxt.clones[node] = _clone(node, size, cxt)
        split_nodes_cnt = len(cxt.devectorized_nodes)
        if split_nodes_cnt > 0:
            return bmg, True, ErrorReport()
        else:
            return bmg_old, False, ErrorReport()

    return vobs_fixer


# a graph fixer is a callable that accepts a list and returns a Tuple[bool, ErrorReport]
def vectorized_graph_fixer() -> GraphFixer:
    sizer = Sizer()
    return vectorized_node_fixer(sizer)
