# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import typing
from typing import Callable, Dict, List, Type

import beanmachine.ppl.compiler.bmg_nodes as bn
import beanmachine.ppl.compiler.bmg_types
import beanmachine.ppl.compiler.broadcast
from beanmachine.ppl.compiler.bmg_types import BMGMatrixType, Untypable
import beanmachine.ppl.compiler.execution_context
import beanmachine.ppl.compiler.lattice_typer
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.error_report import ErrorReport, UnsupportedNode, BadMatrixMultiplication
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
    bn.CategoricalNode,
    bn.CholeskyNode,
    bn.ColumnIndexNode,
    bn.DirichletNode,
    bn.IndexNode,
    bn.LogSumExpNode,
    bn.MatrixMultiplicationNode,
    bn.MatrixScaleNode,
    bn.Observation, # unsure, can observations consume tensor types? See Dirichlet and Category tests
    bn.Query,
    bn.SampleNode, # unsure, can samples consume tensor types? See Dirichlet and Category tests
    bn.SumNode,
    bn.ToRealMatrixNode,
    bn.TransposeNode,
]

_leaves = [
    bn.Query
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
        bn.NegativeRealNode: bmg.add_neg_real,
        bn.NaturalNode: bmg.add_natural,
        bn.ConstantNode: bmg.add_constant,
        bn.RealNode: bmg.add_real,
        bn.PositiveRealNode: bmg.add_pos_real,
        bn.ProbabilityNode: bmg.add_probability,
        bn.ConstantTensorNode: bmg.add_constant_tensor,
        bn.ConstantPositiveRealMatrixNode: bmg.add_pos_real_matrix,
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
        bn.ComplementNode: bmg.add_complement,
        bn.DivisionNode: bmg.add_division,
        bn.EqualNode: bmg.add_equal,
        bn.Exp2Node: bmg.add_exp2,
        bn.ExpNode: bmg.add_exp,
        bn.ExpM1Node: bmg.add_expm1,
        bn.GreaterThanNode: bmg.add_greater_than,
        bn.GreaterThanEqualNode: bmg.add_greater_than_equal,
        bn.ItemNode: bmg.add_item,
        bn.IndexNode: bmg.add_index,
        bn.LessThanNode: bmg.add_less_than,
        bn.LessThanEqualNode: bmg.add_less_than_equal,
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
        bn.MatrixScaleNode: bmg.add_matrix_scale,
        bn.MultiplicationNode: bmg.add_multiplication,
        bn.NegateNode: bmg.add_negate,
        bn.NotEqualNode: bmg.add_not_equal,
        bn.PhiNode: bmg.add_phi,
        bn.PowerNode: bmg.add_power,
        bn.SquareRootNode: bmg.add_squareroot,
        bn.SwitchNode: bmg.add_switch,
        bn.SumNode: bmg.add_sum,
        bn.ToMatrixNode: bmg.add_to_matrix,
        bn.ToPositiveRealMatrixNode: bmg.add_to_positive_real_matrix,
        bn.ToRealMatrixNode: bmg.add_to_real_matrix,
        bn.TransposeNode: bmg.add_transpose,
        bn.ToPositiveRealNode: bmg.add_to_positive_real,
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
        bn.CategoricalNode: bmg.add_categorical,
        bn.CategoricalLogitNode: bmg.add_categorical_logit,
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
    def __init__(self, bmg_original: BMGraphBuilder, sizer: Sizer):
        self.original_context = bmg_original.execution_context
        self.bmg = BMGraphBuilder()
        self.sizer = sizer
        self.original_bmg = bmg_original
        self.dist_factories = _distribution_factories(self.bmg)
        self.node_factories = _node_factories(self.bmg)
        self.value_factories = _constant_factories(self.bmg)
        self.devectorized_nodes: Dict[bn.BMGNode, DevectorizedNode] = {}
        self.clones: Dict[bn.BMGNode, bn.BMGNode] = {}

    def _is_matrix(self, node:bn.BMGNode) -> bool:
        size = self.sizer[node]
        l = len(size)
        if l == 1:
            return size[0] > 1
        elif l == 2:
            return True
        else:
            # either length is 0 or length is greater than 2 so it's a scalar or higher dimensional tensor
            return False

    def _scalar_and_tensor_parents(self, original_node:bn.BMGNode) -> typing.Optional[typing.Tuple[bn.BMGNode, bn.BMGNode]]:
        if isinstance(original_node, bn.MultiplicationNode):
            tensor_parent = None
            scalar_parent = None
            if len(original_node.inputs.inputs) > 2:
                return None
            for parent in original_node.inputs.inputs:
                if self._is_matrix(parent) and tensor_parent == None:
                    tensor_parent = parent
                elif scalar_parent == None:
                    scalar_parent = parent
            if not scalar_parent == None and not tensor_parent == None:
                return scalar_parent, tensor_parent
            return None

    # a node can be tensorized if all its parents satisfy the type requirements
    def can_be_tensorized(self, original_node:bn.BMGNode) -> bool:
        if isinstance(original_node, bn.MultiplicationNode):
            return not self._scalar_and_tensor_parents(original_node) == None

    def try_tensorize(self, original_node:bn.BMGNode) -> typing.Union[bool, bn.BMGNode]:
        if isinstance(original_node, bn.MultiplicationNode):
            parents = self._scalar_and_tensor_parents(original_node)
            if not parents == None:
                scalar_parent = parents[0]
                tensor_parent = parents[1]
                if self.clones.__contains__(scalar_parent) and self.clones.__contains__(tensor_parent):
                    scalar_parent = self.clones[scalar_parent]
                    tensor_parent = self.clones[tensor_parent]
                    return self.bmg.add_matrix_scale(scalar_parent, tensor_parent)
        return False

def _is_fixable_size(s: Size) -> bool:
    dim = len(s)
    if dim == 1:
        return s[0] > 1
    if dim == 2:
        return s[0] > 1 or s[1] > 1
    return False


def _needs_devectorize(node: bn.BMGNode, size: Size, cxt:BuilderContext) -> bool:
    is_eligible_for_devectorize = _is_fixable_size(size) and not _leaves.__contains__(type(node))
    if is_eligible_for_devectorize:
        # I think sample and observations can be devectorized, but they do not have to be
        if isinstance(node, bn.SampleNode) or isinstance(node, bn.Observation):
            if cxt.devectorized_nodes.__contains__(node.inputs.inputs[0]):
                return True

        # since the children of distribution nodes are devectorized depending on the parent,
        # do not check downstream nodes
        if isinstance(node, bn.DistributionNode):
            return not _consumes_tensor_types.__contains__(type(node))

        # this is for operators and tensor nodes, where there could
        # if all downstream consumers either accept tensors or can be tensorized, then no need to devectorize
        for consumer in node.outputs.items:
            if not _consumes_tensor_types.__contains__(type(consumer)) and not cxt.can_be_tensorized(consumer):
                return True
        return False
    else:
        return is_eligible_for_devectorize


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
        item_count = 1
        for i in range(0, len(self.size)):
            item_count *= self.size[i]
        assert item_count == len(elements)


def list_from_parents(
    size: Size, item_count: int, parents: [], creator: Callable
) -> List[bn.BMGNode]:
    return list_from_parents_with_index(
        size, item_count, parents, lambda i, s: creator(*s)
    )


def list_from_parents_with_index(
    size: Size, item_count: int, parents: [], creator: Callable
) -> List[bn.BMGNode]:
    elements: List[bn.BMGNode] = []
    broadcast: Dict[DevectorizedNode, Callable] = {}
    for parent in parents:
        if isinstance(parent, DevectorizedNode):
            broadbast_fnc_maybe = beanmachine.ppl.compiler.broadcast.broadcast_fnc(parent.size, size)
            if isinstance(broadbast_fnc_maybe, Callable):
                broadcast[parent] = broadbast_fnc_maybe
            else:
                raise ValueError(
                    f"The size {parent.size} cannot be broadcast to {size}"
                )

    for i in range(0, item_count):
        reduced_parents = []
        for parent in parents:
            if isinstance(parent, DevectorizedNode):
                new_index = broadcast[parent](i)
                reduced_parents.append(parent.elements[new_index])
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

# we're given a node that we want to devectori
def _clone(node: bn.BMGNode, size: Size, cxt: BuilderContext) -> bn.BMGNode:
    parents = _clone_parents(node, cxt)
    if isinstance(node, bn.SampleNode):
        dist = parents[0]
        assert isinstance(dist, bn.DistributionNode)
        new_node = cxt.bmg.add_sample(operand=dist)
    elif isinstance(node, bn.DistributionNode):
        new_node = cxt.dist_factories[type(node)](*parents)
    elif isinstance(node, bn.Query):
        new_node = cxt.bmg.add_query(parents[0])
        # todo: ask the previous version what the old one was first
        cxt.bmg.query_map[node] = new_node
    elif isinstance(node, bn.TensorNode):
        new_node = cxt.bmg.add_tensor(size, *parents)
    elif isinstance(node, bn.Observation):
        new_node = cxt.bmg.add_observation(parents[0], node.value)
    elif cxt.value_factories.__contains__(type(node)):
        new_node = cxt.value_factories[type(node)](node.value)
    elif cxt.node_factories.__contains__(type(node)):
        new_node = cxt.node_factories[type(node)](*parents)
    else:
        raise NotImplementedError(type(node))
    locations = cxt.original_context.node_locations(node)
    for site in locations:
        cxt.bmg.execution_context.record_node_call(new_node, site)
    return new_node


def split(node: bn.BMGNode, cxt: BuilderContext, size: Size) -> DevectorizedNode:
    item_count = 1
    for i in range(0, len(size)):
        item_count *= size[i]

    # identify parents
    parents = []
    for input in node.inputs.inputs:
        if cxt.devectorized_nodes.__contains__(input):
            devectorized_parent = cxt.devectorized_nodes[input]
            parents.append(devectorized_parent)
        elif cxt.clones.__contains__(input):
            parents.append(cxt.clones[input])
        else:
            raise ValueError("value not found in clone context")

    if isinstance(node, bn.SampleNode):
        new_nodes = list_from_parents(size, item_count, parents, cxt.bmg.add_sample)
        return DevectorizedNode(new_nodes, size)
    if _indexable_node_types.__contains__(type(node)):
        return DevectorizedNode(_node_to_index_list(cxt, size, node), size)
    if isinstance(node, bn.DistributionNode):
        return DevectorizedNode(
            list_from_parents(
                size, item_count, parents, cxt.dist_factories[type(node)]
            ),
            size,
        )
    if isinstance(node, bn.Query):
        return DevectorizedNode(
            list_from_parents(size, item_count, parents, cxt.bmg.add_query), size
        )
    if isinstance(node, bn.OperatorNode):
        return DevectorizedNode(
            list_from_parents(
                size, item_count, parents, cxt.node_factories[type(node)]
            ),
            size,
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
                size,
                item_count,
                parents,
                lambda i, s: cxt.bmg.add_observation(*s, values[i]),
            ),
            size,
        )
    else:
        raise NotImplementedError()


def vectorized_node_fixer(sizer: Sizer) -> GraphFixer:
    def _tensorize(bmg_old: BMGraphBuilder) -> GraphFixerResult:
        cxt = BuilderContext(bmg_old, sizer)
        tensorized_nodes_cnt = 0
        report = ErrorReport()
        for node in bmg_old.all_nodes():

            # check nodes that may cause exceptions. TODO: extract this out
            error = None
            # enable registering size sensitive verification checks for certain nodes
            # otherwise the sizer will be unable to determine a size which is necessary for
            if isinstance(node, bn.MatrixMultiplicationNode):
                lhs = node.inputs.inputs[0]
                rhs = node.inputs.inputs[1]

                lhs_size = sizer[node.inputs.inputs[0]]
                rhs_size = sizer[node.inputs.inputs[1]]

                if not (is_scalar(lhs_size) and is_scalar(rhs_size)):
                    if not (len(lhs_size) == 2 and len(rhs_size) == 2) or lhs_size[1] != rhs_size[0]:
                        typer = beanmachine.ppl.compiler.lattice_typer.LatticeTyper()
                        # type and correct the types. We only care about dimensions so if the
                        # typer cannot type it we just add dummy values for element types that
                        # are undecipherable
                        lt = typer[lhs]
                        if not isinstance(lt, BMGMatrixType):
                            lt = BMGMatrixType(Untypable, "", "", lhs_size[0], lhs_size[1])
                        rt = typer[rhs]
                        if not isinstance(rt, BMGMatrixType):
                            rt = BMGMatrixType(Untypable, "", "", rhs_size[0], rhs_size[1])
                        error = BadMatrixMultiplication(node, lt, rt, cxt.original_context.node_locations(node))
            if not error == None:
                report.add_error(error)
                return bmg_old, False, report

            if cxt.can_be_tensorized(node):
                tensorized_nodes_cnt = tensorized_nodes_cnt + 1
                node_maybe = cxt.try_tensorize(node)
                if isinstance(node_maybe, bn.BMGNode):
                    cxt.clones[node] = node_maybe
                else:
                    raise ValueError("expected a node but tensorize creation failed")
            else:
                cxt.clones[node] = _clone(node, sizer[node], cxt)
        if tensorized_nodes_cnt > 0:
            return cxt.bmg, True, report
        else:
            return bmg_old, False, report

    def _detensorize(bmg_old: BMGraphBuilder) -> GraphFixerResult:
        bmg_old, made_changes, report = _tensorize(bmg_old)
        if len(report.errors) > 0:
            return bmg_old, made_changes, report

        cxt = BuilderContext(bmg_old, sizer)
        report = ErrorReport()
        for node in bmg_old.all_nodes():
            if _needs_devectorize(node, sizer[node], cxt):
                cxt.devectorized_nodes[node] = split(node, cxt, sizer[node])
            else:
                cxt.clones[node] = _clone(node, sizer[node], cxt)
        split_nodes_cnt = len(cxt.devectorized_nodes)
        if split_nodes_cnt > 0:
            return cxt.bmg, True, report
        else:
            return bmg_old, False, report

    return _detensorize


# a graph fixer is a callable that accepts a list and returns a Tuple[bool, ErrorReport]
def vectorized_graph_fixer() -> GraphFixer:
    sizer = Sizer()
    return vectorized_node_fixer(sizer)