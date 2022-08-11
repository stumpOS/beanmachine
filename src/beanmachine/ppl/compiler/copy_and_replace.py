# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import typing
from typing import Callable, Dict, List, Type
from enum import Enum

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
from beanmachine.ppl.compiler.sizer import is_scalar, Sizer, Unsized
class TransformAssessment:
    def __init__(self, needs_transform:bool, errors:ErrorReport):
        self.node_needs_transform = needs_transform
        self.error_report = errors



class NodeTransformer:
    def assess_node(self, node:bn.BMGNode) -> TransformAssessment:
        raise NotImplementedError("this is an abstract base class")

    # a node is either replaced 1-1, 1-many, or deleted
    def transform_node(self, node:bn.BMGNode, new_inputs:List[bn.BMGNode]) -> typing.Union[bn.BMGNode, List[bn.BMGNode], None]:
        raise NotImplementedError("this is an abstract base class")

def _node_factories(bmg: BMGraphBuilder) -> Dict[Type, Callable]:
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
        bn.ExpProductFactorNode: bmg.add_exp_product,
        bn.GreaterThanNode: bmg.add_greater_than,
        bn.GreaterThanEqualNode: bmg.add_greater_than_equal,
        bn.IfThenElseNode: bmg.add_if_then_else,
        bn.IsNode: bmg.add_is,
        bn.IsNotNode: bmg.add_is_not,
        bn.ItemNode: bmg.add_item,
        bn.IndexNode: bmg.add_index,
        bn.InNode: bmg.add_in,
        bn.InvertNode: bmg.add_invert,
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
        bn.LShiftNode: bmg.add_lshift,
        bn.MatrixMultiplicationNode: bmg.add_matrix_multiplication,
        bn.MatrixScaleNode: bmg.add_matrix_scale,
        bn.ModNode: bmg.add_mod,
        bn.MultiplicationNode: bmg.add_multiplication,
        bn.NegateNode: bmg.add_negate,
        bn.NotEqualNode: bmg.add_not_equal,
        bn.NotNode: bmg.add_not,
        bn.NotInNode: bmg.add_not_in,
        bn.PhiNode: bmg.add_phi,
        bn.PowerNode: bmg.add_power,
        bn.Query: bmg.add_query,
        bn.RShiftNode: bmg.add_rshift,
        bn.SampleNode: bmg.add_sample,
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

def flatten(inputs:List[typing.Union[bn.BMGNode, List[bn.BMGNode], None]]) -> List[bn.BMGNode]:
    parents = []
    for input in inputs:
        if input == None:
            continue
        if isinstance(input, List):
            for i in input:
                parents.append(i)
        else:
            parents.append(input)
    return parents

def copy_and_replace(bmg_original:BMGraphBuilder, transformer:NodeTransformer) -> typing.Tuple[BMGraphBuilder, ErrorReport]:
    original_context = bmg_original.execution_context
    bmg = BMGraphBuilder()
    sizer = Sizer()
    node_factories = _node_factories(bmg)
    value_factories = _constant_factories(bmg)

    copies = {}
    for index, original in enumerate(bmg.all_nodes()):
        inputs = []
        for c in original.inputs.inputs:
            inputs.append(copies[c])
        assessment = transformer.assess_node(original)
        if len(assessment.error_report.errors) > 0:
            return bmg, assessment.error_report
        elif assessment.node_needs_transform:
            image = transformer.transform_node(original, inputs)
            copies[original] = image
        else:
            parents = flatten(inputs)
            if value_factories.__contains__(type(original)):
                image = node_factories[type(original)](original.value)
            if isinstance(original, bn.Query):
                assert len(parents) == 1
                image = bmg.add_query(parents[0])
                key = original
                for k, v in bmg_original.query_map.items():
                    if v == original:
                        key = k
                        break
                bmg.query_map[key] = image
            elif isinstance(original, bn.Observation):
                assert len(parents) == 1
                image = bmg.add_observation(parents[0], original.value)
            elif isinstance(original, bn.TensorNode):
                image = bmg.add_tensor(sizer[original], *parents)
            else:
                image = node_factories[type(original)](*parents)

            copies[original] = image
            locations = original_context.node_locations(original)
            for site in locations:
                bmg.execution_context.record_node_call(image, site)



