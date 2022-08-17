# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing
from typing import List, Callable

import beanmachine.ppl.compiler.bmg_nodes as bn
import beanmachine.ppl.compiler.bmg_types
import beanmachine.ppl.compiler.broadcast
import beanmachine.ppl.compiler.copy_and_replace
import beanmachine.ppl.compiler.execution_context
import beanmachine.ppl.compiler.lattice_typer
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_types import BMGMatrixType, unknown_number
from beanmachine.ppl.compiler.copy_and_replace import (
    Cloner,
    NodeTransformer,
    TransformAssessment,
)
from beanmachine.ppl.compiler.error_report import BadMatrixMultiplication, ErrorReport
from beanmachine.ppl.compiler.sizer import is_scalar, Sizer, Unsized
from enum import Enum

class ElementType(Enum):
    TENSOR = 1
    SCALAR = 2
    MATRIX = 3
    UNKNOWN = 4


class Tensorizer(NodeTransformer):
    def __init__(self, cloner: Cloner, sizer: Sizer):
        self.cloner = cloner
        self.sizer = sizer
        self.transform_cache = {}
        # TODO: remove functions and check/clone in impl
        self.can_be_transformed_map = {
            bn.AdditionNode: lambda node: True,
            bn.MultiplicationNode: self.mult_can_be_tensorized,
            bn.DivisionNode: self.div_can_be_tensorized,
            bn.ExpNode: lambda node: True,
            bn.SumNode: lambda node: True,
        }
        self.transform_map = {
            bn.AdditionNode: lambda node, inputs: self._tensorize_binary_elementwise(node, inputs, self.cloner.bmg.add_elementwise_addition),
            bn.MultiplicationNode: self._tensorize_multiply,
            bn.DivisionNode: self._tensorize_div,
            bn.ExpNode: lambda node, inputs: self._tensorize_unary_elementwise(node, inputs, self.cloner.bmg.add_matrix_exp),
            bn.SumNode: self._tensorize_sum
        }

    def _tensorize_div(self, node: bn.DivisionNode, new_inputs:List[bn.BMGNode]) -> bn.BMGNode:
        assert len(node.inputs.inputs) == 2
        tensor_input = new_inputs[0]
        scalar_input = new_inputs[1]
        if self._element_type(tensor_input) is not ElementType.MATRIX:
            raise ValueError("Expected a matrix as first operand")
        if self._element_type(scalar_input) is not ElementType.SCALAR:
            raise ValueError("Expected a scalar as second operand")
        one = self.cloner.bmg.add_pos_real(1.0)
        new_scalar = self.cloner.bmg.add_division(one, scalar_input)
        return self.cloner.bmg.add_matrix_scale(new_scalar, tensor_input)

    def _tensorize_sum(self, node:bn.SumNode, new_inputs:List[bn.BMGNode]) -> bn.BMGNode:
        assert len(new_inputs) >= 1
        # note that scalars can be broadcast
        if any(self._element_type(operand) == ElementType.MATRIX for operand in node.inputs.inputs):
            current = new_inputs[0]
            for i in range(1, len(new_inputs)):
                current = self.cloner.bmg.add_elementwise_addition(current, new_inputs[i])
            return self.cloner.bmg.add_matrix_sum(current)
        return self.cloner.bmg.add_sum(*new_inputs)


    def _tensorize_multiply(self, node: bn.MultiplicationNode, new_inputs:List[bn.BMGNode]) -> bn.BMGNode:
        if len(new_inputs) != 2:
            raise ValueError(
                "Cannot transform a mult into a tensor mult because there are not two operands"
            )
        lhs_sz = self.sizer[new_inputs[0]]
        rhs_sz = self.sizer[new_inputs[1]]
        if lhs_sz == Unsized or rhs_sz == Unsized:
            raise ValueError(
                f"cannot multiply an unsized quantity. Operands: {new_inputs[0]} and {new_inputs[1]}"
            )
        lhs_is_scalar = is_scalar(lhs_sz)
        rhs_is_scalar = is_scalar(rhs_sz)
        if not lhs_is_scalar and not rhs_is_scalar:
            return self.cloner.bmg.add_elementwise_multiplication(new_inputs[0], new_inputs[1])
        if lhs_is_scalar and not rhs_is_scalar:
            scalar_parent_image = new_inputs[0]
            tensor_parent_image = new_inputs[1]
            assert not is_scalar(rhs_sz)
        elif rhs_is_scalar and not lhs_is_scalar:
            tensor_parent_image = new_inputs[0]
            scalar_parent_image = new_inputs[1]
            assert is_scalar(rhs_sz)
        else:
            return self.cloner.bmg.add_multiplication(new_inputs[0], new_inputs[1])
        return self.cloner.bmg.add_matrix_scale(
            scalar_parent_image, tensor_parent_image
        )

    def _tensorize_unary_elementwise(self, node: bn.UnaryOperatorNode, new_inputs:List[bn.BMGNode], creator:Callable) -> bn.BMGNode:
        assert len(new_inputs) == 1
        if self._element_type(new_inputs[0]) == ElementType.MATRIX:
            return creator(new_inputs[0])
        else:
            return self.cloner.clone(node, new_inputs)

    def _tensorize_binary_elementwise(self, node: bn.BinaryOperatorNode, new_inputs:List[bn.BMGNode], creator:Callable) -> bn.BMGNode:
        assert len(new_inputs) == 2
        if self._element_type(new_inputs[0]) == ElementType.MATRIX and self._element_type(new_inputs[1]) == ElementType.MATRIX:
            return creator(new_inputs[0], new_inputs[1])
        else:
            return self.cloner.clone(node, new_inputs)

    def _element_type(self, node: bn.BMGNode) -> ElementType:
        size = self.sizer[node]
        if size == Unsized:
            return ElementType.UNKNOWN
        length = len(size)
        if length == 0 or is_scalar(size):
            return ElementType.SCALAR
        if length == 1 and size[0] > 1:
            return ElementType.MATRIX
        elif length == 2:
            return ElementType.MATRIX
        else:
            return ElementType.TENSOR

    def div_can_be_tensorized(self, node:bn.DivisionNode) -> bool:
        if len(node.inputs.inputs) == 2:
            return self._element_type(node.inputs.inputs[0]) == ElementType.MATRIX and self._element_type(node.inputs.inputs[1]) == ElementType.SCALAR
        return False

    def mult_can_be_tensorized(self, node:bn.MultiplicationNode) -> bool:
        return len(node.inputs.inputs) == 2

    # a node can be tensorized if all its parents satisfy the type requirements
    def can_be_tensorized(self, original_node: bn.BMGNode) -> bool:
        if self.can_be_transformed_map.__contains__(type(original_node)):
            return self.can_be_transformed_map[type(original_node)](original_node)
        else:
            return False

    def assess_node(
        self, node: bn.BMGNode, original: BMGraphBuilder
    ) -> TransformAssessment:
        report = ErrorReport()
        error = None
        # enable registering size sensitive verification checks for certain nodes
        # otherwise the sizer will be unable to determine a size which is necessary for tensorization
        if isinstance(node, bn.MatrixMultiplicationNode):
            lhs = node.inputs.inputs[0]
            rhs = node.inputs.inputs[1]

            lhs_size = self.sizer[node.inputs.inputs[0]]
            rhs_size = self.sizer[node.inputs.inputs[1]]

            if not (is_scalar(lhs_size) or is_scalar(rhs_size)):
                l_rhs = len(rhs_size)
                l_lhs = len(lhs_size)
                rhs_can_be_considered_column = (
                    l_rhs == 1 and l_lhs == 2 and lhs_size[1] == rhs_size[0]
                )
                lhs_can_be_considered_row = (
                    l_lhs == 1 and l_rhs == 2 and lhs_size[0] == rhs_size[0]
                )
                can_be_inner_product = (
                    l_rhs == 1 and l_lhs == 1 and rhs_size[0] == lhs_size[0]
                )
                are_not_matrices_or_not_compatible_matrices = (
                    not (len(lhs_size) == 2 and l_rhs == 2)
                ) or (lhs_size[1] != rhs_size[0])
                if are_not_matrices_or_not_compatible_matrices and not (
                    rhs_can_be_considered_column
                    or lhs_can_be_considered_row
                    or can_be_inner_product
                ):
                    typer = beanmachine.ppl.compiler.lattice_typer.LatticeTyper()
                    # type and correct the types. We only care about dimensions so if the
                    # typer cannot type it we just add dummy values for element types that
                    # are undecipherable
                    lt = typer[lhs]
                    if not isinstance(lt, BMGMatrixType):
                        lt = BMGMatrixType(
                            unknown_number, "", "", lhs_size[0], lhs_size[1]
                        )
                    rt = typer[rhs]
                    if not isinstance(rt, BMGMatrixType):
                        rt = BMGMatrixType(
                            unknown_number, "", "", rhs_size[0], rhs_size[1]
                        )
                    error = BadMatrixMultiplication(
                        node, lt, rt, original.execution_context.node_locations(node)
                    )
        if error is not None:
            report.add_error(error)
        return TransformAssessment(self.can_be_tensorized(node), report)

    # a node is either replaced 1-1, 1-many, or deleted
    def transform_node(
        self, node: bn.BMGNode, new_inputs: List[bn.BMGNode]
    ) -> typing.Optional[typing.Union[bn.BMGNode, List[bn.BMGNode]]]:
        if self.transform_map.__contains__(type(node)):
            return self.transform_map[type(node)](node, new_inputs)
        else:
            return self.cloner.clone(node, new_inputs)
