import typing
from typing import List

import beanmachine.ppl.compiler.bmg_nodes as bn
import beanmachine.ppl.compiler.bmg_types
import beanmachine.ppl.compiler.broadcast
import beanmachine.ppl.compiler.copy_and_replace
import beanmachine.ppl.compiler.execution_context
import beanmachine.ppl.compiler.lattice_typer
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_types import BMGMatrixType, Untypable
from beanmachine.ppl.compiler.copy_and_replace import (
    Cloner,
    NodeTransformer,
    TransformAssessment,
)
from beanmachine.ppl.compiler.error_report import BadMatrixMultiplication, ErrorReport
from beanmachine.ppl.compiler.sizer import is_scalar, Sizer, Unsized


class Tensorizer(NodeTransformer):
    def __init__(self, cloner: Cloner, sizer: Sizer):
        self.cloner = cloner
        self.sizer = sizer

    def _is_matrix(self, node: bn.BMGNode) -> bool:
        size = self.sizer[node]
        length = len(size)
        if length == 1:
            return size[0] > 1
        elif length == 2:
            return True
        else:
            # either length is 0 or length is greater than 2 so it's a scalar or higher dimensional tensor
            return False

    def _scalar_and_tensor_parents(
        self, original_node: bn.BMGNode
    ) -> typing.Optional[typing.Tuple[bn.BMGNode, bn.BMGNode]]:
        if isinstance(original_node, bn.MultiplicationNode):
            tensor_parent = None
            scalar_parent = None
            if len(original_node.inputs.inputs) != 2:
                return None
            for parent in original_node.inputs.inputs:
                if self._is_matrix(parent):
                    if tensor_parent is None:
                        tensor_parent = parent
                elif scalar_parent is None:
                    scalar_parent = parent
            if scalar_parent is not None and tensor_parent is not None:
                return scalar_parent, tensor_parent
            return None

    # a node can be tensorized if all its parents satisfy the type requirements
    def can_be_tensorized(self, original_node: bn.BMGNode) -> bool:
        if isinstance(original_node, bn.MultiplicationNode):
            return not self._scalar_and_tensor_parents(original_node) is None
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
                        lt = BMGMatrixType(Untypable, "", "", lhs_size[0], lhs_size[1])
                    rt = typer[rhs]
                    if not isinstance(rt, BMGMatrixType):
                        rt = BMGMatrixType(Untypable, "", "", rhs_size[0], rhs_size[1])
                    error = BadMatrixMultiplication(
                        node, lt, rt, original.execution_context.node_locations(node)
                    )
        if error is not None:
            report.add_error(error)
        return TransformAssessment(self.can_be_tensorized(node), report)

    # a node is either replaced 1-1, 1-many, or deleted
    def transform_node(
        self, node: bn.BMGNode, new_inputs: List[bn.BMGNode]
    ) -> typing.Union[bn.BMGNode, List[bn.BMGNode], None]:
        if isinstance(node, bn.MultiplicationNode):
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
            elif is_scalar(lhs_sz):
                scalar_parent_image = new_inputs[0]
                tensor_parent_image = new_inputs[1]
                assert not is_scalar(rhs_sz)
            else:
                tensor_parent_image = new_inputs[0]
                scalar_parent_image = new_inputs[1]
                assert is_scalar(rhs_sz)
            return self.cloner.bmg.add_matrix_scale(
                scalar_parent_image, tensor_parent_image
            )
        else:
            return self.cloner.clone(node, new_inputs)
