# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_types import BMGElementType, BMGMatrixType
from beanmachine.ppl.compiler.error_report import BadMatrixMultiplication, BMGError
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper
from beanmachine.ppl.compiler.sizer import is_scalar, Sizer

untypeable_element = BMGElementType("U", "untyped")


class SizeAssessment:
    def __init__(self, sizer: Sizer):
        self.sizer = sizer

    def size_error(
        self, node: bn.BMGNode, context: BMGraphBuilder
    ) -> Optional[BMGError]:
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
                # todo: consider case where the rhs is a row or the lhs is a column
                # todo: make more general functions to improve readability
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
                    typer = LatticeTyper()
                    # type and correct the types. We only care about dimensions so if the
                    # typer cannot type it we just add dummy values for element types that
                    # are undecipherable
                    lt = typer[lhs]
                    if not isinstance(lt, BMGMatrixType):
                        lt = BMGMatrixType(
                            untypeable_element, "", "", lhs_size[0], lhs_size[1]
                        )
                    rt = typer[rhs]
                    if not isinstance(rt, BMGMatrixType):
                        rt = BMGMatrixType(
                            untypeable_element, "", "", rhs_size[0], rhs_size[1]
                        )
                    error = BadMatrixMultiplication(
                        node, lt, rt, context.execution_context.node_locations(node)
                    )

        return error
