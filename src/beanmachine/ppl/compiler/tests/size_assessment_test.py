# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.error_report import BadMatrixMultiplication
from beanmachine.ppl.compiler.size_assessment import SizeAssessment

from beanmachine.ppl.compiler.sizer import Size, Sizer


class SizeAssessmentTests(unittest.TestCase):
    def test_matrix_mult(self):
        bmg = BMGraphBuilder()
        assessor = SizeAssessment(Sizer())
        probs = bmg.add_real_matrix(torch.tensor([[0.75, 0.25], [0.125, 0.875]]))
        tensor_elements = []
        for row in range(0, 2):
            row_node = bmg.add_natural(row)
            row_prob = bmg.add_column_index(probs, row_node)
            for column in range(0, 2):
                col_index = bmg.add_natural(column)
                prob = bmg.add_vector_index(row_prob, col_index)
                bernoulli = bmg.add_bernoulli(prob)
                sample = bmg.add_sample(bernoulli)
                tensor_elements.append(sample)
        matrix2by2 = bmg.add_tensor(Size([2, 2]), *tensor_elements)
        # invalid
        matrix3by3 = bmg.add_real_matrix(
            torch.tensor([[0.21, 0.27, 0.3], [0.5, 0.6, 0.1], [0.8, 0.6, 0.9]])
        )
        mm_invalid = bmg.add_matrix_multiplication(matrix2by2, matrix3by3)

        # can be broadcast
        matrix1by3 = bmg.add_real_matrix(torch.tensor([[0.1, 0.2, 0.3]]))
        matrix2 = bmg.add_real_matrix(torch.tensor([0.1, 0.2]))
        scalar = bmg.add_real(4.5)

        error_size_mismatch = assessor.size_error(mm_invalid, bmg)
        self.assertIsInstance(error_size_mismatch, BadMatrixMultiplication)
        expectation = """
The model uses a matrix multiplication (@) operation unsupported by Bean Machine Graph.
The dimensions of the operands are 2x2 and 3x3.
        """
        self.assertEqual(expectation.strip(), error_size_mismatch.__str__().strip())
        errors = [
            assessor.size_error(bmg.add_matrix_multiplication(matrix2by2, mm), bmg)
            for mm in [matrix1by3, matrix2, scalar]
        ]
        for error in errors:
            self.assertIsNone(error)
