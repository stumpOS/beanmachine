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
import beanmachine.ppl.compiler.copy_and_replace
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
from beanmachine.ppl.compiler.copy_and_replace import NodeTransformer, copy_and_replace, TransformAssessment

class Devectorizer(NodeTransformer):
    def assess_node(self, node:bn.BMGNode) -> TransformAssessment:
        raise NotImplementedError("this is an abstract base class")

    # a node is either replaced 1-1, 1-many, or deleted
    def transform_node(self, node:bn.BMGNode, new_inputs:List[bn.BMGNode]) -> typing.Union[bn.BMGNode, List[bn.BMGNode], None]:
        raise NotImplementedError("this is an abstract base class")



def vectorized_graph_fixer() -> GraphFixer:
    def _tensorize(bmg_old:BMGraphBuilder) -> GraphFixerResult:
        bmg, errors = copy_and_replace(bmg_old, Devectorizer())
        return bmg, True, errors

    def _detensorize(bmg_old:BMGraphBuilder) -> GraphFixerResult:
        bmg, errors = copy_and_replace(bmg_old, Devectorizer())
        return bmg, True, errors

    return sequential_graph_fixer([_tensorize, _detensorize])