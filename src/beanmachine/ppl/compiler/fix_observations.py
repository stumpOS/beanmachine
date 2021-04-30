# Copyright (c) Facebook, Inc. and its affiliates.

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_types import (
    Boolean,
    Natural,
    PositiveReal,
    Probability,
    Real,
    supremum,
    type_of_value,
)
from beanmachine.ppl.compiler.error_report import ErrorReport, ImpossibleObservation
from beanmachine.ppl.compiler.typer_base import TyperBase


class ObservationsFixer:
    """This class takes a Bean Machine Graph builder and attempts to
    fix violations of BMG type system requirements in observation nodes.
    It also finds observations that are impossible -- say, an observation
    that a Boolean node is -3.14 -- and reports them as errors."""

    errors: ErrorReport
    bmg: BMGraphBuilder
    _typer: TyperBase

    def __init__(self, bmg: BMGraphBuilder, typer: TyperBase) -> None:
        # The typer is not actually needed but the caller assumes that
        # all problem fixers need a typer to either get types or propagate
        # updates. This fixer does neither, since it only works on leaf nodes
        # and types of values.
        self.errors = ErrorReport()
        self.bmg = bmg
        self._typer = typer

    def fix_problems(self) -> None:
        for o in self.bmg.all_observations():
            v = o.value
            inf = type_of_value(v)
            gt = o.observed.graph_type
            if supremum(inf, gt) != gt:
                self.errors.add_error(ImpossibleObservation(o))
            elif gt == Boolean:
                o.value = bool(v)
            elif gt == Natural:
                o.value = int(v)
            elif gt in {Probability, PositiveReal, Real}:
                o.value = float(v)
            else:
                # TODO: How should we deal with observations of
                # TODO: matrix-valued samples?
                pass
            # TODO: Handle the case where there are two inconsistent
            # TODO: observations of the same sample
