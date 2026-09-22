"""ParameterBitVecEncoding: the BitVec variables and domain (tau_V) formula for a parameter space, used
by SynthesizerSMPMC.synthesize_one for every synthesis run."""

from __future__ import annotations

from typing import Any

import z3

import paynt.parameter_space.bitvec
import paynt.parameter_space.parameter_space

import logging

logger = logging.getLogger(__name__)


class ParameterBitVecEncoding:
    def __init__(self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace):
        self.parameter_space = parameter_space
        self.variables, self.width, self.name_to_parameter = paynt.parameter_space.bitvec.parameter_bitvec_variables(parameter_space)

    def value(self, option: int) -> Any:
        return z3.BitVecVal(option, self.width)

    def in_range(self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace | None = None) -> Any:
        """tau_V's domain conjunct: every parameter constrained to one of its currently-allowed options.
        Defaults to self.parameter_space, but accepts a narrower one (e.g. a split-off child subspace)."""
        if parameter_space is None:
            parameter_space = self.parameter_space
        clauses = []
        for parameter in range(parameter_space.num_parameters):
            var = self.variables[parameter]
            option_clauses = [var == self.value(option) for option in parameter_space.parameter_options(parameter)]
            clauses.append(option_clauses[0] if len(option_clauses) == 1 else z3.Or(option_clauses))
        return clauses[0] if len(clauses) == 1 else z3.And(clauses)

    def exclude(self, assignment: paynt.parameter_space.parameter_space.ParameterSpace) -> Any:
        """A clause ruling out exactly this one concrete assignment -- used by the optimality loop to
        force Z3 to look elsewhere after recording assignment as the new best, since nothing else about
        the Z3-level formula changes when only the theory's threshold tightens."""
        assert assignment.size == 1, "exclude() expects a single concrete assignment"
        equalities = [
            self.variables[parameter] == self.value(assignment.parameter_options(parameter)[0])
            for parameter in range(assignment.num_parameters)
        ]
        conjunction = equalities[0] if len(equalities) == 1 else z3.And(equalities)
        return z3.Not(conjunction)

    def extract_assignment(
        self, model: Any, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace
    ) -> paynt.parameter_space.parameter_space.ParameterSpace:
        parameter_options = []
        for parameter in range(parameter_space.num_parameters):
            option = model.eval(self.variables[parameter], model_completion=True).as_long()
            parameter_options.append([option])
        return parameter_space.assume_options_copy(parameter_options)
