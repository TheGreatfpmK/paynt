"""SynthesizerSMPMC: SMT-based synthesis over colored MDPs (arXiv:2511.08078).

Unlike AR/CEGIS/Hybrid, this engine does not drive its own search loop and call Z3 as a subordinate
oracle -- Z3's own CDCL search drives the whole process, calling back into PAYNT/Storm (via
paynt/synthesizer/smpmc/theory.py's theory-solver plugin) whenever it needs to know if a partial
parameter assignment is still viable. synthesize_one therefore owns its entire search internally (one
Z3 solver session) rather than participating in PAYNT's AR-style node-splitting.

Scope so far: a single property (a plain threshold constraint, or a single optimality objective -- not
both together, and not multiple constraints -- see the plan), one constraint at a time selected via
--constraint (paynt.parameter_space.constraints, shared with CEGIS), no robust (forall-quantified)
parameters yet.
"""

from __future__ import annotations

from typing import Any

import z3

import paynt.colored_mdp
import paynt.parameter_space.constraints
import paynt.parameter_space.parameter_space
import paynt.specification.property
import paynt.synthesizer.search_node
import paynt.synthesizer.smpmc.checker
import paynt.synthesizer.smpmc.encoding
import paynt.synthesizer.smpmc.theory
import paynt.synthesizer.synthesizer
import paynt.task

import logging

logger = logging.getLogger(__name__)


class SynthesizerSMPMC(paynt.synthesizer.synthesizer.Synthesizer):
    def __init__(self, colored_mdp: paynt.colored_mdp.ColoredMdp, task: paynt.task.SynthesisTask):
        super().__init__(colored_mdp, task)
        spec = task.specification
        assert spec.num_properties == 1, (
            "SMPMC currently supports a single property: either one plain threshold constraint or one "
            "optimality objective, not both together and not several constraints at once"
        )
        self.prop: paynt.specification.property.Property = spec.optimality if spec.has_optimality else spec.constraints[0]
        # unlike CEGIS, SMPMC always needs *some* constraint -- it's what supplies the viable(...) clause
        # the theory-solver integration depends on -- so it defaults to "exists" rather than to no
        # constraint at all
        self.constraint = paynt.parameter_space.constraints.build_constraint(task.constraint_name or "exists")

    @property
    def method_name(self) -> str:
        return "SMPMC"

    def synthesize_one(
        self, node: paynt.synthesizer.search_node.SearchNode
    ) -> paynt.parameter_space.parameter_space.ParameterSpace | None:
        encoding = paynt.synthesizer.smpmc.encoding.ParameterBitVecEncoding(node.parameter_space)
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(self.colored_mdp, self.prop, self.stat)

        sorts = [variable.sort() for variable in encoding.variables]
        viable = z3.PropagateFunction("viable", *sorts, z3.BoolSort())

        solver = z3.Solver()
        # keep the propagator alive for as long as solver is used -- it owns the native callback
        # registration, so letting Python garbage-collect it mid-search is a use-after-free
        propagator = paynt.synthesizer.smpmc.theory.SmpmcPropagator(solver, None, theory, encoding.name_to_parameter)

        # Registering the propagator before this solver.add() call matters -- it's what makes `created`
        # fire correctly for the viable(...) term as it's asserted.
        ctx = paynt.parameter_space.constraints.ConstraintContext(
            colored_mdp=self.colored_mdp, parameter_space=node.parameter_space, task=self.task,
            variables=encoding.variables, width=encoding.width, viable=viable,
        )
        solver.add(*self.constraint.build(ctx))

        is_optimality = self.task.specification.has_optimality
        while not self.resource_limit_reached():
            result = solver.check()
            if result == z3.unsat:
                break
            if result == z3.unknown:
                logger.warning(f"SMPMC returned unknown: {solver.reason_unknown()}")
                break

            assignment = encoding.extract_assignment(solver.model(), node.parameter_space)

            if not is_optimality:
                # plain threshold constraint: the first satisfying assignment is the answer, matching
                # AR/CEGIS/OneByOne's convention of not reporting a value for a non-optimality spec
                self.best_assignment = assignment
                break

            value = self._value_of(assignment)
            if not self.prop.improves_optimum(value):
                # Does happen in practice, not just a theoretical edge case: the theory's own value for a
                # witness (computed while Z3 was still exploring) can disagree with this exact recheck by
                # an amount that straddles model_checking_precision, letting a non-improving candidate
                # through as apparently-viable. That does not mean no better assignment exists elsewhere --
                # only solver.check() returning unsat is a genuine proof of that -- so this witness is
                # excluded like any other explored point and the search continues, exactly as it does
                # below after a genuine improvement (see also checker.py's singleton-eta shortcut, which
                # eliminates the specific solver-disagreement this guards against for future assignments,
                # but this remains the correct response to a non-improving witness regardless of cause).
                solver.add(encoding.exclude(assignment))
                continue
            self.best_assignment = assignment
            self.best_assignment_value = value
            self.prop.update_optimum(value)
            # the threshold just tightened: previously-"inconclusive" verdicts may no longer hold (a
            # refutation, by contrast, only gets more valid as the threshold tightens) -- see
            # PartialModelCache's docstring
            theory.epoch += 1
            # force Z3 to look elsewhere: without this, re-checking would trivially return the same model,
            # since nothing at the Z3/Boolean level has changed -- only self.prop's threshold has, and the
            # theory solver only revisits a literal when asked to
            solver.add(encoding.exclude(assignment))

        self.explored = node.parameter_space.size
        return self.best_assignment

    def _value_of(self, assignment: paynt.parameter_space.parameter_space.ParameterSpace) -> Any:
        model = self.colored_mdp.build_assignment(assignment)
        return model.model_check_property(self.prop).value
