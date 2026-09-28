"""SynthesizerSMPMC: SMT-based synthesis over colored MDPs (arXiv:2511.08078).

Unlike AR/CEGIS/Hybrid, this engine does not drive its own search loop and call Z3 as a subordinate
oracle -- Z3's own CDCL search drives the whole process, calling back into PAYNT/Storm (via
paynt/synthesizer/smpmc/theory.py's theory-solver plugin) whenever it needs to know if a partial
parameter assignment is still viable. synthesize_one therefore owns its entire search internally (one
Z3 solver session) rather than participating in PAYNT's AR-style node-splitting.

Scope so far: a single property (a plain threshold constraint, or a single optimality objective -- not
both together, and not multiple constraints -- see the plan), one constraint at a time selected via
--constraint (paynt.parameter_space.constraints, shared with CEGIS). Robust synthesis (--constraint
exists_forall, Section 4.4) quantifies the parameters selected by --smpmc-forall universally; on an MDP family,
the policy is made explicit as parameters (see _augment.py) and the family's own parameters are the default
universally quantified environment. An optimality objective then optimizes the worst case over those.
"""

from __future__ import annotations

import re
from typing import Any

import z3

import paynt.colored_mdp
import paynt.parameter_space.constraints
import paynt.parameter_space.parameter_space
import paynt.specification.property
import paynt.synthesizer.search_node
import paynt.synthesizer.smpmc._augment
import paynt.synthesizer.smpmc._utils
import paynt.synthesizer.smpmc.checker
import paynt.synthesizer.smpmc.encoding
import paynt.synthesizer.smpmc.result
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
        self.prop: paynt.specification.property.Property = spec.optimality if spec.optimality is not None else spec.constraints[0]
        # unlike CEGIS, SMPMC always needs *some* constraint -- it's what supplies the viable(...) clause
        # the theory-solver integration depends on -- so it defaults to "exists" rather than to no
        # constraint at all
        self.constraint = paynt.parameter_space.constraints.build_constraint(task.constraint_name or "exists")

        # robust synthesis: the universally quantified parameters, and the robust space found by the last run
        self.forall_parameters: list[int] = []
        self.robust_assignment: paynt.parameter_space.parameter_space.ParameterSpace | None = None
        if self.constraint.name == "exists_forall":
            self._init_forall_parameters()

    def _init_forall_parameters(self) -> None:
        if self.colored_mdp.feature_kind == "family":
            # the family's parameters are the environment; making the policy explicit lets it be quantified existentially
            self.colored_mdp, self.forall_parameters = paynt.synthesizer.smpmc._augment.add_policy_parameters(self.colored_mdp)
        parameter_space = self.colored_mdp.parameter_space
        if self.task.forall_pattern is not None:
            pattern = re.compile(self.task.forall_pattern)
            self.forall_parameters = [
                parameter for parameter in range(parameter_space.num_parameters) if pattern.search(parameter_space.parameter_name(parameter))
            ]
            if not self.forall_parameters:
                raise ValueError(f"--smpmc-forall {self.task.forall_pattern!r} matches no parameter")
        elif not self.forall_parameters:
            raise ValueError("--constraint exists_forall needs --smpmc-forall to select the universally quantified parameters")
        forall_names = [parameter_space.parameter_name(parameter) for parameter in self.forall_parameters]
        num_existential = parameter_space.num_parameters - len(self.forall_parameters)
        logger.info(f"robust synthesis: {num_existential} existential parameters, universally quantified: {forall_names}")

    @property
    def method_name(self) -> str:
        return "SMPMC"

    def synthesize_one(self, node: paynt.synthesizer.search_node.SearchNode) -> paynt.parameter_space.parameter_space.ParameterSpace | None:
        encoding = paynt.synthesizer.smpmc.encoding.ParameterBitVecEncoding(node.parameter_space)
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(self.colored_mdp, self.prop, self.stat, self.forall_parameters)

        sorts = [variable.sort() for variable in encoding.variables]
        viable = z3.PropagateFunction("viable", *sorts, z3.BoolSort())
        ctx = paynt.parameter_space.constraints.ConstraintContext(
            colored_mdp=self.colored_mdp,
            parameter_space=node.parameter_space,
            task=self.task,
            variables=encoding.variables,
            width=encoding.width,
            viable=viable,
            forall_parameters=self.forall_parameters,
        )
        clauses = self.constraint.build(ctx)
        # the parameters a witness fixes: all of them, or the policy of a robust search
        policy_parameters = [parameter for parameter in range(node.parameter_space.num_parameters) if parameter not in self.forall_parameters]
        # clauses ruling out witnesses whose value is known, see below
        exclusions: list[Any] = []

        def new_solver() -> tuple[Any, Any]:
            solver = z3.Solver()
            # Registering the propagator before adding the clauses matters -- it's what makes `created` fire correctly for the viable(...) terms as they're
            # asserted. Keep the propagator alive for as long as solver is used -- it owns the native callback registration, so letting Python
            # garbage-collect it mid-search is a use-after-free.
            propagator = paynt.synthesizer.smpmc.theory.SmpmcPropagator(solver, None, theory, encoding.name_to_parameter)
            solver.add(*clauses, *exclusions)
            return solver, propagator

        solver, _propagator = new_solver()
        optimality = self.task.specification.optimality
        while not self.resource_limit_reached():
            result = solver.check()
            if result == z3.unsat:
                break
            if result == z3.unknown:
                logger.warning(f"SMPMC returned unknown: {solver.reason_unknown()}")
                break

            assignment = encoding.extract_assignment(solver.model(), node.parameter_space, self.forall_parameters)

            if optimality is None:
                # plain threshold constraint: the first satisfying assignment is the answer, matching
                # AR/CEGIS/OneByOne's convention of not reporting a value for a non-optimality spec
                if self.forall_parameters:
                    self.robust_assignment = assignment
                self.best_assignment = assignment
                break

            if self.forall_parameters:
                value, exact = self._worst_case_of(theory, assignment, policy_parameters, optimality)
            else:
                value, exact = self._value_of(assignment), True
            # A plain witness can fail to improve, not just in theory: the theory's own value for it (computed while Z3 was still exploring) can disagree
            # with this exact recheck by an amount that straddles model_checking_precision, letting a non-improving candidate through as apparently-viable.
            # That does not mean no better assignment exists elsewhere -- only solver.check() returning unsat is a genuine proof of that -- so it is excluded
            # like any other explored point and the search continues (see also checker.py's singleton-eta shortcut, which eliminates the specific
            # solver-disagreement this guards against for future assignments). A robust witness always improves unless its value is exact, see
            # _worst_case_of.
            improves = optimality.improves_optimum(value)
            if improves:
                self.best_assignment = assignment
                self.best_assignment_value = value
                if self.forall_parameters:
                    self.robust_assignment = assignment
                optimality.update_optimum(value)
                # the threshold just tightened: some cached verdicts may no longer hold -- see PartialModelCache's docstring
                theory.epoch += 1
            if exact:
                # force Z3 to look elsewhere: without this, re-checking could return the same witness, since nothing at the Z3/Boolean level has changed --
                # only self.prop's threshold has, and the theory solver only revisits a literal when asked to. A robust witness is excluded only if its
                # value is exact: if only bounded, it may still turn out better than the tightened threshold.
                exclusions.append(encoding.exclude(assignment, policy_parameters))
            if improves and self.forall_parameters:
                # A tighter threshold can falsify what Z3 learned from refuted `not viable` literals ("every completion meets the threshold"), and a
                # learned clause cannot be retracted: start over, from the theory's cache, which drops exactly those (see PartialModelCache).
                solver, _propagator = new_solver()
            elif exact:
                solver.add(exclusions[-1])

        self.explored = node.parameter_space.size
        return self.best_assignment

    def _worst_case_of(
        self,
        theory: paynt.synthesizer.smpmc.checker.ColoredMdpTheory,
        robust_assignment: paynt.parameter_space.parameter_space.ParameterSpace,
        policy_parameters: list[int],
        optimality: paynt.specification.property.OptimalityProperty,
    ) -> tuple[Any, bool]:
        """The worst case over the environments of the policy robust_assignment fixes, and whether it is exact rather than a bound (see WorstCase).

        Z3 only reports a robust policy after covering every environment by refuted `not viable` literals, whose model-checked values the theory keeps: the
        worst of them bounds the policy's worst case and beats the current threshold, so it is a sound next threshold, and the policy needs no separate
        evaluation. Only if that bookkeeping comes up empty is the worst case computed, by AR.
        """
        policy = {parameter: robust_assignment.parameter_options(parameter)[0] for parameter in policy_parameters}
        bound = theory.worst_case_bound(policy)
        if bound is not None and optimality.improves_optimum(bound.value):
            return bound.value, bound.exact
        logger.warning("no usable worst-case bound for the robust policy, computing its worst case by AR")
        return paynt.synthesizer.smpmc._utils.worst_case_by_ar(self.colored_mdp, optimality, robust_assignment), True

    def run(self, optimum_threshold: Any = None) -> paynt.synthesizer.smpmc.result.SmpmcResult:
        """Also reports the robust space: synthesize() collapses best_assignment to one member of it (and double-checks that one)."""
        self.robust_assignment = None
        result = super().run(optimum_threshold)
        robust_verified = None
        if self.robust_assignment is not None:
            if result.value is None:
                logger.info(f"robust assignment, every member satisfies the specification: {self.robust_assignment}")
            else:
                logger.info(f"robust assignment, worst case {result.value} over its members: {self.robust_assignment}")
            if self.task.verify_robust:
                robust_verified = paynt.synthesizer.smpmc._utils.verify_robust(self.colored_mdp, self.task, self.robust_assignment, result.value)
                logger.info(f"robustness verified by AR: {robust_verified}")
        return paynt.synthesizer.smpmc.result.SmpmcResult(
            success=result.success,
            value=result.value,
            assignment=result.assignment,
            robust_assignment=self.robust_assignment,
            robust_verified=robust_verified,
        )

    def _value_of(self, assignment: paynt.parameter_space.parameter_space.ParameterSpace) -> Any:
        model = self.colored_mdp.build_assignment(assignment)
        return model.model_check_property(self.prop).value
