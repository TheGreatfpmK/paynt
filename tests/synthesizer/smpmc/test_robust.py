"""Integration tests of robust synthesis (--constraint exists_forall), each result checked independently with
verify_robust -- AR on the negated specification, a port of molehill's tests/test_robust.py.

smpmc-tiny (see tests/parameter_space/constraints/test_costs.py) has exactly 3 satisfying assignments:
(x1=1,x2=4), (x1=4,x2=3) and (x1=4,x2=4), with x1 in {1..4} and x2 in {3,4}. So "exists x1 forall x2" holds with
x1=4, while "exists x2 forall x1" does not hold at all.
"""

from __future__ import annotations

import gc

import pytest

import paynt.synthesizer.smpmc
import paynt.synthesizer.smpmc._utils
import paynt.synthesizer.smpmc.theory


def _robust(task, forall_pattern=None, verify_robust=False):
    task.constraint_name = "exists_forall"
    task.forall_pattern = forall_pattern
    task.verify_robust = verify_robust
    return task


class TestRobustSynthesisOnFamily:
    """The environment is the family's own parameters, the policy is synthesized."""

    def test_finds_a_policy_robust_against_every_environment(self, rocks_colored_mdp, rocks_task):
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(rocks_colored_mdp, _robust(rocks_task)).run()
        assert result.success
        robust = result.robust_assignment
        assert [robust.parameter_num_options(parameter) for parameter in range(4)] == [3, 3, 3, 3]
        assert all(robust.parameter_num_options(parameter) == 1 for parameter in range(4, robust.num_parameters))
        assert result.assignment.size == 1

    def test_the_robust_policy_is_confirmed_by_ar(self, rocks_colored_mdp, rocks_task):
        synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(rocks_colored_mdp, _robust(rocks_task))
        result = synthesizer.run()
        assert paynt.synthesizer.smpmc._utils.verify_robust(synthesizer.colored_mdp, rocks_task, result.robust_assignment)

    def test_run_reports_the_verification_when_asked(self, rocks_colored_mdp, rocks_task):
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(rocks_colored_mdp, _robust(rocks_task, verify_robust=True)).run()
        assert result.robust_verified is True

    def test_a_forall_pattern_overrides_the_environment_default(self, rocks_colored_mdp, rocks_task):
        # quantifying the policy universally too: not every policy visits both rocks
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(rocks_colored_mdp, _robust(rocks_task, forall_pattern=".")).run()
        assert not result.success
        assert result.robust_assignment is None


class TestRobustSynthesisOnGenericSketch:
    def test_sat_when_one_option_works_for_every_environment(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, _robust(smpmc_tiny_task, forall_pattern="^x2$", verify_robust=True)).run()
        assert result.success
        assert str(result.robust_assignment) == "x1=4, x2: {3,4}, x3=8"
        assert result.robust_verified is True

    def test_unsat_when_no_option_works_for_every_environment(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, _robust(smpmc_tiny_task, forall_pattern="^x1$")).run()
        assert not result.success
        assert result.robust_assignment is None

    def test_a_forall_pattern_is_required_outside_a_family(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        with pytest.raises(ValueError, match="--smpmc-forall"):
            paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, _robust(smpmc_tiny_task))

    def test_a_pattern_matching_no_parameter_is_an_error(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        with pytest.raises(ValueError, match="matches no parameter"):
            paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, _robust(smpmc_tiny_task, forall_pattern="no_such_hole"))

    def test_an_optimality_objective_is_rejected(self, smpmc_tiny_optimality_colored_mdp, smpmc_tiny_optimality_task):
        with pytest.raises(ValueError, match="threshold property"):
            paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_optimality_colored_mdp, _robust(smpmc_tiny_optimality_task, forall_pattern="^x2$"))

    def test_a_plain_search_reports_no_robust_assignment(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, smpmc_tiny_task).run()
        assert result.success
        assert result.robust_assignment is None
        assert result.robust_verified is None


class TestVerifyRobust:
    """verify_robust on hand-picked candidates: x1 fixed, x2 left open."""

    def _candidate(self, colored_mdp, x1_option):
        return colored_mdp.parameter_space.assume_options_copy([[x1_option], [0, 1], [0]])

    def test_accepts_a_robust_candidate(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        candidate = self._candidate(smpmc_tiny_colored_mdp, x1_option=3)  # x1=4
        assert paynt.synthesizer.smpmc._utils.verify_robust(smpmc_tiny_colored_mdp, smpmc_tiny_task, candidate)

    def test_rejects_a_candidate_violated_by_one_environment(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        candidate = self._candidate(smpmc_tiny_colored_mdp, x1_option=0)  # x1=1: fails for x2=3
        assert not paynt.synthesizer.smpmc._utils.verify_robust(smpmc_tiny_colored_mdp, smpmc_tiny_task, candidate)


class TestFreshPropagators:
    def test_every_term_is_held_through_the_solver_context(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        """Z3 frees an MBQI sub-context once its round ends, while z3's Python API keeps every propagator alive until
        exit: a term held through the sub-context is then released through a dangling pointer. The resulting heap
        corruption only shows up at exit, and not reliably (see test_synthesizer_smpmc_cli.py), so check the invariant."""
        paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, _robust(smpmc_tiny_task, forall_pattern="^x2$")).run()
        propagators = [obj for obj in gc.get_objects() if isinstance(obj, paynt.synthesizer.smpmc.theory.SmpmcPropagator)]
        assert any(propagator.fresh_ctx is not None for propagator in propagators), "expected MBQI to create sub-context propagators"
        for propagator in propagators:
            assert propagator.owner_ctx is not propagator.fresh_ctx
            assert all(term.ctx is propagator.owner_ctx for term in propagator.term_of.values())
