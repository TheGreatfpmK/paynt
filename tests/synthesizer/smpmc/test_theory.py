"""Integration tests of ColoredMdpTheory (Algorithm 1 "CheckMDPs") against a real colored MDP and real
Storm model-checking calls, on the smpmc-tiny fixture (see models/tests/smpmc-tiny)."""

from __future__ import annotations

import pytest

import paynt.parser.sketch
import paynt.synthesizer.smpmc.checker

from helpers.helper import get_sketch_paths


class TestColoredMdpTheoryOnFullAssignments:
    """For a *complete* assignment (every parameter fixed), check()'s verdict has an unambiguous ground
    truth: build the concrete DTMC directly and compare. This sidesteps needing to hand-derive which
    option index corresponds to which sketch-level hole label."""

    def test_every_full_assignment_agrees_with_direct_model_checking(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        prop = smpmc_tiny_task.specification.constraints[0]
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(smpmc_tiny_colored_mdp, prop)

        parameter_space = smpmc_tiny_colored_mdp.parameter_space
        checked_at_least_one_sat = False
        checked_at_least_one_unsat = False
        for combination in parameter_space.all_combinations():
            fixed = dict(enumerate(combination))
            assignment = parameter_space.construct_assignment(combination)
            dtmc = smpmc_tiny_colored_mdp.build_assignment(assignment)
            real_sat = dtmc.model_check_property(prop).sat
            checked_at_least_one_sat |= real_sat
            checked_at_least_one_unsat |= not real_sat

            # viable(fixed) should hold iff real_sat -- so asking for polarity=True must be inconclusive
            # (not refuted) when real_sat is True, and refuted when real_sat is False
            viable_result = theory.check(fixed, True)
            assert (viable_result is None) == real_sat, f"{fixed}: real_sat={real_sat}, viable check={viable_result}"

            # dually for the not-viable literal
            not_viable_result = theory.check(fixed, False)
            assert (not_viable_result is None) == (not real_sat), f"{fixed}: real_sat={real_sat}, not-viable check={not_viable_result}"

        # a meaningful test needs the fixture to actually contain both outcomes
        assert checked_at_least_one_sat
        assert checked_at_least_one_unsat

    def test_refutation_conflict_is_a_subset_of_fixed(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        prop = smpmc_tiny_task.specification.constraints[0]
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(smpmc_tiny_colored_mdp, prop)
        parameter_space = smpmc_tiny_colored_mdp.parameter_space
        for combination in parameter_space.all_combinations():
            fixed = dict(enumerate(combination))
            for polarity in (True, False):
                result = theory.check(fixed, polarity)
                if result is not None:
                    assert set(result.conflict_parameters).issubset(fixed.keys())


class TestColoredMdpTheorySingletonEta:
    """Regression coverage for a real correctness bug found on models/tests/generic-maze: for a fully-
    fixed (singleton) eta, ColoredMdp.build() hands back an MDP-typed model even though it has exactly one
    choice per state, and solving that via minmax/policy iteration can disagree with the exact value
    build_assignment()'s DTMC conversion gives for the mathematically identical process -- by an amount
    that lands just past model_checking_precision. That was enough to make SynthesizerSMPMC's optimality
    loop treat a non-improving witness as still-viable and terminate with the wrong (much worse) optimum:
    real 24-hole run converged to 36001.06 instead of the true minimum around 8.13. check() now uses
    build_assignment() directly whenever eta.size == 1, sidestepping the disagreement rather than working
    around its symptom.
    """

    def test_a_full_assignment_is_checked_via_the_exact_dtmc_value_not_build(self):
        sketch_path, props_path = get_sketch_paths("tests/generic-maze")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        colored_mdp = factory.build()
        prop = task.specification.optimality
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(colored_mdp, prop)

        assignment = colored_mdp.parameter_space.pick_any()
        expected = colored_mdp.build_assignment(assignment).model_check_property(prop, alt=False).value

        fixed = {p: assignment.parameter_options(p)[0] for p in range(assignment.num_parameters)}
        # a threshold exactly at the expected value: if check() ever computed this via build() (MDP-typed,
        # minmax-solved) instead of build_assignment() (DTMC-typed, exact), the two values would disagree
        # by an amount straddling model_checking_precision, and this refutation query would come back as
        # inconclusive (None) instead of correctly refuted -- exactly the bug that let a non-improving
        # witness through in the real run.
        prop.update_optimum(expected)
        result = theory.check(fixed, polarity=True)

        assert result is not None, (
            "expected this witness to be refuted (its value cannot strictly improve on itself), "
            "but check() returned inconclusive -- the singleton-eta shortcut may have regressed"
        )


class TestColoredMdpTheoryOutOfRangeValues:
    """Regression test: an out-of-range option must be treated as inconclusive (letting the plain
    in_range() Boolean constraint reject it), not passed through to ColoredMdp.build() -- which previously
    crashed stormpy's submodel construction with a deadlock-state error for some out-of-range partial
    assignments. See checker.py's check() for the guard this tests."""

    def test_out_of_range_option_is_inconclusive_not_a_crash(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        prop = smpmc_tiny_task.specification.constraints[0]
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(smpmc_tiny_colored_mdp, prop)
        # parameter 0 (x1) only has options [0,1,2,3] -- 9999 is never valid
        assert theory.check({0: 9999}, True) is None
        assert theory.check({0: 9999}, False) is None


class TestColoredMdpTheoryRewardConvergence:
    """Regression coverage for _model_check's value-iteration-to-policy-iteration escalation.

    models/tests/generic-maze is a genuine, naturally-occurring example of the failure mode this guards
    against: R{"steps"}max=?[F "goal"] is a large but finite value (~903.9 million) that value iteration
    converges to pathologically slowly -- capped VI (see Property.initialize's max_minmax_iterations)
    returns a wildly wrong snapshot (order 10^3-10^7, depending on the cap) long before it gets anywhere
    close, while the min direction of the very same property converges immediately with plain VI. This is
    not a synthetic scenario; it's the exact model that originally exposed the bug (see conversation).
    """

    def test_the_slow_converging_direction_escalates_to_the_correct_value(self):
        sketch_path, props_path = get_sketch_paths("tests/generic-maze")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        colored_mdp = factory.build()
        prop = task.specification.optimality
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(colored_mdp, prop)
        sub_mdp, _selected_choices = colored_mdp.build(colored_mdp.parameter_space)

        result = theory._model_check(sub_mdp, alt=True)  # the max (worst-case) direction

        assert theory._reward_use_policy_iteration.get(True) is True
        assert result.value == pytest.approx(903940953.7112827, rel=1e-6)

    def test_the_fast_converging_direction_does_not_escalate(self):
        sketch_path, props_path = get_sketch_paths("tests/generic-maze")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        colored_mdp = factory.build()
        prop = task.specification.optimality
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(colored_mdp, prop)
        sub_mdp, _selected_choices = colored_mdp.build(colored_mdp.parameter_space)

        result = theory._model_check(sub_mdp, alt=False)  # the min (best-case) direction

        assert theory._reward_use_policy_iteration.get(False, False) is False
        assert result.value == pytest.approx(6.889553051515894, rel=1e-6)

    def test_a_second_query_on_an_escalated_direction_skips_reverification(self):
        sketch_path, props_path = get_sketch_paths("tests/generic-maze")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        colored_mdp = factory.build()
        prop = task.specification.optimality
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(colored_mdp, prop)
        sub_mdp, _selected_choices = colored_mdp.build(colored_mdp.parameter_space)

        first = theory._model_check(sub_mdp, alt=True)
        second = theory._model_check(sub_mdp, alt=True)

        assert second.value == first.value

    def test_probability_properties_are_unaffected(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        """The escalation machinery only ever triggers for reward properties -- probability values are
        bounded in [0,1] and don't exhibit this failure mode, so a probability property should never end
        up flagged for policy iteration."""
        prop = smpmc_tiny_task.specification.constraints[0]
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(smpmc_tiny_colored_mdp, prop)
        sub_mdp, _selected_choices = smpmc_tiny_colored_mdp.build(smpmc_tiny_colored_mdp.parameter_space)

        theory._model_check(sub_mdp, alt=False)
        theory._model_check(sub_mdp, alt=True)

        assert theory._reward_convergence_verified == {}
        assert theory._reward_use_policy_iteration == {}


class TestColoredMdpTheoryCache:
    def test_repeated_query_is_served_from_cache_without_a_second_model_check(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        prop = smpmc_tiny_task.specification.constraints[0]
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(smpmc_tiny_colored_mdp, prop)
        fixed = {0: 0, 1: 0, 2: 0}
        theory.check(fixed, True)
        calls_after_first = theory.mc_calls
        theory.check(fixed, True)
        assert theory.mc_calls == calls_after_first, "a repeated query should hit the cache, not call Storm again"

    def test_inconclusive_cache_entry_is_not_reused_across_an_epoch_change(self, smpmc_tiny_colored_mdp, smpmc_tiny_optimality_task):
        prop = smpmc_tiny_optimality_task.specification.optimality
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(smpmc_tiny_colored_mdp, prop)
        fixed = {0: 0, 1: 0, 2: 0}
        theory.check(fixed, True)
        calls_after_first = theory.mc_calls
        theory.epoch += 1
        theory.check(fixed, True)
        assert theory.mc_calls == calls_after_first + 1, "an inconclusive verdict from a stale epoch must not be reused"
