"""Fixture-based tests of SynthesizerSMPMC, mirroring tests/pomdp/test_pomdp_synthesizer.py's style:
instantiate the synthesizer directly, synthesize, and assert against a pinned answer -- plus cross-checks
against AR/OneByOne on the same fixtures, since SMPMC and those engines must agree on feasibility."""

from __future__ import annotations

import pytest

import paynt.parser.sketch
import paynt.synthesizer.smpmc
import paynt.synthesizer.synthesizer

from helpers.helper import get_sketch_paths


class TestSmpmcOnThresholdProperty:
    def test_finds_a_satisfying_assignment(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, smpmc_tiny_task)
        result = synthesizer.run()
        assert result.success is True

    def test_agrees_with_ar_on_feasibility(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        smpmc_result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, smpmc_tiny_task).run()

        # a genuinely separate load, not a reused fixture object: Synthesizer instances mutate
        # task.specification state (optimality threshold/reset), so AR and SMPMC must not share one task
        sketch_path, props_path = get_sketch_paths("tests/smpmc-tiny")
        ar_factory, ar_task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        ar_result = paynt.synthesizer.synthesizer.Synthesizer.for_method(ar_factory.build(), ar_task, "ar").run()

        assert smpmc_result.success == ar_result.success

    def test_infeasible_threshold_is_correctly_refuted(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        smpmc_tiny_task.specification.constraints[0].threshold = 1.5  # P>=1.5 can never hold
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, smpmc_tiny_task).run()
        assert result.success is False


class TestSmpmcOnOptimalityProperty:
    def test_finds_the_known_optimum(self, smpmc_tiny_optimality_colored_mdp, smpmc_tiny_optimality_task):
        synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_optimality_colored_mdp, smpmc_tiny_optimality_task)
        result = synthesizer.run()
        assert result.success is True
        assert result.value == pytest.approx(1.0)


class TestSmpmcOnFamilyModel:
    """mdp-family-avoid-8-2-easy is a feature_kind=="family" sketch: --method ar on it routes to
    PolicyTreeSynthesizer (a robust-by-construction game-abstraction algorithm solving a *different*
    problem), not plain existential search -- so the correct cross-check for Phase 1's plain-exists SMPMC
    is --method onebyone, which api.py also routes through the generic Synthesizer.for_method path."""

    def test_agrees_with_onebyone_on_feasibility(self, mdp_family_colored_mdp, mdp_family_task):
        smpmc_result = paynt.synthesizer.smpmc.SynthesizerSMPMC(mdp_family_colored_mdp, mdp_family_task).run()

        sketch_path, props_path = get_sketch_paths("tests/mdp-family-avoid-8-2-easy")
        onebyone_factory, onebyone_task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        onebyone_result = paynt.synthesizer.synthesizer.Synthesizer.for_method(onebyone_factory.build(), onebyone_task, "onebyone").run()

        assert smpmc_result.success == onebyone_result.success
