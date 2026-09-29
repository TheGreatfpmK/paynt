"""Phase 3: SynthesizerSMPMC as DtSynthesizer's inner engine (over the tree's ColoringGeneral), instead of the default SynthesizerARDt (over ColoringSmt) -- see
paynt.dt._utils and DtColoredMdpFactory.reset_tree.

Depth 0 is compared against SynthesizerARDt directly. Deeper trees are compared against optima pinned from SynthesizerARDt over ColoringSmt (see
tests/dt/test_dt_coloring_general.py's docstring on why running the AR engine over the general coloring is not a fair comparison there), and only became
feasible for SMPMC once its conflicts are minimized with ColoringGeneral.relevantParameters.
"""

from __future__ import annotations

import random

import pytest

import paynt.api
import paynt.parser.sketch
import paynt.dt.synthesizer
import paynt.synthesizer.smpmc.checker
from helpers.helper import get_sketch_paths

PROJECTS = ["tests/dt-orchard", "tests/dt-maze"]


@pytest.mark.parametrize("project", PROJECTS)
class TestSmpmcInnerEngineMatchesArDtAtDepthZero:
    def test_synthesize_tree_reaches_the_same_optimum(self, project):
        sketch_path, props_path = get_sketch_paths(project)
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        smpmc_synth = paynt.dt.synthesizer.DtSynthesizer(factory, task, method="smpmc")
        smpmc_synth.synthesize_tree(0)

        sketch_path, props_path = get_sketch_paths(project)
        factory_ar, task_ar = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        ar_synth = paynt.dt.synthesizer.DtSynthesizer(factory_ar, task_ar, method="ar")
        ar_synth.synthesize_tree(0)

        assert smpmc_synth.best_tree is not None
        assert smpmc_synth.best_tree_value == pytest.approx(ar_synth.best_tree_value, abs=1e-6)

    def test_synthesize_tree_sequence_reaches_the_same_optimum(self, project):
        sketch_path, props_path = get_sketch_paths(project)
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        smpmc_synth = paynt.dt.synthesizer.DtSynthesizer(factory, task, method="smpmc")
        smpmc_synth.synthesize_tree_sequence(opt_result_value=0.0, overall_timeout=20, max_depth=1)

        sketch_path, props_path = get_sketch_paths(project)
        factory_ar, task_ar = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        ar_synth = paynt.dt.synthesizer.DtSynthesizer(factory_ar, task_ar, method="ar")
        ar_synth.synthesize_tree_sequence(opt_result_value=0.0, overall_timeout=20, max_depth=1)

        assert smpmc_synth.best_tree_value == pytest.approx(ar_synth.best_tree_value, abs=1e-6)


class TestApiRouting:
    def test_get_synthesizer_routes_method_smpmc_to_dt_synthesizer_with_smpmc_inner_engine(self):
        sketch_path, props_path = get_sketch_paths("tests/dt-orchard")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        synthesizer = paynt.api.get_synthesizer(factory, task, method="smpmc")
        assert isinstance(synthesizer, paynt.dt.synthesizer.DtSynthesizer)
        assert synthesizer.method == "smpmc"

    def test_dtnest_rejects_method_smpmc(self):
        sketch_path, props_path = get_sketch_paths("tests/dt-orchard")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        with pytest.raises(ValueError, match="dtnest"):
            paynt.api.get_synthesizer(factory, task, method="smpmc", dtnest=True)

    def test_a_generic_method_on_a_dt_sketch_is_rejected_with_a_clear_error(self):
        sketch_path, props_path = get_sketch_paths("tests/dt-orchard")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        with pytest.raises(ValueError, match="decision-tree synthesis supports"):
            paynt.api.get_synthesizer(factory, task, method="onebyone")

    def test_default_method_ar_is_unaffected(self):
        """Regression guard: get_synthesizer's default routing for "dt" (no explicit --method) must still build a ColoringSmt-backed DtSynthesizer, exactly as
        before this feature existed."""
        sketch_path, props_path = get_sketch_paths("tests/dt-orchard")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        synthesizer = paynt.api.get_synthesizer(factory, task)
        assert isinstance(synthesizer, paynt.dt.synthesizer.DtSynthesizer)
        assert synthesizer.method == "ar"
        assert synthesizer.colored_mdp.coloring.__class__.__name__ == "ColoringSmt"


class TestSmpmcOnDeeperTrees:
    """Optima pinned from SynthesizerARDt over ColoringSmt; SMPMC proves optimality, so agreeing with that independent engine is the check.

    With the static per-state supports (every tree parameter, for every state) SMPMC's conflicts never shrink: dt-orchard depth 1 took minutes instead of
    seconds.
    """

    @pytest.mark.parametrize(
        ("project", "depth", "optimum"),
        [
            ("tests/dt-orchard", 1, 0.4882931556949054),
            ("tests/dt-maze", 1, 114.8076923076909),
            ("tests/dt-maze", 2, 20.948148711838865),
        ],
    )
    def test_reaches_the_optimum(self, project, depth, optimum):
        sketch_path, props_path = get_sketch_paths(project)
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        synthesizer = paynt.dt.synthesizer.DtSynthesizer(factory, task, method="smpmc")
        synthesizer.synthesize_tree(depth, timeout=120)
        assert synthesizer.best_tree is not None
        assert synthesizer.best_tree_value == pytest.approx(optimum, rel=1e-9)


class TestConflictsOnTrees:
    """The soundness of ColoredMdpTheory's conflicts on a general coloring: whenever it learns a conflict smaller than the parameters fixed, every full
    assignment agreeing with the conflict must be refuted too (checked by model-checking the extension directly), or the learned clause would prune a real
    solution."""

    @pytest.mark.parametrize(
        ("project", "threshold", "polarity"),
        [
            # only trees needing fewer than 30 steps improve: most partial trees are refuted as `viable`
            ("tests/dt-maze", 30.0, True),
            # every tree improves: many partial trees are refuted as `not viable`, i.e. even their worst case improves
            ("tests/dt-maze", 1e6, False),
            ("tests/dt-orchard", 0.4884, True),
            ("tests/dt-orchard", 0.3, False),
        ],
    )
    def test_every_extension_of_a_smaller_conflict_is_refuted_too(self, project, threshold, polarity):
        sketch_path, props_path = get_sketch_paths(project)
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        colored_mdp = factory.reset_tree(2, general=True)
        parameter_space = colored_mdp.parameter_space
        optimality = task.specification.optimality
        optimality.update_optimum(threshold)
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(colored_mdp, optimality, None)

        rng = random.Random(1)
        smaller = 0
        for _ in range(100):
            fixed = {p: rng.choice(parameter_space.parameter_options(p)) for p in range(parameter_space.num_parameters) if rng.random() < 0.7}
            refutation = theory.check(fixed, polarity)
            if refutation is None or len(refutation.conflict_parameters) == len(fixed):
                continue
            smaller += 1
            conflict = set(refutation.conflict_parameters)
            assert conflict <= set(fixed)
            for _ in range(4):
                extension = parameter_space.copy()
                for p in range(parameter_space.num_parameters):
                    extension.parameter_set_options(p, [fixed[p] if p in conflict else rng.choice(parameter_space.parameter_options(p))])
                result = theory._model_check(colored_mdp.build_assignment(extension), alt=(polarity is False))
                assert result.sat is not polarity, f"extension of conflict {sorted(conflict)} is not refuted"
        assert smaller > 0, "no refutation had a conflict smaller than the fixed parameters: the check proves nothing about minimization"
