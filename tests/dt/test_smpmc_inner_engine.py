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
import paynt.model.model
import paynt.parser.sketch
import paynt.dt.api
import paynt.dt.synthesizer
import paynt.dt.task
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

    def test_mapping_a_scheduler_rejects_method_smpmc(self):
        """What --tree-map-scheduler sets: a scheduler file to map.

        SMPMC has nothing to search there.
        """
        sketch_path, props_path = get_sketch_paths("tests/dt-orchard")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        factory.build_task.scheduler_path = "scheduler.json"
        with pytest.raises(ValueError, match="mapping a scheduler to a tree does not support method 'smpmc'"):
            paynt.api.get_synthesizer(factory, task, method="smpmc")
        assert paynt.api.get_synthesizer(factory, task, method="ar").method == "ar"

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


def load(project):
    sketch_path, props_path = get_sketch_paths(project)
    return paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)


def map_optimal_scheduler(project, max_depth, method="ar"):
    """Map Storm's optimal scheduler of the project's MDP to a tree of at most max_depth, through the library entry point."""
    factory, task = load(project)
    scheduler = paynt.model.model.Mdp(factory.underlying_mdp).model_check_property(task.get_property()).result.scheduler
    build_task = paynt.dt.task.DtTask(tree_depth=max_depth)
    build_task.set_scheduler_to_map(scheduler)
    return paynt.dt.api.synthesize(factory, task, build_task, method=method)


class TestMapScheduler:
    """Mapping a scheduler to a tree is one satisfiability query per depth, answered by ColoringSmt; there is nothing for SMPMC to search, so it is refused.

    That the tree makes exactly the scheduler's choices is checked where the query is answered, see TestAreChoicesConsistentMatchesColoringSmt.
    """

    def test_the_optimal_scheduler_of_the_maze_is_mapped_to_a_tree_of_depth_3(self):
        result = map_optimal_scheduler("tests/dt-maze", 4)
        assert result.success
        assert result.tree.get_depth() == 3
        assert len(result.tree.collect_nonterminals()) == 5
        assert result.value == pytest.approx(6.889153192908649, rel=1e-6)

    @pytest.mark.parametrize(("project", "max_depth"), [("tests/dt-maze", 2), ("tests/dt-orchard", 2)])
    def test_no_tree_is_found_when_the_optimal_scheduler_needs_a_deeper_one(self, project, max_depth):
        assert not map_optimal_scheduler(project, max_depth).success

    def test_method_smpmc_is_refused(self):
        with pytest.raises(ValueError, match="mapping a scheduler to a tree does not support method 'smpmc'"):
            map_optimal_scheduler("tests/dt-maze", 4, "smpmc")

    def test_the_method_is_checked(self):
        factory, task = load("tests/dt-maze")
        with pytest.raises(ValueError, match="decision-tree synthesis supports"):
            paynt.dt.synthesizer.DtSynthesizer(factory, task, method="onebyone")


class TestLibraryEntryPointTakesTheMethod:
    def test_synthesizing_trees_through_paynt_dt_api_with_smpmc(self):
        factory, task = load("tests/dt-maze")
        smpmc = paynt.dt.api.synthesize(factory, task, paynt.dt.task.DtTask(tree_depth=1), use_solver="dtpaynt", method="smpmc")
        factory, task = load("tests/dt-maze")
        ar = paynt.dt.api.synthesize(factory, task, paynt.dt.task.DtTask(tree_depth=1), use_solver="dtpaynt")
        assert smpmc.success and ar.success
        assert smpmc.value == pytest.approx(ar.value, abs=1e-6)
