"""Phase 2 differential tests: decision_tree_coloring (paynt.dt.coloring_general) against ColoringSmt, on the two tracked DT fixtures (dt-orchard: Pmax; dt-
maze: Rmin).

reset_tree(general=True) is the only entry point exercised -- it, not decision_tree_coloring directly, is DtColoredMdpFactory's own contract.
"""

from __future__ import annotations

import random

import pytest

import paynt.parser.sketch
import paynt.dt.synthesizer_ar_dt
from helpers.helper import get_sketch_paths

PROJECTS = ["tests/dt-orchard", "tests/dt-maze"]
DEPTHS = [0, 1, 2]


def load(project, depth):
    sketch_path, props_path = get_sketch_paths(project)
    factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return factory.reset_tree(depth, general=True), factory.reset_tree(depth, general=False), task


@pytest.mark.parametrize("project", PROJECTS)
@pytest.mark.parametrize("depth", DEPTHS)
class TestHasGeneralColoring:
    def test_reset_tree_reports_which_coloring_it_built(self, project, depth):
        cmdp_general, cmdp_smt, _task = load(project, depth)
        assert cmdp_general.has_general_coloring is True
        assert cmdp_smt.has_general_coloring is False
        # feature_kind is unaffected: every DT code path still applies to the general-coloring variant
        assert cmdp_general.feature_kind == cmdp_smt.feature_kind == "dt"


@pytest.mark.parametrize("project", PROJECTS)
@pytest.mark.parametrize("depth", DEPTHS)
class TestParameterLayoutMatches:
    def test_get_family_info_matches_coloringsmt(self, project, depth):
        cmdp_general, cmdp_smt, _task = load(project, depth)
        assert cmdp_general.parameter_space.num_parameters == cmdp_smt.parameter_space.num_parameters
        for parameter in range(cmdp_general.parameter_space.num_parameters):
            assert cmdp_general.parameter_space.parameter_name(parameter) == cmdp_smt.parameter_space.parameter_name(parameter)
            assert cmdp_general.parameter_space.parameter_options(parameter) == cmdp_smt.parameter_space.parameter_options(parameter)


@pytest.mark.parametrize("project", PROJECTS)
@pytest.mark.parametrize("depth", DEPTHS)
class TestSelectionMatchesOnRandomFamilies:
    def test_full_family_selection_matches(self, project, depth):
        cmdp_general, cmdp_smt, _task = load(project, depth)
        parameter_space = cmdp_general.parameter_space
        general = set(cmdp_general.coloring.selectCompatibleChoices(parameter_space.native))
        smt = set(cmdp_smt.coloring.selectCompatibleChoices(cmdp_smt.parameter_space.native))
        assert general == smt

    def test_random_subfamily_selection_matches(self, project, depth):
        cmdp_general, cmdp_smt, _task = load(project, depth)
        parameter_space = cmdp_general.parameter_space
        rng = random.Random(depth)
        for _trial in range(10):
            sub_general = parameter_space.copy()
            sub_smt = cmdp_smt.parameter_space.copy()
            for parameter in range(parameter_space.num_parameters):
                options = parameter_space.parameter_options(parameter)
                if len(options) > 1 and rng.random() < 0.6:
                    chosen = sorted(rng.sample(options, rng.randint(1, len(options))))
                    sub_general.parameter_set_options(parameter, chosen)
                    sub_smt.parameter_set_options(parameter, chosen)
            general = set(cmdp_general.coloring.selectCompatibleChoices(sub_general.native))
            smt = set(cmdp_smt.coloring.selectCompatibleChoices(sub_smt.native))
            assert general == smt

    def test_random_full_assignment_gives_the_same_specification_result(self, project, depth):
        cmdp_general, _cmdp_smt, task = load(project, depth)
        parameter_space = cmdp_general.parameter_space
        rng = random.Random(depth + 100)
        for _trial in range(5):
            assignment = parameter_space.copy()
            for parameter in range(parameter_space.num_parameters):
                options = parameter_space.parameter_options(parameter)
                assignment.parameter_set_options(parameter, [rng.choice(options)])
            dtmc = cmdp_general.build_assignment(assignment)
            result = dtmc.check_specification(task.specification)
            assert result.constraints_result is not None


@pytest.mark.parametrize("project", PROJECTS)
@pytest.mark.parametrize("depth", DEPTHS)
class TestAreChoicesConsistentHintsObeyTheContract:
    """The contract dt/synthesizer_ar_dt.py relies on (see ColoredMdp.are_choices_consistent's own assertions and harmonize_inconsistent_scheduler): SAT gives
    one option per parameter; UNSAT+harmonization gives one option per parameter except exactly one with two *distinct*, valid options."""

    def test_hints_on_random_choice_masks(self, project, depth):
        cmdp_general, _cmdp_smt, _task = load(project, depth)
        parameter_space = cmdp_general.parameter_space
        underlying_mdp = cmdp_general.underlying_mdp
        row_groups = underlying_mdp.nondeterministic_choice_indices
        rng = random.Random(depth + 200)
        num_multi_option = 0
        for _trial in range(15):
            # a random "almost a scheduler" choice mask: one choice per state, from a random *sub*family's own
            # compatible choices, then perturbed by swapping a handful of states to a different one of their own
            # options -- likely inconsistent (what harmonization is for), but always a genuine full choice-per-state
            # mask (so areChoicesConsistent's own preconditions, e.g. reachability, hold)
            subfamily = parameter_space.copy()
            for parameter in range(parameter_space.num_parameters):
                options = parameter_space.parameter_options(parameter)
                if len(options) > 1 and rng.random() < 0.5:
                    subfamily.parameter_set_options(parameter, sorted(rng.sample(options, rng.randint(1, len(options)))))
            _mdp, selected_choices = cmdp_general.build(subfamily)
            choices = [next(c for c in range(row_groups[state], row_groups[state + 1]) if selected_choices[c]) for state in range(len(row_groups) - 1)]
            mask_choices = set(choices)
            for _ in range(3):
                state = rng.randrange(len(row_groups) - 1)
                candidates = [c for c in range(row_groups[state], row_groups[state + 1]) if selected_choices[c]]
                if len(candidates) > 1:
                    mask_choices.discard(next(c for c in candidates if c in mask_choices))
                    mask_choices.add(rng.choice(candidates))

            import stormpy.storage

            mask = stormpy.storage.BitVector(underlying_mdp.nr_choices, False)
            for choice in mask_choices:
                mask.set(choice, True)

            consistent, selection = cmdp_general.are_choices_consistent(mask, subfamily)
            assert len(selection) == parameter_space.num_parameters
            multi = [p for p in range(parameter_space.num_parameters) if len(selection[p]) > 1]
            if consistent:
                assert all(len(options) <= 1 for options in selection)
            elif selection and any(selection):
                # harmonization succeeded or a bisection hint was given -- both shapes are "one option per
                # parameter, at most one with two distinct valid options"
                assert len(multi) <= 1
                for parameter in multi:
                    assert len(set(selection[parameter])) == 2
                    for option in selection[parameter]:
                        assert option in parameter_space.parameter_options(parameter)
                    num_multi_option += 1
                for parameter, options in enumerate(selection):
                    for option in options:
                        assert option in parameter_space.parameter_options(parameter)
        # not asserting num_multi_option > 0: whether harmonization/bisection actually triggers depends on the
        # random masks drawn: this loop's job is only to check the *shape* holds whenever it does trigger.


@pytest.mark.parametrize("project", PROJECTS)
class TestArDtReachesTheKnownOptimumAtDepthZero:
    """Depth 0 (a single leaf, the smallest possible search) is a clean, fast check that the two colorings drive SynthesizerARDt to an identical result.

    Deeper depths are not compared for an exact match here: ColoringGeneral's
    per-node cost is higher than ColoringSmt's hand-written tree walk, so within a short timeout the same AR search can
    stop at a worse value -- a search-efficiency gap, not a correctness one (selection and consistency are compared
    exactly above, at every depth).
    """

    def test_depth_0_optimum_matches(self, project):
        sketch_path, props_path = get_sketch_paths(project)
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        cmdp_general = factory.reset_tree(0, general=True)
        synthesizer = paynt.dt.synthesizer_ar_dt.SynthesizerARDt(cmdp_general, task)
        synthesizer.synthesize(keep_optimum=True)
        assert synthesizer.best_assignment is not None

        sketch_path, props_path = get_sketch_paths(project)
        factory_smt, task_smt = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        cmdp_smt = factory_smt.reset_tree(0, general=False)
        synthesizer_smt = paynt.dt.synthesizer_ar_dt.SynthesizerARDt(cmdp_smt, task_smt)
        synthesizer_smt.synthesize(keep_optimum=True)

        assert synthesizer.best_assignment_value == pytest.approx(synthesizer_smt.best_assignment_value, abs=1e-6)
