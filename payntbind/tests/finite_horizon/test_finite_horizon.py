import numpy as np
import pytest
import stormpy

import payntbind

# A small MDP with a coloring. State 3 is only reachable after two steps and state 4 is not reachable at all.
#
# state 0 (init): choice 0 -> {1: 0.5, 2: 0.5}  colored (h0=0)
#                 choice 1 -> {0: 1}            colored (h0=1, h1=2)
# state 1:        choice 2 -> {3: 1}            uncolored
# state 2:        choice 3 -> {3: 1}            colored (h1=0)
# state 3 (goal): choice 4 -> {3: 1}            uncolored
# state 4:        choice 5 -> {0: 1}            colored (h1=1)
NUM_STATES = 5
NUM_CHOICES = 6
ROW_GROUPS = [0, 2, 3, 4, 5, 6]
SUCCESSORS = [{1: 0.5, 2: 0.5}, {0: 1.0}, {3: 1.0}, {3: 1.0}, {3: 1.0}, {0: 1.0}]
CHOICE_TO_ASSIGNMENT = [[(0, 0)], [(0, 1), (1, 2)], [], [(1, 0)], [], [(1, 1)]]
HOLE_NUM_OPTIONS = [2, 3]
REWARDS = [1.0, 2.0, 3.0, 4.0, 0.0, 5.0]
CHOICE_LABELS = {"a": [0, 2], "b": [1, 5]}
HORIZON = 3
# the states reachable after t steps, t = 0..HORIZON
LAYERS = [[0], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3]]


@pytest.fixture
def mdp():
    builder = stormpy.SparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)
    for state in range(NUM_STATES):
        builder.new_row_group(ROW_GROUPS[state])
        for choice in range(ROW_GROUPS[state], ROW_GROUPS[state + 1]):
            for successor, probability in sorted(SUCCESSORS[choice].items()):
                builder.add_next_value(choice, successor, probability)
    labeling = stormpy.storage.StateLabeling(NUM_STATES)
    for label in ("init", "goal"):
        labeling.add_label(label)
    labeling.add_label_to_state("init", 0)
    labeling.add_label_to_state("goal", 3)

    choice_labeling = stormpy.storage.ChoiceLabeling(NUM_CHOICES)
    for label, choices in CHOICE_LABELS.items():
        choice_labeling.add_label(label)
        for choice in choices:
            choice_labeling.add_label_to_choice(label, choice)

    reward_model = stormpy.SparseRewardModel(optional_state_action_reward_vector=REWARDS)
    components = stormpy.SparseModelComponents(
        transition_matrix=builder.build(overridden_column_count=NUM_STATES), state_labeling=labeling, reward_models={"cost": reward_model}
    )
    components.choice_labeling = choice_labeling
    return stormpy.storage.SparseMdp(components)


@pytest.fixture
def family():
    family = payntbind.synthesis.Family()
    for num_options in HOLE_NUM_OPTIONS:
        family.addHole(num_options)
    return family


@pytest.fixture
def coloring(family):
    return payntbind.synthesis.Coloring(family, ROW_GROUPS, CHOICE_TO_ASSIGNMENT)


def row(matrix, index):
    return {entry.column: entry.value() for entry in matrix.get_row(index)}


def expected_states():
    """(step, state) of the states of the unfolded MDP, in order."""
    return [(step, state) for step in range(HORIZON + 1) for state in LAYERS[step]]


class TestUnfolder:
    def test_sizes_count_only_reachable_states(self, mdp):
        unfolder = payntbind.synthesis.FiniteHorizonUnfolder(mdp, HORIZON)
        num_states = sum(len(layer) for layer in LAYERS)
        # the choices of the states of the layers before the last one and a self-loop for each state of the last layer
        num_choices = sum(ROW_GROUPS[s + 1] - ROW_GROUPS[s] for layer in LAYERS[:-1] for s in layer) + len(LAYERS[-1])
        num_transitions = sum(len(SUCCESSORS[c]) for layer in LAYERS[:-1] for s in layer for c in range(ROW_GROUPS[s], ROW_GROUPS[s + 1])) + len(LAYERS[-1])
        assert (num_states, num_choices, num_transitions) == (12, 15, 18)
        assert (unfolder.num_states, unfolder.num_choices, unfolder.num_transitions) == (num_states, num_choices, num_transitions)
        unfolded = unfolder.construct_unfolded_model()
        assert (unfolded.nr_states, unfolded.nr_choices, unfolded.nr_transitions) == (num_states, num_choices, num_transitions)
        # the full unfolding would have (H+1)*S states, which is more
        assert unfolded.nr_states < (HORIZON + 1) * NUM_STATES

    def test_state_maps(self, mdp):
        unfolder = payntbind.synthesis.FiniteHorizonUnfolder(mdp, HORIZON)
        unfolder.construct_unfolded_model()
        assert list(zip(unfolder.state_step.tolist(), unfolder.state_base.tolist(), strict=True)) == expected_states()
        # the unreachable state is never built
        assert 4 not in unfolder.state_base.tolist()

    def test_choice_maps(self, mdp):
        unfolder = payntbind.synthesis.FiniteHorizonUnfolder(mdp, HORIZON)
        unfolded = unfolder.construct_unfolded_model()
        expected = []
        for step in range(HORIZON):
            for state in LAYERS[step]:
                expected += [(step, choice) for choice in range(ROW_GROUPS[state], ROW_GROUPS[state + 1])]
        expected += [(HORIZON, -1) for _ in LAYERS[HORIZON]]
        assert list(zip(unfolder.choice_step.tolist(), unfolder.choice_base.tolist(), strict=True)) == expected
        assert len(expected) == unfolded.nr_choices

    def test_transitions_lead_to_the_copies_of_the_successors_in_the_next_step(self, mdp):
        unfolded = payntbind.synthesis.FiniteHorizonUnfolder(mdp, HORIZON).construct_unfolded_model()
        state_index = {pair: index for index, pair in enumerate(expected_states())}
        row_groups = list(unfolded.nondeterministic_choice_indices)
        for unfolded_state, (step, state) in enumerate(expected_states()):
            num_choices = row_groups[unfolded_state + 1] - row_groups[unfolded_state]
            # the choices of a state are the choices of the state it copies, in the same order; the last layer has a self-loop
            assert num_choices == (ROW_GROUPS[state + 1] - ROW_GROUPS[state] if step < HORIZON else 1)
            for offset in range(num_choices):
                transitions = row(unfolded.transition_matrix, row_groups[unfolded_state] + offset)
                if step == HORIZON:
                    assert transitions == {unfolded_state: 1.0}
                else:
                    original = SUCCESSORS[ROW_GROUPS[state] + offset]
                    assert transitions == {state_index[(step + 1, successor)]: p for successor, p in original.items()}

    def test_every_unfolded_state_is_reachable(self, mdp):
        unfolded = payntbind.synthesis.FiniteHorizonUnfolder(mdp, HORIZON).construct_unfolded_model()
        row_groups = list(unfolded.nondeterministic_choice_indices)
        reached, queue = {0}, [0]
        while queue:
            state = queue.pop()
            for choice in range(row_groups[state], row_groups[state + 1]):
                for successor in row(unfolded.transition_matrix, choice):
                    if successor not in reached:
                        reached.add(successor)
                        queue.append(successor)
        assert len(reached) == unfolded.nr_states

    def test_state_labels(self, mdp):
        unfolded = payntbind.synthesis.FiniteHorizonUnfolder(mdp, HORIZON).construct_unfolded_model()
        assert set(unfolded.labeling.get_labels()) == {"init", "goal"}
        # init only in the first layer, other labels wherever the state is copied
        assert list(unfolded.labeling.get_states("init")) == [0]
        assert list(unfolded.initial_states) == [0]
        assert list(unfolded.labeling.get_states("goal")) == [index for index, (_, state) in enumerate(expected_states()) if state == 3]

    def test_final_layer_labels(self, mdp):
        unfolded = payntbind.synthesis.FiniteHorizonUnfolder(mdp, HORIZON).construct_unfolded_model(["goal", "timeout"])
        last_layer = [index for index, (step, _) in enumerate(expected_states()) if step == HORIZON]
        goal = [index for index, (_, state) in enumerate(expected_states()) if state == 3]
        # an existing label is extended, a missing one is created
        assert list(unfolded.labeling.get_states("goal")) == sorted(set(goal) | set(last_layer))
        assert list(unfolded.labeling.get_states("timeout")) == last_layer

    def test_choice_labels(self, mdp):
        unfolder = payntbind.synthesis.FiniteHorizonUnfolder(mdp, HORIZON)
        unfolded = unfolder.construct_unfolded_model()
        assert set(unfolded.choice_labeling.get_labels()) == set(CHOICE_LABELS)
        for label, choices in CHOICE_LABELS.items():
            expected = [x for x, base in enumerate(unfolder.choice_base.tolist()) if base in choices]
            assert expected
            assert list(unfolded.choice_labeling.get_choices(label)) == expected

    def test_rewards(self, mdp):
        unfolder = payntbind.synthesis.FiniteHorizonUnfolder(mdp, HORIZON)
        unfolded = unfolder.construct_unfolded_model()
        rewards = list(unfolded.reward_models["cost"].state_action_rewards)
        assert rewards == [REWARDS[base] if base >= 0 else 0.0 for base in unfolder.choice_base.tolist()]

    def test_nonpositive_horizon_is_rejected(self, mdp):
        with pytest.raises(RuntimeError):
            payntbind.synthesis.FiniteHorizonUnfolder(mdp, 0)

    def test_the_maps_are_numpy_arrays(self, mdp):
        unfolder = payntbind.synthesis.FiniteHorizonUnfolder(mdp, HORIZON)
        unfolder.construct_unfolded_model()
        for array in (unfolder.state_step, unfolder.state_base, unfolder.choice_step, unfolder.choice_base):
            assert isinstance(array, np.ndarray)
            assert array.dtype == np.int64

    def test_values_of_unfolded_mdp_are_the_bounded_values(self, mdp):
        """Reaching the goal within H steps: from state 0 only the choice 0 can reach it, in exactly 2 steps."""
        formula = stormpy.parse_properties_without_context('Pmax=? [F "goal"]')[0]
        for horizon, expected in [(1, 0.0), (2, 1.0), (3, 1.0)]:
            unfolded = payntbind.synthesis.FiniteHorizonUnfolder(mdp, horizon).construct_unfolded_model()
            result = stormpy.model_checking(unfolded, formula)
            assert result.at(unfolded.initial_states[0]) == pytest.approx(expected)

    def test_alternating_states_have_one_state_per_layer(self):
        """Two states that lead to each other: only one of them is reachable in each step."""
        builder = stormpy.SparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)
        builder.new_row_group(0)
        builder.add_next_value(0, 1, 1.0)
        builder.new_row_group(1)
        builder.add_next_value(1, 0, 1.0)
        labeling = stormpy.storage.StateLabeling(2)
        labeling.add_label("init")
        labeling.add_label_to_state("init", 0)
        mdp = stormpy.storage.SparseMdp(stormpy.SparseModelComponents(transition_matrix=builder.build(), state_labeling=labeling))
        unfolder = payntbind.synthesis.FiniteHorizonUnfolder(mdp, 4)
        assert unfolder.num_states == 5
        unfolder.construct_unfolded_model()
        assert unfolder.state_base.tolist() == [0, 1, 0, 1, 0]


class TestUnfoldFamily:
    def test_each_hole_is_copied_for_every_step(self, family):
        unfolded = payntbind.synthesis.unfoldFamilyFiniteHorizon(family, 4)
        assert unfolded.numHoles() == 4 * len(HOLE_NUM_OPTIONS)
        for step in range(4):
            for hole, num_options in enumerate(HOLE_NUM_OPTIONS):
                assert unfolded.holeNumOptionsTotal(step * len(HOLE_NUM_OPTIONS) + hole) == num_options
                assert unfolded.holeOptions(step * len(HOLE_NUM_OPTIONS) + hole) == list(range(num_options))


class TestTimeUnfoldedColoring:
    @pytest.fixture
    def unfolded(self, mdp, coloring, family):
        unfolder = payntbind.synthesis.FiniteHorizonUnfolder(mdp, HORIZON)
        unfolder.construct_unfolded_model()
        return payntbind.synthesis.TimeUnfoldedColoring(coloring, family, unfolder), payntbind.synthesis.unfoldFamilyFiniteHorizon(family, HORIZON), unfolder

    def test_sizes(self, unfolded):
        coloring, _, unfolder = unfolded
        assert coloring.horizon == HORIZON
        assert coloring.num_holes == HORIZON * len(HOLE_NUM_OPTIONS)
        assert coloring.num_choices == unfolder.num_choices

    def test_requires_the_constructed_model(self, mdp, coloring, family):
        unfolder = payntbind.synthesis.FiniteHorizonUnfolder(mdp, HORIZON)
        with pytest.raises(RuntimeError):
            payntbind.synthesis.TimeUnfoldedColoring(coloring, family, unfolder)

    def test_colors_get_the_time_step(self, unfolded):
        coloring, _, unfolder = unfolded
        num_holes = len(HOLE_NUM_OPTIONS)
        for choice, (step, base) in enumerate(zip(unfolder.choice_step.tolist(), unfolder.choice_base.tolist(), strict=True)):
            expected = [] if base < 0 else [(hole + step * num_holes, option) for hole, option in CHOICE_TO_ASSIGNMENT[base]]
            assert coloring.getChoiceAssignment(choice) == expected
        # colors of different steps never share a parameter
        for choice, step in enumerate(unfolder.choice_step.tolist()):
            for hole, _ in coloring.getChoiceAssignment(choice):
                assert step * num_holes <= hole < (step + 1) * num_holes

    def test_out_of_range_choice_is_rejected(self, unfolded):
        coloring, _, _ = unfolded
        with pytest.raises(RuntimeError):
            coloring.getChoiceAssignment(coloring.num_choices)

    def compatible(self, unfolder, assignment):
        """Choices compatible with the assignment (an option of each hole in each step), computed from the maps."""
        selected = set()
        for choice, (step, base) in enumerate(zip(unfolder.choice_step.tolist(), unfolder.choice_base.tolist(), strict=True)):
            if base < 0 or all(assignment[step][hole] == option for hole, option in CHOICE_TO_ASSIGNMENT[base]):
                selected.add(choice)
        return selected

    def test_select_compatible_choices(self, unfolded):
        coloring, family, unfolder = unfolded
        num_holes = len(HOLE_NUM_OPTIONS)
        assignment = [(0, 2), (1, 2), (1, 0)]
        for step, options in enumerate(assignment):
            for hole, option in enumerate(options):
                family.holeSetOptions(step * num_holes + hole, [option])
        selected = set(coloring.selectCompatibleChoices(family))
        assert selected == self.compatible(unfolder, assignment)
        # uncolored choices and the last layer are always selected, colored ones only if they agree with the assignment of their step
        steps = unfolder.choice_step.tolist()
        assert {c for c in range(coloring.num_choices) if steps[c] == HORIZON} <= selected
        assert len(selected) < coloring.num_choices

    def test_select_compatible_choices_restricted_to_base_choices(self, unfolded):
        coloring, family, unfolder = unfolded
        num_holes = len(HOLE_NUM_OPTIONS)
        for step in range(HORIZON):
            family.holeSetOptions(step * num_holes, [0])
            family.holeSetOptions(step * num_holes + 1, [0])
        everything = set(coloring.selectCompatibleChoices(family))
        # exclude the choices of step 1 from the candidates: its colored choices are lost, but its uncolored choices cannot be excluded
        steps, bases = unfolder.choice_step.tolist(), unfolder.choice_base.tolist()
        candidates = [c for c in range(coloring.num_choices) if steps[c] != 1]
        restricted = set(coloring.selectCompatibleChoices(family, payntbind.synthesis.bitVectorFromIndices(coloring.num_choices, candidates)))
        lost = {c for c in everything if steps[c] == 1 and CHOICE_TO_ASSIGNMENT[bases[c]]}
        assert lost
        assert restricted == everything - lost

    def test_collect_hole_options(self, unfolded):
        coloring, _, unfolder = unfolded
        num_holes = len(HOLE_NUM_OPTIONS)
        steps, bases = unfolder.choice_step.tolist(), unfolder.choice_base.tolist()

        def find(step, base):
            return next(c for c in range(coloring.num_choices) if steps[c] == step and bases[c] == base)

        chosen = [find(0, 1), find(2, 0), find(2, 3), find(1, 2)]
        options = coloring.collectHoleOptions(payntbind.synthesis.bitVectorFromIndices(coloring.num_choices, chosen))
        assert len(options) == coloring.num_holes
        assert options[0 * num_holes + 0] == [1] and options[0 * num_holes + 1] == [2]
        assert options[1 * num_holes + 0] == [] and options[1 * num_holes + 1] == []
        assert options[2 * num_holes + 0] == [0] and options[2 * num_holes + 1] == [0]


class TestExports:
    def test_sparse_matrix_to_csr(self, mdp):
        row_pointers, columns, values = payntbind.synthesis.sparseMatrixToCsr(mdp.transition_matrix)
        assert row_pointers.tolist() == [0, 2, 3, 4, 5, 6, 7]
        assert columns.tolist() == [1, 2, 0, 3, 3, 3, 0]
        assert values.tolist() == [0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]

    def test_coloring_to_csr(self, coloring):
        pointers, holes, options = payntbind.synthesis.coloringToCsr(coloring)
        assert pointers.tolist() == [0, 1, 3, 3, 4, 4, 5]
        assert holes.tolist() == [hole for assignment in CHOICE_TO_ASSIGNMENT for hole, _ in assignment]
        assert options.tolist() == [option for assignment in CHOICE_TO_ASSIGNMENT for _, option in assignment]

    def test_bit_vector_from_indices(self):
        bit_vector = payntbind.synthesis.bitVectorFromIndices(10, np.array([7, 2, 2, 9]))
        assert bit_vector.size() == 10
        assert list(bit_vector) == [2, 7, 9]
        assert payntbind.synthesis.bitVectorFromIndices(4, []).number_of_set_bits() == 0

    def test_bit_vector_from_indices_rejects_out_of_range(self):
        with pytest.raises(RuntimeError):
            payntbind.synthesis.bitVectorFromIndices(3, [0, 3])
        with pytest.raises(RuntimeError):
            payntbind.synthesis.bitVectorFromIndices(3, [-1])
