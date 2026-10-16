"""Tests of the conditional reachability probability in finite-horizon-time-dependent.py, for models that are already unfolded in time: the solver and the exact
search are compared to a brute force over all color consistent policies of small random colored MDPs, and the values are double-checked by Storm."""

import itertools
import json
import logging
import os
import time

import numpy as np
import pytest
import scipy.sparse
import stormpy
from click.testing import CliRunner

import payntbind

import paynt.colored_mdp
import paynt.parameter_space.parameter_space
import paynt.parser.sketch
import paynt.task

from helpers.helper import get_sketch_paths

LUKO_MODELS = os.path.join(os.path.dirname(__file__), "..", "..", "models", "luko-models")
EVENT = 'F "goal" || F "condition"'


class LayeredInstance:
    """A random acyclic colored MDP in layers, like the product of a hidden Markov chain and a monitor that is unrolled in time.

    The states 0, 1 and 2 are absorbing: the sink, a condition state and a goal state (which is a condition state too). The other states form layers, the first
    one has the initial state only, and the choices of a state lead to states of the next layer and to the absorbing states. The states of a layer depend on
    the parameters of the layer: a parameter (a class of the layer) per class, whose options are the choices of the states. Everything is kept as plain python
    structures, so a brute force can use it without sharing any code with the solver.
    """

    SINK, STOP, GOAL = 0, 1, 2

    def __init__(self, seed, layers=3, width=2, max_options=3, classes=1, forced=False, free_choices=False):
        rng = np.random.default_rng(seed)
        self.layer_states, self.hole_num_options = [], []
        self.state_choices, self.choice_color, self.successors = [], [], []
        self.num_states = 3
        for layer in range(layers):
            count = 1 if layer == 0 else int(rng.integers(1, width + 1))
            self.layer_states.append(list(range(self.num_states, self.num_states + count)))
            self.num_states += count
        self.init = self.layer_states[0][0]
        for state in range(3):
            self.add_state([({}, {state: 1.0})])
        for layer, states in enumerate(self.layer_states):
            holes = []
            for _ in range(classes):
                holes.append(len(self.hole_num_options))
                self.hole_num_options.append(int(rng.integers(2, max_options + 1)))
            targets = (self.layer_states[layer + 1] if layer + 1 < len(self.layer_states) else []) + [self.SINK, self.STOP, self.GOAL]
            for _ in states:
                kind = rng.random()
                hole = holes[int(rng.integers(classes))]
                if kind < 0.15 and forced:
                    colors = [{}]
                elif kind < 0.3 and free_choices:
                    colors = [{} for _ in range(int(rng.integers(2, 4)))]
                else:
                    colors = [{hole: option} for option in range(self.hole_num_options[hole])]
                choices = []
                for color in colors:
                    successors = rng.choice(targets, size=int(rng.integers(1, min(3, len(targets)) + 1)), replace=False)
                    choices.append((color, {int(s): float(p) for s, p in zip(successors, rng.dirichlet(np.ones(len(successors))), strict=True)}))
                self.add_state(choices)

    @classmethod
    def explicit(cls, layers, hole_num_options, states):
        """An instance with the given layers (lists of state numbers, which start at 3) and numbers of options of the parameters.

        The states are given in the order of their numbers by their choices, which are pairs of a color and the successors.
        """
        instance = cls.__new__(cls)
        instance.layer_states = [list(layer) for layer in layers]
        instance.num_states = 3 + sum(len(layer) for layer in layers)
        instance.init = instance.layer_states[0][0]
        instance.hole_num_options = list(hole_num_options)
        instance.state_choices, instance.choice_color, instance.successors = [], [], []
        for state in range(3):
            instance.add_state([({}, {state: 1.0})])
        for choices in states:
            instance.add_state(choices)
        return instance

    @classmethod
    def dense(cls, seed, layers, width, options):
        """An instance in which every choice leads to all states of the next layer (and to the absorbing states) with generic probabilities, so that the reach
        vectors of different policies are different: in the layer l there are options^l distinct ones."""
        rng = np.random.default_rng(seed)
        layer_states = [[3]] + [list(range(4 + width * (layer - 1), 4 + width * layer)) for layer in range(1, layers)]
        states = []
        for layer, layer_of_states in enumerate(layer_states):
            targets = (layer_states[layer + 1] if layer + 1 < layers else []) + [0, 1, 2]
            for _ in layer_of_states:
                states.append([({layer: option}, dict(zip(targets, rng.dirichlet(np.ones(len(targets))).tolist(), strict=True))) for option in range(options)])
        return cls.explicit(layer_states, [options] * layers, states)

    def add_state(self, choices):
        self.state_choices.append(list(range(len(self.choice_color), len(self.choice_color) + len(choices))))
        for color, successors in choices:
            self.choice_color.append(color)
            self.successors.append(successors)

    @property
    def absorbing(self):
        return {self.SINK, self.STOP, self.GOAL}

    def ratio(self, policy):
        """The probabilities of reaching the goal and the condition, and so the conditional probability, of the policy (the choice of every state)."""
        mass = {self.init: 1.0}
        for state in [s for states in self.layer_states for s in states]:
            for successor, probability in self.successors[policy[state]].items():
                mass[successor] = mass.get(successor, 0.0) + mass.get(state, 0.0) * probability
        goal, condition = mass.get(self.GOAL, 0.0), mass.get(self.GOAL, 0.0) + mass.get(self.STOP, 0.0)
        return goal / condition if condition > 1e-12 else None

    def color_consistent_policies(self):
        """All policies that execute a parameter assignment (and choose arbitrarily in the states that have no colors)."""
        free_states = [s for s in range(self.num_states) if s not in self.absorbing and not self.choice_color[self.state_choices[s][0]]]
        domains = [range(num_options) for num_options in self.hole_num_options] + [range(len(self.state_choices[s])) for s in free_states]
        for combination in itertools.product(*domains):
            assignment, free = combination[: len(self.hole_num_options)], combination[len(self.hole_num_options) :]
            policy = {s: self.state_choices[s][0] for s in self.absorbing}
            for state in range(self.num_states):
                if state in self.absorbing:
                    continue
                if state in free_states:
                    policy[state] = self.state_choices[state][free[free_states.index(state)]]
                else:
                    policy[state] = next(c for c in self.state_choices[state] if all(assignment[h] == o for h, o in self.choice_color[c].items()))
            yield policy

    def best(self, maximize):
        """The optimal conditional probability over the color consistent policies (None if there is none that reaches the condition)."""
        values = [r for r in map(self.ratio, self.color_consistent_policies()) if r is not None]
        return (max if maximize else min)(values) if values else None

    def best_unconstrained(self, maximize):
        """The optimal conditional probability over all policies, which do not need memory."""
        states = [s for s in range(self.num_states) if s not in self.absorbing]
        values = []
        for choices in itertools.product(*[self.state_choices[s] for s in states]):
            policy = {s: self.state_choices[s][0] for s in self.absorbing} | dict(zip(states, choices, strict=True))
            if (value := self.ratio(policy)) is not None:
                values.append(value)
        return (max if maximize else min)(values) if values else None

    def arrays(self, fh):
        """The instance as the arrays of the solver, together with the goal and condition states."""
        row_groups = np.concatenate(([0], np.cumsum([len(choices) for choices in self.state_choices])))
        rows, columns, values = [], [], []
        for choice, successors in enumerate(self.successors):
            for successor, probability in sorted(successors.items()):
                rows.append(choice)
                columns.append(successor)
                values.append(probability)
        matrix = scipy.sparse.csr_matrix((values, (rows, columns)), shape=(len(self.choice_color), self.num_states))
        goal = np.zeros(self.num_states, dtype=bool)
        goal[self.GOAL] = True
        condition = goal.copy()
        condition[self.STOP] = True
        mdp = fh.MdpArrays(row_groups, np.repeat(np.arange(self.num_states), np.diff(row_groups)), matrix, np.zeros(len(self.choice_color)), goal, self.init)
        return mdp, condition

    def coloring(self):
        family = payntbind.synthesis.Family()
        for num_options in self.hole_num_options:
            family.addHole(num_options)
        row_groups = [0] + list(np.cumsum([len(choices) for choices in self.state_choices]))
        return payntbind.synthesis.Coloring(family, row_groups, [list(color.items()) for color in self.choice_color])

    def colored_mdp(self):
        """The instance as a stormpy MDP with the coloring, the way PAYNT would build it."""
        builder = stormpy.SparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)
        row = 0
        for choices in self.state_choices:
            builder.new_row_group(row)
            for choice in choices:
                for successor, probability in sorted(self.successors[choice].items()):
                    builder.add_next_value(row, successor, probability)
                row += 1
        labeling = stormpy.storage.StateLabeling(self.num_states)
        for label in ("init", "goal", "condition"):
            labeling.add_label(label)
        labeling.add_label_to_state("init", self.init)
        labeling.add_label_to_state("goal", self.GOAL)
        labeling.add_label_to_state("condition", self.GOAL)
        labeling.add_label_to_state("condition", self.STOP)
        components = stormpy.SparseModelComponents(transition_matrix=builder.build(overridden_column_count=self.num_states), state_labeling=labeling)
        mdp = stormpy.storage.SparseMdp(components)
        space = paynt.parameter_space.parameter_space.ParameterSpace()
        for hole, num_options in enumerate(self.hole_num_options):
            space.add_parameter(f"h{hole}", [f"o{option}" for option in range(num_options)])
        coloring = payntbind.synthesis.Coloring(space.native, mdp.nondeterministic_choice_indices, [list(color.items()) for color in self.choice_color])
        return paynt.colored_mdp.ColoredMdp(mdp, space, coloring, False, feature_kind="generic")


def make_task(maximize=True, line=None):
    return paynt.task.SynthesisTask([stormpy.parse_properties_without_context(line or f"P{'max' if maximize else 'min'}=? [{EVENT}]")[0]])


def solve(fh, instance, maximize=True, verify=False, **kwargs):
    return fh.solve_conditional(instance.colored_mdp(), make_task(maximize), tolerance=1e-12, verify=verify, **kwargs)


def layered_mdp(fh, instance):
    mdp, condition = instance.arrays(fh)
    colors = fh.ColorStructure.analyze(mdp, instance.coloring(), np.array(instance.hole_num_options, dtype=np.int64))
    return fh.LayeredMdp(mdp, colors, mdp.target, condition)


class TestConditionalObjective:
    @pytest.mark.parametrize(
        ("line", "maximize"),
        [
            (f"Pmax=? [{EVENT}]", True),
            (f"Pmin=? [{EVENT}]", False),
            (f"P>0.3 [{EVENT}]", True),
            (f"P>=0.3 [{EVENT}]", True),
            (f"P<0.3 [{EVENT}]", False),
            (f"P<=0.3 [{EVENT}]", False),
        ],
    )
    def test_direction_and_labels(self, fh, line, maximize):
        objective = fh.ConditionalObjective.from_task(make_task(line=line))
        assert (objective.maximize, objective.goal_label, objective.condition_label) == (maximize, "goal", "condition")
        assert fh.ConditionalObjective.is_conditional(make_task(line=line))

    def test_a_plain_reachability_property_is_not_conditional(self, fh):
        task = make_task(line='Pmax=? [F "goal"]')
        assert not fh.ConditionalObjective.is_conditional(task)
        with pytest.raises(fh.UnsupportedProblemError, match="conditional property"):
            fh.ConditionalObjective.from_task(task)

    @pytest.mark.parametrize("line", ['Pmax=? [F<=5 "goal" || F "condition"]', 'Pmax=? [G "goal" || F "condition"]', 'R{"r"}max=? [F "goal" || F "condition"]'])
    def test_unsupported_properties_are_rejected(self, fh, line):
        with pytest.raises(fh.UnsupportedProblemError):
            fh.ConditionalObjective.from_task(make_task(line=line))

    def test_multiple_properties_are_rejected(self, fh):
        task = paynt.task.SynthesisTask([stormpy.parse_properties_without_context(line)[0] for line in (f"Pmax=? [{EVENT}]", 'P>=0.5 [F "bad"]')])
        assert not fh.ConditionalObjective.is_conditional(task)
        with pytest.raises(fh.UnsupportedProblemError, match="exactly one property"):
            fh.ConditionalObjective.from_task(task)


class TestLayeredMdp:
    @pytest.mark.parametrize("seed", range(20))
    def test_states_are_layered_by_the_longest_path(self, fh, seed):
        instance = LayeredInstance(seed, layers=4, width=3, classes=2, forced=True, free_choices=True)
        layered = layered_mdp(fh, instance)
        assert layered.absorbing.tolist() == [s in instance.absorbing for s in range(instance.num_states)]
        reached = {instance.init}
        for states in instance.layer_states:
            reached |= {t for s in states if s in reached for c in instance.state_choices[s] for t in instance.successors[c]}
        level = {s: layered.level[s] for s in range(instance.num_states) if layered.level[s] >= 0}
        # the layer of a state is the length of the longest path to it, and the states of the layers are the reachable states that are not absorbing
        assert set(level) == reached - instance.absorbing
        for state, depth in level.items():
            predecessors = [p for p in level if any(state in instance.successors[c] for c in instance.state_choices[p])]
            assert depth == max((level[p] + 1 for p in predecessors), default=0)
        assert [sorted(layer.states.tolist()) for layer in layered.layers] == [sorted(s for s in level if level[s] == d) for d in range(len(layered.layers))]
        for layer in layered.layers:
            assert layer.counts.tolist() == [len(instance.state_choices[s]) for s in layer.states]
            assert layer.matrix.shape == (len(layer.choices), instance.num_states)

    def test_unreachable_states_are_not_in_layers(self, fh):
        instance = LayeredInstance(3, layers=3)
        instance.add_state([({}, {instance.GOAL: 1.0})])  # nothing leads to the new state
        instance.num_states += 1
        layered = layered_mdp(fh, instance)
        assert not layered.reached[-1]
        assert layered.level[-1] == -1
        assert all(instance.num_states - 1 not in layer.states for layer in layered.layers)

    def test_initial_state_that_is_absorbing_has_no_layers(self, fh):
        instance = LayeredInstance(1, layers=2)
        instance.init = instance.GOAL
        assert layered_mdp(fh, instance).layers == []

    @staticmethod
    def chain(first, second):
        """Two layers (the states 3 and 4): the first state has the choices first, the second one the choices second."""
        return LayeredInstance.explicit([[3], [4]], [2, 2], [first, second])

    def test_goal_states_have_to_be_condition_states(self, fh):
        instance = self.chain([({0: 0}, {4: 1.0}), ({0: 1}, {4: 1.0})], [({1: 0}, {0: 1.0}), ({1: 1}, {2: 1.0})])
        mdp, condition = instance.arrays(fh)
        colors = fh.ColorStructure.analyze(mdp, instance.coloring(), np.array(instance.hole_num_options, dtype=np.int64))
        goal = mdp.target.copy()
        goal[instance.SINK] = True  # the sink is reached by the first choice of the second state
        with pytest.raises(fh.UnsupportedProblemError, match="subset of the condition"):
            fh.LayeredMdp(mdp, colors, goal, condition)

    def test_condition_states_have_to_be_absorbing(self, fh):
        instance = self.chain([({0: 0}, {4: 1.0}), ({0: 1}, {4: 1.0})], [({1: 0}, {2: 1.0}), ({1: 1}, {1: 1.0})])
        mdp, condition = instance.arrays(fh)
        colors = fh.ColorStructure.analyze(mdp, instance.coloring(), np.array(instance.hole_num_options, dtype=np.int64))
        condition[4] = True  # a state that has choices
        with pytest.raises(fh.UnsupportedProblemError, match="condition states have to be absorbing"):
            fh.LayeredMdp(mdp, colors, mdp.target, condition)

    def test_cycles_are_rejected(self, fh):
        instance = self.chain([({0: 0}, {4: 1.0}), ({0: 1}, {4: 1.0})], [({1: 0}, {3: 1.0}), ({1: 1}, {2: 1.0})])
        with pytest.raises(fh.UnsupportedProblemError, match="not acyclic"):
            layered_mdp(fh, instance)

    def test_self_loops_of_states_that_are_not_absorbing_are_rejected(self, fh):
        instance = self.chain([({0: 0}, {4: 1.0}), ({0: 1}, {4: 1.0})], [({1: 0}, {4: 0.5, 2: 0.5}), ({1: 1}, {2: 1.0})])
        with pytest.raises(fh.UnsupportedProblemError, match="not acyclic"):
            layered_mdp(fh, instance)

    def test_a_component_in_several_layers_is_rejected(self, fh):
        # both states depend on the parameter 0, so a decision in the first layer would have to be made again in the second
        instance = LayeredInstance.explicit([[3], [4]], [2], [[({0: 0}, {4: 1.0}), ({0: 1}, {4: 1.0})], [({0: 0}, {2: 1.0}), ({0: 1}, {1: 1.0})]])
        with pytest.raises(fh.UnsupportedProblemError, match="different layers"):
            layered_mdp(fh, instance)

    def test_a_layer_is_the_longest_path_not_the_shortest(self, fh):
        # the state 5 can be reached in one step, but the longest path to it has two
        instance = LayeredInstance.explicit(
            [[3], [4], [5]],
            [2, 2, 2],
            [[({0: 0}, {4: 0.5, 5: 0.5}), ({0: 1}, {4: 1.0})], [({1: 0}, {5: 1.0}), ({1: 1}, {5: 1.0})], [({2: 0}, {2: 1.0}), ({2: 1}, {1: 1.0})]],
        )
        layered = layered_mdp(fh, instance)
        assert [layer.states.tolist() for layer in layered.layers] == [[3], [4], [5]]


class TestBestChoices:
    def test_ties_prefer_the_previous_choice_and_then_the_first(self, fh):
        instance = LayeredInstance(7, layers=2, max_options=3)
        layer = layered_mdp(fh, instance).layers[0]
        scores = np.zeros(len(layer.choices))
        tolerance = np.zeros(len(layer.choices))
        assert layer.best_choices(scores, tolerance, None).tolist() == [0]
        assert layer.best_choices(scores, tolerance, np.array([len(layer.choices) - 1])).tolist() == [len(layer.choices) - 1]
        scores[1] = 1.0
        assert layer.best_choices(scores, tolerance, None).tolist() == [1]
        assert layer.best_choices(scores, tolerance, np.array([0])).tolist() == [1]  # the previous choice is not among the best ones
        # choices within the tolerance of the best one are equally good
        assert layer.best_choices(scores, tolerance + 2.0, np.array([2])).tolist() == [2]

    def test_joint_option_tolerance_is_given_by_the_scale(self, fh):
        instance = LayeredInstance(8, layers=2, max_options=2)
        layered = layered_mdp(fh, instance)
        colors = layered.colors
        values = np.zeros(colors.num_joint_options)
        values[colors.comp_offset[0] + 1] = 1e-14  # noise next to terms of the size 1
        assert colors.best_joint_options(values, None)[0] == 1  # relative to the best value, the difference counts
        assert colors.best_joint_options(values, None, np.ones(colors.num_components))[0] == 0  # relative to the terms, it is nothing
        assert colors.best_joint_options(values, np.zeros(colors.num_components, dtype=np.int64), np.ones(colors.num_components))[0] == 0


class TestConditionalSolver:
    @pytest.mark.parametrize("seed", range(40))
    def test_the_solver_is_between_the_brute_force_optimum_and_the_unconstrained_one(self, fh, seed):
        rng = np.random.default_rng(seed)
        maximize = bool(rng.integers(2))
        instance = LayeredInstance(
            seed, layers=int(rng.integers(1, 5)), width=3, classes=int(rng.integers(1, 3)), forced=True, free_choices=bool(rng.integers(2))
        )
        optimum, unconstrained = instance.best(maximize), instance.best_unconstrained(maximize)
        if unconstrained is None:
            with pytest.raises(fh.UnsupportedProblemError, match="cannot be reached"):
                solve(fh, instance, maximize)
            return
        sign = 1 if maximize else -1
        result = solve(fh, instance, maximize)
        # the unconstrained value is the optimum over all policies
        assert result["unconstrained_value"] == pytest.approx(unconstrained, abs=1e-9)
        # the color consistent value is in between: the heuristic cannot beat the optimum
        assert optimum is not None
        assert sign * (result["value"] - optimum) <= 1e-9
        assert sign * (result["unconstrained_value"] - result["value"]) >= -1e-9
        # the value is the conditional probability of the policy, which is color consistent
        solution = result["solution"]
        policy = {state: int(choice) for state, choice in enumerate(solution.policy)}
        assert instance.ratio(policy) == pytest.approx(solution.value, abs=1e-9)
        assert policy in list(instance.color_consistent_policies()) or not instance.hole_num_options
        history = result["history"]
        assert all(sign * (new - old) >= -1e-9 for old, new in zip(history, history[1:], strict=False)), history

    @pytest.mark.parametrize("seed", range(100, 112))
    def test_a_single_decision_is_solved_exactly(self, fh, seed):
        """With a single layer the first reachability probabilities are the exact ones, and the iterations on lambda find the best option."""
        rng = np.random.default_rng(seed)
        maximize = bool(rng.integers(2))
        instance = LayeredInstance(seed, layers=1, max_options=5)
        optimum = instance.best(maximize)
        if optimum is not None:
            assert solve(fh, instance, maximize)["value"] == pytest.approx(optimum, abs=1e-9)

    @pytest.mark.parametrize("seed", range(200, 224))
    def test_storm_agrees_on_the_optimum_and_on_the_value_of_the_policy(self, fh, seed):
        rng = np.random.default_rng(seed)
        maximize = bool(rng.integers(2))
        instance = LayeredInstance(
            seed, layers=int(rng.integers(2, 5)), width=3, classes=int(rng.integers(1, 3)), forced=True, free_choices=bool(rng.integers(2))
        )
        if instance.best_unconstrained(maximize) is None:
            pytest.skip("the condition cannot be reached")
        result = solve(fh, instance, maximize, verify=True)
        assert result["storm_unconstrained_value"] == pytest.approx(result["unconstrained_value"], abs=1e-5)
        assert result["storm_value"] == pytest.approx(result["value"], abs=1e-5)
        assert result["consistent"]

    def test_without_colors_the_unconstrained_value_is_achieved(self, fh):
        instance = LayeredInstance(11, layers=3, free_choices=True)
        instance.choice_color = [{} for _ in instance.choice_color]
        instance.hole_num_options = []
        result = solve(fh, instance, True)
        assert result["value"] == pytest.approx(result["unconstrained_value"], abs=1e-12)

    def test_the_condition_must_be_reachable(self, fh):
        instance = LayeredInstance(12, layers=2)
        instance.successors = [{instance.SINK: 1.0} if c >= 3 else s for c, s in enumerate(instance.successors)]
        with pytest.raises(fh.UnsupportedProblemError, match="cannot be reached"):
            solve(fh, instance)

    def test_initial_state_in_the_condition(self, fh):
        instance = LayeredInstance(13, layers=2)
        instance.init = instance.GOAL
        assert solve(fh, instance)["value"] == 1.0
        instance.init = instance.STOP
        assert solve(fh, instance)["value"] == 0.0

    def test_verification_can_be_skipped(self, fh):
        result = solve(fh, LayeredInstance(14, layers=2), verify=False)
        assert "storm_value" not in result

    def test_a_model_of_an_unsupported_kind_is_rejected(self, fh):
        colored_mdp = LayeredInstance(15).colored_mdp()
        colored_mdp.feature_kind = "dt"
        with pytest.raises(fh.UnsupportedProblemError, match="not supported"):
            fh.solve_conditional(colored_mdp, make_task())

    def test_a_reachability_property_is_rejected(self, fh):
        with pytest.raises(fh.UnsupportedProblemError, match="conditional property"):
            fh.solve_conditional(LayeredInstance(16).colored_mdp(), make_task(line='Pmax=? [F "goal"]'))

    def test_threshold_properties_are_optimized_like_optimality_properties(self, fh):
        instance = LayeredInstance(17, layers=3)
        threshold = fh.solve_conditional(instance.colored_mdp(), make_task(line=f"P>0.3 [{EVENT}]"), tolerance=1e-12)
        assert threshold["value"] == pytest.approx(solve(fh, instance, True)["value"], abs=1e-12)
        below = fh.solve_conditional(instance.colored_mdp(), make_task(line=f"P<0.3 [{EVENT}]"), tolerance=1e-12)
        assert below["value"] == pytest.approx(solve(fh, instance, False)["value"], abs=1e-12)

    @pytest.mark.parametrize("seed", range(10))
    def test_reach_probabilities_are_those_of_the_policy_and_of_the_uniform_policy(self, fh, seed):
        instance = LayeredInstance(seed, layers=4, width=3, classes=2, forced=True, free_choices=True)
        solver = fh.ConditionalSolver(layered_mdp(fh, instance), fh.ConditionalObjective(None, True, "goal", "condition"))
        rng = np.random.default_rng(seed)
        policy = np.array([instance.state_choices[s][int(rng.integers(len(instance.state_choices[s])))] for s in range(instance.num_states)])
        for uniform in (False, True):
            expected = np.zeros(instance.num_states)
            expected[instance.init] = 1.0
            for state in [s for layer in instance.layer_states for s in layer]:
                choices = instance.state_choices[state] if uniform else [policy[state]]
                for choice in choices:
                    for successor, probability in instance.successors[choice].items():
                        expected[successor] += expected[state] * probability / len(choices)
            assert solver.reach_probabilities(None if uniform else policy) == pytest.approx(expected, abs=1e-12)
        numerator, evidence = solver.evaluate(policy)
        assert numerator == pytest.approx(expected_goal := solver.reach_probabilities(policy)[instance.GOAL])
        assert evidence == pytest.approx(expected_goal + solver.reach_probabilities(policy)[instance.STOP])

    def test_a_flat_objective_does_not_start_in_a_policy_without_evidence(self, fh):
        # with lambda = 0 both options are worth nothing, and the first one never reaches the condition: the solver has to start from the other one
        instance = LayeredInstance.explicit([[3]], [2], [[({0: 0}, {0: 1.0}), ({0: 1}, {1: 1.0})]])
        result = solve(fh, instance, True, verify=True)
        assert result["value"] == 0.0
        assert result["unconstrained_value"] == 0.0
        assert result["solution"].options.tolist() == [1]
        assert result["storm_value"] == pytest.approx(0.0, abs=1e-9)

    def test_the_minimum_with_a_flat_objective(self, fh):
        # with lambda = 1 both options are worth the same (nothing, and 1 - 1), and the first one never reaches the condition
        instance = LayeredInstance.explicit([[3]], [2], [[({0: 0}, {0: 1.0}), ({0: 1}, {2: 1.0})]])
        result = solve(fh, instance, False, verify=True)
        assert result["value"] == 1.0
        assert result["solution"].options.tolist() == [1]
        assert result["storm_value"] == pytest.approx(1.0, abs=1e-9)

    def test_the_policy_is_described_layer_by_layer(self, fh):
        instance = LayeredInstance(18, layers=3, classes=1)
        colored_mdp = instance.colored_mdp()
        result = fh.solve_conditional(colored_mdp, make_task(), tolerance=1e-12)
        policy = fh.describe_conditional_policy(colored_mdp, result["solver"], result["solution"])
        assert len(policy) == len(result["layered"].layers)
        for layer_index, assignment in enumerate(policy):
            # the parameter of the layer is the one of its states, and its option is the one the solution chose
            for name, label in assignment.items():
                hole = int(name[1:])
                assert hole == layer_index
                assert label == f"o{result['solution'].options[hole]}"


class TestExactSearch:
    @pytest.mark.parametrize("seed", range(60))
    def test_the_optimum_is_found(self, fh, seed):
        rng = np.random.default_rng(seed)
        maximize = bool(rng.integers(2))
        instance = LayeredInstance(seed, layers=int(rng.integers(1, 5)), width=3, classes=1, forced=bool(rng.integers(2)))
        optimum = instance.best(maximize)
        if optimum is None:
            pytest.skip("the condition cannot be reached")
        result = solve(fh, instance, maximize, dp=True)
        dp = result["dp"]
        assert dp.complete
        assert dp.solution.value == pytest.approx(optimum, abs=1e-9)
        assert result["value"] == pytest.approx(optimum, abs=1e-9)
        assert result["optimal"]
        # the policy is color consistent and has the value
        policy = {state: int(choice) for state, choice in enumerate(dp.solution.policy)}
        assert instance.ratio(policy) == pytest.approx(optimum, abs=1e-9)
        # the heuristic cannot be better
        assert (1 if maximize else -1) * (result["heuristic_value"] - optimum) <= 1e-9

    @pytest.mark.parametrize("seed", range(300, 312))
    def test_storm_agrees_with_the_exact_policy(self, fh, seed):
        instance = LayeredInstance(seed, layers=4, width=3, classes=1, forced=True)
        if instance.best(True) is None:
            pytest.skip("the condition cannot be reached")
        result = solve(fh, instance, True, verify=True, dp=True)
        assert result["storm_value"] == pytest.approx(result["value"], abs=1e-5)
        assert result["consistent"]

    @staticmethod
    def dp(fh, instance, maximize=True, **limits):
        solver = fh.ConditionalSolver(layered_mdp(fh, instance), fh.ConditionalObjective(None, maximize, "goal", "condition"))
        return fh.ReachVectorDp(solver).run(**{"max_states": 10**6} | limits)

    def test_policies_that_lead_to_the_same_reach_vector_are_one_state(self, fh):
        # three options of the initial state: the second one leads to the same distribution as the first one, but loses a part of the probability in the sink
        # (the conditional probability does not depend on that), the third one to another distribution
        instance = LayeredInstance.explicit(
            [[3], [4, 5]],
            [3, 2],
            [
                [({0: 0}, {4: 0.5, 5: 0.5}), ({0: 1}, {4: 0.25, 5: 0.25, 0: 0.5}), ({0: 2}, {4: 0.9, 5: 0.1})],
                [({1: 0}, {2: 1.0}), ({1: 1}, {1: 1.0})],
                [({1: 0}, {2: 1.0}), ({1: 1}, {1: 1.0})],
            ],
        )
        dp = self.dp(fh, instance)
        assert dp.complete
        assert dp.sizes == [1, 2]  # three policies of the first layer, two distinct reach vectors
        assert dp.solution.value == pytest.approx(instance.best(True))

    def test_the_search_does_not_depend_on_how_many_policies_there_are(self, fh):
        # eight layers with three options each: 3^8 = 6561 policies, but all options lead to the same state
        layers = [[3 + layer] for layer in range(8)]
        states = [[({layer: option}, {3 + layer + 1: 1.0} if layer < 7 else {2: 0.5, 1: 0.5}) for option in range(3)] for layer in range(8)]
        instance = LayeredInstance.explicit(layers, [3] * 8, states)
        dp = self.dp(fh, instance)
        assert dp.sizes == [1] * 8
        assert dp.solution.value == pytest.approx(0.5)

    def test_prefixes_that_ended_are_told_apart(self, fh):
        # a policy may end half of the mass in the goal and continue with the other half: a different continuation changes the result, so these states differ
        # from the ones that have not ended anything even if their reach vectors are proportional
        instance = LayeredInstance.explicit(
            [[3], [4]],
            [2, 2],
            [
                [({0: 0}, {4: 1.0}), ({0: 1}, {4: 0.5, 2: 0.5})],
                [({1: 0}, {1: 1.0}), ({1: 1}, {2: 0.2, 1: 0.8})],
            ],
        )
        assert self.dp(fh, instance).sizes == [1, 2]
        for maximize in (True, False):
            assert self.dp(fh, instance, maximize).solution.value == pytest.approx(instance.best(maximize))

    def test_the_limits_stop_the_search(self, fh):
        instance = LayeredInstance.dense(22, layers=4, width=2, options=3)
        complete = self.dp(fh, instance)
        assert complete.complete
        assert complete.sizes == [1, 3, 9, 27]
        for limits in ({"max_states": 1}, {"time_limit": 0.0}):
            stopped = self.dp(fh, instance, **limits)
            assert not stopped.complete
            assert len(stopped.sizes) < len(complete.sizes)
            # what has been found before the limit is a color consistent policy and has the value that it was found with
            if stopped.solution is not None:
                policy = {state: int(choice) for state, choice in enumerate(stopped.solution.policy)}
                assert instance.ratio(policy) == pytest.approx(stopped.solution.value, abs=1e-9)
                assert stopped.solution.value <= complete.solution.value + 1e-9

    def test_the_initial_state_that_is_absorbing(self, fh):
        instance = LayeredInstance(23, layers=2)
        instance.init = instance.GOAL
        result = solve(fh, instance, True, dp=True)
        assert result["dp"].complete
        assert result["dp"].solution.value == 1.0
        assert result["dp"].sizes == []

    def test_states_without_colors_may_have_only_one_choice(self, fh):
        # the second state has a single choice that no parameter restricts
        instance = LayeredInstance.explicit(
            [[3], [4, 5]],
            [2, 2],
            [
                [({0: 0}, {4: 0.7, 5: 0.3}), ({0: 1}, {4: 0.2, 5: 0.8})],
                [({}, {2: 0.3, 1: 0.7})],
                [({1: 0}, {2: 0.9, 1: 0.1}), ({1: 1}, {2: 0.1, 1: 0.9})],
            ],
        )
        for maximize in (True, False):
            result = solve(fh, instance, maximize, dp=True, verify=True)
            assert result["dp"].complete
            assert result["dp"].solution.value == pytest.approx(instance.best(maximize), abs=1e-12)
            assert result["storm_value"] == pytest.approx(result["value"], abs=1e-6)

    def test_free_states_with_several_choices_are_not_supported(self, fh):
        instance = LayeredInstance.explicit([[3], [4]], [2], [[({0: 0}, {4: 1.0}), ({0: 1}, {4: 1.0})], [({}, {2: 1.0}), ({}, {1: 1.0})]])
        with pytest.raises(fh.UnsupportedProblemError, match="no parameter restricts"):
            solve(fh, instance, True, dp=True)

    def test_several_components_with_probability_in_a_layer_are_not_supported(self, fh):
        # the two states of the second layer depend on different parameters and both are reached
        instance = LayeredInstance.explicit(
            [[3], [4, 5]],
            [2, 2, 2],
            [
                [({0: 0}, {4: 0.5, 5: 0.5}), ({0: 1}, {4: 0.5, 5: 0.5})],
                [({1: 0}, {2: 1.0}), ({1: 1}, {1: 1.0})],
                [({2: 0}, {2: 1.0}), ({2: 1}, {1: 1.0})],
            ],
        )
        with pytest.raises(fh.UnsupportedProblemError, match="several components"):
            solve(fh, instance, True, dp=True)
        # but the heuristic does not mind
        assert solve(fh, instance, True)["value"] == pytest.approx(instance.best(True), abs=1e-9)

    def test_transitions_have_to_lead_to_the_next_layer(self, fh):
        # the state 5 is in the third layer (it follows the state 4), but the first state leads to it as well
        instance = LayeredInstance.explicit(
            [[3], [4], [5]],
            [2, 2, 2],
            [
                [({0: 0}, {4: 0.5, 5: 0.5}), ({0: 1}, {4: 1.0})],
                [({1: 0}, {5: 1.0}), ({1: 1}, {5: 1.0})],
                [({2: 0}, {2: 1.0}), ({2: 1}, {1: 1.0})],
            ],
        )
        layered_mdp(fh, instance)  # a valid layered MDP
        with pytest.raises(fh.UnsupportedProblemError, match="next layer"):
            solve(fh, instance, True, dp=True)


class TestReport:
    """What the exact search tells about a property that is a threshold on the conditional probability."""

    @staticmethod
    def run(fh, caplog, line, dp=True):
        # the best policy has the conditional probability 0.4: a single choice that ends in the goal with the probability 0.3 or 0.4 (and in the condition else)
        instance = LayeredInstance.explicit([[3]], [2], [[({0: 0}, {2: 0.3, 1: 0.7}), ({0: 1}, {2: 0.4, 1: 0.6})]])
        with caplog.at_level(logging.INFO, logger="finite-horizon"):
            return fh.solve_conditional(instance.colored_mdp(), make_task(line=line), dp=dp)

    def test_it_proves_that_no_policy_satisfies_the_property(self, fh, caplog):
        result = self.run(fh, caplog, f"P>0.45 [{EVENT}]")
        assert result["value"] == pytest.approx(0.4)
        assert "proved that no color consistent policy satisfies the property" in caplog.text
        assert "the heuristic is optimal" in caplog.text

    def test_it_finds_a_policy_that_satisfies_the_property(self, fh, caplog):
        self.run(fh, caplog, f"P>0.35 [{EVENT}]")
        assert "found a policy that satisfies the property" in caplog.text

    def test_the_heuristic_alone_does_not_prove_anything(self, fh, caplog):
        self.run(fh, caplog, f"P>0.45 [{EVENT}]", dp=False)
        assert "proved" not in caplog.text
        assert "is not satisfied by the policy" in caplog.text


class TestWindows:
    """The backward passes change one layer at a time and can get stuck; windows change several layers at once when they do."""

    # (seed, maximize, layers, width, classes) of the instances whose backward passes get stuck in a policy that is not optimal; a window of several layers
    # leaves it
    STUCK = [
        (13, True, 5, 3, 2),
        (53, True, 3, 3, 2),
        (85, True, 3, 3, 1),
        (165, True, 4, 3, 2),
        (380, True, 3, 3, 1),
        (398, True, 5, 2, 1),
        (495, False, 4, 3, 2),
        (526, True, 4, 1, 1),
    ]

    @pytest.mark.parametrize(("seed", "maximize", "layers", "width", "classes"), STUCK)
    def test_windows_leave_the_local_optimum_of_the_backward_passes(self, fh, seed, maximize, layers, width, classes):
        instance = LayeredInstance(seed, layers=layers, width=width, max_options=3, classes=classes)
        sign = 1 if maximize else -1
        optimum = instance.best(maximize)
        assert optimum is not None
        stuck = solve(fh, instance, maximize)
        assert sign * (optimum - stuck["value"]) > 1e-6
        result = solve(fh, instance, maximize, verify=True, max_window=layers)
        assert result["value"] == pytest.approx(optimum, abs=1e-9)
        # every window improved the conditional probability, the values of the iterations never decrease
        solver = result["solver"]
        assert solver.window_moves
        assert all(sign * (move["value"] - stuck["value"]) > 0 and move["gain"] > 0 and 2 <= move["window"] <= layers for move in solver.window_moves)
        assert all(sign * (new - old) >= -1e-9 for old, new in zip(result["history"], result["history"][1:], strict=False)), result["history"]
        # the policy is color consistent and has the conditional probability, which Storm agrees with
        solution = result["solution"]
        policy = {state: int(choice) for state, choice in enumerate(solution.policy)}
        assert instance.ratio(policy) == pytest.approx(solution.value, abs=1e-9)
        assert policy in list(instance.color_consistent_policies())
        assert result["storm_value"] == pytest.approx(result["value"], abs=1e-5)
        assert result["consistent"]

    def test_a_default_run_does_not_use_windows(self, fh):
        seed, maximize, layers, width, classes = self.STUCK[0]
        instance = LayeredInstance(seed, layers=layers, width=width, max_options=3, classes=classes)
        result = solve(fh, instance, maximize)
        assert result["solver"].window_moves == []
        assert result["solver"].window_stats == {}
        assert solve(fh, instance, maximize, max_window=1, changes=2)["value"] == result["value"]

    @pytest.mark.parametrize("seed", range(600, 624))
    def test_the_value_does_not_decrease_with_the_size_of_the_window(self, fh, seed):
        rng = np.random.default_rng(seed)
        maximize = bool(rng.integers(2))
        instance = LayeredInstance(
            seed, layers=int(rng.integers(2, 6)), width=3, classes=int(rng.integers(1, 3)), forced=True, free_choices=bool(rng.integers(2))
        )
        optimum = instance.best(maximize)
        if optimum is None or instance.best_unconstrained(maximize) is None:
            return
        sign = 1 if maximize else -1
        for changes in (1, 2):
            values = [solve(fh, instance, maximize, max_window=window, changes=changes)["value"] for window in range(1, 6)]
            # a larger limit continues the search of a smaller one, which is a part of it (the windows of the smaller sizes are tried first)
            assert all(sign * (new - old) >= -1e-9 for old, new in zip(values, values[1:], strict=False)), values
            assert sign * (optimum - values[-1]) >= -1e-9

    def gain_is_exact(self, fh, instance, maximize):
        """Compare the gain of every window with the change of the objective N - lambda*D of the policy that it gives; returns the number of windows."""
        solver = solve(fh, instance, maximize)["solver"]
        best, _, _, _ = solver.solve(100, 1e-12)
        lam = best.value
        reach = solver.reach_probabilities(best.policy)
        values = solver.values_of(best.policy, 1.0 - lam, -lam)
        layers, colors = solver.layered.layers, solver.colors
        count = 0
        for start, size in itertools.product(range(len(layers)), range(1, len(layers) + 1)):
            if start + size > len(layers):
                continue
            inflow = solver.inflow(best.policy, reach, start)
            for index, layer in enumerate(layers[start : start + size]):
                for component in solver._reaching_components(layer, reach):
                    for option in range(int(colors.comp_size[component])):
                        if option == best.joint_options[component]:
                            continue
                        gain, joint, policy = solver.window_gain(best, values, inflow, start, size, [(index, int(component), option)])
                        new = solver.solution_of_policy(policy, joint)
                        # the objective of the solution is zero, since lambda is its conditional probability
                        assert solver.sign * (new.numerator - lam * new.evidence) == pytest.approx(gain, abs=1e-12)
                        count += 1
        return count

    @pytest.mark.parametrize("seed", range(700, 716))
    def test_the_gain_of_a_window_is_the_gain_of_the_objective(self, fh, seed):
        rng = np.random.default_rng(seed)
        maximize = bool(rng.integers(2))
        instance = LayeredInstance(
            seed, layers=int(rng.integers(2, 6)), width=3, classes=int(rng.integers(1, 3)), forced=True, free_choices=bool(rng.integers(2))
        )
        if instance.best_unconstrained(maximize) is None:
            return
        self.gain_is_exact(fh, instance, maximize)

    def test_the_gain_is_exact_if_the_paths_skip_layers(self, fh):
        """The first layer leads to the last one as well, which a window of the last two layers has to count as a path that enters it."""
        instance = LayeredInstance.explicit(
            [[3], [4, 5], [6]],
            [2, 2, 2],
            [
                [({0: 0}, {4: 0.5, 5: 0.5}), ({0: 1}, {4: 0.3, 6: 0.4, 1: 0.3})],
                [({1: 0}, {6: 0.6, 2: 0.2, 0: 0.2}), ({1: 1}, {6: 0.2, 1: 0.5, 2: 0.3})],
                [({1: 0}, {6: 0.5, 1: 0.5}), ({1: 1}, {6: 0.1, 2: 0.9})],
                [({2: 0}, {2: 0.7, 1: 0.1, 0: 0.2}), ({2: 1}, {2: 0.2, 1: 0.7, 0: 0.1})],
            ],
        )
        layered = layered_mdp(fh, instance)
        assert [layer.states.tolist() for layer in layered.layers] == [[3], [4, 5], [6]]
        for maximize in (True, False):
            assert self.gain_is_exact(fh, instance, maximize) > 0

    @pytest.mark.parametrize("seed", range(800, 810))
    def test_the_probability_of_a_state_is_the_inflow_and_the_flow_of_the_window(self, fh, seed):
        instance = LayeredInstance(seed, layers=4, width=3, classes=2, forced=True)
        solver = solve(fh, instance)["solver"]
        policy = solver.solve(100, 1e-12)[0].policy
        reach = solver.reach_probabilities(policy)
        layers = solver.layered.layers
        for start in range(len(layers)):
            flow = solver.inflow(policy, reach, start)
            for layer in layers[start:]:
                flow += solver._flow(layer, policy, reach[layer.states])
            states = np.concatenate([layer.states for layer in layers[start:]])
            assert flow[states] == pytest.approx(reach[states], abs=1e-12)

    def test_the_search_stops_at_the_deadline(self, fh):
        seed, maximize, layers, width, classes = self.STUCK[0]
        instance = LayeredInstance(seed, layers=layers, width=width, max_options=3, classes=classes)
        solver = solve(fh, instance, maximize)["solver"]
        solution = solver.solve(100, 1e-12)[0]
        assert solver.search_windows(solution, 2, 1e-12, deadline=time.time() - 1.0) is None
        assert solver.widen(solution, 3, 1e-12, deadline=time.time() - 1.0) is None

    def test_the_statistics_count_the_candidates_of_each_size(self, fh):
        seed, maximize, layers, width, classes = self.STUCK[0]
        instance = LayeredInstance(seed, layers=layers, width=width, max_options=3, classes=classes)
        solver = solve(fh, instance, maximize, max_window=3)["solver"]
        assert set(solver.window_stats) <= {2, 3}
        assert all(entry["candidates"] > 0 and entry["seconds"] >= 0 for entry in solver.window_stats.values())
        assert sum(entry["moves"] for entry in solver.window_stats.values()) == len(solver.window_moves) > 0
        assert {move["reference"] for move in solver.window_moves} <= {"unconstrained", "uniform"}

    def test_windows_larger_than_the_model_and_models_without_parameters_are_skipped(self, fh):
        seed, maximize, layers, width, classes = self.STUCK[0]
        instance = LayeredInstance(seed, layers=layers, width=width, max_options=3, classes=classes)
        assert solve(fh, instance, maximize, max_window=layers + 5)["solver"].window_stats.get(layers + 1) is None
        no_colors = LayeredInstance(11, layers=3, free_choices=True)
        no_colors.choice_color = [{} for _ in no_colors.choice_color]
        no_colors.hole_num_options = []
        result = solve(fh, no_colors, True, max_window=3)
        assert result["solver"].window_stats == {}
        assert result["value"] == pytest.approx(result["unconstrained_value"], abs=1e-12)


class TestCommandLine:
    @staticmethod
    def luko_files(name):
        path = os.path.join(LUKO_MODELS, name)
        if not os.path.exists(path):
            pytest.skip(f"the model {name} is not available")
        return os.path.join(path, "model.drn"), os.path.join(path, "property.props")

    def test_run_and_export_policy(self, fh, tmp_path):
        model, props = self.luko_files("05-icy-h40-fp")
        export_path = tmp_path / "policy.json"
        result = CliRunner().invoke(fh.main, [model, props, "--dp", "--show-policy", "--export-policy", str(export_path)])
        assert result.exit_code == 0, result.output
        exported = json.loads(export_path.read_text())
        assert exported["value"] == pytest.approx(0.9, abs=1e-9)
        assert exported["optimal"]
        assert exported["dp"]["complete"]
        assert len(exported["policy"]) == 41
        assert exported["unconstrained_value"] == pytest.approx(1.0)
        assert set(exported["references"]) == {"unconstrained", "uniform"}

    def test_a_conditional_property_is_solved_without_horizon_and_without_the_exact_search(self, fh, tmp_path):
        model, props = self.luko_files("05-icy-h40-fp")
        export_path = tmp_path / "policy.json"
        result = CliRunner().invoke(fh.main, [model, props, "--no-unfold", "--export-policy", str(export_path)])
        assert result.exit_code == 0, result.output
        assert "dp" not in json.loads(export_path.read_text())

    def test_a_conditional_property_does_not_take_a_horizon(self, fh):
        model, props = self.luko_files("05-icy-h40-fp")
        result = CliRunner().invoke(fh.main, [model, props, "5"])
        assert result.exit_code == 2
        assert "HORIZON" in result.output

    def test_a_reachability_property_needs_a_horizon(self, fh):
        sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
        result = CliRunner().invoke(fh.main, [sketch_path, props_path])
        assert result.exit_code == 2
        assert "HORIZON" in result.output

    def test_the_exact_search_is_for_conditional_properties(self, fh):
        sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
        result = CliRunner().invoke(fh.main, [sketch_path, props_path, "3", "--dp"])
        assert result.exit_code == 2
        assert "--dp" in result.output

    @pytest.mark.parametrize("option", [["--dp-timeout", "5"], ["--dp-max-states", "10"]])
    def test_the_limits_of_the_exact_search_need_it(self, fh, option):
        model, props = self.luko_files("05-icy-h40-fp")
        result = CliRunner().invoke(fh.main, [model, props, *option])
        assert result.exit_code == 2
        assert "--dp" in result.output

    def test_alpha_vectors_do_not_support_conditional_properties(self, fh):
        model, props = self.luko_files("05-icy-h40-fp")
        result = CliRunner().invoke(fh.main, [model, props, "--alpha-vectors", "8"])
        assert result.exit_code == 2
        assert "--alpha-vectors" in result.output

    def test_ar_does_not_support_conditional_properties(self, fh):
        model, props = self.luko_files("05-icy-h40-fp")
        result = CliRunner().invoke(fh.main, [model, props, "--ar"])
        assert result.exit_code == 2
        assert "--ar" in result.output

    def test_the_limits_of_the_exact_search_are_used(self, fh, tmp_path):
        model, props = self.luko_files("01-airportB7-h9-fn")
        export_path = tmp_path / "policy.json"
        result = CliRunner().invoke(fh.main, [model, props, "--dp", "--dp-max-states", "10", "--export-policy", str(export_path)])
        assert result.exit_code == 0, result.output
        exported = json.loads(export_path.read_text())
        assert not exported["dp"]["complete"]
        assert not exported["optimal"]

    def test_windows_are_exported(self, fh, tmp_path):
        model, props = self.luko_files("02-airportB7-h8-fn")
        export_path = tmp_path / "policy.json"
        result = CliRunner().invoke(fh.main, [model, props, "--no-unfold", "--window", "3", "--window-changes", "2", "--export-policy", str(export_path)])
        assert result.exit_code == 0, result.output
        exported = json.loads(export_path.read_text())
        assert exported["windows"]["max_window"] == 3
        assert exported["windows"]["changes"] == 2
        assert exported["windows"]["moves"]
        assert set(exported["windows"]["sizes"]) <= {"2", "3"}
        assert exported["value"] == pytest.approx(0.333317, abs=1e-5)

    def test_the_changes_of_a_window_need_windows(self, fh):
        model, props = self.luko_files("05-icy-h40-fp")
        result = CliRunner().invoke(fh.main, [model, props, "--window-changes", "2"])
        assert result.exit_code == 2
        assert "--window" in result.output

    @pytest.mark.parametrize("option", ["--window", "--window-changes"])
    def test_windows_must_be_positive(self, fh, option):
        model, props = self.luko_files("05-icy-h40-fp")
        assert CliRunner().invoke(fh.main, [model, props, option, "0"]).exit_code == 2
