"""Tests of finite-horizon-time-dependent.py: the solver is compared to a brute force over all time-dependent color consistent policies on small random colored
MDPs, and the values are double-checked by Storm on the unfolded colored MDP built by payntbind."""

import itertools
import logging
import time
import types

import numpy as np
import pytest
import scipy.sparse
import stormpy

import payntbind

import paynt.colored_mdp
import paynt.parameter_space.parameter_space
import paynt.parser.sketch
import paynt.synthesizer.synthesizer
import paynt.task

from helpers.helper import get_sketch_paths


class Instance:
    """A random colored MDP with a few components of parameters.

    Each state is either free (uncolored choices, possibly several) or colored: its choices then cover all combinations of options of the parameters of one of
    the components. Everything is kept as plain python structures, so a brute force can use it without sharing any code with the solver.
    """

    def __init__(self, seed, num_states, max_holes=3, max_hole_options=3, max_component_holes=2, init_is_target=False):
        rng = np.random.default_rng(seed)
        self.num_states = num_states
        self.init = 0
        self.hole_num_options = []
        components = []
        for _ in range(int(rng.integers(1, 4))):
            num_holes = min(int(rng.integers(1, max_component_holes + 1)), max_holes - len(self.hole_num_options))
            if num_holes <= 0:
                break
            components.append(list(range(len(self.hole_num_options), len(self.hole_num_options) + num_holes)))
            self.hole_num_options += [int(rng.integers(1, max_hole_options + 1)) for _ in range(num_holes)]
        # shuffle the numbering of parameters
        renumbering = rng.permutation(len(self.hole_num_options))
        components = [[int(renumbering[hole]) for hole in component] for component in components]
        options = [0] * len(self.hole_num_options)
        for old, num_options in enumerate(self.hole_num_options):
            options[renumbering[old]] = num_options
        self.hole_num_options = options

        self.state_choices = []
        self.choice_color = []
        for _ in range(num_states):
            first = len(self.choice_color)
            component = int(rng.integers(-1, len(components))) if components else -1
            if component >= 0:
                combinations = list(itertools.product(*[range(self.hole_num_options[hole]) for hole in components[component]]))
                rng.shuffle(combinations)
                for combination in combinations:
                    pairs = list(zip(components[component], combination, strict=True))
                    rng.shuffle(pairs)
                    self.choice_color.append(dict(pairs))
            else:
                self.choice_color += [{} for _ in range(int(rng.integers(1, 4)))]
            self.state_choices.append(list(range(first, len(self.choice_color))))

        self.successors, self.reward = [], []
        for _ in self.choice_color:
            successors = rng.choice(num_states, size=int(rng.integers(1, 4)), replace=False)
            probabilities = rng.dirichlet(np.ones(len(successors)))
            self.successors.append({int(s): float(p) for s, p in zip(successors, probabilities, strict=True)})
            self.reward.append(float(rng.choice([0.0, 0.5, 1.0, 2.0])))
        self.target = {state for state in range(1, num_states) if rng.random() < 0.25} or {num_states - 1}
        if init_is_target:
            self.target.add(self.init)

    @classmethod
    def explicit(cls, hole_num_options, state_colors, target=None):
        """An instance with the given colors of the choices of each state; every choice leads to the last state, which is the (only) target."""
        instance = cls.__new__(cls)
        instance.num_states = len(state_colors) + 1
        instance.init = 0
        instance.hole_num_options = list(hole_num_options)
        instance.state_choices, instance.choice_color = [], []
        for colors in list(state_colors) + [[{}]]:
            instance.state_choices.append(list(range(len(instance.choice_color), len(instance.choice_color) + len(colors))))
            instance.choice_color += colors
        instance.successors = [{instance.num_states - 1: 1.0} for _ in instance.choice_color]
        instance.reward = [1.0] * len(instance.choice_color)
        instance.target = {instance.num_states - 1} if target is None else set(target)
        return instance

    @property
    def num_holes(self):
        return len(self.hole_num_options)

    def arrays(self, fh, objective):
        """The instance as the arrays of the solver."""
        row_groups = np.concatenate(([0], np.cumsum([len(choices) for choices in self.state_choices])))
        rows, columns, values = [], [], []
        for choice, successors in enumerate(self.successors):
            for successor, probability in sorted(successors.items()):
                rows.append(choice)
                columns.append(successor)
                values.append(probability)
        matrix = scipy.sparse.csr_matrix((values, (rows, columns)), shape=(len(self.choice_color), self.num_states))
        target = np.zeros(self.num_states, dtype=bool)
        target[list(self.target)] = True
        reward = np.array(self.reward) if objective.is_reward else np.zeros(len(self.reward))
        return fh.MdpArrays(row_groups, np.repeat(np.arange(self.num_states), np.diff(row_groups)), matrix, reward, target, self.init)

    def coloring(self):
        family = payntbind.synthesis.Family()
        for num_options in self.hole_num_options:
            family.addHole(num_options)
        row_groups = [0] + list(np.cumsum([len(choices) for choices in self.state_choices]))
        return payntbind.synthesis.Coloring(family, row_groups, [list(color.items()) for color in self.choice_color])

    def colored_mdp(self, is_reward):
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
        for label in ("init", "goal"):
            labeling.add_label(label)
        labeling.add_label_to_state("init", self.init)
        for state in self.target:
            labeling.add_label_to_state("goal", state)
        reward_model = stormpy.SparseRewardModel(optional_state_action_reward_vector=self.reward)
        components = stormpy.SparseModelComponents(
            transition_matrix=builder.build(overridden_column_count=self.num_states),
            state_labeling=labeling,
            reward_models={"r": reward_model} if is_reward else {},
        )
        mdp = stormpy.storage.SparseMdp(components)
        space = paynt.parameter_space.parameter_space.ParameterSpace()
        for hole, num_options in enumerate(self.hole_num_options):
            space.add_parameter(f"h{hole}", [str(option) for option in range(num_options)])
        coloring = payntbind.synthesis.Coloring(space.native, mdp.nondeterministic_choice_indices, [list(color.items()) for color in self.choice_color])
        return paynt.colored_mdp.ColoredMdp(mdp, space, coloring, False, feature_kind="generic")


def make_task(maximize, is_reward):
    direction = "max" if maximize else "min"
    text = f'R{{"r"}}{direction}=? [F "goal"]' if is_reward else f'P{direction}=? [F "goal"]'
    return paynt.task.SynthesisTask([stormpy.parse_properties_without_context(text)[0]])


def brute_force(instance, horizon, maximize, is_reward):
    """The optimal value over all sequences of parameter assignments, one for each step; free choices are optimized."""
    terminal = [0.0 if is_reward else float(state in instance.target) for state in range(instance.num_states)]
    assignments = list(itertools.product(*[range(num_options) for num_options in instance.hole_num_options]))

    def step(values, assignment):
        new_values = []
        for state in range(instance.num_states):
            if state in instance.target:
                new_values.append(terminal[state] if not is_reward else 0.0)
                continue
            candidates = [
                (instance.reward[choice] if is_reward else 0.0) + sum(p * values[s] for s, p in instance.successors[choice].items())
                for choice in instance.state_choices[state]
                if all(assignment[hole] == option for hole, option in instance.choice_color[choice].items())
            ]
            new_values.append(max(candidates) if maximize else min(candidates))
        return new_values

    best = None
    for sequence in itertools.product(assignments, repeat=horizon):
        values = [1.0 if (not is_reward and state in instance.target) else 0.0 for state in range(instance.num_states)]
        for assignment in reversed(sequence):
            values = step(values, assignment)
        best = values[instance.init] if best is None else (max if maximize else min)(best, values[instance.init])
    return best


def solve(fh, instance, horizon, maximize, is_reward, verify=False, **kwargs):
    task = make_task(maximize, is_reward)
    return fh.solve_time_dependent(instance.colored_mdp(is_reward), task, horizon, max_iterations=100, tolerance=1e-12, verify=verify, **kwargs)


@pytest.fixture
def pomdp_maze():
    sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
    factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path, precision=1e-6)
    return factory.build(), task


class TestSolverAgainstBruteForce:
    @pytest.mark.parametrize("seed", range(24))
    def test_value_is_between_optimum_and_unconstrained_value(self, fh, seed):
        rng = np.random.default_rng(seed)
        maximize, is_reward, horizon = bool(rng.integers(2)), bool(rng.integers(2)), int(rng.integers(1, 4))
        instance = Instance(seed, int(rng.integers(3, 7)))
        result = solve(fh, instance, horizon, maximize, is_reward)
        sign = 1 if maximize else -1
        optimum = brute_force(instance, horizon, maximize, is_reward)
        assert sign * (result["value"] - optimum) <= 1e-9
        assert sign * (result["unconstrained_value"] - optimum) >= -1e-9
        if horizon == 1:
            # a single step is solved exactly
            assert result["value"] == pytest.approx(optimum, abs=1e-9)

    @pytest.mark.parametrize("seed", range(24, 40))
    def test_larger_components_and_single_option_parameters(self, fh, seed):
        rng = np.random.default_rng(seed)
        maximize, is_reward, horizon = bool(rng.integers(2)), bool(rng.integers(2)), int(rng.integers(1, 3))
        instance = Instance(seed, int(rng.integers(3, 7)), max_holes=4, max_component_holes=4, init_is_target=bool(rng.random() < 0.2))
        result = solve(fh, instance, horizon, maximize, is_reward)
        sign = 1 if maximize else -1
        assert sign * (result["value"] - brute_force(instance, horizon, maximize, is_reward)) <= 1e-9


class TestSolverInvariants:
    @pytest.mark.parametrize("seed", range(100, 124))
    def test_iterations_improve_and_agree_with_storm(self, fh, seed):
        rng = np.random.default_rng(seed)
        maximize, is_reward, horizon = bool(rng.integers(2)), bool(rng.integers(2)), int(rng.integers(1, 16))
        instance = Instance(seed, int(rng.integers(8, 26)), max_holes=10)
        result = solve(fh, instance, horizon, maximize, is_reward, verify=True)
        sign = 1 if maximize else -1
        history = result["history"]
        assert all(sign * (new - old) >= -1e-9 for old, new in zip(history, history[1:], strict=False)), history
        # the value computed backward is the value of the policy computed forward
        assert result["forward_value"] == pytest.approx(result["value"], abs=1e-9)
        # Storm agrees on the optimum of the unfolded MDP and on the value of the policy, which is color consistent in every step
        assert result["storm_unconstrained_value"] == pytest.approx(result["unconstrained_value"], abs=1e-5)
        assert result["storm_value"] == pytest.approx(result["value"], abs=1e-5)
        assert result["consistent"]
        assert sign * (result["unconstrained_value"] - result["value"]) >= -1e-9

    def test_without_colors_the_unconstrained_value_is_achieved(self, fh):
        instance = Instance(7, 8, max_holes=0)
        assert instance.num_holes == 0
        result = solve(fh, instance, 6, True, False, verify=True)
        assert result["value"] == pytest.approx(result["unconstrained_value"], abs=1e-12)
        assert result["storm_value"] == pytest.approx(result["value"], abs=1e-5)

    def test_initial_state_in_target(self, fh):
        instance = Instance(3, 5, init_is_target=True)
        assert solve(fh, instance, 4, True, False)["value"] == 1.0
        assert solve(fh, instance, 4, False, True)["value"] == 0.0


class TestColorStructure:
    def make(self, fh, instance, is_reward=False):
        objective = fh.Objective(None, True, is_reward, "goal", "r")
        mdp = instance.arrays(fh, objective)
        return mdp, fh.ColorStructure.analyze(mdp, instance.coloring(), np.array(instance.hole_num_options, dtype=np.int64))

    def test_joint_options_are_indexed_consistently(self, fh):
        instance = Instance(5, 6, max_holes=4, max_component_holes=3)
        mdp, colors = self.make(fh, instance)
        assert colors.num_joint_options == colors.comp_offset[-1] == colors.comp_size.sum()
        # decoding the joint option of each choice gives back the options of its color
        for index, choice in enumerate(colors.colored_choices):
            joint = np.zeros((1, colors.num_components), dtype=np.int64)
            joint[0, colors.choice_component[index]] = colors.choice_joint[index] - colors.comp_offset[colors.choice_component[index]]
            options = colors.decode(joint)[0]
            for hole, option in instance.choice_color[choice].items():
                assert options[hole] == option

    def test_pomdp_maze_has_one_parameter_per_component(self, fh, pomdp_maze):
        colored_mdp, task = pomdp_maze
        objective = fh.Objective.from_task(task)
        mdp = fh.MdpArrays.from_mdp(colored_mdp.underlying_mdp, objective)
        num_options = np.array([len(labels) for labels in colored_mdp.parameter_space.parameter_to_option_labels])
        colors = fh.ColorStructure.analyze(mdp, colored_mdp.coloring, num_options)
        assert colors.num_components == colored_mdp.parameter_space.num_parameters
        assert colors.num_joint_options == num_options.sum()

    def test_state_with_colored_and_uncolored_choices_is_rejected(self, fh):
        instance = Instance.explicit([2], [[{0: 0}, {0: 1}, {}]])
        with pytest.raises(fh.UnsupportedProblemError, match="both colored and uncolored"):
            self.make(fh, instance)

    def test_choices_with_the_same_color_are_rejected(self, fh):
        instance = Instance.explicit([2], [[{0: 0}, {0: 0}]])
        with pytest.raises(fh.UnsupportedProblemError, match="same color"):
            self.make(fh, instance)

    def test_state_missing_a_combination_of_options_is_rejected(self, fh):
        instance = Instance.explicit([2, 2], [[{0: 0, 1: 0}, {0: 0, 1: 1}, {0: 1, 1: 0}]])
        with pytest.raises(fh.UnsupportedProblemError, match="do not cover all options"):
            self.make(fh, instance)

    def test_choices_of_a_state_depending_on_different_parameters_are_rejected(self, fh):
        instance = Instance.explicit([2, 2], [[{0: 0}, {1: 0}]])
        with pytest.raises(fh.UnsupportedProblemError, match="different sets of parameters"):
            self.make(fh, instance)

    def test_states_depending_on_different_parameters_of_a_component_are_rejected(self, fh):
        # the states depend on {h0,h1} and {h1}, which makes {h0,h1} a component that the second state does not depend on in full
        instance = Instance.explicit([2, 2], [[{0: 0, 1: 0}, {0: 0, 1: 1}, {0: 1, 1: 0}, {0: 1, 1: 1}], [{1: 0}, {1: 1}]])
        with pytest.raises(fh.UnsupportedProblemError, match="depends on only some of the parameters"):
            self.make(fh, instance)

    def test_valid_explicit_structure_is_accepted(self, fh):
        instance = Instance.explicit(
            [2, 3], [[{0: 0, 1: 0}, {0: 0, 1: 1}, {0: 0, 1: 2}, {0: 1, 1: 0}, {0: 1, 1: 1}, {0: 1, 1: 2}], [{}, {}], [{0: 1, 1: 2}, {0: 0, 1: 1}]]
        )
        with pytest.raises(fh.UnsupportedProblemError):
            self.make(fh, instance)  # the third state depends on both parameters but misses combinations
        instance = Instance.explicit([2, 3], [[{0: 0, 1: 0}, {0: 0, 1: 1}, {0: 0, 1: 2}, {0: 1, 1: 0}, {0: 1, 1: 1}, {0: 1, 1: 2}], [{}, {}]])
        _, colors = self.make(fh, instance)
        assert colors.num_components == 1
        assert colors.comp_size.tolist() == [6]


class TestObjective:
    def load_task(self, line):
        return paynt.task.SynthesisTask([stormpy.parse_properties_without_context(line)[0]])

    def test_reachability_probability(self, fh):
        objective = fh.Objective.from_task(self.load_task('Pmax=? [F "goal"]'))
        assert (objective.maximize, objective.is_reward, objective.target_label) == (True, False, "goal")
        assert objective.target_value == 1.0

    def test_reachability_reward(self, fh):
        objective = fh.Objective.from_task(self.load_task('R{"steps"}min=? [F "goal"]'))
        assert (objective.maximize, objective.is_reward, objective.target_label, objective.reward_name) == (False, True, "goal", "steps")
        assert objective.target_value == 0.0

    @pytest.mark.parametrize("line", ['Pmax=? [F<=5 "goal"]', 'R{"r"}max=? [C<=5]', 'R{"r"}max=? [Cdiscount=0.9]', 'Pmax=? [G "safe"]'])
    def test_unsupported_properties_are_rejected(self, fh, line):
        with pytest.raises(fh.UnsupportedProblemError):
            fh.Objective.from_task(self.load_task(line))

    def test_multiple_properties_are_rejected(self, fh):
        task = paynt.task.SynthesisTask([stormpy.parse_properties_without_context(line)[0] for line in ('Pmax=? [F "goal"]', 'P>=0.5 [F "bad"]')])
        with pytest.raises(fh.UnsupportedProblemError, match="exactly one property"):
            fh.Objective.from_task(task)


class TestPomdp:
    def test_unfolded_colored_mdp_and_storm_agree_with_the_solver(self, fh, pomdp_maze):
        colored_mdp, task = pomdp_maze
        horizon = 8
        result = fh.solve_time_dependent(colored_mdp, task, horizon)
        # the values are the ones of the unfolded MDP checked by Storm
        assert result["storm_unconstrained_value"] == pytest.approx(result["unconstrained_value"], abs=1e-5)
        assert result["storm_value"] == pytest.approx(result["value"], abs=1e-5)
        assert result["consistent"]
        # minimizing the expected number of steps: consistency cannot help
        assert result["value"] >= result["unconstrained_value"] - 1e-9
        # the policy uses a different observation-based assignment in each step, which is only possible with the parameters of every step
        assert result["solution"].options.shape == (horizon, colored_mdp.parameter_space.num_parameters)

    def test_unfolded_colored_mdp_has_a_parameter_per_step(self, fh, pomdp_maze):
        colored_mdp, task = pomdp_maze
        unfolded = fh.UnfoldedColoredMdp.build(colored_mdp, 5, [])
        base = colored_mdp.parameter_space
        assert unfolded.parameter_space.num_parameters == 5 * base.num_parameters
        assert unfolded.parameter_space.parameter_name(3 * base.num_parameters + 2) == f"{base.parameter_name(2)}@4"
        # only the reachable pairs (s,t) are built
        assert unfolded.mdp.nr_states <= 6 * colored_mdp.underlying_mdp.nr_states

    def test_memory_gives_components_of_two_parameters(self, fh):
        sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path, precision=1e-6, task_kwargs={"memory_size": 2})
        colored_mdp = factory.build()
        objective = fh.Objective.from_task(task)
        mdp = fh.MdpArrays.from_mdp(colored_mdp.underlying_mdp, objective)
        num_options = np.array([len(labels) for labels in colored_mdp.parameter_space.parameter_to_option_labels])
        colors = fh.ColorStructure.analyze(mdp, colored_mdp.coloring, num_options)
        assert colors.num_components < colored_mdp.parameter_space.num_parameters
        result = fh.solve_time_dependent(colored_mdp, task, 6)
        assert result["storm_value"] == pytest.approx(result["value"], abs=1e-5)
        assert result["consistent"]

    def test_decision_tree_colorings_are_not_supported(self, fh):
        sketch_path, props_path = get_sketch_paths("tests/dt-maze")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        with pytest.raises(fh.UnsupportedProblemError, match="kind 'dt'"):
            fh.solve_time_dependent(factory.build(), task, 3)


class TestUnfoldedColoredMdp:
    @staticmethod
    def reachable_layers(instance, horizon):
        """The states that are reachable in t steps, for each t, found by a breadth-first search that is independent of the unfolder."""
        layers = [{instance.init}]
        for _ in range(horizon):
            layers.append({successor for state in layers[-1] for choice in instance.state_choices[state] for successor in instance.successors[choice]})
        return layers

    @pytest.mark.parametrize("seed", range(120, 150))
    def test_only_reachable_states_are_built(self, fh, seed):
        rng = np.random.default_rng(seed)
        instance = Instance(seed, int(rng.integers(4, 14)))
        horizon = int(rng.integers(1, 8))
        unfolded = fh.UnfoldedColoredMdp.build(instance.colored_mdp(False), horizon, [])
        layers = self.reachable_layers(instance, horizon)
        assert unfolded.mdp.nr_states == sum(len(layer) for layer in layers)
        assert unfolded.mdp.nr_choices == sum(len(instance.state_choices[s]) for layer in layers[:-1] for s in layer) + len(layers[-1])
        transitions = sum(len(instance.successors[c]) for layer in layers[:-1] for s in layer for c in instance.state_choices[s]) + len(layers[-1])
        assert unfolded.mdp.nr_transitions == transitions
        assert unfolded.state_step.tolist() == [step for step, layer in enumerate(layers) for _ in layer]
        assert unfolded.state_base.tolist() == [state for layer in layers for state in sorted(layer)]
        # a pair (s,t) is built if and only if it is reachable, and so no more states are built than there are pairs
        assert unfolded.mdp.nr_states <= (horizon + 1) * instance.num_states

    def test_unreachable_states_are_not_built(self, fh):
        # the second state cannot be reached from the first one
        instance = Instance.explicit([2], [[{0: 0}, {0: 1}], [{}]])
        instance.successors = [{0: 1.0}, {0: 1.0}, {1: 1.0}, {2: 1.0}]
        unfolded = fh.UnfoldedColoredMdp.build(instance.colored_mdp(False), 5, [])
        assert set(unfolded.state_base.tolist()) == {0}
        assert unfolded.mdp.nr_states == 6

    @pytest.mark.parametrize("seed", range(150, 160))
    def test_policy_selects_one_choice_in_each_state(self, fh, seed):
        rng = np.random.default_rng(seed)
        instance = Instance(seed, int(rng.integers(4, 12)))
        horizon = int(rng.integers(1, 6))
        result = solve(fh, instance, horizon, True, False, verify=True)
        unfolded = result["unfolded"]
        policy = result["solution"].policy
        choices = unfolded.policy_choices(policy)
        assert choices.number_of_set_bits() == unfolded.mdp.nr_states
        first_choice = np.asarray(unfolded.mdp.nondeterministic_choice_indices)
        selected = np.array(list(choices))
        # exactly one per state, namely the choice of the policy (or the self-loop in the last layer)
        assert np.array_equal(np.searchsorted(first_choice, selected, side="right") - 1, np.arange(unfolded.mdp.nr_states))
        inner = unfolded.state_step < horizon
        assert np.array_equal(unfolded.choice_base[selected][inner], policy[unfolded.state_step[inner], unfolded.state_base[inner]])
        assert np.all(unfolded.choice_base[selected][~inner] == -1)


class TestAr:
    """The standard AR synthesis of PAYNT on the unfolded colored MDP (option --ar) finds the optimal time-dependent color consistent policy."""

    @pytest.mark.parametrize(("project", "horizon"), [("tests/pomdp-maze", 3), ("pomdp/grid/avoid", 4), ("pomdp/grid/avoid", 7), ("pomdp/maze/cheese", 3)])
    def test_pomdps(self, fh, project, horizon):
        sketch_path, props_path = get_sketch_paths(project)
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path, precision=1e-6)
        colored_mdp = factory.build()
        result = fh.solve_time_dependent(colored_mdp, task, horizon, tolerance=1e-12, ar=True)
        ar = result["ar"]
        assert ar.complete
        assert ar.options.shape == (horizon, colored_mdp.parameter_space.num_parameters)
        sign = 1 if fh.Objective.from_task(task).maximize else -1
        # the heuristic cannot beat the optimum and on these models it finds it
        assert sign * (result["value"] - ar.value) <= 1e-6
        assert result["value"] == pytest.approx(ar.value, abs=1e-5)
        # the solver computes the same value for the assignment of AR
        assert ar.solution.value == pytest.approx(ar.value, abs=1e-5)
        # AR works on a copy of the specification
        assert task.specification.optimality.optimum is None

    @pytest.mark.parametrize("seed", range(40, 70))
    def test_random_instances_have_the_brute_force_optimum(self, fh, seed):
        """The instances have states with free choices (nondeterminism that no parameter restricts), which the induced model of an assignment keeps."""
        rng = np.random.default_rng(seed)
        maximize, is_reward, horizon = bool(rng.integers(2)), bool(rng.integers(2)), int(rng.integers(1, 4))
        instance = Instance(seed, int(rng.integers(3, 7)))
        result = solve(fh, instance, horizon, maximize, is_reward, ar=True)
        ar = result["ar"]
        optimum = brute_force(instance, horizon, maximize, is_reward)
        assert ar.complete
        assert ar.value == pytest.approx(optimum, abs=1e-3)
        assert ar.solution.value == pytest.approx(ar.value, abs=1e-3)
        assert (1 if maximize else -1) * (result["value"] - optimum) <= 1e-9

    @pytest.mark.parametrize("satisfiable", [True, False])
    @pytest.mark.parametrize("seed", range(70, 86))
    def test_constraint_is_satisfied_by_ar_iff_some_policy_satisfies_it(self, fh, seed, satisfiable):
        rng = np.random.default_rng(seed)
        maximize, horizon = bool(rng.integers(2)), int(rng.integers(1, 4))
        instance = Instance(seed, int(rng.integers(3, 7)))
        sign = 1 if maximize else -1
        threshold = brute_force(instance, horizon, maximize, True) + sign * (-0.05 if satisfiable else 0.05)
        line = f'R{{"r"}}{">=" if maximize else "<="}{threshold:.6f} [F "goal"]'
        task = paynt.task.SynthesisTask([stormpy.parse_properties_without_context(line)[0]])
        result = fh.solve_time_dependent(instance.colored_mdp(True), task, horizon, ar=True)
        ar = result["ar"]
        assert ar.complete
        if satisfiable:
            assert ar.options is not None
            assert task.get_property().satisfies_threshold(ar.value)
        else:
            assert ar.options is None
            assert not task.get_property().satisfies_threshold(result["value"])

    def test_model_without_parameters(self, fh):
        instance = Instance(7, 6, max_holes=0)
        result = solve(fh, instance, 4, True, False, ar=True)
        assert result["ar"].value == pytest.approx(result["unconstrained_value"], abs=1e-3)

    def test_time_limit_stops_the_search(self, fh):
        instance = Instance(5, 6)
        ar = solve(fh, instance, 3, True, False, ar=True, ar_timeout=1e-9)["ar"]
        assert not ar.complete
        assert ar.options is None
        assert solve(fh, instance, 3, True, False, ar=True, ar_timeout=600)["ar"].complete

    def test_unfolded_model_is_needed(self, fh, pomdp_maze):
        colored_mdp, task = pomdp_maze
        with pytest.raises(fh.UnsupportedProblemError, match="above the limit"):
            fh.solve_time_dependent(colored_mdp, task, 5, ar=True, max_unfolded_states=10)

    def test_memory_needed_by_the_full_coloring_is_checked(self, fh, pomdp_maze, monkeypatch):
        colored_mdp, task = pomdp_maze
        monkeypatch.setattr(fh.psutil, "virtual_memory", lambda: types.SimpleNamespace(available=1024))
        with pytest.raises(fh.UnsupportedProblemError, match="GiB"):
            fh.solve_time_dependent(colored_mdp, task, 5, ar=True)

    def test_memory_estimate_grows_with_the_horizon(self, fh, pomdp_maze):
        colored_mdp, _ = pomdp_maze
        estimates = [fh.estimate_ar_memory(fh.UnfoldedColoredMdp.build(colored_mdp, horizon, [])) for horizon in (10, 20, 40)]
        assert estimates[0] < estimates[1] < estimates[2]
        # the number of choices grows linearly and the masks over all parameters make the cost of each of them grow as well
        assert estimates[2] > 1.9 * estimates[1]

    def test_solver_valuates_the_assignment_of_its_own_solution(self, fh):
        instance = Instance(11, 6)
        result = solve(fh, instance, 3, True, True)
        again = result["solver"].solution_of_options(result["solution"].options)
        assert again.value == pytest.approx(result["value"], abs=1e-12)

    def test_feature_kind_of_the_unfolded_colored_mdp(self, fh):
        """An assignment of parameters induces a Markov chain only if no state keeps several choices: that is the difference between the kinds."""
        with_free_choices = Instance.explicit([2], [[{}, {}], [{0: 0}, {0: 1}]]).colored_mdp(False)
        without_free_choices = Instance.explicit([2], [[{0: 0}, {0: 1}]]).colored_mdp(False)
        assert fh.UnfoldedColoredMdp.build(with_free_choices, 2, []).to_colored_mdp().feature_kind == "family"
        assert fh.UnfoldedColoredMdp.build(without_free_choices, 2, []).to_colored_mdp().feature_kind == "generic"

    def test_light_unfolded_colored_mdp_builds_the_markov_chain_of_an_assignment(self, fh, pomdp_maze):
        colored_mdp, task = pomdp_maze
        result = fh.solve_time_dependent(colored_mdp, task, 6, verify=False)
        unfolded = fh.UnfoldedColoredMdp.build(colored_mdp, 6, [fh.Objective.from_task(task).target_label])
        dtmc = unfolded.to_colored_mdp().build_assignment(unfolded.assignment(result["solution"].options))
        assert dtmc.check_specification(task.specification).optimality_result.value == pytest.approx(result["value"], abs=1e-5)


class TestWindows:
    """The backward passes change one step at a time and can get stuck; windows change several steps at once when they do."""

    # (seed, maximize, is_reward, horizon) of the instances whose backward passes get stuck in a policy that is not optimal; a window of several steps leaves it
    STUCK = [(98, False, True, 4), (105, False, True, 4), (193, True, True, 4), (219, False, True, 4), (255, False, True, 4), (318, False, True, 4)]

    @staticmethod
    def stuck_instance(seed):
        rng = np.random.default_rng(seed)
        maximize, is_reward, horizon = bool(rng.integers(2)), bool(rng.integers(2)), int(rng.integers(2, 5))
        return Instance(seed, int(rng.integers(4, 8)), max_holes=4, max_component_holes=2), maximize, is_reward, horizon

    @pytest.mark.parametrize(("seed", "maximize", "is_reward", "horizon"), STUCK)
    def test_windows_leave_the_local_optimum_of_the_backward_passes(self, fh, seed, maximize, is_reward, horizon):
        instance, expected_maximize, expected_is_reward, expected_horizon = self.stuck_instance(seed)
        assert (maximize, is_reward, horizon) == (expected_maximize, expected_is_reward, expected_horizon)
        sign = 1 if maximize else -1
        optimum = brute_force(instance, horizon, maximize, is_reward)
        stuck = solve(fh, instance, horizon, maximize, is_reward)
        assert sign * (optimum - stuck["value"]) > 1e-6
        result = solve(fh, instance, horizon, maximize, is_reward, verify=True, max_window=horizon, changes=2)
        assert result["value"] == pytest.approx(optimum, abs=1e-9)
        # every window improved the value, the values of the iterations never decrease, and Storm agrees on the policy that is color consistent
        solver = result["solver"]
        assert solver.window_moves
        assert all(move["gain"] > 0 and 2 <= move["window"] <= horizon for move in solver.window_moves)
        assert all(sign * (new - old) >= -1e-9 for old, new in zip(result["history"], result["history"][1:], strict=False)), result["history"]
        assert result["forward_value"] == pytest.approx(result["value"], abs=1e-9)
        assert result["storm_value"] == pytest.approx(result["value"], abs=1e-5)
        assert result["consistent"]

    def test_a_default_run_does_not_use_windows(self, fh):
        instance, maximize, is_reward, horizon = self.stuck_instance(98)
        result = solve(fh, instance, horizon, maximize, is_reward)
        assert result["solver"].window_moves == []
        assert result["solver"].window_stats == {}
        assert solve(fh, instance, horizon, maximize, is_reward, max_window=1, changes=2)["value"] == result["value"]

    def test_a_window_may_have_to_change_two_parameters_at_once(self, fh):
        instance, maximize, is_reward, horizon = self.stuck_instance(193)
        stuck = solve(fh, instance, horizon, maximize, is_reward)["value"]
        # changing one parameter at the beginning of a window does not help here, however long the window is
        assert solve(fh, instance, horizon, maximize, is_reward, max_window=horizon)["value"] == pytest.approx(stuck, abs=1e-12)
        two = solve(fh, instance, horizon, maximize, is_reward, max_window=horizon, changes=2)
        assert two["value"] == pytest.approx(brute_force(instance, horizon, maximize, is_reward), abs=1e-9)
        assert any(len(move["cells"]) == 2 for move in two["solver"].window_moves)

    @pytest.mark.parametrize("seed", range(300, 324))
    def test_the_value_does_not_decrease_with_the_size_of_the_window(self, fh, seed):
        rng = np.random.default_rng(seed)
        maximize, is_reward, horizon = bool(rng.integers(2)), bool(rng.integers(2)), int(rng.integers(2, 5))
        instance = Instance(seed, int(rng.integers(4, 8)), max_holes=4, max_component_holes=2)
        sign = 1 if maximize else -1
        results = [solve(fh, instance, horizon, maximize, is_reward, max_window=window, changes=2) for window in range(1, horizon + 1)]
        values = [result["value"] for result in results]
        # a larger limit continues the search of a smaller one, which is a part of it (the windows of the smaller sizes are tried first)
        assert all(sign * (new - old) >= -1e-9 for old, new in zip(values, values[1:], strict=False)), values
        assert sign * (results[0]["unconstrained_value"] - values[-1]) >= -1e-9

    @pytest.mark.parametrize("seed", range(400, 412))
    def test_windows_of_random_instances_agree_with_storm(self, fh, seed):
        rng = np.random.default_rng(seed)
        maximize, is_reward, horizon = bool(rng.integers(2)), bool(rng.integers(2)), int(rng.integers(2, 12))
        instance = Instance(seed, int(rng.integers(8, 20)), max_holes=8)
        result = solve(fh, instance, horizon, maximize, is_reward, verify=True, max_window=3, changes=2)
        assert result["forward_value"] == pytest.approx(result["value"], abs=1e-9)
        assert result["storm_value"] == pytest.approx(result["value"], abs=1e-5)
        assert result["consistent"]
        plain = solve(fh, instance, horizon, maximize, is_reward)["value"]
        assert (1 if maximize else -1) * (result["value"] - plain) >= -1e-9

    def test_the_gain_of_a_window_is_the_gain_of_the_value(self, fh):
        """The gain is exact if the window is the only change, and a lower bound when states without colors can respond outside of the window."""
        exact = 0
        for seed in range(500, 540):
            rng = np.random.default_rng(seed)
            maximize, is_reward, horizon = bool(rng.integers(2)), bool(rng.integers(2)), int(rng.integers(2, 6))
            instance = Instance(seed, int(rng.integers(4, 9)), max_holes=5)
            solver = solve(fh, instance, horizon, maximize, is_reward)["solver"]
            solution = solver.solve(100, 1e-12)[0]
            reach, values = solver.reach_probabilities(solution.policy), solver.values_of(solution.policy)
            colors = solver.colors
            for start, size in itertools.product(range(horizon), range(1, horizon + 1)):
                if start + size > horizon:
                    continue
                for component, option in itertools.product(range(colors.num_components), range(int(colors.comp_size.max()))):
                    if option >= colors.comp_size[component] or option == solution.joint_options[start, component]:
                        continue
                    gain, rows = solver.window_gain(solution, reach, values, start, size, [(0, component, option)])
                    joint_options = solution.joint_options.copy()
                    joint_options[start : start + size] = rows
                    actual = solver.sign * (solver.improve(None, fixed=joint_options).value - solution.value)
                    assert actual >= gain - 1e-9
                    if not len(solver.free_states):
                        assert actual == pytest.approx(gain, abs=1e-9)
                        exact += 1
        assert exact > 100

    def test_seeds_start_in_the_first_step_and_continue_where_the_changes_reach(self, fh):
        first = [(0, 0, 1), (0, 1, 1)]
        calls = []

        def later(seed):
            calls.append(seed)
            return [(seed[-1][0] + 1, 2, 1), (seed[-1][0] + 1, 3, 1)] if seed[-1][0] < 2 else []

        assert list(fh.window_seeds(first, later, 1)) == [((0, 0, 1),), ((0, 1, 1),)]
        assert calls == []  # nothing is added to a seed that has the changes that it may have
        seeds = list(fh.window_seeds(first, later, 3))
        assert seeds[:5] == [((0, 0, 1),), ((0, 0, 1), (1, 2, 1)), ((0, 0, 1), (1, 2, 1), (2, 2, 1)), ((0, 0, 1), (1, 2, 1), (2, 3, 1)), ((0, 0, 1), (1, 3, 1))]
        assert len(seeds) == 2 * (1 + 2 * (1 + 2))
        assert all(len(seed) <= 3 and seed[0] in first for seed in seeds)

    def test_the_statistics_count_the_candidates_of_each_size(self, fh):
        instance, maximize, is_reward, horizon = self.stuck_instance(98)
        solver = solve(fh, instance, horizon, maximize, is_reward, max_window=3)["solver"]
        assert set(solver.window_stats) <= {2, 3}
        assert all(entry["candidates"] > 0 and entry["seconds"] >= 0 for entry in solver.window_stats.values())
        assert sum(entry["moves"] for entry in solver.window_stats.values()) == len(solver.window_moves) > 0

    def test_windows_larger_than_the_horizon_and_models_without_parameters_are_skipped(self, fh):
        instance, maximize, is_reward, horizon = self.stuck_instance(98)
        assert solve(fh, instance, horizon, maximize, is_reward, max_window=horizon + 5)["solver"].window_stats.get(horizon + 1) is None
        no_colors = Instance(7, 8, max_holes=0)
        result = solve(fh, no_colors, 5, True, False, max_window=3)
        assert result["solver"].window_stats == {}
        assert result["value"] == pytest.approx(result["unconstrained_value"], abs=1e-12)

    def test_the_search_stops_at_the_deadline(self, fh):
        instance, maximize, is_reward, horizon = self.stuck_instance(98)
        solver = solve(fh, instance, horizon, maximize, is_reward)["solver"]
        solution = solver.solve(100, 1e-12)[0]
        assert solver.search_windows(solution, 2, 1e-12, deadline=time.time() - 1.0) is None
        assert solver.widen(solution, 3, 1e-12, deadline=time.time() - 1.0) is None


def gadget():
    """A model in which the backward passes get stuck (models/fh-counterexample, which has the explanation), with the horizon 3 and the probability of reaching
    the goal to maximize. Only one of the parameters is relevant in each step: h0 in the first, h1 in the second (u and v have the same observation), h2 in the
    third (the four payoff states have the same observation). The policies (a0,a1,a2) have the values aba 0.098 (where the backward passes stop), bab 0.1 (the
    optimum), abb 0.095, aab 0.09 and 0 for the others."""
    instance = Instance.__new__(Instance)
    instance.num_states, instance.init, instance.hole_num_options = 9, 0, [2, 2, 2]
    states = [
        [({0: 0}, {1: 0.1, 2: 0.9}), ({0: 1}, {2: 1.0})],  # the first step: a goes to u with 0.1, to v with 0.9; b to v
        [({1: 0}, {3: 1.0}), ({1: 1}, {4: 1.0})],  # u: a to the payoff state Pu0 (3), b to Pu1 (4)
        [({1: 0}, {5: 1.0}), ({1: 1}, {6: 1.0})],  # v: a to Pv0 (5), b to Pv1 (6)
        [({2: 0}, {7: 1.0}), ({2: 1}, {7: 1.0})],  # Pu0 pays nothing
        [({2: 0}, {8: 0.98, 7: 0.02}), ({2: 1}, {8: 0.95, 7: 0.05})],  # Pu1 pays well whatever is done: the lure
        [({2: 0}, {7: 1.0}), ({2: 1}, {8: 0.1, 7: 0.9})],  # Pv0 pays only if b is done
        [({2: 0}, {7: 1.0}), ({2: 1}, {7: 1.0})],  # Pv1 pays nothing
        [({}, {7: 1.0})],  # the sink
        [({}, {8: 1.0})],  # the goal
    ]
    instance.state_choices, instance.choice_color, instance.successors = [], [], []
    for choices in states:
        instance.state_choices.append(list(range(len(instance.choice_color), len(instance.choice_color) + len(choices))))
        instance.choice_color += [color for color, _ in choices]
        instance.successors += [successors for _, successors in choices]
    instance.reward = [0.0] * len(instance.choice_color)
    instance.target = {8}
    return instance


def gated_gadgets(copies=2, theta=0.099):
    """A choice in the first step in front of copies of the gadget (see gadget): the option 0 reaches the goal with the probability theta and the option 1
    goes to one of the copies at random, each with its own parameters. The best policy of a copy has the probability 0.1 and the backward passes get 0.098 in
    each of them, so the option 1 is worth 0.1 at best and less than theta (0.099) until more than half of the copies are changed, as well as the option of the
    first step. The policy that chooses the option 0 is a local optimum of value theta: no change of one or two parameters leads to a better policy if there
    are two copies (the first step and a parameter in each copy are three changes), and the optimum is 0.1. The horizon is 5."""
    instance = Instance.__new__(Instance)
    sink, goal = 2 + 7 * copies, 3 + 7 * copies
    instance.num_states, instance.init, instance.hole_num_options = goal + 1, 0, [2] * (1 + 3 * copies)
    states = [[({0: 0}, {goal: theta, sink: 1 - theta}), ({0: 1}, {1: 1.0})], [({}, {2 + 7 * copy: 1 / copies for copy in range(copies)})]]
    for copy in range(copies):
        base, (h1, h2, h3) = 2 + 7 * copy, (1 + 3 * copy, 2 + 3 * copy, 3 + 3 * copy)
        states += [
            [({h1: 0}, {base + 1: 0.1, base + 2: 0.9}), ({h1: 1}, {base + 2: 1.0})],
            [({h2: 0}, {base + 3: 1.0}), ({h2: 1}, {base + 4: 1.0})],
            [({h2: 0}, {base + 5: 1.0}), ({h2: 1}, {base + 6: 1.0})],
            [({h3: 0}, {sink: 1.0}), ({h3: 1}, {sink: 1.0})],
            [({h3: 0}, {goal: 0.98, sink: 0.02}), ({h3: 1}, {goal: 0.95, sink: 0.05})],
            [({h3: 0}, {sink: 1.0}), ({h3: 1}, {goal: 0.1, sink: 0.9})],
            [({h3: 0}, {sink: 1.0}), ({h3: 1}, {sink: 1.0})],
        ]
    states += [[({}, {sink: 1.0})], [({}, {goal: 1.0})]]
    instance.state_choices, instance.choice_color, instance.successors = [], [], []
    for choices in states:
        instance.state_choices.append(list(range(len(instance.choice_color), len(instance.choice_color) + len(choices))))
        instance.choice_color += [color for color, _ in choices]
        instance.successors += [successors for _, successors in choices]
    instance.reward = [0.0] * len(instance.choice_color)
    instance.target = {goal}
    return instance


class TestAlphaVectors:
    """The backward pass keeps one value vector per step; the pooled backward pass keeps one for each of several policies, and a step can use the suffix of
    another policy (option --alpha-vectors)."""

    @staticmethod
    def instance_of(seed, free=None, horizon=None):
        rng = np.random.default_rng(seed)
        maximize, is_reward, steps = bool(rng.integers(2)), bool(rng.integers(2)), int(rng.integers(2, 5))
        instance = Instance(seed, int(rng.integers(4, 8)), max_holes=4, max_component_holes=2)
        return instance, maximize, is_reward, horizon or steps

    @staticmethod
    def prepared(fh, seed, **kwargs):
        instance, maximize, is_reward, horizon = TestAlphaVectors.instance_of(seed, **kwargs)
        solver = solve(fh, instance, horizon, maximize, is_reward)["solver"]
        return instance, solver, maximize, is_reward, horizon

    @staticmethod
    def single_anchors(fh, solver, reach):
        return fh.Anchors([scipy.sparse.csr_matrix(reach[step][None, :]) for step in range(solver.horizon)], [[0] * solver.horizon])

    @pytest.mark.parametrize("seed", range(30))
    def test_one_anchor_per_step_is_the_backward_pass(self, fh, seed):
        _, solver, *_ = self.prepared(fh, seed)
        if not solver.colors.num_components:
            pytest.skip("no parameters")
        reach = solver.reach_probabilities(solver.solve_unconstrained().policy)
        reference = solver.improve(reach, None)
        pool = solver.alpha_sweep(self.single_anchors(fh, solver, reach))
        policy = solver.improve(None, fixed=pool.suffix(0, 0))
        assert np.array_equal(policy.joint_options, reference.joint_options)
        assert policy.value == pytest.approx(reference.value, abs=1e-12)
        # the vector of the initial state is the value of the policy
        assert pool.values[0][solver.mdp.initial_state] == pytest.approx(reference.value, abs=1e-9)

    @pytest.mark.parametrize("seed", range(40))
    def test_the_pooled_backup_agrees_with_the_backup_of_every_suffix(self, fh, seed):
        _, solver, *_ = self.prepared(fh, seed)
        if not solver.colors.num_components:
            pytest.skip("no parameters")
        rng = np.random.default_rng(seed)
        n = solver.mdp.num_states
        vectors = rng.random((5, n)) * 3
        vectors[:, solver.mdp.target] = solver.terminal[solver.mdp.target]
        anchors = rng.random((4, n)) * (rng.random((4, n)) < 0.6)
        previous = np.array([rng.integers(size) for size in solver.colors.comp_size])
        for previous_joint in (None, previous):
            parent, joint, new_vectors = solver._pool_backup(vectors, anchors, previous_joint)
            for index, anchor in enumerate(anchors):
                results = [solver._decide_step(vector, anchor, previous_joint) for vector in vectors]
                scores = np.array([solver.sign * np.dot(anchor * solver.nontarget, new_values) for _, _, new_values in results])
                best = int(np.flatnonzero(scores >= scores.max() - fh.TIE_TOLERANCE * abs(scores.max()))[0])
                assert parent[index] == best
                assert np.array_equal(joint[index], results[best][1])
                assert new_vectors[index] == pytest.approx(results[best][2], abs=1e-12)

    def test_the_pooled_backup_chooses_among_suffixes_that_differ(self, fh):
        """Two suffixes, each of which is the better one for one of two anchors."""
        solver = self.prepared(fh, 98)[1]
        rng = np.random.default_rng(0)
        n = solver.mdp.num_states
        vectors = rng.random((6, n)) * 3
        vectors[:, solver.mdp.target] = solver.terminal[solver.mdp.target]
        anchors = np.eye(n)[: min(n, 6)]
        parent = solver._pool_backup(vectors, anchors, None)[0]
        assert len(set(parent.tolist())) > 1

    def test_all_reach_vectors_as_anchors_give_the_optimum(self, fh):
        """The pooled backward pass is the dynamic programming over the reach vectors if all of them are anchors, so it is exact."""
        checked = 0
        for seed in range(400):
            instance, solver, maximize, is_reward, _ = self.prepared(fh, seed, horizon=3)
            colors = solver.colors
            options = [np.array(option) for option in itertools.product(*[range(int(size)) for size in colors.comp_size])]
            if len(solver.free_states) or not colors.num_components or len(options) > 8:
                continue
            solution = solver.solve(100, 1e-12)[0]
            joints = [np.array(rows) for rows in itertools.product(options, repeat=3)]
            anchors = solver._alpha_anchors(solution, joints, solver.solve_unconstrained())
            pool = solver.alpha_sweep(anchors)
            optimum = brute_force(instance, 3, maximize, is_reward)
            assert pool.values[0][solver.mdp.initial_state] == pytest.approx(optimum, abs=1e-9)
            assert solver.improve(None, fixed=pool.suffix(0, 0)).value == pytest.approx(optimum, abs=1e-9)
            checked += 1
        assert checked >= 10

    @pytest.mark.parametrize("seed", range(24))
    def test_more_anchors_never_hurt(self, fh, seed):
        _, solver, *_ = self.prepared(fh, seed)
        if not solver.colors.num_components:
            pytest.skip("no parameters")
        solution = solver.solve(100, 1e-12)[0]
        unconstrained = solver.solve_unconstrained()
        joints = [change[0] for change in solver._alpha_changes(solution, 12)]
        values = []
        for count in (0, 1, 3, 6, len(joints)):
            anchors = solver._alpha_anchors(solution, [solution.joint_options] + joints[:count], unconstrained)
            pool = solver.alpha_sweep(anchors, solution.joint_options)
            values.append(pool.values[0][solver.mdp.initial_state])
        assert all(solver.sign * (new - old) >= -1e-9 for old, new in zip(values, values[1:], strict=False)), values
        assert solver.sign * (values[0] - solution.value) >= -1e-9  # the solution is not lost

    def test_the_counter_example_of_the_backward_passes(self, fh):
        instance, maximize, is_reward, horizon = gadget(), True, False, 3
        assert brute_force(instance, horizon, maximize, is_reward) == pytest.approx(0.1, abs=1e-12)
        stuck = solve(fh, instance, horizon, maximize, is_reward)
        assert stuck["value"] == pytest.approx(0.098, abs=1e-12)
        assert stuck["unconstrained_value"] == pytest.approx(0.188, abs=1e-12)
        solver = stuck["solver"]
        # the policies of the other windows of 3 steps cannot leave it either, a pool of the incumbent and the unconstrained optimum cannot, the changes can
        solution, unconstrained = solver.solve(100, 1e-12)[0], solver.solve_unconstrained()
        none = solver.alpha_sweep(solver._alpha_anchors(solution, [solution.joint_options], unconstrained), solution.joint_options)
        assert none.values[0][solver.mdp.initial_state] == pytest.approx(0.098, abs=1e-12)
        escaped = solve(fh, instance, horizon, maximize, is_reward, verify=True, alpha_vectors=8)
        assert escaped["value"] == pytest.approx(0.1, abs=1e-12)
        assert escaped["solver"].alpha_moves and escaped["solver"].alpha_moves[0]["gain"] == pytest.approx(0.002, abs=1e-12)
        assert escaped["storm_value"] == pytest.approx(0.1, abs=1e-6)
        assert escaped["consistent"]

    @pytest.mark.parametrize(("seed", "maximize", "is_reward", "horizon"), TestWindows.STUCK)
    def test_the_pool_leaves_the_local_optimum_of_the_backward_passes(self, fh, seed, maximize, is_reward, horizon):
        instance, *_ = TestWindows.stuck_instance(seed)
        sign = 1 if maximize else -1
        optimum = brute_force(instance, horizon, maximize, is_reward)
        stuck = solve(fh, instance, horizon, maximize, is_reward)
        assert sign * (optimum - stuck["value"]) > 1e-6
        result = solve(fh, instance, horizon, maximize, is_reward, verify=True, alpha_vectors=64)
        assert result["value"] == pytest.approx(optimum, abs=1e-9)
        solver = result["solver"]
        assert solver.alpha_moves
        assert all(move["gain"] > 0 and 1 <= move["rounds"] <= fh.ALPHA_ROUNDS and move["policies"] <= 64 for move in solver.alpha_moves)
        assert solver.alpha_stats["moves"] == len(solver.alpha_moves)
        assert all(sign * (new - old) >= -1e-9 for old, new in zip(result["history"], result["history"][1:], strict=False)), result["history"]
        assert result["forward_value"] == pytest.approx(result["value"], abs=1e-9)
        assert result["storm_value"] == pytest.approx(result["value"], abs=1e-5)
        assert result["consistent"]

    @pytest.mark.parametrize("seed", range(600, 624))
    def test_the_pool_agrees_with_storm_and_never_loses_value(self, fh, seed):
        rng = np.random.default_rng(seed)
        maximize, is_reward, horizon = bool(rng.integers(2)), bool(rng.integers(2)), int(rng.integers(2, 12))
        instance = Instance(seed, int(rng.integers(8, 20)), max_holes=8)
        result = solve(fh, instance, horizon, maximize, is_reward, verify=True, alpha_vectors=16)
        assert result["forward_value"] == pytest.approx(result["value"], abs=1e-9)
        assert result["storm_value"] == pytest.approx(result["value"], abs=1e-5)
        assert result["consistent"]
        sign = 1 if maximize else -1
        assert sign * (result["value"] - solve(fh, instance, horizon, maximize, is_reward)["value"]) >= -1e-9
        assert sign * (result["unconstrained_value"] - result["value"]) >= -1e-9

    def test_a_default_run_does_not_pool_anything(self, fh):
        instance, maximize, is_reward, horizon = TestWindows.stuck_instance(98)
        result = solve(fh, instance, horizon, maximize, is_reward)
        assert result["solver"].alpha_moves == []
        assert result["solver"].alpha_stats == {}
        assert solve(fh, instance, horizon, maximize, is_reward, alpha_vectors=1)["value"] == result["value"]

    def test_the_pool_is_tried_before_the_windows(self, fh):
        instance, maximize, is_reward, horizon = TestWindows.stuck_instance(98)
        both = solve(fh, instance, horizon, maximize, is_reward, alpha_vectors=64, max_window=3, changes=2)["solver"]
        assert both.alpha_moves and not both.window_moves  # the pool improved it, so the windows were not needed
        only_windows = solve(fh, instance, horizon, maximize, is_reward, max_window=3, changes=2)["solver"]
        assert only_windows.window_moves and not only_windows.alpha_moves

    @pytest.mark.parametrize("seed", range(10))
    def test_the_changes_are_single_ones_first_then_pairs(self, fh, seed):
        _, solver, *_ = self.prepared(fh, seed)
        if not solver.colors.num_components:
            pytest.skip("no parameters")
        solution = solver.solve(100, 1e-12)[0]
        changes = solver._alpha_changes(solution, 10_000)
        differing = [int(np.count_nonzero(joint != solution.joint_options)) for joint, _ in changes]
        assert set(differing) <= {1, 2}
        assert differing == sorted(differing)
        singles = [step for (_, step), count in zip(changes, differing, strict=True) if count == 1]
        assert singles == sorted(singles)  # from the first step to the last
        assert all(0 <= step < solver.horizon for _, step in changes)
        for limit in (0, 1, 3):
            assert len(solver._alpha_changes(solution, limit)) == min(limit, len(changes))
        # up to three components: the smaller changes come first, and the ones of one or two components are the same as before
        more = solver._alpha_changes(solution, 10_000, changes=3)
        counts = [int(np.count_nonzero(joint != solution.joint_options)) for joint, _ in more]
        assert set(counts) <= {1, 2, 3}
        assert counts == sorted(counts)
        assert [joint.tolist() for joint, _ in more[: len(changes)]] == [joint.tolist() for joint, _ in changes]
        assert len(solver._alpha_changes(solution, 10_000, changes=1)) == counts.count(1)

    def test_the_policies_of_the_pool_have_their_own_choices_of_the_states_without_colors(self, fh):
        """The anchors of a change are the reach vectors of the policy that is best with the change, which is not the solution with its choices of the states
        without colors if the change makes another choice better."""
        compared = other_choices = 0
        for seed in range(120):
            _, solver, *_ = self.prepared(fh, seed)
            if not len(solver.free_states) or not solver.colors.num_components:
                continue
            solution = solver.solve(100, 1e-12)[0]
            for joint, _ in solver._alpha_changes(solution, 12):
                policy = solver.improve(None, fixed=joint).policy
                other_choices += bool(np.any(policy[:, solver.free_states] != solution.policy[:, solver.free_states]))
                anchors = solver._alpha_anchors(solution, [joint], solver.solve_unconstrained())
                reach = solver.reach_probabilities(policy)
                for step in range(solver.horizon):
                    expected = reach[step] * solver.nontarget
                    if expected.sum() > 1e-14:
                        assert anchors.rows[step].toarray()[anchors.index[0][step]] == pytest.approx(expected, abs=1e-12)
                        compared += 1
        assert compared > 100
        assert other_choices > 0  # the changes do change the choices of the states without colors in some instances, so the comparison is not empty

    @pytest.mark.parametrize("free", [False, True])
    def test_the_anchors_are_distinct_and_the_first_policy_comes_first(self, fh, free):
        """The anchors are computed differently for models with states that have several choices without colors, so both kinds of models are checked."""
        seed = next(seed for seed in range(200) if self.single_kind(fh, seed) == free)
        _, solver, *_ = self.prepared(fh, seed)
        solution = solver.solve(100, 1e-12)[0]
        joints = [solution.joint_options] * 3 + [change[0] for change in solver._alpha_changes(solution, 20)]
        unconstrained = solver.solve_unconstrained()
        anchors = solver._alpha_anchors(solution, joints, unconstrained)
        assert len(anchors.index) == len(joints) + 1
        for step in range(solver.horizon):
            rows = [row for row in anchors.rows[step].toarray() if row.sum() > 0]
            assert anchors.rows[step].shape[1] == solver.mdp.num_states
            assert len({tuple(np.round(row / row.sum(), 9)) for row in rows}) == len(rows)
            assert anchors.index[0][step] == anchors.index[1][step] == anchors.index[2][step]  # the same policy
            assert anchors.index[0][step] in (0, -1)
        assert anchors.index[0][0] == 0  # the initial state is reached by every policy
        # the reach vectors of the policies of the changes, and of the last one, which is the unconstrained optimum
        policies = [solver.improve(None, fixed=joint).policy for joint in joints] + [unconstrained.policy]
        for member, policy in enumerate(policies):
            reach = solver.reach_probabilities(policy)
            for step in range(solver.horizon):
                expected = reach[step] * solver.nontarget
                if expected.sum() > 1e-14:
                    row = anchors.rows[step].toarray()[anchors.index[member][step]]
                    assert row / row.sum() == pytest.approx(expected / expected.sum(), abs=1e-9)
                else:
                    assert anchors.index[member][step] == -1

    @staticmethod
    def single_kind(fh, seed):
        """Whether the instance has states with several choices and no colors (free states), None if it has no parameters or the unconstrained optimum
        reaches the states in the same way as the solution (then the two cannot be told apart by their reach vectors)."""
        _, solver, *_ = TestAlphaVectors.prepared(fh, seed)
        if not solver.colors.num_components:
            return None
        solution, unconstrained = solver.solve(100, 1e-12)[0], solver.solve_unconstrained()
        same = np.allclose(solver.reach_probabilities(solution.policy), solver.reach_probabilities(unconstrained.policy), atol=1e-9)
        return None if same else bool(len(solver.free_states))

    @pytest.mark.parametrize("seed", range(12))
    def test_a_change_responds_with_the_suffix_of_its_vector(self, fh, seed):
        _, solver, *_ = self.prepared(fh, seed)
        if not solver.colors.num_components:
            pytest.skip("no parameters")
        solution, unconstrained = solver.solve(100, 1e-12)[0], solver.solve_unconstrained()
        members = [[solution.joint_options.copy(), -1]] + [[joint, step] for joint, step in solver._alpha_changes(solution, 10)]
        before = [(joint.copy(), step) for joint, step in members]
        anchors = solver._alpha_anchors(solution, [joint for joint, _ in members], unconstrained)
        pool = solver.alpha_sweep(anchors, solution.joint_options)
        responded = solver._alpha_respond(members, anchors, pool)
        assert np.array_equal(members[0][0], before[0][0])  # the solution does not respond
        changed = False
        for index, ((joint, step), (old, _)) in enumerate(zip(members, before, strict=True)):
            assert np.array_equal(joint[: step + 1], old[: step + 1])  # the steps up to the first change are kept
            if 0 <= step < solver.horizon - 1 and anchors.index[index][step + 1] >= 0:
                assert np.array_equal(joint[step + 1 :], pool.suffix(step + 1, anchors.index[index][step + 1]))
            else:
                assert np.array_equal(joint, old)
            changed |= not np.array_equal(joint, old)
        assert responded == changed

    def test_a_move_may_need_the_policies_of_the_changes_to_respond(self, fh, monkeypatch):
        """The policies of the changes are the anchors of the first pass as they are, with the steps after the change as in the solution; the improvement is
        found only in the second pass, in which they have the suffixes that the first pass found for them."""
        instance, maximize, is_reward, horizon = self.instance_of(2420)
        optimum = brute_force(instance, horizon, maximize, is_reward)
        plain = solve(fh, instance, horizon, maximize, is_reward)["value"]
        assert (1 if maximize else -1) * (optimum - plain) > 1e-6
        result = solve(fh, instance, horizon, maximize, is_reward, alpha_vectors=6)
        assert [move["rounds"] for move in result["solver"].alpha_moves] == [2]
        assert result["solver"].alpha_stats["sweeps"] >= 2
        assert (1 if maximize else -1) * (result["value"] - plain) > 1e-6
        monkeypatch.setattr(fh, "ALPHA_ROUNDS", 1)
        one_round = solve(fh, instance, horizon, maximize, is_reward, alpha_vectors=6)
        assert one_round["solver"].alpha_moves == []
        assert one_round["value"] == pytest.approx(plain, abs=1e-12)

    def test_a_trap_that_needs_three_changes_at_once(self, fh):
        """The first step that leads to the copies of the gadget is better only if the parameters of more than half of the copies are changed too, so the three
        changes (the first step and a parameter of each of the two copies) have to be made together; changes of two are not enough however many policies the
        pool has, and a pool of the changes of three is."""
        instance, maximize, is_reward, horizon = gated_gadgets(), True, False, 5
        plain = solve(fh, instance, horizon, maximize, is_reward)
        assert plain["value"] == pytest.approx(0.099, abs=1e-12)
        for vectors in (64, 4096):
            assert solve(fh, instance, horizon, maximize, is_reward, alpha_vectors=vectors)["value"] == pytest.approx(0.099, abs=1e-12)
        # the windows do not help either, with up to two changes
        assert solve(fh, instance, horizon, maximize, is_reward, max_window=4, changes=2)["value"] == pytest.approx(0.099, abs=1e-12)
        three = solve(fh, instance, horizon, maximize, is_reward, verify=True, alpha_vectors=64, alpha_changes=3)
        assert three["value"] == pytest.approx(0.1, abs=1e-12)
        assert three["storm_value"] == pytest.approx(0.1, abs=1e-6)
        assert three["consistent"]
        assert three["solver"].alpha_moves
        # 0.1 is the optimum: the policy that does not take the first step is worth 0.099, and the other one is the average of the copies, which is 0.1 at best
        assert brute_force(gadget(), 3, True, False) == pytest.approx(0.1, abs=1e-12)

    def test_the_changes_of_the_pool_are_not_limited_by_the_windows(self, fh):
        instance, maximize, is_reward, horizon = TestWindows.stuck_instance(98)
        one = solve(fh, instance, horizon, maximize, is_reward, alpha_vectors=64, alpha_changes=1)["value"]
        two = solve(fh, instance, horizon, maximize, is_reward, alpha_vectors=64, alpha_changes=2)["value"]
        three = solve(fh, instance, horizon, maximize, is_reward, alpha_vectors=64, alpha_changes=3)["value"]
        assert (1 if maximize else -1) * (two - one) >= -1e-9
        assert two == pytest.approx(brute_force(instance, horizon, maximize, is_reward), abs=1e-9)
        assert three == pytest.approx(two, abs=1e-9)

    def test_the_search_stops_at_the_deadline(self, fh):
        instance, maximize, is_reward, horizon = TestWindows.stuck_instance(98)
        solver = solve(fh, instance, horizon, maximize, is_reward)["solver"]
        solution = solver.solve(100, 1e-12)[0]
        assert solver.search_alpha(solution, 16, 1e-12, deadline=time.time() - 1.0) is None
        assert solver.alpha_sweep(solver._alpha_anchors(solution, [solution.joint_options], solver.solve_unconstrained()), deadline=time.time() - 1.0) is None

    def test_one_vector_and_models_without_parameters_pool_nothing(self, fh):
        instance, maximize, is_reward, horizon = TestWindows.stuck_instance(98)
        solver = solve(fh, instance, horizon, maximize, is_reward)["solver"]
        assert solver.search_alpha(solver.solve(100, 1e-12)[0], 1, 1e-12) is None
        assert solver.alpha_stats == {}
        no_colors = Instance(7, 8, max_holes=0)
        result = solve(fh, no_colors, 5, True, False, alpha_vectors=16)
        assert result["solver"].alpha_stats == {}
        assert result["value"] == pytest.approx(result["unconstrained_value"], abs=1e-12)

    def test_the_pool_is_reduced_to_fit_the_memory(self, fh, monkeypatch, caplog):
        _, solver, *_ = self.prepared(fh, 98)
        assert solver._alpha_budget(1000) == 1000
        monkeypatch.setattr(fh, "ALPHA_MEMORY", 1)
        with caplog.at_level(logging.WARNING):
            assert solver._alpha_budget(1000) == 2
            assert solver._alpha_budget(1000) == 2
        assert sum("reduced" in record.message for record in caplog.records) == 1  # it is said once
        solution = solver.solve(100, 1e-12)[0]
        solver.search_alpha(solution, 1000, 1e-12)
        assert solver.alpha_stats["policies"] == 2

    def test_the_statistics_count_the_work(self, fh):
        instance, maximize, is_reward, horizon = TestWindows.stuck_instance(98)
        solver = solve(fh, instance, horizon, maximize, is_reward, alpha_vectors=32)["solver"]
        statistics = solver.alpha_stats
        assert statistics["attempts"] >= statistics["moves"] > 0
        assert statistics["sweeps"] >= statistics["moves"]
        assert statistics["policies"] >= 2 and statistics["seconds"] > 0


class TestArWarmStart:
    """AR that starts with the value of the backward passes as the optimum so far (option --ar-warm-start): it prunes the families that cannot beat it from the
    beginning, and only looks for better policies."""

    @pytest.mark.parametrize(("project", "horizon"), [("tests/pomdp-maze", 3), ("tests/pomdp-maze", 8), ("pomdp/grid/avoid", 7), ("pomdp/maze/cheese", 3)])
    def test_pomdps_have_the_optimum_of_plain_ar_in_a_search_that_is_not_larger(self, fh, project, horizon):
        sketch_path, props_path = get_sketch_paths(project)
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path, precision=1e-6)
        colored_mdp = factory.build()
        plain = fh.solve_time_dependent(colored_mdp, task, horizon, tolerance=1e-12, ar=True)["ar"]
        result = fh.solve_time_dependent(colored_mdp, task, horizon, tolerance=1e-12, ar=True, ar_warm_start=True)
        warm = result["ar"]
        sign = 1 if fh.Objective.from_task(task).maximize else -1
        assert plain.complete and warm.complete
        assert warm.initial == pytest.approx(result["value"])
        assert warm.value == pytest.approx(plain.value, abs=1e-5)
        assert warm.solution.value == pytest.approx(warm.value, abs=1e-5)
        if warm.found:
            assert sign * (warm.value - warm.initial) > 0
        else:
            # nothing is better than the policy that AR started from, which is the result
            assert warm.value == warm.initial
            assert np.array_equal(warm.options, result["solution"].options)
        # the search of AR is the same, apart from the families that the better optimum so far prunes
        assert 1 <= warm.families <= plain.families

    def test_ar_finds_a_better_policy_than_the_heuristic_that_is_stuck(self, fh, pomdp_maze):
        colored_mdp, task = pomdp_maze
        result = fh.solve_time_dependent(colored_mdp, task, 8, tolerance=1e-12, ar=True, ar_warm_start=True)
        ar = result["ar"]
        sign = 1 if fh.Objective.from_task(task).maximize else -1
        assert ar.complete and ar.found
        assert sign * (ar.value - result["value"]) > 1e-3
        # with windows the heuristic is optimal: AR proves it, and does not find anything
        windows = fh.solve_time_dependent(colored_mdp, task, 8, tolerance=1e-12, ar=True, ar_warm_start=True, max_window=3, changes=2)
        assert windows["value"] == pytest.approx(ar.value, abs=1e-5)
        assert windows["ar"].complete and not windows["ar"].found
        assert windows["ar"].families <= ar.families

    @pytest.mark.parametrize("seed", range(40, 70))
    def test_random_instances_have_the_brute_force_optimum(self, fh, seed):
        rng = np.random.default_rng(seed)
        maximize, is_reward, horizon = bool(rng.integers(2)), bool(rng.integers(2)), int(rng.integers(1, 4))
        instance = Instance(seed, int(rng.integers(3, 7)))
        result = solve(fh, instance, horizon, maximize, is_reward, ar=True, ar_warm_start=True)
        ar = result["ar"]
        optimum = brute_force(instance, horizon, maximize, is_reward)
        assert ar.complete
        assert ar.initial == pytest.approx(result["value"])
        # AR has a policy that is at least as good as the one that it started from, and it is optimal
        assert ar.options is not None and ar.value == pytest.approx(optimum, abs=1e-3)
        assert ar.solution.value == pytest.approx(ar.value, abs=1e-3)
        if ar.found:
            assert (1 if maximize else -1) * (ar.value - ar.initial) > 0
        else:
            assert abs(result["value"] - optimum) <= 1e-3
            assert ar.value == ar.initial and np.array_equal(ar.options, result["solution"].options)

    @pytest.mark.parametrize(("seed", "maximize", "is_reward", "horizon"), TestWindows.STUCK)
    def test_instances_that_are_stuck_have_a_better_policy_found_by_ar(self, fh, seed, maximize, is_reward, horizon):
        instance = TestWindows.stuck_instance(seed)[0]
        optimum = brute_force(instance, horizon, maximize, is_reward)
        ar = solve(fh, instance, horizon, maximize, is_reward, ar=True, ar_warm_start=True)["ar"]
        assert ar.complete and ar.found
        assert ar.value == pytest.approx(optimum, abs=1e-3)
        # the windows get to the optimum, from which AR has nothing to find
        windows = solve(fh, instance, horizon, maximize, is_reward, ar=True, ar_warm_start=True, max_window=horizon, changes=2)
        assert windows["value"] == pytest.approx(optimum, abs=1e-9)
        assert windows["ar"].complete and not windows["ar"].found

    def test_a_poor_policy_to_start_from_is_improved_to_the_optimum(self, fh, pomdp_maze):
        colored_mdp, task = pomdp_maze
        horizon = 6
        result = fh.solve_time_dependent(colored_mdp, task, horizon, tolerance=1e-12, ar=True)
        objective = fh.Objective.from_task(task)
        unfolded = fh.UnfoldedColoredMdp.build(colored_mdp, horizon, [objective.target_label] if objective.is_reward else [])
        # the first option of every parameter in every step
        poor = result["solver"].solution_of_options(np.zeros((horizon, colored_mdp.parameter_space.num_parameters), dtype=np.int64))
        assert (1 if objective.maximize else -1) * (result["ar"].value - poor.value) > 1e-3
        warm = fh.run_ar(unfolded, task, None, poor)
        assert warm.complete and warm.found and warm.initial == pytest.approx(poor.value)
        assert warm.value == pytest.approx(result["ar"].value, abs=1e-5)

    def test_a_model_without_parameters_is_proved_optimal_at_once(self, fh):
        instance = Instance(7, 6, max_holes=0)
        result = solve(fh, instance, 4, True, False, ar=True, ar_warm_start=True)
        ar = result["ar"]
        # the value of the policy is the unconstrained optimum, so the first family has nothing better
        assert ar.complete and not ar.found and ar.families == 1
        assert ar.value == pytest.approx(result["unconstrained_value"], abs=1e-9)

    def test_the_time_limit_leaves_the_policy_that_ar_started_from(self, fh):
        result = solve(fh, Instance(5, 6), 3, True, False, ar=True, ar_warm_start=True, ar_timeout=1e-9)
        ar = result["ar"]
        assert not ar.complete and not ar.found and ar.families == 0
        assert ar.value == pytest.approx(result["value"]) and np.array_equal(ar.options, result["solution"].options)

    def test_a_constraint_has_no_optimum_to_start_with(self, fh, caplog):
        rng = np.random.default_rng(70)
        maximize, horizon = bool(rng.integers(2)), int(rng.integers(1, 4))
        instance = Instance(70, int(rng.integers(3, 7)))
        threshold = brute_force(instance, horizon, maximize, True) + (1 if maximize else -1) * -0.05
        line = f'R{{"r"}}{">=" if maximize else "<="}{threshold:.6f} [F "goal"]'
        task = paynt.task.SynthesisTask([stormpy.parse_properties_without_context(line)[0]])
        with caplog.at_level(logging.WARNING, logger="finite-horizon"):
            ar = fh.solve_time_dependent(instance.colored_mdp(True), task, horizon, ar=True, ar_warm_start=True)["ar"]
        assert "no optimum to start AR with" in caplog.text
        # AR runs as it does without the option
        assert ar.initial is None and ar.complete and ar.found
        assert task.get_property().satisfies_threshold(ar.value)

    def test_the_policy_of_the_heuristic_is_not_changed_by_ar(self, fh, pomdp_maze):
        colored_mdp, task = pomdp_maze
        plain = fh.solve_time_dependent(colored_mdp, task, 8, tolerance=1e-12, verify=False)
        warm = fh.solve_time_dependent(colored_mdp, task, 8, tolerance=1e-12, ar=True, ar_warm_start=True)
        assert warm["value"] == plain["value"] and np.array_equal(warm["solution"].options, plain["solution"].options)


class TestCommandLine:
    def test_run_and_export_policy(self, fh, tmp_path):
        from click.testing import CliRunner

        sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
        export_path = tmp_path / "policy.json"
        result = CliRunner().invoke(fh.main, [sketch_path, props_path, "5", "--export-policy", str(export_path)])
        assert result.exit_code == 0, result.output
        import json

        exported = json.loads(export_path.read_text())
        assert exported["horizon"] == 5
        assert len(exported["policy"]) == 5
        assert exported["value"] >= exported["unconstrained_value"] - 1e-9

    def test_ar_option(self, fh, tmp_path):
        from click.testing import CliRunner

        sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
        export_path = tmp_path / "policy.json"
        result = CliRunner().invoke(fh.main, [sketch_path, props_path, "3", "--ar", "--ar-timeout", "600", "--export-policy", str(export_path)])
        assert result.exit_code == 0, result.output
        import json

        exported = json.loads(export_path.read_text())
        assert exported["ar"]["complete"]
        assert exported["ar"]["value"] == pytest.approx(exported["value"], abs=1e-5)
        assert len(exported["ar"]["policy"]) == 3

    def test_ar_warm_start_option(self, fh, tmp_path):
        from click.testing import CliRunner

        sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
        export_path = tmp_path / "policy.json"
        args = [sketch_path, props_path, "8", "--ar", "--ar-warm-start", "--ar-timeout", "600", "--export-policy", str(export_path)]
        result = CliRunner().invoke(fh.main, args)
        assert result.exit_code == 0, result.output
        import json

        exported = json.loads(export_path.read_text())
        # the backward passes are stuck in this model, AR finds the better policy from their value
        assert exported["ar"]["complete"] and exported["ar"]["found"]
        assert exported["ar"]["initial"] == pytest.approx(exported["value"])
        assert exported["ar"]["value"] != pytest.approx(exported["value"], abs=1e-3)
        assert len(exported["ar"]["policy"]) == 8 and exported["ar"]["families"] > 0
        # with the windows AR proves that the policy is optimal
        result = CliRunner().invoke(fh.main, [*args, "--window", "3", "--window-changes", "2"])
        assert result.exit_code == 0, result.output
        exported = json.loads(export_path.read_text())
        assert exported["ar"]["complete"] and not exported["ar"]["found"] and exported["ar"]["policy"] is None
        assert exported["ar"]["value"] == pytest.approx(exported["value"])

    def test_ar_warm_start_needs_ar(self, fh):
        from click.testing import CliRunner

        sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
        result = CliRunner().invoke(fh.main, [sketch_path, props_path, "3", "--ar-warm-start"])
        assert result.exit_code == 2
        assert "--ar" in result.output

    def test_ar_cannot_be_combined_with_no_unfold(self, fh):
        from click.testing import CliRunner

        sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
        result = CliRunner().invoke(fh.main, [sketch_path, props_path, "3", "--ar", "--no-unfold"])
        assert result.exit_code == 2
        assert "--no-unfold" in result.output

    def test_ar_timeout_needs_ar(self, fh):
        from click.testing import CliRunner

        sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
        result = CliRunner().invoke(fh.main, [sketch_path, props_path, "3", "--ar-timeout", "5"])
        assert result.exit_code == 2
        assert "--ar" in result.output

    def test_unsupported_model_is_reported(self, fh):
        from click.testing import CliRunner

        sketch_path, props_path = get_sketch_paths("tests/dt-maze")
        result = CliRunner().invoke(fh.main, [sketch_path, props_path, "5"])
        assert result.exit_code != 0
        assert "not supported" in result.output

    def test_horizon_must_be_positive(self, fh):
        from click.testing import CliRunner

        sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
        assert CliRunner().invoke(fh.main, [sketch_path, props_path, "0"]).exit_code != 0

    def test_windows_are_exported(self, fh, tmp_path):
        from click.testing import CliRunner

        sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
        export_path = tmp_path / "policy.json"
        result = CliRunner().invoke(fh.main, [sketch_path, props_path, "6", "--window", "3", "--window-changes", "2", "--export-policy", str(export_path)])
        assert result.exit_code == 0, result.output
        import json

        windows = json.loads(export_path.read_text())["windows"]
        assert windows["max_window"] == 3
        assert windows["changes"] == 2
        assert set(windows["sizes"]) <= {"2", "3"}
        assert len(windows["moves"]) == sum(entry["moves"] for entry in windows["sizes"].values())

    def test_windows_are_not_exported_without_them(self, fh, tmp_path):
        from click.testing import CliRunner

        sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
        export_path = tmp_path / "policy.json"
        assert CliRunner().invoke(fh.main, [sketch_path, props_path, "4", "--export-policy", str(export_path)]).exit_code == 0
        import json

        assert "windows" not in json.loads(export_path.read_text())

    def test_alpha_vectors_are_exported(self, fh, tmp_path):
        from click.testing import CliRunner

        sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
        export_path = tmp_path / "policy.json"
        result = CliRunner().invoke(fh.main, [sketch_path, props_path, "8", "--alpha-vectors", "32", "--export-policy", str(export_path)])
        assert result.exit_code == 0, result.output
        import json

        alpha = json.loads(export_path.read_text())["alpha"]
        assert alpha["alpha_vectors"] == 32
        assert len(alpha["moves"]) == alpha["statistics"]["moves"]
        assert alpha["statistics"]["attempts"] >= 1

    def test_alpha_vectors_are_not_exported_without_them(self, fh, tmp_path):
        from click.testing import CliRunner

        sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
        export_path = tmp_path / "policy.json"
        assert CliRunner().invoke(fh.main, [sketch_path, props_path, "4", "--export-policy", str(export_path)]).exit_code == 0
        import json

        assert "alpha" not in json.loads(export_path.read_text())

    def test_the_changes_of_the_pool_are_exported(self, fh, tmp_path):
        from click.testing import CliRunner

        sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
        export_path = tmp_path / "policy.json"
        result = CliRunner().invoke(
            fh.main, [sketch_path, props_path, "6", "--alpha-vectors", "16", "--alpha-changes", "3", "--export-policy", str(export_path)]
        )
        assert result.exit_code == 0, result.output
        import json

        alpha = json.loads(export_path.read_text())["alpha"]
        assert (alpha["alpha_vectors"], alpha["alpha_changes"]) == (16, 3)

    def test_the_changes_of_the_pool_need_the_pool(self, fh):
        from click.testing import CliRunner

        sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
        result = CliRunner().invoke(fh.main, [sketch_path, props_path, "3", "--alpha-changes", "2"])
        assert result.exit_code == 2
        assert "--alpha-vectors" in result.output
        assert CliRunner().invoke(fh.main, [sketch_path, props_path, "3", "--alpha-vectors", "4", "--alpha-changes", "0"]).exit_code == 2

    def test_alpha_vectors_must_be_positive(self, fh):
        from click.testing import CliRunner

        sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
        assert CliRunner().invoke(fh.main, [sketch_path, props_path, "3", "--alpha-vectors", "0"]).exit_code == 2

    def test_the_changes_of_a_window_need_windows(self, fh):
        from click.testing import CliRunner

        sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
        result = CliRunner().invoke(fh.main, [sketch_path, props_path, "3", "--window-changes", "2"])
        assert result.exit_code == 2
        assert "--window" in result.output

    @pytest.mark.parametrize("option", ["--window", "--window-changes"])
    def test_windows_must_be_positive(self, fh, option):
        from click.testing import CliRunner

        sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
        assert CliRunner().invoke(fh.main, [sketch_path, props_path, "3", option, "0"]).exit_code == 2
