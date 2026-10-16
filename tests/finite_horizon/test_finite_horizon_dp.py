"""Tests of finite-horizon-dp.py, the exact search over beliefs (normalized reach vectors).

The optimum is compared to a brute force over all color consistent policies of small random colored MDPs: unfolded here with a horizon (reachability), and
unfolded already (layered, reachability and conditional reachability, also with transitions that skip layers). The parts of the algorithm are checked
against their definitions (the enumeration of the stage decisions, the consistency of the upper bound, the point-based backup, the bounds of a node, the
contract of the depth-first search, the merging of proportional reach vectors, and the bounds of a search that is stopped), and the luko models (monitor
verification, conditional) against their known optima.
"""

import itertools
import json
import os

import numpy as np
import pytest
from test_conditional_reachability import LUKO_MODELS, LayeredInstance
from test_conditional_reachability import make_task as line_task
from test_finite_horizon_time_dependent import Instance, brute_force, gadget, gated_gadgets, make_task


def random_instance(seed):
    """A random colored MDP with a reachability objective, small enough for the brute force (None if it is not)."""
    rng = np.random.default_rng(seed)
    maximize, horizon = bool(rng.integers(2)), int(rng.integers(1, 5))
    instance = Instance(seed, int(rng.integers(3, 8)), max_holes=4, max_component_holes=2)
    if int(np.prod(instance.hole_num_options)) ** horizon > 20000:
        return None
    return instance, maximize, horizon


def with_single_choices(instance):
    """The instance in which every state without colors keeps only its first choice: its mass moves without a decision (as part of a sum, in fractions)."""
    state_choices, choice_color, successors, reward = [], [], [], []
    for choices in instance.state_choices:
        kept = choices if instance.choice_color[choices[0]] else choices[:1]
        state_choices.append(list(range(len(choice_color), len(choice_color) + len(kept))))
        choice_color += [instance.choice_color[choice] for choice in kept]
        successors += [instance.successors[choice] for choice in kept]
        reward += [instance.reward[choice] for choice in kept]
    instance.state_choices, instance.choice_color, instance.successors, instance.reward = state_choices, choice_color, successors, reward
    return instance


def problem_of(fhdp, instance, maximize, horizon):
    return fhdp.Problem.from_colored_mdp(instance.colored_mdp(False), make_task(maximize, False), horizon)


# (seed, maximize, horizon) of instances in which phase 1 without the beam (one sweep at the root) is not optimal, so phase 2 has to improve the incumbent
WEAK = [(9, False, 4), (31, True, 4), (58, True, 2), (66, True, 4), (72, True, 4), (125, False, 4), (143, True, 2), (278, True, 3)]


# --- layered models (unfolded already) ---


def random_layered(seed):
    """A random layered colored MDP (possibly with states without colors, also with several choices, and with two classes in a layer) and a direction."""
    rng = np.random.default_rng(seed)
    instance = LayeredInstance(
        seed, layers=int(rng.integers(2, 5)), width=3, classes=int(rng.integers(1, 3)), forced=bool(rng.integers(2)), free_choices=bool(rng.integers(2))
    )
    free = [s for s in range(instance.num_states) if s not in instance.absorbing and not instance.choice_color[instance.state_choices[s][0]]]
    if int(np.prod(instance.hole_num_options)) * int(np.prod([len(instance.state_choices[s]) for s in free])) > 20000:
        return None
    return instance, bool(rng.integers(2))


def skipping():
    """A layered model whose first layer leads to the third one directly (and through the second one), so that some mass waits during a step."""
    sink, stop, goal = LayeredInstance.SINK, LayeredInstance.STOP, LayeredInstance.GOAL
    states = [
        [({0: 0}, {4: 0.6, 6: 0.4}), ({0: 1}, {5: 0.7, 7: 0.3})],
        [({1: 0}, {6: 0.5, goal: 0.5}), ({1: 1}, {7: 1.0})],
        [({1: 0}, {6: 1.0}), ({1: 1}, {stop: 0.5, 7: 0.5})],
        [({2: 0}, {goal: 0.8, sink: 0.2}), ({2: 1}, {stop: 1.0})],
        [({2: 0}, {goal: 0.3, stop: 0.7}), ({2: 1}, {goal: 0.6, sink: 0.4})],
    ]
    return LayeredInstance.explicit([[3], [4, 5], [6, 7]], [2, 2, 2], states)


def reachability_line(maximize):
    return f'P{"max" if maximize else "min"}=? [F "goal"]'


def layered_problem(fhdp, instance, maximize, conditional):
    task = line_task(maximize) if conditional else line_task(line=reachability_line(maximize))
    return fhdp.Problem.from_colored_mdp(instance.colored_mdp(), task, None)


def goal_probability(instance, policy):
    """The probability that the policy (the choice of every state) reaches the goal of a layered instance."""
    mass = {instance.init: 1.0}
    for state in [s for states in instance.layer_states for s in states]:
        for successor, probability in instance.successors[policy[state]].items():
            mass[successor] = mass.get(successor, 0.0) + mass.get(state, 0.0) * probability
    return mass.get(instance.GOAL, 0.0)


def best_goal_probability(instance, maximize):
    return (max if maximize else min)(goal_probability(instance, policy) for policy in instance.color_consistent_policies())


# --- brute force from a node ---


def run_from(problem, vector, step, suffix):
    """The payoff that the decisions of the steps step, step+1, ... collect from the mass vector (with the waiting mass of the states of later steps)."""
    reach, collected = vector.copy(), 0.0
    for offset, decision in enumerate(suffix):
        acting = problem.acting(step + offset)
        moving = np.where(problem.terminal, 0.0, reach)
        executed = np.zeros(problem.matrix.shape[0])
        executed[problem.choices_of(np.asarray(decision, dtype=np.int64))[acting]] = moving[acting]
        reach = problem.matrix.T @ executed + np.where(acting, 0.0, moving)
        collected += float(reach[problem.terminal] @ problem.payoff[problem.terminal])
    return collected


def stage_decisions(problem, step):
    """All the stage decisions of the groups that choose in the step (the other groups take the option 0)."""
    groups = [g for g in range(len(problem.groups)) if problem.group_step[g] in (-1, step)]
    for options in itertools.product(*[range(problem.groups[g].num_options) for g in groups]):
        decision = np.zeros(len(problem.groups), dtype=np.int64)
        decision[groups] = options
        yield decision


def completion_optimum(problem, node):
    """W(node): the best value of a suffix from the node, by enumerating all the suffixes."""
    suffixes = itertools.product(*[list(stage_decisions(problem, step)) for step in range(node.step, problem.num_steps)])
    values = [run_from(problem, node.dense(problem.num_states), node.step, suffix) for suffix in suffixes]
    return (max if problem.maximize else min)(values)


def nodes_of_a_beam(fhdp, graph, width=4):
    return [node for layer in fhdp.beam(graph, width) for node in layer]


class TestOptimality:
    @pytest.mark.parametrize("seed", range(80))
    def test_the_optimum_is_found_and_proved(self, fhdp, seed):
        case = random_instance(seed)
        if case is None:
            pytest.skip("too large for the brute force")
        instance, maximize, horizon = case
        optimum = brute_force(instance, horizon, maximize, False)
        problem = problem_of(fhdp, instance, maximize, horizon)
        for width, rounds in ((0, 1), (8, 10)):
            result = fhdp.solve(problem, width=width, rounds=rounds)
            assert result.optimal
            assert result.value == pytest.approx(optimum, abs=1e-9)
            assert result.bound == pytest.approx(optimum, abs=1e-9)
            assert problem.evaluate(result.decisions) == pytest.approx(optimum, abs=1e-9)

    @pytest.mark.parametrize("seed", range(40))
    def test_the_optimum_with_states_that_have_a_single_choice(self, fhdp, seed):
        case = random_instance(seed)
        if case is None:
            pytest.skip("too large for the brute force")
        instance, maximize, horizon = case
        instance = with_single_choices(instance)
        problem = problem_of(fhdp, instance, maximize, horizon)
        result = fhdp.solve(problem, width=4, rounds=3)
        assert result.optimal and result.value == pytest.approx(brute_force(instance, horizon, maximize, False), abs=1e-9)

    @pytest.mark.parametrize(("seed", "maximize", "horizon"), WEAK)
    def test_phase_two_improves_a_policy_that_is_not_optimal(self, fhdp, seed, maximize, horizon):
        instance, expected_maximize, expected_horizon = random_instance(seed)
        assert (maximize, horizon) == (expected_maximize, expected_horizon)
        optimum = brute_force(instance, horizon, maximize, False)
        problem = problem_of(fhdp, instance, maximize, horizon)
        result = fhdp.solve(problem, width=0, rounds=1)
        assert problem.better(result.value, result.statistics["phase 1 value"])
        assert result.statistics["improved"]
        assert result.optimal and result.value == pytest.approx(optimum, abs=1e-9)

    @pytest.mark.parametrize("seed", range(20))
    def test_folding_does_not_change_the_optimum(self, fhdp, seed):
        case = random_instance(seed)
        if case is None:
            pytest.skip("too large for the brute force")
        problem = problem_of(fhdp, *case)
        assert fhdp.solve(problem, fold=True).value == pytest.approx(fhdp.solve(problem, fold=False).value, abs=1e-12)

    @pytest.mark.parametrize("seed", range(8))
    def test_storm_agrees_on_the_policy(self, fhdp, seed):
        case = random_instance(seed)
        if case is None:
            pytest.skip("too large")
        instance, maximize, horizon = case
        problem = problem_of(fhdp, instance, maximize, horizon)
        result = fhdp.solve(problem)
        value, consistent = fhdp.verify_with_storm(problem, instance.colored_mdp(False), make_task(maximize, False), result.decisions)
        assert consistent
        assert value == pytest.approx(result.value, abs=1e-5)

    def test_the_counter_example_of_the_backward_passes(self, fhdp):
        problem = problem_of(fhdp, gadget(), True, 3)
        for width in (0, 32):
            result = fhdp.solve(problem, width=width)
            assert result.optimal and result.value == pytest.approx(0.1, abs=1e-12)

    def test_a_trap_that_needs_three_changes_at_once(self, fhdp):
        """The heuristics stop at 0.099 here (see the tests of finite-horizon-time-dependent.py); the exact search finds and proves 0.1."""
        problem = problem_of(fhdp, gated_gadgets(), True, 5)
        for width in (0, 32):
            result = fhdp.solve(problem, width=width)
            assert result.optimal and result.value == pytest.approx(0.1, abs=1e-12)

    def test_the_initial_state_in_the_target(self, fhdp):
        instance = Instance(3, 5, max_holes=2, init_is_target=True)
        result = fhdp.solve(problem_of(fhdp, instance, False, 3))
        assert result.optimal and result.value == 1.0

    def test_a_model_without_parameters(self, fhdp):
        instance = Instance(7, 8, max_holes=0)
        result = fhdp.solve(problem_of(fhdp, instance, True, 5))
        assert result.optimal
        assert result.value == pytest.approx(brute_force(instance, 5, True, False), abs=1e-12)


class TestLayeredModels:
    """Models that are unfolded already: their layers are the steps."""

    @pytest.mark.parametrize("seed", range(60))
    def test_reachability_is_optimal(self, fhdp, seed):
        case = random_layered(seed)
        if case is None:
            pytest.skip("too large for the brute force")
        instance, maximize = case
        problem = layered_problem(fhdp, instance, maximize, conditional=False)
        assert problem.layered and problem.num_steps <= len(instance.layer_states)  # the layers that can be reached
        result = fhdp.solve(problem, width=4, rounds=3)
        assert result.optimal and result.value == pytest.approx(best_goal_probability(instance, maximize), abs=1e-9)

    @pytest.mark.parametrize("maximize", [True, False])
    def test_transitions_that_skip_a_layer(self, fhdp, maximize):
        instance = skipping()
        for conditional in (False, True):
            problem = layered_problem(fhdp, instance, maximize, conditional)
            assert problem.step[[3, 4, 5, 6, 7]].tolist() == [0, 1, 1, 2, 2]  # the longest paths: 6 and 7 are also reached from 3 directly
            if conditional:
                result = fhdp.solve_conditional(problem, width=0, rounds=1)
                assert result.optimal and result.value == pytest.approx(instance.best(maximize), abs=1e-9)
            else:
                result = fhdp.solve(problem, width=0, rounds=1)
                assert result.optimal and result.value == pytest.approx(best_goal_probability(instance, maximize), abs=1e-9)

    @pytest.mark.parametrize("seed", range(6))
    def test_storm_agrees_on_the_policy(self, fhdp, seed):
        instance, maximize = random_layered(seed) or (skipping(), True)
        for conditional in (False, True):
            problem = layered_problem(fhdp, instance, maximize, conditional)
            task = line_task(maximize) if conditional else line_task(line=reachability_line(maximize))
            result = fhdp.solve_conditional(problem) if conditional else fhdp.solve(problem)
            value, consistent = fhdp.verify_with_storm(problem, instance.colored_mdp(), task, result.decisions)
            assert consistent and value == pytest.approx(result.value, abs=1e-5)

    def test_a_cycle_is_rejected(self, fhdp):
        instance = LayeredInstance.explicit([[3], [4]], [2], [[({0: 0}, {4: 1.0}), ({0: 1}, {2: 1.0})], [({}, {3: 0.5, 2: 0.5})]])
        with pytest.raises(fhdp.UnsupportedProblemError, match="acyclic"):
            layered_problem(fhdp, instance, True, conditional=False)

    def test_a_component_in_two_layers_is_rejected(self, fhdp):
        instance = LayeredInstance.explicit([[3], [4]], [2], [[({0: 0}, {4: 1.0}), ({0: 1}, {2: 1.0})], [({0: 0}, {2: 1.0}), ({0: 1}, {1: 1.0})]])
        with pytest.raises(fhdp.UnsupportedProblemError, match="steps 0 and 1"):
            layered_problem(fhdp, instance, True, conditional=False)

    def test_condition_states_have_to_be_absorbing(self, fhdp):
        instance = skipping()
        instance.successors[instance.state_choices[instance.STOP][0]] = {instance.GOAL: 1.0}
        with pytest.raises(fhdp.UnsupportedProblemError, match="absorbing"):
            layered_problem(fhdp, instance, True, conditional=True)
        layered_problem(fhdp, instance, True, conditional=False)  # for reachability the goal just ends the run

    def test_a_conditional_property_needs_a_model_that_is_unfolded_already(self, fhdp):
        with pytest.raises(fhdp.UnsupportedProblemError, match="unfolded already"):
            fhdp.Problem.from_colored_mdp(skipping().colored_mdp(), line_task(True), 3)


class TestConditional:
    @pytest.mark.parametrize("seed", range(60))
    def test_the_optimum_is_found_and_proved(self, fhdp, seed):
        case = random_layered(seed)
        if case is None:
            pytest.skip("too large for the brute force")
        instance, maximize = case
        optimum = instance.best(maximize)
        problem = layered_problem(fhdp, instance, maximize, conditional=True)
        if optimum is None:
            with pytest.raises(fhdp.UnsupportedProblemError, match="not defined"):
                fhdp.solve_conditional(problem)
            return
        result = fhdp.solve_conditional(problem, width=4, rounds=3)
        assert result.optimal and result.value == pytest.approx(optimum, abs=1e-9)
        assert result.value == pytest.approx(result.numerator / result.evidence, abs=1e-12)
        # Dinkelbach's method improves strictly
        assert all(problem.better(b, a) for a, b in itertools.pairwise(result.lambdas))
        assert result.lambdas[-1] == pytest.approx(result.value, abs=1e-12)

    def test_a_condition_that_cannot_happen(self, fhdp):
        instance = LayeredInstance.explicit([[3]], [2], [[({0: 0}, {0: 1.0}), ({0: 1}, {0: 1.0})]])
        with pytest.raises(fhdp.UnsupportedProblemError, match="not defined"):
            fhdp.solve_conditional(layered_problem(fhdp, instance, True, conditional=True))

    def test_a_search_that_is_stopped_gives_a_policy(self, fhdp):
        instance = LayeredInstance.dense(5, layers=4, width=3, options=3)
        problem = layered_problem(fhdp, instance, True, conditional=True)
        result = fhdp.solve_conditional(problem, width=0, rounds=1, max_nodes=2)
        assert result.value <= instance.best(True) + 1e-9
        assert result.optimal or any("stopped" in search for search in result.statistics["iterations"])
        assert problem.evaluate(result.decisions, problem.goal.astype(float)) == pytest.approx(result.numerator, abs=1e-12)


class TestStoppedSearch:
    @pytest.mark.parametrize(("seed", "maximize", "horizon"), WEAK)
    def test_a_stopped_search_brackets_the_optimum(self, fhdp, seed, maximize, horizon):
        instance, *_ = random_instance(seed)
        optimum = brute_force(instance, horizon, maximize, False)
        problem = problem_of(fhdp, instance, maximize, horizon)
        for max_nodes in (1, 3, 10):
            result = fhdp.solve(problem, width=0, rounds=1, max_nodes=max_nodes)
            low, high = sorted((result.value, result.bound))
            assert low - 1e-9 <= optimum <= high + 1e-9
            assert problem.evaluate(result.decisions) == pytest.approx(result.value, abs=1e-9)
            assert result.optimal == (not problem.better(result.bound, result.value))
            if not result.optimal:
                assert result.statistics["stopped"] == "nodes"

    @pytest.mark.parametrize("seed", range(40))
    def test_the_bracket_of_a_stopped_search_holds(self, fhdp, seed):
        """Stopped after a few nodes (in the middle of a descent of the thresholds, or of a child), the search still brackets the optimum."""
        for problem, optimum in optima(fhdp, seed):
            for max_nodes in (2, 5, 20):
                result = fhdp.solve(problem, width=0, rounds=1, max_nodes=max_nodes)
                low, high = sorted((result.value, result.bound))
                assert low - 1e-9 <= optimum <= high + 1e-9
                assert problem.evaluate(result.decisions) == pytest.approx(result.value, abs=1e-9)

    def test_the_time_limit_stops_the_search(self, fhdp):
        instance, maximize, horizon = random_instance(WEAK[0][0])
        problem = problem_of(fhdp, instance, maximize, horizon)
        result = fhdp.solve(problem, width=0, rounds=1, time_limit=0.0)
        low, high = sorted((result.value, result.bound))
        assert low - 1e-9 <= brute_force(instance, horizon, maximize, False) <= high + 1e-9


def optima(fhdp, seed):
    """Problems of the seed (unfolded here, and layered) with their optima by brute force."""
    out = []
    case = random_instance(seed)
    if case is not None:
        instance, maximize, horizon = case
        out.append((problem_of(fhdp, instance, maximize, horizon), brute_force(instance, horizon, maximize, False)))
    case = random_layered(seed)
    if case is not None:
        instance, maximize = case
        out.append((layered_problem(fhdp, instance, maximize, conditional=False), best_goal_probability(instance, maximize)))
    return out


def naive_edges(problem, relaxation, node):
    """(child key, reward, scale) of the edges of the node under all the stage decisions of the groups that choose in the step and that its mass touches."""
    acting = problem.acts(node.step, node.support)
    present = sorted({int(group) for group in problem.group_of[node.support[acting]] if group >= 0})
    out = set()
    for options in itertools.product(*[range(problem.groups[g].num_options) for g in present]):
        decision = np.zeros(len(problem.groups), dtype=np.int64)
        decision[present] = options
        executed = np.zeros(problem.matrix.shape[0])
        executed[problem.choices_of(decision)[node.support[acting]]] = node.mass[acting]
        reached = problem.matrix.T @ executed
        reached[node.support[~acting]] += node.mass[~acting]
        moving = np.where(relaxation.alive[node.step + 1], reached, 0.0)
        scale = float(moving.sum())
        child = moving / scale if scale > 0 else moving
        support = np.flatnonzero(child)
        out.add((support.tobytes(), np.round(child[support], 9).tobytes(), round(float(reached @ relaxation.cash[node.step + 1]), 9), round(scale, 9)))
    return out


def signature(edge):
    child = edge.child
    return child.support.tobytes(), np.round(child.mass, 9).tobytes(), round(edge.reward, 9), round(edge.scale, 9)


def small_problems(fhdp, seed):
    """The problems of a seed: unfolded here (with and without states that have a single choice) and layered (reachability, and N - D/2 of a conditional
    probability, whose payoffs are negative too)."""
    problems = []
    case = random_instance(seed)
    if case is not None:
        instance, maximize, horizon = case
        problems.append(problem_of(fhdp, instance, maximize, horizon))
        problems.append(problem_of(fhdp, with_single_choices(random_instance(seed)[0]), maximize, horizon))
    case = random_layered(seed)
    instance, maximize = case if case is not None else (skipping(), True)
    problems.append(layered_problem(fhdp, instance, maximize, conditional=False))
    conditional = layered_problem(fhdp, instance, maximize, conditional=True)
    goal, condition = conditional.goal.astype(float), conditional.condition.astype(float)
    problems.append(conditional.with_payoff(goal - 0.5 * condition))
    return problems


class TestParts:
    def test_combinations_come_in_the_order_of_decreasing_scores(self, fhdp):
        rng = np.random.default_rng(0)
        for _ in range(30):
            scores = [sorted(((float(rng.normal()), o) for o in range(int(rng.integers(1, 4)))), reverse=True) for _ in range(int(rng.integers(0, 4)))]
            base = float(rng.normal())
            generated = list(fhdp.best_combinations(scores, base, -np.inf))
            expected = sorted((base + sum({o: v for v, o in group}[option] for group, option in zip(scores, combination, strict=True)), combination)
                              for combination in itertools.product(*[[o for _, o in group] for group in scores]))  # fmt: skip
            assert [c for _, c in generated] and sorted(c for _, c in generated) == sorted(c for _, c in expected)
            totals = [t for t, _ in generated]
            assert all(a >= b - 1e-12 for a, b in zip(totals, totals[1:], strict=False))
            assert totals[0] == pytest.approx(max(t for t, _ in expected))
            threshold = float(np.median(totals))
            assert [c for t, c in generated if t > threshold] == [c for _, c in fhdp.best_combinations(scores, base, threshold)]

    @pytest.mark.parametrize("seed", range(30))
    def test_edges_are_complete_ordered_and_the_upper_bound_is_consistent(self, fhdp, seed):
        for problem in small_problems(fhdp, seed):
            graph = fhdp.Graph(problem, fhdp.Relaxation.compute(problem))
            layer = [graph.root()[0]]
            for _ in range(problem.num_steps):
                following = []
                for node in layer[:5]:
                    edges = list(graph.children(node, -np.inf))
                    # every edge of some stage decision is generated (the options of a group that lead to the same edge are generated once)
                    assert {signature(edge) for edge in edges} == naive_edges(problem, graph.relaxation, node)
                    bounds = [edge.bound for edge in edges]
                    assert all(problem.sign * (a - b) >= -1e-12 for a, b in itertools.pairwise(bounds))
                    for edge in edges:
                        assert edge.reward + edge.scale * graph.upper(edge.child) == pytest.approx(edge.bound, abs=1e-12)
                        assert problem.sign * (edge.bound - graph.upper(node)) <= 1e-12  # consistent: the bound does not grow along an edge
                        assert edge.child.support.size == 0 or edge.child.mass.sum() == pytest.approx(1.0, abs=1e-12)
                    following += [edge.child for edge in edges]
                layer = following
            assert all(node.support.size == 0 for node in layer)  # in the step H nothing is moving anymore

    @pytest.mark.parametrize("seed", range(12))
    def test_the_bounds_of_a_node_hold(self, fhdp, seed):
        """lower <= W(node) <= upper for the nodes of a beam; the lower bound is the value of the suffix of its vector."""
        for problem in small_problems(fhdp, seed):
            graph = fhdp.Graph(problem, fhdp.Relaxation.compute(problem))
            _, pool = fhdp.lower_bound_phase(graph, 4, 3)
            for node in nodes_of_a_beam(fhdp, graph):
                if node.support.size == 0:
                    continue
                best = completion_optimum(problem, node)
                lower, vector = pool.lower(node)
                assert problem.sign * (best - lower) >= -1e-9
                assert problem.sign * (graph.upper(node) - best) >= -1e-9
                assert run_from(problem, node.dense(problem.num_states), node.step, pool.suffix(node.step, vector)) == pytest.approx(lower, abs=1e-9)

    @pytest.mark.parametrize("seed", range(12))
    def test_the_search_of_a_node_is_exact_or_below_the_threshold(self, fhdp, seed):
        rng = np.random.default_rng(seed)
        for problem in small_problems(fhdp, seed):
            graph = fhdp.Graph(problem, fhdp.Relaxation.compute(problem))
            _, pool = fhdp.lower_bound_phase(graph, 0, 1)
            search = fhdp.Search(graph, pool, max_nodes=10**6)
            for node in nodes_of_a_beam(fhdp, graph):
                if node.support.size == 0:
                    continue
                best = problem.sign * completion_optimum(problem, node)
                threshold = best + float(rng.normal()) * 0.1
                entry = search.value(node, threshold)
                assert entry.lower - 1e-9 <= best <= entry.upper + 1e-9
                assert entry.upper - entry.lower <= 1e-9 or entry.upper <= threshold + 1e-9
                if best > threshold + 1e-9:
                    assert entry.lower == pytest.approx(best, abs=1e-9)

    @pytest.mark.parametrize("seed", [*range(6), *(case[0] for case in WEAK)])
    def test_children_below_their_threshold_are_not_searched(self, fhdp, seed, monkeypatch):
        """The search goes into a child only if the bound of its edge beats the target, i.e. the upper bound of the child beats its threshold."""
        for problem in small_problems(fhdp, seed):
            graph = fhdp.Graph(problem, fhdp.Relaxation.compute(problem))
            incumbent, pool = fhdp.lower_bound_phase(graph, 0, 1)
            calls = []
            value = fhdp.Search.value

            def recording(self, node, threshold, calls=calls, value=value):
                calls.append((node, threshold))
                return value(self, node, threshold)

            monkeypatch.setattr(fhdp.Search, "value", recording)
            fhdp.search(graph, pool, incumbent, max_nodes=10**6)
            monkeypatch.undo()
            assert all(problem.sign * graph.upper(node) > threshold for node, threshold in calls[1:])

    def test_proportional_reach_vectors_are_one_node(self, fhdp):
        """Two decisions at the root lead to the reach vectors 0.5 * b and 1.0 * b: one child. The states of b share a parameter and prefer different
        options, so the relaxation is loose there and the search has to go into the child."""
        sink, goal, stop = LayeredInstance.SINK, LayeredInstance.GOAL, LayeredInstance.STOP
        states = [
            [({0: 0}, {4: 0.25, 5: 0.25, sink: 0.5}), ({0: 1}, {4: 0.5, 5: 0.5})],
            [({1: 0}, {goal: 1.0}), ({1: 1}, {stop: 1.0})],
            [({1: 0}, {stop: 1.0}), ({1: 1}, {goal: 0.9, sink: 0.1})],
        ]
        instance = LayeredInstance.explicit([[3], [4, 5]], [2, 2], states)
        problem = layered_problem(fhdp, instance, True, conditional=False)
        graph = fhdp.Graph(problem, fhdp.Relaxation.compute(problem))
        root, _ = graph.root()
        edges = list(graph.children(root, -np.inf))
        assert len({edge.child.key for edge in edges}) == 1 and sorted(edge.scale for edge in edges) == [0.5, 1.0]
        assert graph.upper(root) == pytest.approx(0.95)
        _, pool = fhdp.lower_bound_phase(graph, 0, 1)
        search = fhdp.Search(graph, pool, max_nodes=10**6)
        entry = search.value(root, -np.inf)
        assert entry.lower == pytest.approx(0.5) and entry.upper == pytest.approx(0.5) and len(search.table) == 2

    @pytest.mark.parametrize("seed", range(20))
    def test_the_point_based_backup_is_the_best_decision_and_suffix(self, fhdp, seed):
        rng = np.random.default_rng(seed)
        for problem in small_problems(fhdp, seed):
            relaxation = fhdp.Relaxation.compute(problem)
            for step in range(problem.num_steps):
                alive = relaxation.alive[step]
                anchors = rng.random((3, problem.num_states)) * alive * (rng.random((3, problem.num_states)) < 0.7)
                following = rng.random((4, problem.num_states))
                following[:, problem.terminal] = problem.payoff[problem.terminal]
                layer = fhdp.point_backup(problem, relaxation, step, anchors, following)
                stages = list(stage_decisions(problem, step))
                acting = problem.acting(step)
                for anchor in anchors:
                    naive = -np.inf
                    for stage in stages:
                        executed = np.zeros(problem.matrix.shape[0])
                        executed[problem.choices_of(stage)[acting]] = anchor[acting]
                        reached = problem.matrix.T @ executed + np.where(acting, 0.0, anchor)
                        naive = max(naive, *(problem.sign * float(reached @ vector) for vector in following))
                    mine = max(problem.sign * float(anchor @ vector) for vector in layer.vectors)
                    assert mine == pytest.approx(naive, abs=1e-9)

    def test_rewards_are_not_supported(self, fhdp):
        instance = Instance(3, 5, max_holes=2)
        with pytest.raises(fhdp.UnsupportedProblemError, match="probabilities"):
            fhdp.Problem.from_colored_mdp(instance.colored_mdp(True), make_task(True, True), 3)


class TestCommandLine:
    def test_run_verify_and_export(self, fhdp, tmp_path):
        from click.testing import CliRunner

        export_path = tmp_path / "policy.json"
        arguments = ["models/fh-counterexample/sketch.templ", "models/fh-counterexample/sketch.props", "3", "--verify", "--show-policy", "--export-policy"]
        result = CliRunner().invoke(fhdp.main, [*arguments, str(export_path)])
        assert result.exit_code == 0, result.output
        exported = json.loads(export_path.read_text())
        assert exported["optimal"] and exported["value"] == pytest.approx(0.1, abs=1e-12)
        assert len(exported["policy"]) == 3

    def test_a_conditional_property_of_a_luko_model(self, fhdp, tmp_path):
        from click.testing import CliRunner

        path = os.path.join(LUKO_MODELS, "05-icy-h40-fp")
        if not os.path.exists(path):
            pytest.skip("the luko models are not available")
        export_path = tmp_path / "policy.json"
        arguments = [os.path.join(path, "model.drn"), os.path.join(path, "property.props"), "--verify", "--export-policy", str(export_path)]
        result = CliRunner().invoke(fhdp.main, arguments)
        assert result.exit_code == 0, result.output
        exported = json.loads(export_path.read_text())
        assert exported["optimal"] and exported["layered"] and exported["value"] == pytest.approx(0.9, abs=1e-9)
        assert len(exported["policy"]) == exported["steps"] == 41

    def test_a_conditional_property_with_a_horizon_is_reported(self, fhdp):
        from click.testing import CliRunner

        path = os.path.join(LUKO_MODELS, "05-icy-h40-fp")
        if not os.path.exists(path):
            pytest.skip("the luko models are not available")
        result = CliRunner().invoke(fhdp.main, [os.path.join(path, "model.drn"), os.path.join(path, "property.props"), "3"])
        assert result.exit_code == 1 and "without a HORIZON" in " ".join(result.output.replace("│", " ").split())

    def test_a_reward_property_is_reported(self, fhdp):
        from click.testing import CliRunner

        result = CliRunner().invoke(fhdp.main, ["models/tests/pomdp-maze/sketch.templ", "models/tests/pomdp-maze/sketch.props", "3"])
        assert result.exit_code == 1
        assert "probabilities" in result.output
