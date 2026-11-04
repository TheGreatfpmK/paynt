#!/usr/bin/env python3
"""Verify a finite-state controller (FSC) on an interval POMDP with PAYNT, nature choosing extreme points.

1. Build the interval POMDP with Storm; find_ipomdp_vertices.py gives, for every choice (s,a), the extreme
   points (vertices) of {p : l <= p <= u, sum(p) = 1}.
2. Unfold the FSC: product states (s,n). At (s,n) the FSC fixes the action a = act(n, z(s)) and the next
   node n' = upd(n, z(s)); nature picks a vertex v of (s,a) and the product moves to (t_i, n') with
   probability v_i. Target and avoid states of the property are absorbing.
3. Color the product MDP: the row of vertex j at (s,n) gets parameter option j, where the parameter is
     memory  one per original choice (s,a): every memory copy (s,n) must use the same vertex
     none    one per product state: the standard (rectangular) robust semantics
4. Run PAYNT's abstraction refinement (AR) on the colored MDP for nature's objective.

Objectives: P=? [F b], P=? [a U b], or R=? [F b] (expected cost; nature maximizes it by default).
Reported per tie mode (nature minimizing; mirrored for nature maximizing):
    U  value of the quotient MDP = nature free in every copy = Storm's robust value of the product iDTMC
    V  PAYNT's optimum over tied vertex assignments
With ties, the true tied value T (nature may pick any point of the polytope, the same in every copy) lies
in [U, V] and need not equal V: extreme points are not enough once rows are tied (coin-tie.prism with
coin-tie-min.fsc.json: U = 0.36, T = 0.75, V = 0.84).

Examples:
    python ipomdp-fsc-paynt.py models/robust-pomdps/coin-tie.prism 'P=? [F "won"]' --fsc json:models/robust-pomdps/coin-tie-min.fsc.json
    python ipomdp-fsc-paynt.py models/robust-pomdps/evade-large.prism 'P=? ["notbad" U "goal"]' --fsc random:2:0
    python ipomdp-fsc-paynt.py models/robust-pomdps/evade-large.prism 'P=? ["notbad" U "goal"]' --fsc paynt:2:60
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
import time

import numpy as np
import stormpy

import find_ipomdp_vertices
import paynt.colored_mdp
import paynt.parameter_space.parameter_space
import paynt.specification.property
import paynt.synthesizer.synthesizer_ar
import paynt.task
import paynt.utils.timer
import payntbind

logger = logging.getLogger("ipomdp-fsc-paynt")


# ----------------------------------------------------------------- the interval POMDP


def load(path: str, prop: str, assignments: list[str]):
    """Build the interval POMDP with respect to the property (target states are not explored further)."""
    with open(path) as f:
        text = f.read()
    for assignment in assignments:
        name, value = assignment.split("=", 1)
        text, count = re.subn(rf"(const\s+(?:int|double|bool)\s+{re.escape(name)}\s*=\s*)[^;]+;", lambda m: f"{m.group(1)}{value};", text)
        assert count == 1, f"constant {name} is not defined in the model"
    # Storm refuses observation valuations whose label clashes with a variable name (avoid-large: "dir")
    text = re.sub(r'observable\s+"(\w+)"', r'observable "o_\1"', text)
    handle, tmp = tempfile.mkstemp(suffix=".prism")
    with os.fdopen(handle, "w") as f:
        f.write(text)
    try:
        program = stormpy.parse_prism_program(tmp)
    finally:
        os.remove(tmp)
    properties = stormpy.parse_properties(prop, program)
    options = stormpy.BuilderOptions([p.get_raw_formula() for p in properties])
    options.set_build_choice_labels(True)
    options.set_build_observation_valuations(True)
    options.set_build_all_labels(True)
    model = stormpy.build_model[stormpy.Interval](program, options=options)
    return model, properties[0].get_raw_formula()


def state_set(model, formula) -> np.ndarray:
    """States satisfying a propositional formula over labels (its string is evaluated as a numpy expression)."""
    n = model.nr_states
    masks = {}
    for label in model.labeling.get_labels():
        mask = np.zeros(n, dtype=bool)
        mask[list(model.labeling.get_states(label))] = True
        masks[label] = mask
    expression = re.sub(r'"([^"]+)"', r'L["\1"]', str(formula)).replace("!", "~")
    expression = re.sub(r"\btrue\b", "T", re.sub(r"\bfalse\b", "F", expression))
    assert re.fullmatch(r'[\sLTF~&|()\[\]"\w-]*', expression), f"unsupported state formula {formula}"
    return np.asarray(eval(expression, {"L": masks, "T": np.ones(n, bool), "F": np.zeros(n, bool)}), dtype=bool) & np.ones(n, bool)


def target_and_avoid(model, formula):
    """P[F b] or P[a U b]: target = b, avoid = !a & !b."""
    path = formula.subformula
    if isinstance(path, stormpy.logic.EventuallyFormula):
        target = state_set(model, path.subformula)
        return target, np.zeros(model.nr_states, dtype=bool)
    assert isinstance(path, stormpy.logic.UntilFormula), f"only P[F b] and P[a U b] are supported, not {formula}"
    target = state_set(model, path.right_subformula)
    return target, ~state_set(model, path.left_subformula) & ~target


def choice_costs(model, formula, table):
    """R[F b]: the cost of every choice (its state's reward plus its own reward), None for P[...]."""
    if not isinstance(formula, stormpy.logic.RewardOperatorFormula):
        return None
    assert len(model.reward_models) == 1, "R=? [F b] needs exactly one reward model"
    (reward_model,) = model.reward_models.values()
    assert not reward_model.has_transition_rewards, "transition rewards are not supported"

    def point(vector):
        lower, upper = np.array([v.lower() for v in vector]), np.array([v.upper() for v in vector])
        assert np.array_equal(lower, upper), "interval rewards are not supported"
        return lower

    cost = np.zeros(int(table.state_choice_offsets[-1]))
    if reward_model.has_state_action_rewards:
        cost += point(reward_model.state_action_rewards)
    if reward_model.has_state_rewards:
        cost += np.repeat(point(reward_model.state_rewards), np.diff(table.state_choice_offsets))
    return cost


def actions_at_observation(table) -> list[list[str]]:
    """Action names available at every observation (states with one observation offer the same actions)."""
    result: list[list[str] | None] = [None] * (int(table.observations.max()) + 1)
    for s in range(table.nr_states):
        names = [table.action_names[table.choice_action[c]] for c in table.choices_of_state(s)]
        z = table.observations[s]
        if result[z] is None:
            result[z] = names
        else:
            assert sorted(result[z]) == sorted(names), f"states with observation {z} offer different actions"
    return [r or [] for r in result]


# ----------------------------------------------------------------- FSCs


def random_fsc(table, nodes: int, seed: int):
    rng = np.random.default_rng(seed)
    actions = actions_at_observation(table)
    action = [[actions[z][rng.integers(len(actions[z]))] if actions[z] else None for z in range(len(actions))] for _ in range(nodes)]
    update = [[int(rng.integers(nodes)) for _ in range(len(actions))] for _ in range(nodes)]
    return action, update


def json_fsc(model, path: str):
    """{"action": [per node: {observation label: action name}], "update": [per node: {observation label: node}]}."""
    with open(path) as f:
        spec = json.load(f)
    labels = [model.observation_valuations.to_string(z) for z in range(model.nr_observations)]

    def by_observation(table):
        if isinstance(table, list):
            return table
        missing = [label for label in labels if label not in table]
        assert not missing, f"the FSC misses observations {missing[:5]}"
        return [table[label] for label in labels]

    return [by_observation(t) for t in spec["action"]], [by_observation(t) for t in spec["update"]]


def nominal_pomdp(model, table, target, avoid, cost=None):
    """The POMDP whose every choice uses the centroid of its vertices; target and avoid states absorbing."""
    stop = target | avoid
    builder = stormpy.datastructures.SparseMatrixBuilder(model.nr_choices, model.nr_states, 0, True, True, model.nr_states)
    for s in range(model.nr_states):
        builder.new_row_group(int(table.state_choice_offsets[s]))
        for c in table.choices_of_state(s):
            if stop[s]:
                builder.add_next_value(c, s, 1.0)
                continue
            successors, vertices = table.choice_vertices(c)
            centroid = vertices.mean(axis=0)
            for t, p in sorted(zip(successors, centroid)):
                if p > 0:
                    builder.add_next_value(c, int(t), float(p))
    reward_models = {} if cost is None else {"cost": stormpy.models.SparseRewardModel(optional_state_action_reward_vector=[float(c) for c in cost])}
    components = stormpy.models.SparseModelComponents(transition_matrix=builder.build(), state_labeling=model.labeling, reward_models=reward_models)
    components.observability_classes = list(model.observations)
    components.choice_labeling = model.choice_labeling
    return stormpy.models.SparsePomdp(components)


def paynt_fsc(model, table, target, avoid, nodes: int, timeout: int, cost=None):
    """An FSC synthesized by PAYNT for the nominal POMDP (agent maximizing the probability to reach the target,
    or minimizing the expected cost to reach it)."""
    import paynt.pomdp
    import paynt.pomdp._utils
    import paynt.pomdp.task

    pomdp = nominal_pomdp(model, table, target, avoid, cost)
    labeling = pomdp.labeling
    if not labeling.contains_label("__target__"):
        labeling.add_label("__target__")
    for s in np.flatnonzero(target):
        labeling.add_label_to_state("__target__", int(s))
    paynt.utils.timer.GlobalTimer.start(None)
    factory = paynt.pomdp.PomdpColoredMdpFactory(pomdp, paynt.pomdp.task.PomdpTask(memory_size=nodes))
    colored_mdp = factory.build()
    task = task_for('Pmax=? [F "__target__"]' if cost is None else 'R{"cost"}min=? [F "__target__"]')
    synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(colored_mdp, task)
    assignment = synthesizer.synthesize(timeout=timeout, keep_optimum=True, print_stats=False)
    assert assignment is not None, "PAYNT found no FSC"
    fsc = paynt.pomdp._utils.assignment_to_fsc(colored_mdp.feature_info, assignment)
    assert list(colored_mdp.feature_info.pomdp.observations) == list(model.observations), "PAYNT's POMDP numbers observations differently"
    action = [[None if a is None else fsc.action_labels[a] for a in fsc.action_function[node]] for node in range(fsc.num_nodes)]
    update = [list(fsc.update_function[node]) for node in range(fsc.num_nodes)]
    action = [[None if a is None else ("" if a == paynt.colored_mdp.ColoredMdp.EMPTY_LABEL else a) for a in row] for row in action]
    return action, update, synthesizer.best_assignment_value


# ----------------------------------------------------------------- the colored product


class Product:
    """The FSC-unfolded product; one row per vertex of the choice the FSC takes."""

    def __init__(self, table, target, avoid, fsc_action, fsc_update, cost=None):
        stop = target | avoid
        choice_of_action = []
        for s in range(table.nr_states):
            choice_of_action.append({table.action_names[table.choice_action[c]]: c for c in table.choices_of_state(s)})
        init = (int(table.initial_states[0]), 0)
        index = {init: 0}
        self.states = [init]
        self.rows = []  # per product state: list of (successor product states, probabilities)
        self.choice = []  # per product state: original choice (-1 for absorbing states)
        frontier = [init]
        while frontier:
            state = frontier.pop()
            s, node = state
            p = index[state]
            while len(self.rows) <= p:
                self.rows.append(None)
                self.choice.append(-1)
            if stop[s]:
                self.rows[p] = [([p], np.array([1.0]))]
                continue
            z = table.observations[s]
            c = choice_of_action[s][fsc_action[node][z]]
            next_node = fsc_update[node][z]
            successors, vertices = table.choice_vertices(c)
            columns = []
            for t in successors:
                key = (int(t), next_node)
                j = index.get(key)
                if j is None:
                    j = index[key] = len(self.states)
                    self.states.append(key)
                    frontier.append(key)
                columns.append(j)
            self.rows[p] = [(columns, v) for v in vertices]
            self.choice[p] = c
        self.n = len(self.states)
        self.target = np.array([target[s] for s, _ in self.states])
        self.branch_lower, self.branch_upper = table.branch_lower, table.branch_upper
        self.choice_branch_offsets = table.choice_branch_offsets
        # expected cost: the cost of the choice the FSC takes (none in target states); None = reachability
        self.cost = None if cost is None else np.array([0.0 if c < 0 else float(cost[c]) for c in self.choice])
        if self.cost is not None:
            # then which states reach the target almost surely must not depend on nature (TiedChain.solve)
            assert all(np.all(v > 0) for rows in self.rows for _, v in rows), "expected cost: every extreme point must keep all successors"

    def prop(self, direction: str) -> str:
        """The property on the product, for direction min, max or "" (a Markov chain)."""
        operator = "P" if self.cost is None else 'R{"cost"}'
        return f'{operator}{direction}=? [F "goal"]'

    def mdp(self):
        nr_rows = sum(len(r) for r in self.rows)
        builder = stormpy.datastructures.SparseMatrixBuilder(nr_rows, self.n, 0, True, True, self.n)
        row = 0
        for p in range(self.n):
            builder.new_row_group(row)
            for columns, probabilities in self.rows[p]:
                for j, value in sorted(zip(columns, probabilities)):
                    if value > 0:
                        builder.add_next_value(row, j, float(value))
                row += 1
        reward_models = {}
        if self.cost is not None:
            row_cost = [float(self.cost[p]) for p in range(self.n) for _ in self.rows[p]]
            reward_models["cost"] = stormpy.models.SparseRewardModel(optional_state_action_reward_vector=row_cost)
        components = stormpy.models.SparseModelComponents(transition_matrix=builder.build(), state_labeling=self.labeling(), reward_models=reward_models)
        return stormpy.models.SparseMdp(components)

    def interval_dtmc(self):
        """The product as an interval DTMC (Storm resolves the intervals; for validating U)."""
        builder = stormpy.datastructures.SparseMatrixBuilder[stormpy.Interval](self.n, self.n, 0, True)
        for p in range(self.n):
            if self.choice[p] < 0:
                builder.add_next_value(p, p, stormpy.Interval(1.0, 1.0))
                continue
            columns = self.rows[p][0][0]
            b0 = self.choice_branch_offsets[self.choice[p]]
            for i, j in sorted(enumerate(columns), key=lambda x: x[1]):
                builder.add_next_value(p, j, stormpy.Interval(float(self.branch_lower[b0 + i]), float(self.branch_upper[b0 + i])))
        reward_models = {}
        if self.cost is not None:
            state_cost = [stormpy.Interval(float(c), float(c)) for c in self.cost]
            reward_models["cost"] = stormpy.models.SparseRewardModel[stormpy.Interval](optional_state_reward_vector=state_cost)
        components = stormpy.models.SparseModelComponents(transition_matrix=builder.build(), state_labeling=self.labeling(), reward_models=reward_models)
        return stormpy.models.SparseDtmc(components)

    def labeling(self):
        labeling = stormpy.models.StateLabeling(self.n)
        for label in ("init", "goal"):
            labeling.add_label(label)
        labeling.add_label_to_state("init", 0)
        for p in np.flatnonzero(self.target):
            labeling.add_label_to_state("goal", int(p))
        return labeling

    def colored_mdp(self, ties: str):
        """PAYNT colored MDP: one parameter per tie group, its options are the vertex indices."""
        mdp = self.mdp()
        parameter_space = paynt.parameter_space.parameter_space.ParameterSpace()
        parameter_of_key: dict = {}
        choice_to_parameter_options = []
        for p in range(self.n):
            k = len(self.rows[p])
            if k == 1:
                choice_to_parameter_options.append([])
                continue
            key = self.choice[p] if ties == "memory" else p
            parameter = parameter_of_key.get(key)
            if parameter is None:
                parameter = parameter_of_key[key] = parameter_space.num_parameters
                name = f"c{self.choice[p]}" if ties == "memory" else f"s{self.states[p][0]}n{self.states[p][1]}"
                parameter_space.add_parameter(name, [f"v{j}" for j in range(k)])
            for j in range(k):
                choice_to_parameter_options.append([(parameter, j)])
        coloring = payntbind.synthesis.Coloring(parameter_space.native, mdp.nondeterministic_choice_indices, choice_to_parameter_options)
        parameter_of_state = np.full(self.n, -1)
        for p in range(self.n):
            if len(self.rows[p]) > 1:
                parameter_of_state[p] = parameter_of_key[self.choice[p] if ties == "memory" else p]
        copies = np.bincount(parameter_of_state[parameter_of_state >= 0], minlength=parameter_space.num_parameters)
        return paynt.colored_mdp.ColoredMdp(mdp, parameter_space, coloring), parameter_of_state, copies


# ----------------------------------------------------------------- points inside the polytopes


class TiedChain:
    """The product with every parameter (tie group) at a point of its polytope, written as a convex combination
    of the group's extreme points with weights w; exact values and gradients by sparse LU and its adjoint."""

    def __init__(self, product: Product, parameter_of_state, nr_vertices: list[int]):
        self.n, self.target, self.cost = product.n, product.target, product.cost
        self.offsets = np.concatenate([[0], np.cumsum(nr_vertices)]).astype(np.int64)
        self.constant = int(self.offsets[-1])  # weight slot fixed to 1, for rows with a single vertex
        self.group_of_slot = np.repeat(np.arange(len(nr_vertices)), nr_vertices)
        rows, cols, vals, slots = [], [], [], []
        for p in range(self.n):
            if product.choice[p] < 0:
                continue  # target or avoid state: absorbing, value 1 or 0
            for j, (columns, probabilities) in enumerate(product.rows[p]):
                slot = self.constant if len(product.rows[p]) == 1 else int(self.offsets[parameter_of_state[p]] + j)
                for col, value in zip(columns, probabilities):
                    if value > 0:
                        rows.append(p)
                        cols.append(col)
                        vals.append(value)
                        slots.append(slot)
        self.rows, self.cols = np.array(rows), np.array(cols)
        self.vals, self.slots = np.array(vals), np.array(slots)
        self.transposed = None

    def vertex_weights(self, options):
        w = np.zeros(self.constant + 1)
        w[self.offsets[:-1] + np.asarray(options, dtype=np.int64)] = 1.0
        w[self.constant] = 1.0
        return w

    @staticmethod
    def closure(graph, start):
        """start and every state with a path into it (graph: successor x predecessor)."""
        found = start.copy()
        frontier = np.flatnonzero(start)
        while len(frontier):
            predecessors = np.unique(graph[frontier].indices)
            predecessors = predecessors[~found[predecessors]]
            found[predecessors] = True
            frontier = predecessors
        return found

    def solve(self, w):
        """Values x (reachability probability, or expected cost: infinite where the target is missed with
        positive probability), the LU of the linear system, the states it covers and their index in it."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        data = self.vals * w[self.slots]
        positive = data > 0
        graph = sp.csr_matrix((np.ones(positive.sum()), (self.cols[positive], self.rows[positive])), shape=(self.n, self.n))
        reach = self.closure(graph, self.target)
        if self.cost is None:
            maybe = reach & ~self.target
        else:
            maybe = ~self.closure(graph, ~reach) & ~self.target  # reach the target almost surely
        index = np.full(self.n, -1)
        states = np.flatnonzero(maybe)
        index[states] = np.arange(len(states))
        inside = maybe[self.rows]
        to_maybe = inside & maybe[self.cols]
        A = sp.identity(len(states), format="csc") - sp.csc_matrix((data[to_maybe], (index[self.rows[to_maybe]], index[self.cols[to_maybe]])), shape=(len(states),) * 2)
        if self.cost is None:
            to_target = inside & self.target[self.cols]
            b = np.bincount(index[self.rows[to_target]], weights=data[to_target], minlength=len(states))
            x = np.zeros(self.n)
            x[self.target] = 1.0
        else:
            b = self.cost[states]
            x = np.full(self.n, np.inf)
            x[self.target] = 0.0
        lu = spla.splu(A.tocsc()) if len(states) else None
        if lu is not None:
            x[states] = lu.solve(b)
        return x, lu, maybe, index

    def value_grad(self, w):
        x, lu, maybe, index = self.solve(w)
        grad = np.zeros_like(w)
        if lu is not None and maybe[0]:
            e = np.zeros(lu.shape[0])
            e[index[0]] = 1.0
            y = np.zeros(self.n)
            y[maybe] = lu.solve(e, trans="T")  # expected visits from the initial state
            finite = np.where(np.isfinite(x), x, 0.0)  # infinite only where the initial state never goes
            grad = np.bincount(self.slots, weights=y[self.rows] * self.vals * finite[self.cols], minlength=len(w))
        grad[self.constant] = 0.0
        return x[0], grad

    def project(self, w):
        """Every group's weights onto its simplex (sort-based projection)."""
        w = w.copy()
        for g in range(len(self.offsets) - 1):
            a, b = self.offsets[g], self.offsets[g + 1]
            v = w[a:b]
            u = np.sort(v)[::-1]
            cumulative = np.cumsum(u) - 1.0
            k = np.nonzero(u - cumulative / np.arange(1, len(u) + 1) > 0)[0][-1]
            w[a:b] = np.maximum(v - cumulative[k] / (k + 1), 0.0)
        w[self.constant] = 1.0
        return w

    def interior_search(self, w, sense: int, iterations: int) -> dict:
        """From the best vertex assignment: which groups improve when moved inside their polytope (first order),
        then projected gradient descent of sense * value over all polytope points."""
        f0, grad = self.value_grad(w)
        noise = 1e-12 * max(abs(f0), 1e-300)
        inward = 0
        for g in range(len(self.offsets) - 1):
            a, b = self.offsets[g], self.offsets[g + 1]
            chosen = a + int(np.argmax(w[a:b]))
            # moving weight from the chosen vertex to vertex k changes the value at rate grad_k - grad_chosen
            if np.any(sense * (grad[a:b] - grad[chosen]) < -noise):
                inward += 1
        best, f = w, f0
        step = None
        for _ in range(iterations):
            direction = -sense * grad
            direction[np.abs(direction) <= noise] = 0.0
            if not np.any(direction):
                break
            if step is None:
                step = 0.5 / np.abs(direction).max()
            improved = False
            while step * np.abs(direction).max() > 1e-12:
                candidate = self.project(best + step * direction)
                move = candidate - best
                if not np.any(move):
                    break
                fc, gc = self.value_grad(candidate)
                if sense * fc <= sense * f - max(1e-4 * float(np.dot(-sense * grad, move)), noise):
                    best, f, grad = candidate, fc, gc
                    step *= 2
                    improved = True
                    break
                step /= 2
            if not improved:
                break
        inside = 0
        for g in range(len(self.offsets) - 1):
            if best[self.offsets[g] : self.offsets[g + 1]].max() < 1 - 1e-9:
                inside += 1
        return {"V_exact_solve": f0, "T_interior": f, "groups_improving_inward_at_V": inward, "groups_inside": inside}


class TiedBranchAndBound:
    """The true tied value T to a relative tolerance: nature may pick any point of every polytope, the same in
    all memory copies of an original choice. A tied group (an original choice with two extreme points, used in
    several copies) gets an interval [lo, hi] of positions on its segment; a box of intervals is bounded by the
    untied value over it (every copy picks an end of its interval on its own; robust policy iteration), its
    tied projection gives an incumbent, and a box is split at the group whose copies disagree most."""

    def __init__(self, product: Product, sense: int):
        self.sense = sense
        _, state_parameter, _ = product.colored_mdp("none")  # one parameter (weight slots) per product state
        nr = [len(product.rows[p]) for p in range(product.n) if state_parameter[p] >= 0]
        self.chain = TiedChain(product, state_parameter, nr)
        offsets = self.chain.offsets
        copies: dict = {}
        for p in range(product.n):
            if state_parameter[p] >= 0:
                copies.setdefault(product.choice[p], []).append(p)
        tied = [members for members in copies.values() if len(members) > 1]
        for members in tied:
            assert len(product.rows[members[0]]) == 2, "exact tied value: only segments (two extreme points) are supported"
        self.G = len(tied)
        member_state = [p for members in tied for p in members]
        self.member_group = np.repeat(np.arange(self.G), [len(m) for m in tied])
        self.member_slot = offsets[state_parameter[member_state]]  # slot of vertex 0; vertex 1 is the next slot
        self.member_state = np.array(member_state, dtype=np.int64)
        is_member = np.zeros(len(offsets) - 1, dtype=bool)
        is_member[state_parameter[member_state]] = True
        # free parameters: product states whose choice has a single copy; nature picks any extreme point there
        self.free = np.flatnonzero(~is_member)
        self.free_start, self.free_count = offsets[self.free], np.diff(offsets)[self.free]
        self.nodes = 0

    def weights(self, member_position, free_vertex):
        w = np.zeros(self.chain.constant + 1)
        w[self.member_slot] = 1.0 - member_position
        w[self.member_slot + 1] = member_position
        w[self.free_start + free_vertex] = 1.0
        w[self.chain.constant] = 1.0
        return w

    def relax(self, lo, hi, member_position, free_vertex):
        """Untied value over the box, nature's per-copy choice, the tied projection and the disagreement per group."""
        sense = self.sense
        for _ in range(10000):
            x, lu, maybe, index = self.chain.solve(self.weights(member_position, free_vertex))
            x = np.where(np.isfinite(x), x, 0.0)  # infinite cost only where the initial state never goes
            scores = np.bincount(self.chain.slots, weights=self.chain.vals * x[self.chain.cols], minlength=self.chain.constant + 1)
            # noise thresholds relative to each row's own value: values span many orders of magnitude (avoid-large: 1e-11)
            scale = 1e-12 * np.maximum(np.abs(scores[self.member_slot]), np.abs(scores[self.member_slot + 1])) + 1e-300
            d = sense * (scores[self.member_slot + 1] - scores[self.member_slot])  # rate of sense*value along the segment
            want = np.where(d > scale, lo[self.member_group], np.where(d < -scale, hi[self.member_group], member_position))
            change_members = np.abs(want - member_position) * np.abs(d) > scale
            new_free = free_vertex.copy()
            for i, (a, k) in enumerate(zip(self.free_start, self.free_count)):
                if k > 1:
                    s_ = sense * scores[a : a + k]
                    j = int(np.argmin(s_))
                    if s_[j] < s_[free_vertex[i]] - 1e-12 * abs(s_[free_vertex[i]]):
                        new_free[i] = j
            if not change_members.any() and np.array_equal(new_free, free_vertex):
                break
            member_position = np.where(change_members, want, member_position)
            free_vertex = new_free
        y = np.zeros(self.chain.n)
        if lu is not None and maybe[0]:
            e = np.zeros(lu.shape[0])
            e[index[0]] = 1.0
            y[maybe] = lu.solve(e, trans="T")
        weight = y[self.member_state] * np.abs(d)
        at_lo = member_position <= lo[self.member_group]
        w_lo = np.bincount(self.member_group, weights=weight * at_lo, minlength=self.G)
        w_hi = np.bincount(self.member_group, weights=weight * ~at_lo, minlength=self.G)
        projection = np.where(w_lo >= w_hi, lo, hi)
        disagreement = np.minimum(w_lo, w_hi) * (hi - lo)
        disagreement[np.minimum(w_lo, w_hi) <= 1e-13 * max(abs(x[0]), 1e-300)] = 0.0
        return x[0], member_position, free_vertex, projection, disagreement

    def tied_value(self, position, free_vertex):
        return self.chain.solve(self.weights(position[self.member_group], free_vertex))[0][0]

    def run(self, incumbent: float, time_limit: float, eps: float = 1e-6) -> dict:
        import heapq

        sense = self.sense
        start = time.time()
        lo, hi = np.zeros(self.G), np.ones(self.G)
        best = sense * np.inf if incumbent is None else incumbent
        member_position = np.zeros(len(self.member_group))
        free_vertex = np.zeros(len(self.free), dtype=np.int64)

        def tolerance():
            return eps * max(abs(best), 1e-300) if np.isfinite(best) else 0.0

        heap, counter = [], 0
        bound, member_position, free_vertex, projection, disagreement = self.relax(lo, hi, member_position, free_vertex)
        root_bound = bound
        heapq.heappush(heap, (sense * bound, counter, lo, hi, member_position, free_vertex, projection, disagreement))
        exhausted = True
        while heap:
            key, _, lo, hi, member_position, free_vertex, projection, disagreement = heapq.heappop(heap)
            if key >= sense * best - tolerance():
                heap = []
                break
            if time.time() - start > time_limit:
                heapq.heappush(heap, (key, -1, lo, hi, member_position, free_vertex, projection, disagreement))
                exhausted = False
                break
            self.nodes += 1
            value = self.tied_value(projection, free_vertex)
            if sense * value < sense * best:
                best = value
            if not disagreement.any():
                continue  # the bound is attained by the tied projection
            g = int(np.argmax(disagreement))
            mid = (lo[g] + hi[g]) / 2
            for child_lo, child_hi in ((lo[g], mid), (mid, hi[g])):
                clo, chi = lo.copy(), hi.copy()
                clo[g], chi[g] = child_lo, child_hi
                cbound, cpos, cfree, cproj, cdis = self.relax(clo, chi, np.clip(member_position, clo[self.member_group], chi[self.member_group]), free_vertex)
                if sense * cbound < sense * best - tolerance():
                    counter += 1
                    heapq.heappush(heap, (sense * cbound, counter, clo, chi, cpos, cfree, cproj, cdis))
        lower = min(best, heap[0][0] * sense) if (heap and sense > 0) else (max(best, heap[0][0] * sense) if heap else best)
        return {"T_best": best, "T_bound": lower, "T_exact": exhausted, "T_nodes": self.nodes, "T_time": time.time() - start, "T_root_bound": root_bound, "T_groups": self.G}


# ----------------------------------------------------------------- verification


# "vi": native optimistic value iteration, accurate to the precision; "pi": policy iteration with gmm++ for the
# linear equations (its own default precision, stormpy cannot set it), for large expected costs
SOLVER = "vi"


def configure_solver():
    """Make PAYNT's model checking accurate to the requested precision. Its defaults lost up to 1e-4 relative:
      - the topological linear equation solver's gmm++/Eigen sub-solver keeps its own default precision
        (stormpy 2.0 exposes only the native and min-max settings), and the native default method (Jacobi)
        stops on a small step, not a small error (1e-7 off near 1 on evade-large): use native optimistic VI;
      - min-max solving is capped at 10000 iterations unless Property.sound is set (main sets it).
    SOLVER = "pi" instead: value iteration needs minutes per check once expected costs are large (collision,
    1e6: over 120 s, any value iteration variant), policy iteration with gmm++ 1.3 s, matching the exact value.
    Every SynthesisTask re-initializes the environment, so call this after creating one."""
    solver = paynt.specification.property.Property.environment.solver()
    if SOLVER == "pi":
        solver.set_linear_equation_solver_type(stormpy.environments.EquationSolverType.GMMXX)
        solver.min_max().method = stormpy.environments.MinMaxMethod.POLICY_ITERATION
    else:
        solver.set_linear_equation_solver_type(stormpy.environments.EquationSolverType.NATIVE)
        solver.native().method = stormpy.environments.NativeLinearEquationSolverMethod.OPTIMISTIC_VALUE_ITERATION
    solver.native().relative_termination_criterion = True
    solver.min_max().relative_termination_criterion = True


def relative_comparison(a, b) -> bool:
    """Replacement for Property.above_model_checking_precision, which compares |a - b| with the precision
    absolutely: with values below the precision (avoid-large: 1e-11) no assignment ever counts as an
    improvement and AR stops after the first one."""
    if isinstance(a, stormpy.Rational):
        return True
    return abs(a - b) > paynt.specification.property.Property.model_checking_precision * max(abs(a), abs(b), 1e-300)


def task_for(prop: str):
    task = paynt.task.SynthesisTask(stormpy.parse_properties(prop))
    configure_solver()
    return task


def check(model, prop: str, mode=None):
    """Value at the initial state, in PAYNT's model checking environment (same method and precision)."""
    property_ = stormpy.parse_properties(prop)[0]
    environment = paynt.specification.property.Property.environment
    result = stormpy.model_checking(model, property_, only_initial_states=True, uncertainty_resolution_mode=mode, environment=environment)
    return result.get_value(model.initial_states[0])


def verify(product: Product, ties: str, direction: str, args) -> dict:
    colored_mdp, parameter_of_state, copies = product.colored_mdp(ties)
    space = colored_mdp.parameter_space
    prop = product.prop(direction)
    U = check(colored_mdp.underlying_mdp, prop)
    record = {
        "ties": ties,
        "direction": direction,
        "parameters": space.num_parameters,
        "copies_per_parameter": int(copies.max()) if len(copies) else 0,
        "tied_parameters": int(np.sum(copies > 1)),
        "family_log10": float(sum(np.log10(len(space.parameter_options(h))) for h in range(space.num_parameters))),
        "U": U,
    }
    if not np.isfinite(U):
        # infinite expected cost: the target is missed with positive probability whatever nature does (the
        # support does not depend on nature, see Product)
        record.update({"V": U, "finished": True, "paynt_time": 0.0, "paynt_iterations": 0})
        return record
    t0 = time.time()
    paynt.utils.timer.GlobalTimer.start(None)
    task = task_for(prop)
    synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(colored_mdp, task)
    assignment = synthesizer.synthesize(timeout=args.timeout, keep_optimum=True, print_stats=False)
    record["paynt_time"] = time.time() - t0
    record["paynt_iterations"] = synthesizer.stat.iterations_mdp if synthesizer.stat is not None else None
    record["V"] = synthesizer.best_assignment_value
    # PAYNT times itself with perf_counter, which drifts from time.time on WSL: ask it whether the whole family was explored
    record["finished"] = synthesizer.explored is not None and synthesizer.explored >= space.size
    if assignment is not None:
        dtmc = colored_mdp.build_assignment(assignment)
        record["V_check"] = check(dtmc.model, product.prop(""))
    if assignment is not None:
        chain = TiedChain(product, parameter_of_state, [len(space.parameter_options(h)) for h in range(space.num_parameters)])
        start = chain.vertex_weights([assignment.parameter_options(h)[0] for h in range(space.num_parameters)])
        record["V_exact_solve"] = chain.solve(start)[0][0]
        if args.interior:
            record.update(chain.interior_search(start, 1 if direction == "min" else -1, args.interior))
    if args.exact and ties == "memory":
        # exact values only: PAYNT's own value carries Storm's solver error (gmm++ under --solver pi: 4e-5 on collision)
        known = [v for v in (record.get("V_exact_solve"), record.get("T_interior")) if v is not None]
        incumbent = (min(known) if direction == "min" else max(known)) if known else None
        record.update(TiedBranchAndBound(product, 1 if direction == "min" else -1).run(incumbent, args.exact))
    if 0 < space.size <= args.brute_force:
        # every tied vertex assignment, one Markov chain each
        values = [check(colored_mdp.build_assignment(space.construct_assignment(c)).model, product.prop("")) for c in space.all_combinations()]
        record["V_brute_force"] = min(values) if direction == "min" else max(values)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="PRISM file of an interval POMDP")
    parser.add_argument("prop", help='P=? [F b], P=? [a U b] or R=? [F b] (labels; one reward model); the FSC is verified for this objective')
    parser.add_argument("--fsc", default="random:2:0", help="random:NODES:SEED, paynt:NODES:TIMEOUT or json:PATH")
    parser.add_argument("--ties", default="memory,none", help="comma-separated: memory (copies share the vertex), none")
    parser.add_argument(
        "--nature", choices=["robust", "min", "max", "both"], default="robust", help="robust = pessimistic: min for P, max for R (expected cost)"
    )
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="override a constant of the model")
    parser.add_argument("--timeout", type=int, default=300, help="PAYNT time limit per run, seconds")
    parser.add_argument("--precision", type=float, default=1e-10, help="model checking precision (PAYNT compares values with it, absolutely)")
    parser.add_argument("--interior", type=int, default=200, metavar="ITERATIONS", help="search inside the polytopes from PAYNT's assignment (0 = off)")
    parser.add_argument("--exact", type=float, default=0, metavar="SECONDS", help="also compute the true tied value T by branch and bound (0 = off)")
    parser.add_argument("--brute-force", type=int, default=4096, help="also enumerate families up to this size")
    parser.add_argument(
        "--solver", choices=["auto", "vi", "pi"], default="auto", help="Storm/PAYNT: value iteration or policy iteration (auto: pi for expected costs)"
    )
    parser.add_argument("--no-storm-check", action="store_true", help="skip Storm's robust check of the product iDTMC (robust value iteration)")
    parser.add_argument("--json", help="append a JSON line with the results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("paynt", "stormpy"):
        logging.getLogger(name).setLevel(logging.WARNING)

    paynt.specification.property.Property.model_checking_precision = args.precision
    paynt.specification.property.Property.sound = True  # no cap on min-max iterations
    paynt.specification.property.Property.above_model_checking_precision = staticmethod(relative_comparison)
    paynt.specification.property.Property.initialize()

    t0 = time.time()
    model, formula = load(args.model, args.prop, args.set)
    target, avoid = target_and_avoid(model, formula)
    table = find_ipomdp_vertices.compute_vertex_table(model)
    cost = choice_costs(model, formula, table)
    global SOLVER
    SOLVER = args.solver if args.solver != "auto" else ("vi" if cost is None else "pi")
    configure_solver()
    counts, freq = np.unique(table.nr_vertex_choices, return_counts=True)
    logger.info(
        f"{os.path.basename(args.model)} {args.set}: {model.nr_states} states, {model.nr_observations} observations, "
        f"vertices per choice: {dict(zip(counts.tolist(), freq.tolist()))} ({time.time() - t0:.1f}s)"
    )
    record = {"model": args.model, "set": args.set, "prop": args.prop, "fsc": args.fsc}

    kind, *rest = args.fsc.split(":", 1)
    if kind == "random":
        nodes, seed = (int(x) for x in rest[0].split(":"))
        action, update = random_fsc(table, nodes, seed)
    elif kind == "paynt":
        nodes, timeout = (int(x) for x in rest[0].split(":"))
        action, update, nominal = paynt_fsc(model, table, target, avoid, nodes, timeout, cost)
        record["fsc_nominal_value"] = nominal
        logger.info(f"PAYNT FSC with {nodes} nodes, value {nominal} on the nominal POMDP (vertex centroids)")
    else:
        action, update = json_fsc(model, rest[0])
    product = Product(table, target, avoid, action, update, cost)
    used_nodes = len({n for _, n in product.states})
    logger.info(f"product with the FSC: {product.n} states, {used_nodes} memory nodes used, {sum(len(r) > 1 for r in product.rows)} states where nature chooses")

    robust = "min" if cost is None else "max"
    directions = {"robust": [robust], "min": ["min"], "max": ["max"], "both": ["min", "max"]}[args.nature]
    mode = {"min": stormpy.UncertaintyResolutionMode.MINIMIZE, "max": stormpy.UncertaintyResolutionMode.MAXIMIZE}
    storm = {d: None if args.no_storm_check else check(product.interval_dtmc(), product.prop(""), mode[d]) for d in directions}
    record["storm_robust"] = storm
    record["results"] = []
    for ties in args.ties.split(","):
        for direction in directions:
            r = verify(product, ties, direction, args)
            record["results"].append(r)
            extra = f", brute force {r['V_brute_force']:.10g}" if "V_brute_force" in r else ""
            if "T_best" in r:
                extra += f"; T in [{r['T_bound']:.13g}, {r['T_best']:.13g}] ({'exact' if r['T_exact'] else 'time limit'}, {r['T_nodes']} nodes, {r['T_groups']} tied groups, {r['T_time']:.0f}s)"
            if "T_interior" in r:
                extra += (
                    f"; exact solve at V {r['V_exact_solve']:.13g}, inside the polytopes {r['T_interior']:.13g} ({r['groups_inside']} groups inside, "
                    f"{r['groups_improving_inward_at_V']} improve when moved inward at V)"
                )
            logger.info(
                f"ties={ties:6} nature {direction}: {r['parameters']} parameters ({r['tied_parameters']} shared by several memory copies), "
                f"family 1e{r['family_log10']:.1f}; U={r['U']:.10g} (Storm robust {'skipped' if storm[direction] is None else format(storm[direction], '.10g')}), "
                f"PAYNT V={'none found' if r['V'] is None else format(r['V'], '.10g')} in {r['paynt_time']:.1f}s{'' if r['finished'] else ' (timeout)'}{extra}"
            )
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
