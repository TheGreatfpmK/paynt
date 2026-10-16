#!/usr/bin/env python3
"""Optimal color consistent policies over a finite number of steps (reachability and conditional reachability): an exact search over reach vectors
with two bounds.

    python finite-horizon-dp.py MODEL PROPERTIES [HORIZON] [--width 32] [--max-nodes 2000000] [--timeout 600] [--verify] [--show-policy]

MODEL and PROPERTIES are loaded as by paynt (for a POMDP the parameters are the observations). The property is a probability P[F goal] or a conditional
probability P[F goal || F condition], to be maximized or minimized (a threshold P>b is answered by the optimum). With a HORIZON the model is unfolded here:
the policy chooses a parameter assignment in every one of the HORIZON steps. Without it the model has to be unfolded already (acyclic, its observations
include the time, like the models of monitor verification), and its layers are the steps; a conditional property needs such a model. The model and the
coloring are read with the code of finite-horizon-time-dependent.py; the search below is independent of it (that script is a heuristic, this one is exact).


THE PROBLEM. A colored MDP has states S, for every state choices with distributions P(c, .) over S, and terminal states T in which a run ends and collects
the payoff of the state (for P[F goal]: 1 in the goal, 0 in the absorbing states that are not goal states). The coloring groups the colored states into
components g (e.g. the observations of a POMDP): every state s of the component g has one choice a(s, o) for every option o of g. A state without colors
with several choices is a component of its own (its choice is free), a state with a single choice executes it. A policy makes a *stage decision* d_t in
every step t = 0..H-1: an option for every component. The steps come in two kinds:
  - a model unfolded here (a horizon H is given): every state chooses in every step, and the components of the steps are separate (the parameter p is a
    separate parameter p@t in every step: a time-dependent color consistent policy);
  - a model unfolded already (acyclic): the step of a state is its layer, the length of the longest path to it from the initial state, so every transition
    leads to a later step. A state chooses in its own step only, all states of a component are in the same step, and H is the number of layers.
The task is a policy that maximizes (or minimizes) the expected payoff J(pi). Deciding whether J(pi) >= lambda for some pi is NP-hard (it contains
memoryless observation-based POMDP policies of a bounded horizon).

REACH VECTORS. A policy moves the probability mass step by step: rho_0 = 1_{s0}; in the step t the mass of the states of the step moves along the choices
that d_t selects (the matrix P_{d_t}), the mass of the states of later steps waits, and the mass that enters T is collected: c_{t+1} = c_t + <mass that
enters T, payoff>; J(pi) = c_H. The reach vector rho_t is an unnormalized belief: the distribution of the state in the step t, knowing nothing but the
policy. Four facts make a dynamic program over reach vectors exact and efficient:
  (1) Separation. For every step t, J(pi) = c_t + <rho_t, alpha_t>, where alpha_t(s) is the expected payoff of the suffix (d_t, ..., d_{H-1}) from s.
  (2) Prefix independence. The components of different steps are different, so any prefix d_0..d_{t-1} can be followed by any suffix. Hence
      OPT = max { c + W_t(rho) : (rho, c) reached in the step t by a prefix }, where W_t(rho) = max over the suffixes of <rho, alpha_t>: two prefixes that
      reach the same rho can be followed by the same suffixes. (A stationary policy of a model unfolded here, one assignment for all steps, does not have
      this property: the suffix would depend on the prefix.)
  (3) Separability of a stage. For a fixed continuation v (a vector over S), <P_d^T rho, v> is a sum over the components of terms that depend on the
      option of a single component, plus a constant. The number of stage decisions is the product of the numbers of options, but for a given v the best
      one is found per component, and the decisions can be enumerated in the order of decreasing value. Only the components that rho touches matter, and
      among their options only those that move the mass of rho differently (two options that differ in states outside the support of rho are one edge).
  (4) Homogeneity. W_t is the maximum of finitely many linear functions (one per suffix), so it is convex and positively homogeneous: W_t(k rho) =
      k W_t(rho). So the nodes of the search are the *normalized* reach vectors b = rho / |rho| (beliefs), and an edge d from b collects the reward
      r = <mass that enters T, payoff> and leaves the alive mass k b' (scale k, normalized child b'):
          W_t(b) = max_d [ r(b, d) + k(b, d) W_{t+1}(b') ],      W_H = 0,      OPT = c_0 + W_0(1_{s0}).
      This is a deterministic finite-horizon DP over the beliefs that the policies reach. Prefixes that reach proportional reach vectors (e.g. the observation
      words of a monitor that lead to the same belief about the hidden state, with different probabilities) share one node.

BOUNDS. W_t is bounded on both sides by functions of b that are cheap to evaluate:
  - Upper bound U_t(b) = <b, V^_t>, where V^_t(s) is the optimum of the MDP that ignores the colors (every state chooses for itself; every color consistent
    suffix is one of its policies), computed once by backward induction. It is consistent: U_t(b) >= r(b, d) + k(b, d) U_{t+1}(b') for every edge d.
  - Lower bound L_t(b) = max_{alpha in Gamma_t} <b, alpha>, where Gamma_t is a set of value vectors of concrete suffix policies (alpha vectors). Every
    vector is the value of a policy that can be executed, so L_t(b) is achieved by a suffix.
  - The states whose value does not depend on the policy (V^_t(s) equals the value V_t(s) of the worst policy that ignores the colors; e.g. the states that
    cannot reach the goal anymore) are *folded*: their mass is collected with its value at once and removed from b. This loses nothing and makes more
    beliefs equal. If the initial state is folded, every policy is optimal.

CONDITIONAL PROBABILITIES. Pr(F goal | F condition) = N / D with N = Pr(F goal) and D = Pr(F condition); the goal states have to be condition states, and
the condition states have to be absorbing. The ratio has no Bellman equation, but its threshold question does (Theorem 1 of Ceska, Junges, van der Maas,
Macak, Quatmann: arXiv 2605.11897): max_pi N/D > lambda iff max_pi N - lambda D > 0, and N - lambda D is the expected payoff with the payoff 1 - lambda
in the goal and -lambda in the other condition states. Dinkelbach's method solves a sequence of such problems exactly: start with a policy pi_0 with D > 0
(the best policy of phase 1 for Pr(F condition)), and with lambda_i = N(pi_i) / D(pi_i) take pi_{i+1} = argmax_pi N - lambda_i D. If the maximum is not
positive, lambda_i is the optimum, otherwise lambda_{i+1} > lambda_i. There are finitely many policies, so this ends; it converges superlinearly and takes a
few iterations. The payoffs of the iteration i are divided by D(pi_i), so that the tolerance of the search is relative to the evidence of the policy.

ALGORITHM (for one expected payoff).
  Phase 1, the lower bound (point-based value iteration). A beam search of the belief graph (in every step the W beliefs with the largest upper bound
  c + K U(b) of a prefix that reaches K b, among the children of the previous ones) gives W *anchors* in every step. A backward sweep computes, for every
  anchor of the step t, the best stage decision and the best suffix among the vectors of Gamma_{t+1}, i.e. the decision and the vector that maximize
  r(b, d) + <P_d^T b, alpha'> (a point-based backup, done per component by (3)); the resulting value vectors form Gamma_t. The vector of the initial belief
  is the value of a policy, the *incumbent*. The reach vectors of the incumbent are added to the anchors and the sweep is repeated as long as the incumbent
  improves.
  Phase 2, optimality (depth-first branch and bound with a table of bounds). search(t, b, beta) returns bounds lo <= W_t(b) <= hi such that either
  lo = hi (exact) or hi <= beta (b cannot beat the threshold beta); the bounds are kept in a table under the key (t, b) and improved when b is reached
  again. A call
     (a) starts from lo = L_t(b) (achieved by the suffix of a vector) and hi = U_t(b), and returns at once if they meet or if hi <= beta;
     (b) otherwise enumerates the edges by (3) in the order of decreasing r + k U_{t+1}(b'), and stops when that does not beat max(beta, lo): no later edge
         can matter. An edge is searched with the threshold (max(beta, lo) - r) / k for its child, and r + k lo' improves lo.
  The lower bound of a belief acts as a local incumbent, the threshold carries the global one down, and the table shares the work between the prefixes
  that reach the same belief (also in its proportional forms). The root is searched with a descending sequence of thresholds: the middle of the gap between
  the incumbent and the upper bound proved so far (6 times), then the incumbent - c_0 itself. A search that fails low proves a smaller upper bound, and the
  table keeps its work for the next one; a search whose root's bounds meet has found the optimum.

THEOREM. The search returns the optimum: if the root's bounds meet, lo = hi = W_0 and the policy is read from the table (the edges and vectors that
achieve the lower bounds); if hi <= beta, the incumbent is optimal. Proof sketch: by (2) and (4), W satisfies the recursion above. Every returned lo is the
value of a suffix, and hi is an upper bound because the edges that were not searched have bounds r + k U_{t+1}(b') <= max(beta, lo), and the searched ones
return either their exact value or a bound below their threshold, i.e. r + k hi' <= max(beta, lo). So hi <= max(beta, lo): either lo > beta and then
lo = hi, or hi <= beta. A search that is stopped (by --max-nodes or --timeout) unwinds with valid bounds, so it returns a policy and a certified bracket of
the optimum, [incumbent, smallest upper bound proved by the descent].

COMPLEXITY. Phase 1 costs, per sweep, O(H * (anchors * suffixes * (colored choices) + choices * suffixes * |S|)) (matrix products). Phase 2 visits at most
the distinct beliefs whose upper bound beats the threshold that reaches them, each in time linear in its support times the options of the components it
touches, plus the edges it generates. This is exponential in the worst case (as it must be), and small when the beliefs are few (e.g. the hidden state of
a monitored Markov chain) or when the relaxation is tight on the reachable beliefs. The upper bound is loose when partial observability costs much (the
colors-ignored optimum is far above the optimum), and the search is wide when many components are reachable in one step.

RELATION TO POMDP PLANNING. The belief graph is the belief MDP of a POMDP with *open-loop* beliefs: the policy is memoryless (time-dependent and
observation-based), so the beliefs do not branch on observations, and the "action" of a step is a whole parameter assignment (exponentially many, but
separable by (3)). The lower bound is that of point-based value iteration (PBVI, HSVI), the upper bound is the MDP (QMDP-like) bound, and phase 2 is a
depth-first heuristic search over the beliefs with a transposition table (the descent of the thresholds is that of MTD(f) and iterative deepening).

EXPERIENCE. Reachability, models unfolded here (the 21 reachability models of the repository, H = 4, 8, 16, 120 s each): 43 of the 63 instances have a
value that does not depend on the policy (mostly 0) and are proved at once. Of the other 20, 17 are proved optimal, all within 1.1 s: e.g. refuel-08 H=16
(0.268724 in 0.9 s; the plain backward passes of finite-horizon-time-dependent.py reach 0.2018, AR does not finish within 120 s), liveness H=16 (11535
states, 0.8494 in 0.6 s), grid H=16 (0.8494 in 0.4 s); in all of them phase 1 already found the optimum and phase 2 proved it. Not finished: refuel-06
H=16 (the optimum is in [0.4114, 0.4556]) and crypt-4 H=8, 16 ([0.3333, 1]: the optimum without colors is 1, so the upper bound prunes nothing and the steps
are wide). Conditional probabilities, models unfolded already (the six luko models of monitor verification): all optima are proved and equal the known
ones (info.json), and Storm confirms the policies: airport models 01, 02, 03, 06 in 1.6 - 3.7 s (2634 - 7813 beliefs over 6 - 9 Dinkelbach iterations,
which start from the policy that makes the condition most likely), icy 05 in 1.0 s (820 beliefs), refuel 04 in 72 s (225k of its 707k beliefs, 2
iterations). The exhaustive belief enumeration of the previous script (--dp, one component per layer) is faster on these models (0.05 - 0.1 s and 19 s):
the cost here is the per-belief work of the search in Python. On 5637 random colored MDPs (8536 problems, 17072 runs: unfolded here, layered with
reachability and with conditional probabilities, maximum and minimum, with and without the beam) the result agrees with a brute force over all color
consistent policies.
"""

from __future__ import annotations

import dataclasses
import heapq
import importlib.util
import json
import logging
import os
import sys
import time
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import rich_click as click
import scipy.sparse


def _load_model_layer() -> Any:
    """finite-horizon-time-dependent.py, as a module (it reads the colored MDP into arrays and checks its coloring; its file name is not a module name)."""
    name = "finite_horizon_time_dependent"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), "finite-horizon-time-dependent.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look the defining module up
    spec.loader.exec_module(module)
    return module


fh = _load_model_layer()
UnsupportedProblemError = fh.UnsupportedProblemError

import payntbind  # noqa: E402
import stormpy  # noqa: E402

import paynt.cli  # noqa: E402
import paynt.model.model  # noqa: E402
import paynt.parser.sketch  # noqa: E402
import paynt.specification.property  # noqa: E402

logger = logging.getLogger("finite-horizon-dp")

# relative tolerance of the comparisons of values: a bound must beat the incumbent by more than this to count
TOLERANCE = 1e-9
# beliefs that agree in this many decimal digits are the same node
KEY_DIGITS = 12
# a policy whose probability of the condition is not above this does not have a conditional probability
DEGENERATE = 1e-300
# phase 2 halves the gap between the incumbent and the upper bound this many times before it tries the incumbent as the threshold
DESCENT = 6


def tolerance(value: float) -> float:
    return TOLERANCE * max(1.0, abs(value))


# ----------------------------------------------------------------------------------------------------
# the problem
# ----------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Group:
    """A set of states that take their choices together: the states of a component of the coloring, or a single state without colors that has several choices.

    choices[i, o] is the choice that states[i] executes under the option o.
    """

    states: np.ndarray
    choices: np.ndarray
    component: int  # the component of the coloring, or -1 for a state without colors

    @property
    def num_options(self) -> int:
        return self.choices.shape[1]


def absorbing_states(matrix: scipy.sparse.csr_matrix, row_groups: np.ndarray) -> np.ndarray:
    """The states whose every choice is a self-loop."""
    num_states = len(row_groups) - 1
    choice_state = np.repeat(np.arange(num_states), np.diff(row_groups))
    starts = matrix.indptr[:-1]
    loop = (np.diff(matrix.indptr) == 1) & (matrix.indices[starts] == choice_state) & (matrix.data[starts] >= 1 - 1e-12)
    return np.add.reduceat(loop.astype(np.int64), row_groups[:-1]) == np.diff(row_groups)


def layers(matrix: scipy.sparse.csr_matrix, row_groups: np.ndarray, initial: int, terminal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For a model that is unfolded already: the step of every state, which is the length of the longest path to it from the initial state (-1 for the
    terminal states and the states that cannot be reached), and the states that can be reached (also the terminal ones)."""
    num_states = len(row_groups) - 1
    source = np.repeat(np.repeat(np.arange(num_states), np.diff(row_groups)), np.diff(matrix.indptr))
    inner = ~terminal[source]
    graph = scipy.sparse.csr_matrix((np.ones(int(inner.sum())), (source[inner], matrix.indices[inner])), shape=(num_states, num_states))
    reached = np.zeros(num_states, dtype=bool)
    reached[initial] = True
    frontier = np.zeros(0, dtype=np.int64) if terminal[initial] else np.array([initial])
    while len(frontier):
        successors = np.unique(graph[frontier].indices)
        frontier = successors[~reached[successors]]
        reached[frontier] = True
        frontier = frontier[~terminal[frontier]]
    choosing = reached & ~terminal
    # the longest paths: take the states without predecessors again and again
    edges = graph.tocoo()
    inside = choosing[edges.row] & choosing[edges.col]
    indegree = np.bincount(edges.col[inside], minlength=num_states)
    step = np.full(num_states, -1, dtype=np.int64)
    frontier, depth = np.flatnonzero(choosing & (indegree == 0)), 0
    while len(frontier):
        step[frontier] = depth
        successors = graph[frontier].indices
        successors, counts = np.unique(successors[choosing[successors]], return_counts=True)
        indegree[successors] -= counts
        frontier, depth = successors[indegree[successors] == 0], depth + 1
    if np.any(choosing & (step < 0)):
        raise UnsupportedProblemError("the model is not acyclic (apart from the self-loops of absorbing states), so it is not unfolded: give a HORIZON")
    return step, reached


@dataclasses.dataclass
class Problem:
    """A colored MDP with a payoff in its terminal states, as arrays (see the top of this file).

    step is None for a model that is unfolded here (every state that is not terminal chooses in every step), and for a model that is unfolded already the
    step in which each state chooses (-1: never). Besides the list of groups, the states of all the groups are stored one after the other (the group i has
    the positions start[i]..start[i+1]-1), with their choices in a table that is padded with -1 for the groups with fewer options; this lets the steps that
    concern all the groups run without a loop.
    """

    num_steps: int
    maximize: bool
    num_states: int
    initial: int
    goal: np.ndarray
    condition: np.ndarray | None  # the condition states of a conditional probability, None for a probability
    terminal: np.ndarray  # the states in which a run ends: the goal, the condition and the absorbing states
    payoff: np.ndarray  # what a run collects when it enters a terminal state
    step: np.ndarray | None
    matrix: scipy.sparse.csr_matrix  # choices x states
    row_groups: np.ndarray  # the choices of the state s are row_groups[s]..row_groups[s+1]-1
    groups: list[Group]
    group_step: np.ndarray  # the step of every group (-1: every step)
    start: np.ndarray  # the positions of the states of the groups in the arrays below
    grouped_states: np.ndarray
    grouped_group: np.ndarray
    grouped_choices: np.ndarray  # (grouped states) x (most options)
    fixed_states: np.ndarray  # the states that choose and have a single choice without colors
    position: np.ndarray  # the position of a state in the arrays of the grouped states (-1: not in a group)
    group_of: np.ndarray  # the group of each state (-1: not in a group)
    colors: Any  # the ColorStructure, which translates the options of components into parameter assignments
    # all the colored states, also the terminal ones, with their component and their choices under each option (padded): a policy executes the option of
    # the component in every colored state, which makes it color consistent also in the states where the choice does not matter
    colored_states: np.ndarray
    colored_component: np.ndarray
    colored_choices: np.ndarray
    group_of_component: np.ndarray  # the group of every component, -1 if it has no state that chooses

    @property
    def num_components(self) -> int:
        return len(self.group_of_component)

    @property
    def layered(self) -> bool:
        """Whether the model is unfolded already (its layers are the steps)."""
        return self.step is not None

    @property
    def sign(self) -> float:
        return 1.0 if self.maximize else -1.0

    def better(self, value: float, than: float) -> bool:
        """Whether the value beats the other one by more than the tolerance (in the direction of the objective)."""
        return self.sign * (value - than) > tolerance(than)

    def acting(self, step: int) -> np.ndarray:
        """The mask of the states that choose in the step."""
        return ~self.terminal if self.step is None else self.step == step

    def acts(self, step: int, states: np.ndarray) -> np.ndarray:
        """Whether each of the states chooses in the step (or waits for a later one)."""
        return ~self.terminal[states] if self.step is None else self.step[states] == step

    def with_payoff(self, payoff: np.ndarray, maximize: bool | None = None) -> Problem:
        return dataclasses.replace(self, payoff=payoff, maximize=self.maximize if maximize is None else maximize)

    @classmethod
    def from_colored_mdp(cls, colored_mdp: Any, task: Any, horizon: int | None) -> Problem:
        """:param horizon: the number of steps of a model that is unfolded here, None for a model that is unfolded already"""
        fh.check_supported_colored_mdp(colored_mdp)
        conditional = fh.ConditionalObjective.is_conditional(task)
        if conditional and horizon is not None:
            raise UnsupportedProblemError("a conditional property is only supported for a model that is unfolded already (acyclic), without a HORIZON")
        objective = fh.ConditionalObjective.from_task(task) if conditional else fh.Objective.from_task(task)
        if objective.is_reward:
            raise UnsupportedProblemError(f"only probabilities P[F goal] and P[F goal || F condition] are supported for now, not {objective.prop}")
        underlying = colored_mdp.underlying_mdp
        mdp = fh.MdpArrays.from_mdp(underlying, objective)
        space = colored_mdp.parameter_space
        colors = fh.ColorStructure.analyze(mdp, colored_mdp.coloring, np.array([len(labels) for labels in space.parameter_to_option_labels], dtype=np.int64))
        matrix, n, goal = mdp.matrix.tocsr(), mdp.num_states, mdp.target
        condition = fh.label_mask(underlying, objective.condition_label, "condition states") if conditional else None
        absorbing = absorbing_states(matrix, mdp.row_groups)
        terminal = goal | absorbing | (condition if condition is not None else False)
        initial = int(mdp.initial_state)

        if horizon is None:
            step, reached = layers(matrix, mdp.row_groups, initial, terminal)
            if condition is not None and np.any(goal & reached & ~condition):
                raise UnsupportedProblemError("the goal states have to be condition states")
            if condition is not None and np.any(condition & reached & ~absorbing):
                raise UnsupportedProblemError("the condition states have to be absorbing")
            num_steps, choosing = int(step.max()) + 1, step >= 0
        else:
            step, num_steps, choosing = None, horizon, ~terminal
        num_choices = np.diff(mdp.row_groups)
        width = max([int(colors.comp_size.max()) if colors.num_components else 1, int(num_choices.max())])

        # the components: sort the colored choices by component, state and option; the choices of a state are then consecutive (one for each option)
        component = colors.choice_component
        option = colors.choice_joint - colors.comp_offset[component]
        state = mdp.choice_state[colors.colored_choices]
        order = np.lexsort((option, state, component))
        component, state, option, choice = component[order], state[order], option[order], colors.colored_choices[order]
        first = np.ones(len(state), dtype=bool)
        first[1:] = state[1:] != state[:-1]
        colored_states, colored_component = state[first], component[first]
        colored_choices = np.full((len(colored_states), width), -1, dtype=np.int64)
        colored_choices[np.cumsum(first) - 1, option] = choice
        starts = np.searchsorted(colored_component, np.arange(colors.num_components + 1))
        groups: list[Group] = []
        for g in range(colors.num_components):
            states, table = colored_states[starts[g] : starts[g + 1]], colored_choices[starts[g] : starts[g + 1], : int(colors.comp_size[g])]
            keep = choosing[states]
            if keep.any():
                groups.append(Group(states[keep], table[keep], g))
        # the states without colors: those with several choices are groups of their own, those with a single choice are fixed
        colored = np.zeros(n, dtype=bool)
        colored[colored_states] = True
        for s in np.flatnonzero(~colored & choosing & (num_choices > 1)):
            groups.append(Group(np.array([s]), np.arange(mdp.row_groups[s], mdp.row_groups[s + 1])[None, :], -1))
        fixed_states = np.flatnonzero(~colored & choosing & (num_choices == 1))

        group_step = np.full(len(groups), -1, dtype=np.int64)
        if step is not None:
            for index, group in enumerate(groups):
                steps = np.unique(step[group.states])
                if len(steps) > 1:
                    raise UnsupportedProblemError(
                        f"a component of parameters is used in the steps {int(steps[0])} and {int(steps[1])} (the layers of the model): the parameters "
                        "of a model that is unfolded already have to depend on the step, i.e. its observations have to include the time"
                    )
                group_step[index] = steps[0]
        sizes = np.array([len(group.states) for group in groups], dtype=np.int64)
        start = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
        grouped_states = np.concatenate([group.states for group in groups]) if groups else np.zeros(0, dtype=np.int64)
        grouped_group = np.repeat(np.arange(len(groups)), sizes)
        grouped_choices = np.full((len(grouped_states), width), -1, dtype=np.int64)
        for index, group in enumerate(groups):
            grouped_choices[start[index] : start[index + 1], : group.num_options] = group.choices
        position = np.full(n, -1, dtype=np.int64)
        position[grouped_states] = np.arange(len(grouped_states))
        group_of = np.full(n, -1, dtype=np.int64)
        group_of[grouped_states] = grouped_group
        group_of_component = np.full(colors.num_components, -1, dtype=np.int64)
        for index, group in enumerate(groups):
            if group.component >= 0:
                group_of_component[group.component] = index
        return cls(
            num_steps, objective.maximize, n, initial, goal, condition, terminal, goal.astype(float), step, matrix, mdp.row_groups, groups, group_step,
            start, grouped_states, grouped_group, grouped_choices, fixed_states, position, group_of, colors, colored_states, colored_component,
            colored_choices, group_of_component,
        )  # fmt: skip

    # --- policies: a stage decision is an option for every group, a policy is a decision for every step (an array steps x groups) ---

    def component_options(self, decision: np.ndarray) -> np.ndarray:
        """The option of every component of the coloring under a stage decision (0 for the components that have no group)."""
        options = np.zeros(self.num_components, dtype=np.int64)
        has_group = self.group_of_component >= 0
        options[has_group] = decision[self.group_of_component[has_group]]
        return options

    def choices_of(self, decision: np.ndarray) -> np.ndarray:
        """The choice of every state under a stage decision: the colored states execute the option of their component, the other states of groups their
        option, and the remaining ones their first choice."""
        choices = self.row_groups[:-1].copy()
        choices[self.grouped_states] = self.grouped_choices[np.arange(len(self.grouped_states)), decision[self.grouped_group]]
        options = self.component_options(decision)
        choices[self.colored_states] = self.colored_choices[np.arange(len(self.colored_states)), options[self.colored_component]]
        return choices

    def evaluate(self, decisions: np.ndarray, payoff: np.ndarray | None = None) -> float:
        """The expected payoff of the policy (by default the one of the problem), computed forward (independently of the search)."""
        payoff = self.payoff if payoff is None else payoff
        if self.terminal[self.initial]:
            return float(payoff[self.initial])
        reach = np.zeros(self.num_states)
        reach[self.initial] = 1.0
        collected = 0.0
        for step in range(self.num_steps):
            acting = self.acting(step)
            moving = np.where(self.terminal, 0.0, reach)  # the mass in the terminal states has been collected
            executed = np.zeros(self.matrix.shape[0])
            executed[self.choices_of(decisions[step])[acting]] = moving[acting]
            reach = self.matrix.T @ executed + np.where(acting, 0.0, moving)  # the mass of the states of later steps waits
            collected += float(reach[self.terminal] @ payoff[self.terminal])
        return collected

    def joint_options(self, decisions: np.ndarray) -> np.ndarray:
        """The joint option of every component of the coloring in every step (steps x components), e.g. for colors.decode."""
        return np.array([self.component_options(decision) for decision in decisions], dtype=np.int64).reshape(self.num_steps, self.num_components)

    def final_options(self, decisions: np.ndarray) -> np.ndarray:
        """For a model that is unfolded already: the option of every component, which is chosen in the step of the component."""
        options = np.zeros(self.num_components, dtype=np.int64)
        has_group = self.group_of_component >= 0
        groups = self.group_of_component[has_group]
        options[has_group] = decisions[self.group_step[groups], groups]
        return options

    def state_choices(self, decisions: np.ndarray) -> np.ndarray:
        """For a model that is unfolded already: the choice of every state under the policy (a state executes the decision of its step)."""
        choices = self.row_groups[:-1].copy()
        option = decisions[self.group_step[self.grouped_group], self.grouped_group]
        choices[self.grouped_states] = self.grouped_choices[np.arange(len(self.grouped_states)), option]
        options = self.final_options(decisions)
        choices[self.colored_states] = self.colored_choices[np.arange(len(self.colored_states)), options[self.colored_component]]
        return choices


# ----------------------------------------------------------------------------------------------------
# the upper bound: the relaxation that ignores the colors
# ----------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Relaxation:
    """The values of the states when every state chooses for itself (the colors are ignored), in every step t = 0..H.

    best[t][s] is the best expected payoff (in the direction of the objective) from s in the step t, worst[t][s] the worst one. A state is *alive* in the
    step if it can hold mass that moves on (it is not terminal and chooses in this step or a later one) and its value depends on the policy
    (best != worst); cash[t][s] is the value of the states that are not alive (the payoff for the terminal states), 0 for the alive ones. The upper bound
    of a belief is <b, best[t]>.
    """

    best: list[np.ndarray]
    worst: list[np.ndarray]
    alive: list[np.ndarray]
    cash: list[np.ndarray]

    @classmethod
    def compute(cls, problem: Problem, fold: bool = True) -> Relaxation:
        """:param fold: whether the states whose value does not depend on the policy are folded (only the terminal states are folded otherwise)"""
        steps = problem.num_steps
        final = np.where(problem.terminal, problem.payoff, 0.0)
        best, worst = [final] * (steps + 1), [final] * (steps + 1)
        starts = problem.row_groups[:-1]
        better, worse = (np.maximum, np.minimum) if problem.maximize else (np.minimum, np.maximum)

        def backup(step: int, following: np.ndarray, choose: Any) -> np.ndarray:
            """The states that choose in the step take their best (or worst) choice, the states of later steps wait."""
            chosen = np.where(problem.acting(step), choose.reduceat(problem.matrix @ following, starts), following)
            return np.where(problem.terminal, problem.payoff, chosen)

        for step in range(steps - 1, -1, -1):
            best[step], worst[step] = backup(step, best[step + 1], better), backup(step, worst[step + 1], worse)
        alive, cash = [], []
        for step in range(steps + 1):
            moves = ~problem.terminal & (step < steps) & (True if problem.step is None else problem.step >= step)
            undecided = np.abs(best[step] - worst[step]) > 1e-12 if fold else np.ones(problem.num_states, dtype=bool)
            alive.append(moves & undecided)
            cash.append(np.where(alive[-1], 0.0, best[step]))
        return cls(best, worst, alive, cash)


# ----------------------------------------------------------------------------------------------------
# the belief graph
# ----------------------------------------------------------------------------------------------------


@dataclasses.dataclass(eq=False)
class Node:
    """A node of the belief graph: a step and the distribution of the mass that is still moving (a sparse vector over the alive states of the step that sums
    to 1, or empty if nothing moves anymore). The reach vector of a prefix that reaches the node is a multiple of it."""

    step: int
    support: np.ndarray
    mass: np.ndarray

    @property
    def key(self) -> tuple[int, bytes, bytes]:
        return self.step, self.support.tobytes(), np.round(self.mass, KEY_DIGITS).tobytes()

    def dense(self, num_states: int) -> np.ndarray:
        """The belief as an array over all the states."""
        vector = np.zeros(num_states)
        vector[self.support] = self.mass
        return vector


@dataclasses.dataclass
class Edge:
    """A stage decision in a node: the options of the groups that the node's mass reaches (decision), the expected payoff that the step collects (reward),
    and the alive mass after the step, scale * child.mass (scale 0: nothing moves anymore). bound = reward + scale * U(child) (the upper bound)."""

    decision: tuple[tuple[int, int], ...]
    reward: float
    scale: float
    child: Node
    bound: float


def best_combinations(scores: Sequence[Sequence[tuple[float, int]]], base: float, threshold: float) -> Iterator[tuple[float, tuple[int, ...]]]:
    """The combinations of one option per group by decreasing total score (base + the sum of the scores), as long as the total exceeds the threshold.

    scores[i] lists the (score, option) of the group i, best first. A combination is a vector of positions in these lists; it is generated from the combination
    that has a smaller position at the last coordinate that is not 0 (at most once), so every combination comes once and after the ones that are better.
    """
    if not scores:
        if base > threshold:
            yield base, ()
        return
    heap = [(-(base + sum(group[0][0] for group in scores)), (0,) * len(scores), 0)]
    while heap:
        negative, positions, last = heapq.heappop(heap)
        if -negative <= threshold:
            return
        yield -negative, tuple(scores[i][p][1] for i, p in enumerate(positions))
        for i in range(last, len(scores)):
            if positions[i] + 1 < len(scores[i]):
                successor = positions[:i] + (positions[i] + 1,) + positions[i + 1 :]
                heapq.heappush(heap, (negative + scores[i][positions[i]][0] - scores[i][positions[i] + 1][0], successor, i))


def gather(matrix: Any, rows: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The entries of the given rows of a sparse matrix (each row multiplied by its weight), concatenated: their columns and their values. The matrix is
    anything with the arrays indptr, indices and data of the compressed rows (a scipy matrix, Moves)."""
    starts, lengths = matrix.indptr[rows], matrix.indptr[rows + 1] - matrix.indptr[rows]
    entries = np.repeat(starts - (np.cumsum(lengths) - lengths), lengths) + np.arange(lengths.sum())
    values = matrix.data[entries] if weights is None else matrix.data[entries] * np.repeat(weights, lengths)
    return matrix.indices[entries].astype(np.int64), values  # the states as int64, like everywhere else (the keys of the nodes are their bytes)


@dataclasses.dataclass
class Moves:
    """Where the mass of the groups that a node touches moves under each of their options, as the compressed rows of a sparse matrix: the row i*W+o is the
    mass that the i-th of the groups moves under its option o (W is the largest number of options; the rows of the options that a group lacks are empty).
    A state may come several times in a row (the mass of several states of the group that moves to it)."""

    groups: np.ndarray
    rows: np.ndarray  # the row of every entry
    indices: np.ndarray
    data: np.ndarray
    indptr: np.ndarray

    def row_sums(self, vector: np.ndarray) -> np.ndarray:
        """<row, vector> for every row."""
        return np.bincount(self.rows, weights=self.data * vector[self.indices], minlength=len(self.indptr) - 1)


class Graph:
    """The belief graph of a problem: the root, the upper bound of a node and the edges of a node."""

    def __init__(self, problem: Problem, relaxation: Relaxation):
        self.problem, self.relaxation = problem, relaxation
        # two random projections of the alive states of each step, which tell apart the options that move the mass differently (see children)
        rng = np.random.default_rng(0)
        self.projections = [(rng.random((2, problem.num_states)) + 0.5) * alive for alive in relaxation.alive]

    def root(self) -> tuple[Node, float]:
        """The root and the value that is collected before the first step (all of it if the value of the initial state does not depend on the policy)."""
        problem, relaxation = self.problem, self.relaxation
        if problem.num_steps > 0 and relaxation.alive[0][problem.initial]:
            return Node(0, np.array([problem.initial], dtype=np.int64), np.ones(1)), 0.0
        return Node(0, np.zeros(0, dtype=np.int64), np.zeros(0)), float(relaxation.cash[0][problem.initial])

    def upper(self, node: Node) -> float:
        """U(b) = <b, best>: no suffix does better from the node."""
        return float(node.mass @ self.relaxation.best[node.step][node.support])

    def moves(self, states: np.ndarray, mass: np.ndarray) -> Moves:
        """Where the mass in the states (of groups) moves under every option."""
        problem = self.problem
        width = problem.grouped_choices.shape[1]
        present, local = np.unique(problem.group_of[states], return_inverse=True)
        choices = problem.grouped_choices[problem.position[states]]  # states x options
        which, option = np.nonzero(choices >= 0)
        chosen = choices[which, option]
        columns, values = gather(problem.matrix, chosen, mass[which])
        rows = np.repeat(local[which] * width + option, problem.matrix.indptr[chosen + 1] - problem.matrix.indptr[chosen])
        order = np.argsort(rows, kind="stable")
        rows = rows[order]
        return Moves(present, rows, columns[order], values[order], np.searchsorted(rows, np.arange(len(present) * width + 1)))

    def edge(self, node: Node, columns: np.ndarray, values: np.ndarray, decision: tuple[tuple[int, int], ...], bound: float) -> Edge:
        """The edge of the decision under which the mass values moves to the states columns (a state may come several times): the alive mass moves on
        (normalized in the child), the rest is collected."""
        alive, cash = self.relaxation.alive[node.step + 1], self.relaxation.cash[node.step + 1]
        states, where = np.unique(columns, return_inverse=True)
        reached = np.bincount(where, weights=values, minlength=len(states))
        moving = alive[states] & (reached != 0)
        scale = float(reached[moving].sum())
        child = Node(node.step + 1, states[moving], reached[moving] / scale if scale > 0 else np.zeros(0))
        return Edge(decision, float(reached @ cash[states]), scale, child, bound)

    def children(self, node: Node, threshold: float, limit: int | None = None) -> Iterator[Edge]:
        """The edges of the node by decreasing upper bound, as long as sign*bound exceeds the threshold (at most limit of them).

        The mass of a group under an option moves along the rows of its choices; the bound of an edge is the sum of the contributions of the groups (and of
        the states with a single choice and the waiting ones) weighted with the best values of the next step, so the stage decisions are enumerated by
        best_combinations. Two options of a group are equivalent if they move the same mass to every alive state of the next step and collect the same
        value (e.g. they differ only in states that the mass does not touch): only the first one is kept. They are told apart by two random projections of
        the alive mass and by the collected value (rounded), which differ for different mass with probability 1.
        """
        problem, relaxation = self.problem, self.relaxation
        sign, step, width = problem.sign, node.step, problem.grouped_choices.shape[1]
        value_next = relaxation.best[step + 1]
        acting = problem.acts(step, node.support)
        grouped = acting & (problem.group_of[node.support] >= 0)
        fixed = acting & ~grouped
        # the mass of the states with a single choice moves along it, the mass of the states of later steps waits
        columns, values = gather(problem.matrix, problem.row_groups[node.support[fixed]], node.mass[fixed])
        base_columns, base_values = np.concatenate([columns, node.support[~acting]]), np.concatenate([values, node.mass[~acting]])
        base_score = sign * float(base_values @ value_next[base_columns])
        # the groups: a list of the distinct options of every group by their contribution to the upper bound
        moves = self.moves(node.support[grouped], node.mass[grouped])
        gains = sign * moves.row_sums(value_next)
        projections = [moves.row_sums(projection) for projection in self.projections[step + 1]]
        fingerprints = np.round(np.column_stack([*projections, moves.row_sums(relaxation.cash[step + 1])]), KEY_DIGITS)
        num_options = (problem.grouped_choices[problem.start[moves.groups]] >= 0).sum(axis=1)
        scores = []
        for i, count in enumerate(num_options):
            _, distinct = np.unique(fingerprints[i * width : i * width + count], axis=0, return_index=True)
            scores.append(sorted(((float(gains[i * width + o]), int(o)) for o in distinct), reverse=True))
        for number, (total, options) in enumerate(best_combinations(scores, base_score, threshold)):
            if limit is not None and number >= limit:
                return
            columns, values = gather(moves, np.arange(len(moves.groups)) * width + np.array(options, dtype=np.int64))
            decision = tuple(zip(moves.groups.tolist(), options, strict=True))
            yield self.edge(node, np.concatenate([base_columns, columns]), np.concatenate([base_values, values]), decision, sign * total)


# ----------------------------------------------------------------------------------------------------
# the lower bound: value vectors of suffix policies (alpha vectors)
# ----------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class PoolLayer:
    """Value vectors of suffix policies from the step t on: vectors[j][s] is the expected payoff of the suffix j from s; the suffix j executes decisions[j]
    (an option per group) in the step t and continues with the suffix parents[j] of the step t+1."""

    vectors: np.ndarray
    decisions: np.ndarray
    parents: np.ndarray


class Pool:
    """A set of suffix policies for every step, built by point-based backups at anchors (beliefs); the lower bound of a node is the best of them."""

    def __init__(self, problem: Problem, relaxation: Relaxation, anchors: Sequence[scipy.sparse.csr_matrix]):
        """:param anchors: for every step t = 0..H-1 the anchors as the rows of a sparse matrix (the first anchor of the step 0 is the root)"""
        self.problem = problem
        self.layers: list[PoolLayer] = [None] * problem.num_steps  # type: ignore[list-item]
        following = relaxation.cash[problem.num_steps][None, :]  # in the step H only the terminal states are worth anything
        for step in range(problem.num_steps - 1, -1, -1):
            self.layers[step] = point_backup(problem, relaxation, step, anchors[step], following)
            following = self.layers[step].vectors

    def lower(self, node: Node) -> tuple[float, int]:
        """The best value <b, alpha> over the vectors of the step, and the vector."""
        values = self.problem.sign * (self.layers[node.step].vectors[:, node.support] @ node.mass)
        best = int(np.argmax(values))
        return self.problem.sign * float(values[best]), best

    def suffix(self, step: int, index: int) -> np.ndarray:
        """The decisions of the suffix policy of a vector, for the steps step..H-1."""
        rows = []
        for t in range(step, self.problem.num_steps):
            rows.append(self.layers[t].decisions[index])
            index = int(self.layers[t].parents[index])
        return np.array(rows, dtype=np.int64).reshape(len(rows), len(self.problem.groups))


def option_sums(problem: Problem, positions: np.ndarray, weights: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every group with states at the (sorted) positions of the grouped states: the sum over these states of weight * values[the choice of the state under
    the option], for every option and every column of values (an array groups x options x columns; -inf for the options that the group does not have)."""
    width = problem.grouped_choices.shape[1]
    if len(positions) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, width, values.shape[1]))
    choices, groups = problem.grouped_choices[positions], problem.grouped_group[positions]
    starts = np.flatnonzero(np.concatenate(([True], groups[1:] != groups[:-1])))  # the positions of a group are consecutive
    sums = np.add.reduceat(weights[:, None, None] * values[np.maximum(choices, 0)], starts, axis=0)
    sums[choices[starts] < 0] = -np.inf
    return groups[starts], sums


def point_backup(problem: Problem, relaxation: Relaxation, step: int, anchors: Any, following: np.ndarray) -> PoolLayer:
    """The point-based backup of the step: for every anchor b the decision d and the vector alpha' of the next step that maximize the value of b under d
    followed by alpha'.

    For a fixed alpha' the value is a sum over the groups that choose in the step (fact (3)), so the best option of every group is chosen independently, and
    alpha' is the best of the vectors; Q[c, j] is the value of the choice c when the suffix j follows. A group that the anchor does not reach takes the
    option that is the best on average over its states (the vector is then also a good lower bound elsewhere). The mass of the states of later steps waits.
    """
    anchors = scipy.sparse.csr_matrix(anchors)
    values = problem.matrix @ following.T  # Q: choices x suffixes
    signed = problem.sign * values
    acting = problem.acting(step)
    choosing = np.flatnonzero(acting[problem.grouped_states])  # the positions of the grouped states that choose in the step
    fixed = problem.fixed_states[acting[problem.fixed_states]]
    waiting = np.flatnonzero(relaxation.alive[step] & ~acting)
    fixed_values, waiting_values = signed[problem.row_groups[fixed]], problem.sign * following[:, waiting].T
    count = anchors.shape[0]
    decisions = np.zeros((count, len(problem.groups)), dtype=np.int64)
    parents = np.zeros(count, dtype=np.int64)
    on_average: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # for a suffix: the option of every group that is the best on average over its states
    for a in range(count):
        anchor = anchors.getrow(a).toarray().ravel()
        positions = choosing[anchor[problem.grouped_states[choosing]] != 0]
        groups, sums = option_sums(problem, positions, anchor[problem.grouped_states[positions]], signed)
        total = anchor[fixed] @ fixed_values + anchor[waiting] @ waiting_values + sums.max(axis=1).sum(axis=0)
        parent = parents[a] = int(np.argmax(total))
        if parent not in on_average:
            every, average = option_sums(problem, choosing, np.ones(len(choosing)), signed[:, [parent]])
            on_average[parent] = every, average[:, :, 0].argmax(axis=1)
        decisions[a, on_average[parent][0]] = on_average[parent][1]
        decisions[a, groups] = sums[:, :, parent].argmax(axis=1)
    vectors = following[parents].copy()  # the states of later steps keep the value of the next step (their mass waits)
    executed = problem.grouped_choices[choosing[None, :], decisions[:, problem.grouped_group[choosing]]]  # anchors x choosing grouped states
    vectors[:, problem.grouped_states[choosing]] = values[executed, parents[:, None]]
    vectors[:, fixed] = values[problem.row_groups[fixed]][:, parents].T
    settled = ~relaxation.alive[step]
    vectors[:, settled] = relaxation.cash[step][settled]  # the terminal states and the states whose value does not depend on the policy
    # equal suffixes (the same decision and the same continuation) are kept once
    _, unique = np.unique(np.column_stack([decisions, parents]), axis=0, return_index=True)
    unique = np.sort(unique)
    return PoolLayer(vectors[unique], decisions[unique], parents[unique])


# ----------------------------------------------------------------------------------------------------
# phase 1: the lower bound
# ----------------------------------------------------------------------------------------------------


def beam(graph: Graph, width: int, deadline: float | None = None) -> list[list[Node]]:
    """The nodes of a beam search: in every step the `width` beliefs with the largest upper bound c + K U(b) of a prefix that reaches K b, among the children
    of the nodes of the previous step."""
    problem = graph.problem
    sign = problem.sign
    root, collected = graph.root()
    layers_of_nodes, frontier = [[root]], [(root, collected, 1.0)]
    for _ in range(problem.num_steps - 1 if width > 0 else 0):
        candidates: dict[tuple[int, bytes, bytes], tuple[float, Node, float, float]] = {}
        for node, value, scale in frontier:
            if deadline is not None and time.time() > deadline:
                break
            for edge in graph.children(node, -np.inf, limit=width):
                bound = value + scale * edge.bound
                known = candidates.get(edge.child.key)
                if known is None or sign * (bound - known[0]) > 0:
                    candidates[edge.child.key] = (bound, edge.child, value + scale * edge.reward, scale * edge.scale)
        ranked = sorted(candidates.values(), key=lambda entry: -sign * entry[0])[:width]
        frontier = [(node, value, scale) for _, node, value, scale in ranked]
        layers_of_nodes.append([node for node, _, _ in frontier])
    return layers_of_nodes


def reach_vectors(problem: Problem, relaxation: Relaxation, decisions: np.ndarray) -> list[np.ndarray]:
    """The alive mass of the policy in every step t = 0..H-1 (the reach vectors after folding)."""
    reach = np.zeros(problem.num_states)
    reach[problem.initial] = 1.0
    out = []
    for step in range(problem.num_steps):
        reach = np.where(relaxation.alive[step], reach, 0.0)
        out.append(reach)
        acting = problem.acting(step)
        executed = np.zeros(problem.matrix.shape[0])
        executed[problem.choices_of(decisions[step])[acting]] = reach[acting]
        reach = problem.matrix.T @ executed + np.where(acting, 0.0, reach)
    return out


def normalized(step: int, vector: np.ndarray) -> Node:
    support = np.flatnonzero(vector)
    total = float(vector[support].sum())
    return Node(step, support, vector[support] / total if total > 0 else vector[support])


def as_matrix(nodes: Sequence[Node], num_states: int) -> scipy.sparse.csr_matrix:
    """The beliefs as the rows of a sparse matrix."""
    indptr = np.cumsum([0] + [len(node.support) for node in nodes])
    data = np.concatenate([node.mass for node in nodes]) if nodes else np.zeros(0)
    indices = np.concatenate([node.support for node in nodes]) if nodes else np.zeros(0, dtype=np.int64)
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=(len(nodes), num_states))


@dataclasses.dataclass
class Incumbent:
    """The best policy known: its decisions (steps x groups) and its value."""

    decisions: np.ndarray
    value: float


def lower_bound_phase(graph: Graph, width: int, rounds: int, deadline: float | None = None, start: np.ndarray | None = None) -> tuple[Incumbent, Pool]:
    """Phase 1: the pool of suffix policies at the anchors of a beam search and of the incumbents, the sweep repeated while the incumbent improves.

    :param start: a policy that is known already (it is the first incumbent, and its reach vectors are anchors)
    """
    problem, relaxation = graph.problem, graph.relaxation
    # the distinct anchors of every step (by their keys) in the order in which they were found; the root is the first anchor of the step 0
    anchors: list[dict[tuple[int, bytes, bytes], Node]] = [{} for _ in range(problem.num_steps)]

    def add(decisions: np.ndarray) -> None:
        for step, vector in enumerate(reach_vectors(problem, relaxation, decisions)):
            node = normalized(step, vector)
            anchors[step].setdefault(node.key, node)

    for step, layer in enumerate(beam(graph, width, deadline)):
        for node in layer:
            anchors[step].setdefault(node.key, node)
    incumbent = None
    if start is not None:
        incumbent = Incumbent(start, problem.evaluate(start))
        add(start)
    pool = None
    empty = Node(0, np.zeros(0, dtype=np.int64), np.zeros(0))
    for _ in range(max(1, rounds)):
        # a step without anchors (e.g. without the beam, before the first incumbent) gets the empty one: its backup is still a suffix policy
        pool = Pool(problem, relaxation, [as_matrix(list(nodes.values()) or [empty], problem.num_states) for nodes in anchors])
        decisions = pool.suffix(0, 0)
        value = problem.evaluate(decisions)
        if incumbent is not None and not problem.better(value, incumbent.value):
            break
        incumbent = Incumbent(decisions, value)
        add(decisions)
        if deadline is not None and time.time() > deadline:
            break
    assert incumbent is not None and pool is not None
    return incumbent, pool


# ----------------------------------------------------------------------------------------------------
# phase 2: optimality
# ----------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Result:
    """The best policy found, its value, a bound on the optimum (the value itself if the policy is proved optimal) and what the search did."""

    decisions: np.ndarray
    value: float
    bound: float
    optimal: bool
    statistics: dict[str, Any]


@dataclasses.dataclass
class Entry:
    """What the search knows about a node, in the direction of the objective (sign * value): lower <= W(node) <= upper. The lower bound is achieved by the
    edge decision followed by the child (whose key is child; None if nothing moves after the edge), or by the suffix of the vector of the pool."""

    lower: float
    upper: float
    decision: tuple[tuple[int, int], ...] | None = None
    child: tuple[int, bytes, bytes] | None = None
    vector: int | None = None


class Search:
    """Phase 2: depth-first branch and bound over the beliefs with a table of the bounds of the nodes (see the top of this file)."""

    def __init__(self, graph: Graph, pool: Pool, max_nodes: int, deadline: float | None = None):
        self.graph, self.pool, self.problem = graph, pool, graph.problem
        self.max_nodes, self.deadline = max_nodes, deadline
        self.table: dict[tuple[int, bytes, bytes], Entry] = {}
        self.stopped: str | None = None
        self.statistics = {"expanded": 0, "edges": 0, "closed": 0}
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 4 * self.problem.num_steps + 1000))

    def entry(self, node: Node) -> Entry:
        """The entry of the node, which starts with the bounds of the pool and the relaxation."""
        entry = self.table.get(node.key)
        if entry is None:
            if node.support.size == 0:
                entry = Entry(0.0, 0.0)
            else:
                sign = self.problem.sign
                lower, vector = self.pool.lower(node)
                entry = Entry(sign * lower, sign * self.graph.upper(node), vector=vector)
                self.statistics["closed"] += entry.lower >= entry.upper - tolerance(entry.upper)
            self.table[node.key] = entry
        return entry

    def over_budget(self) -> bool:
        if self.stopped is None and len(self.table) > self.max_nodes:
            self.stopped = "nodes"
        if self.stopped is None and self.deadline is not None and time.time() > self.deadline:
            self.stopped = "time"
        return self.stopped is not None

    def value(self, node: Node, threshold: float) -> Entry:
        """The bounds of W(node) (in the direction of the objective): exact (lower = upper), or upper <= threshold (unless the search has been stopped)."""
        entry = self.entry(node)
        if entry.lower >= entry.upper - tolerance(entry.upper) or entry.upper <= threshold + tolerance(threshold) or self.stopped is not None:
            return entry
        self.statistics["expanded"] += 1
        sign = self.problem.sign
        best = entry.lower
        target = max(threshold, best)
        # every edge either is not searched (its bound does not beat the target), or returns its exact value, or a value that does not beat its target:
        # so nothing beats max(best, target), unless the search is stopped in an edge, which is then bounded by its bound (and so are the following ones)
        rest = target
        for edge in self.graph.children(node, target + tolerance(target)):
            self.statistics["edges"] += 1
            target = max(threshold, best)
            bound = sign * edge.bound
            if bound <= target + tolerance(target):
                break  # this edge and all the following ones cannot beat the target
            reward = sign * edge.reward
            low = reward + edge.scale * self.value(edge.child, (target - reward) / edge.scale).lower if edge.scale > 0 else reward
            if low > best:
                best = low
                entry.decision, entry.child, entry.vector = edge.decision, edge.child.key if edge.scale > 0 else None, None
            if self.over_budget():
                rest = bound
                break
        entry.lower = best
        entry.upper = min(entry.upper, max(best, rest))
        return entry

    def decisions(self, root: Node) -> np.ndarray:
        """The policy that achieves the lower bound of the root: the edges of the table, then the suffix of a vector of the pool."""
        decisions = np.zeros((self.problem.num_steps, len(self.problem.groups)), dtype=np.int64)
        key: tuple[int, bytes, bytes] | None = root.key
        step = 0
        while key is not None:
            entry = self.table[key]
            if entry.vector is not None:
                decisions[step:] = self.pool.suffix(step, entry.vector)
                break
            for group, option in entry.decision or ():
                decisions[step, group] = option
            key, step = entry.child, step + 1
        return decisions


def search(graph: Graph, pool: Pool, incumbent: Incumbent, max_nodes: int, deadline: float | None = None) -> Result:
    """Phase 2: the depth-first search from the root, first with thresholds that halve the gap between the incumbent and the upper bound, then with the
    incumbent itself. A search that fails low proves a smaller upper bound (the table keeps the work for the next one), so a search that is stopped has a
    tight bracket; a search whose root's bounds meet has found the optimum."""
    problem = graph.problem
    sign = problem.sign
    root, collected = graph.root()
    searcher = Search(graph, pool, max_nodes, deadline)
    # the bounds of W(root) in the direction of the objective: the incumbent and the upper bound proved so far
    low, high = sign * (incumbent.value - collected), sign * graph.upper(root)
    improved, thresholds = False, 0
    while searcher.stopped is None and high > low + tolerance(low):
        final = thresholds >= DESCENT
        entry = searcher.value(root, low if final else low + (high - low) / 2)
        thresholds += 1
        high = min(high, entry.upper)
        if entry.lower > low + tolerance(low):
            low, improved = entry.lower, True
            incumbent = Incumbent(searcher.decisions(root), collected + sign * low)
        if final:
            break  # with the incumbent as the threshold the search is exact or proves the incumbent optimal
    statistics: dict[str, Any] = dict(searcher.statistics, nodes=len(searcher.table), thresholds=thresholds, improved=improved)
    if searcher.stopped is not None:
        statistics["stopped"] = searcher.stopped
    bound = collected + sign * max(high, low)
    optimal = not problem.better(bound, incumbent.value)
    return Result(incumbent.decisions, incumbent.value, incumbent.value if optimal else bound, optimal, statistics)


def solve(
    problem: Problem,
    width: int = 32,
    rounds: int = 10,
    max_nodes: int = 2_000_000,
    time_limit: float | None = None,
    fold: bool = True,
    start: np.ndarray | None = None,
    prove: bool = True,
) -> Result:
    """The whole procedure for the expected payoff of the problem: the relaxation, phase 1 and phase 2.

    :param start: a policy that is known already (the first incumbent)
    :param prove: whether to run phase 2 (otherwise the result is the policy of phase 1 with the upper bound of the root)
    """
    started = time.time()
    deadline = None if time_limit is None else started + time_limit
    relaxation = Relaxation.compute(problem, fold)
    graph = Graph(problem, relaxation)
    root, collected = graph.root()
    upper = collected + graph.upper(root)
    statistics: dict[str, Any] = {"root bound": upper}
    if root.support.size == 0:
        # the best and the worst policy of the relaxation agree in the initial state: every policy has this value
        decisions = np.zeros((problem.num_steps, len(problem.groups)), dtype=np.int64) if start is None else start
        statistics.update({"phase 1 value": collected, "phase 1 seconds": 0.0, "seconds": round(time.time() - started, 3), "determined": True})
        return Result(decisions, collected, collected, True, statistics)
    incumbent, pool = lower_bound_phase(graph, width, rounds, deadline, start)
    phase1 = time.time() - started
    logger.debug(f"phase 1 (point-based lower bound, beam width {width}): value {incumbent.value:.8g}, upper bound {upper:.8g} [{phase1:.2f}s]")
    if prove:
        result = search(graph, pool, incumbent, max_nodes, deadline)
    else:
        optimal = not problem.better(upper, incumbent.value)
        result = Result(incumbent.decisions, incumbent.value, incumbent.value if optimal else upper, optimal, {})
    result.statistics.update(statistics)
    result.statistics.update({"phase 1 value": incumbent.value, "phase 1 seconds": round(phase1, 3), "seconds": round(time.time() - started, 3)})
    value = problem.evaluate(result.decisions)
    if abs(value - result.value) > 1e-8 * max(1.0, abs(value)):
        logger.warning(f"WARNING: the value of the policy computed again ({value:.10g}) differs from the one of the search ({result.value:.10g})")
    return result


# ----------------------------------------------------------------------------------------------------
# conditional probabilities: Dinkelbach's method
# ----------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class ConditionalResult:
    """The best policy found for a conditional probability: its value numerator / evidence, whether it is proved optimal, the values lambda_0, lambda_1, ...
    of the policies of Dinkelbach's method, and the statistics of the searches."""

    decisions: np.ndarray
    value: float
    numerator: float
    evidence: float
    optimal: bool
    lambdas: list[float]
    statistics: dict[str, Any]


def solve_conditional(
    problem: Problem, width: int = 32, rounds: int = 10, max_nodes: int = 2_000_000, time_limit: float | None = None, fold: bool = True
) -> ConditionalResult:
    """The optimal conditional probability Pr(F goal | F condition) by Dinkelbach's method (see the top of this file)."""
    assert problem.condition is not None
    started = time.time()
    deadline = None if time_limit is None else started + time_limit
    goal, condition = problem.goal.astype(float), problem.condition.astype(float)

    def remaining() -> float | None:
        return None if deadline is None else max(0.0, deadline - time.time())

    def ratio(decisions: np.ndarray) -> tuple[float, float]:
        return problem.evaluate(decisions, goal), problem.evaluate(decisions, condition)

    # a first policy under which the condition can happen: the best one for Pr(F condition) of phase 1 (or of the search, if that one does not reach it)
    likely = problem.with_payoff(condition, maximize=True)
    first = solve(likely, width, rounds, max_nodes, remaining(), fold, prove=False)
    if ratio(first.decisions)[1] <= DEGENERATE:
        first = solve(likely, width, rounds, max_nodes, remaining(), fold)
        if ratio(first.decisions)[1] <= DEGENERATE:
            raise UnsupportedProblemError("no color consistent policy reaches the condition: the conditional probability is not defined")
    decisions = first.decisions
    numerator, evidence = ratio(decisions)
    lambdas = [numerator / evidence]
    iterations: list[dict[str, Any]] = []
    while True:
        lam = lambdas[-1]
        # N - lambda D in units of the evidence of the policy, whose value is 0 in it
        result = solve(problem.with_payoff(((1 - lam) * goal - lam * (condition - goal)) / evidence), width, rounds, max_nodes, remaining(), fold, decisions)
        iterations.append({"lambda": lam, "value": result.value, "bound": result.bound} | result.statistics)
        logger.debug(f"Dinkelbach: lambda {lam:.10g}, max N - lambda D = {result.value:.4g} (in units of the evidence) [{time.time() - started:.2f}s]")
        optimal = result.optimal
        if not problem.better(result.value, 0.0):
            break  # nothing beats lambda (if the search has finished)
        new_numerator, new_evidence = ratio(result.decisions)
        if new_evidence <= DEGENERATE or not problem.better(new_numerator / new_evidence, lam):
            logger.warning("WARNING: the improvement of Dinkelbach's method is below the precision, the last policy is kept")
            break
        decisions, numerator, evidence = result.decisions, new_numerator, new_evidence
        lambdas.append(numerator / evidence)
        if deadline is not None and time.time() > deadline:
            optimal = False
            break
    statistics = {"iterations": iterations, "seconds": round(time.time() - started, 3)}
    return ConditionalResult(decisions, numerator / evidence, numerator, evidence, optimal, lambdas, statistics)


# ----------------------------------------------------------------------------------------------------
# output and the command line
# ----------------------------------------------------------------------------------------------------


def describe(problem: Problem, colored_mdp: Any, decisions: np.ndarray) -> list[dict[str, str]]:
    """For every step, the options of the parameters (and the choices of the states without colors) that the policy can reach in it."""
    space = colored_mdp.parameter_space
    relaxation = Relaxation.compute(problem, fold=False)
    out = []
    for step, reach in enumerate(reach_vectors(problem, relaxation, decisions)):
        options = problem.colors.decode(problem.component_options(decisions[step])[None, :])[0]
        reached = (reach != 0) & problem.acting(step)
        assignment = {}
        for index, group in enumerate(problem.groups):
            if not reached[group.states].any():
                continue
            if group.component < 0:
                assignment[f"state {int(group.states[0])}"] = f"choice {int(decisions[step, index])}"
                continue
            for hole in np.flatnonzero(problem.colors.hole_component == group.component):
                assignment[space.parameter_name(int(hole))] = str(space.parameter_to_option_labels[hole][options[hole]])
        out.append(assignment)
    return out


def reachability_by_storm(mdp: Any, label: str, choices: Any) -> float:
    """The probability of reaching the label in the Markov chain induced by the policy that executes the choices, computed by Storm."""
    builder = paynt.model.model.SubmodelBuilder
    dtmc = builder.mdp_to_dtmc(builder.build_submdp(mdp, choices, builder.default_builder_options()).model)
    result = stormpy.model_checking(dtmc, stormpy.parse_properties_without_context(f'P=? [F "{label}"]')[0], only_initial_states=True)
    return float(result.at(dtmc.initial_states[0]))


def verify_with_storm(problem: Problem, colored_mdp: Any, task: Any, decisions: np.ndarray) -> tuple[float, bool]:
    """The value of the policy computed by Storm (on the time unfolding of the colored MDP, or on the model itself if it is unfolded already), and whether
    the policy is color consistent there."""
    if not problem.layered:
        unfolded = fh.UnfoldedColoredMdp.build(colored_mdp, problem.num_steps, [])
        policy = np.array([problem.choices_of(decisions[step]) for step in range(problem.num_steps)])
        choices = unfolded.policy_choices(policy)
        options = problem.colors.decode(problem.joint_options(decisions))
        return unfolded.value_of_policy(choices, fh.Objective.from_task(task).prop), unfolded.check_consistent(choices, options)
    selected = payntbind.synthesis.bitVectorFromIndices(problem.matrix.shape[0], problem.state_choices(decisions).tolist())
    consistent = fh.is_color_consistent(colored_mdp, selected, problem.colors.decode(problem.final_options(decisions)[None, :])[0])
    if problem.condition is not None:
        return fh.conditional_probability_by_storm(colored_mdp.underlying_mdp, fh.ConditionalObjective.from_task(task), selected), consistent
    return reachability_by_storm(colored_mdp.underlying_mdp, fh.Objective.from_task(task).target_label, selected), consistent


def report_threshold(task: Any, value: float, optimal: bool) -> None:
    """For a property with a threshold (e.g. P>0.35 [...]): whether some policy satisfies it."""
    prop = task.specification.all_properties()[0]
    if type(prop) is not paynt.specification.property.Property:
        return
    if prop.satisfies_threshold(value):
        logger.info(f"the policy satisfies the property {prop}")
    elif optimal:
        logger.info(f"no color consistent policy satisfies the property {prop} (the optimum is proved)")
    else:
        logger.info(f"the best policy found does not satisfy the property {prop} (the search did not finish)")


@click.command()
@click.argument("model", type=click.Path(exists=True, dir_okay=False))
@click.argument("properties", type=click.Path(exists=True, dir_okay=False))
@click.argument("horizon", type=click.IntRange(min=1), required=False)
@click.option("--fsc-memory-size", type=click.IntRange(min=1), default=1, show_default=True, help="implicit memory size for POMDP policies")
@click.option(
    "--width", type=click.IntRange(min=0), default=32, show_default=True, help="beam width of phase 1 (anchors of the lower bound per step; 0: no beam)"
)
@click.option("--rounds", type=click.IntRange(min=1), default=10, show_default=True, help="maximum number of sweeps of phase 1")
@click.option("--max-nodes", type=click.IntRange(min=1), default=2_000_000, show_default=True, help="stop phase 2 (with a bound) if it has more nodes")
@click.option("--timeout", type=click.FLOAT, help="time limit (s); phase 2 then stops with the incumbent and a bound on the optimum")
@click.option("--no-fold", is_flag=True, default=False, help="do not fold the states whose value does not depend on the policy (for comparison)")
@click.option("--verify", is_flag=True, default=False, help="double check the value and the color consistency of the policy with Storm")
@click.option("--show-policy", is_flag=True, default=False, help="print the parameter assignment of each step")
@click.option("--export-policy", type=click.Path(), help="write the result and the parameter assignment of each step to this JSON file")
@click.option("--precision", type=click.FLOAT, default=1e-6, show_default=True, help="model checking precision of Storm")
def main(
    model: str,
    properties: str,
    horizon: int | None,
    fsc_memory_size: int,
    width: int,
    rounds: int,
    max_nodes: int,
    timeout: float | None,
    no_fold: bool,
    verify: bool,
    show_policy: bool,
    export_policy: str | None,
    precision: float,
) -> None:
    """Find an optimal color consistent policy of MODEL for the probability P[F goal] or the conditional probability P[F goal || F condition] of
    PROPERTIES.

    With a HORIZON, the policy chooses a parameter assignment in every one of HORIZON steps. Without it, the model has to be unfolded already (acyclic, its
    observations include the time) and its layers are the steps; a conditional property needs such a model.
    """
    paynt.cli.setup_logger()
    started = time.time()
    factory, task = paynt.parser.sketch.Sketch.load_sketch(model, properties, precision=precision, task_kwargs={"memory_size": fsc_memory_size})
    colored_mdp = factory.build()
    try:
        problem = Problem.from_colored_mdp(colored_mdp, task, horizon)
    except UnsupportedProblemError as e:
        raise click.ClickException(str(e)) from e
    kind = "steps (the layers of the model)" if problem.layered else "steps"
    logger.info(f"{problem.num_states} states, {len(problem.groups)} groups of states that choose together, {problem.num_steps} {kind}")
    exported: dict[str, Any] = {"steps": problem.num_steps, "layered": problem.layered}
    if problem.condition is None:
        result = solve(problem, width, rounds, max_nodes, timeout, fold=not no_fold)
        statistics = result.statistics
        logger.info(
            f"the optimum of the MDP that ignores the colors: {statistics['root bound']:.8g}; phase 1 (point-based lower bound, beam width {width}): "
            f"{statistics['phase 1 value']:.8g} [{statistics['phase 1 seconds']:.2f}s]"
        )
        if not statistics.get("determined"):
            logger.info(
                f"phase 2 (depth-first search): {statistics['nodes']} beliefs, {statistics['expanded']} expanded, {statistics['edges']} edges, "
                f"{statistics['closed']} closed by the bounds [{statistics['seconds']:.2f}s]"
            )
        decisions, value, optimal = result.decisions, result.value, result.optimal
        if optimal:
            logger.info(f"the optimal value is {value:.10g} (proved)")
        else:
            low, high = sorted((result.value, result.bound))
            logger.info(f"the best policy found has the value {value:.10g}, the optimum is in [{low:.10g}, {high:.10g}] (stopped: {statistics['stopped']})")
        exported.update(value=value, bound=result.bound, optimal=optimal, statistics=statistics)
    else:
        conditional = solve_conditional(problem, width, rounds, max_nodes, timeout, fold=not no_fold)
        decisions, value, optimal = conditional.decisions, conditional.value, conditional.optimal
        searches = conditional.statistics["iterations"]
        logger.info(
            f"Dinkelbach's method: {len(conditional.lambdas)} policies with the conditional probabilities "
            + ", ".join(f"{lam:.8g}" for lam in conditional.lambdas)
            + f"; {sum(s.get('nodes', 0) for s in searches)} beliefs in the searches [{conditional.statistics['seconds']:.2f}s]"
        )
        status = "(proved)" if optimal else "(the search did not finish: the optimum is at least this)"
        logger.info(f"the optimal conditional probability is {value:.10g} {status}; the probability of the condition is {conditional.evidence:.6g}")
        exported.update(value=value, numerator=conditional.numerator, evidence=conditional.evidence, optimal=optimal, lambdas=conditional.lambdas)
        exported["statistics"] = conditional.statistics
    report_threshold(task, value, optimal)
    if verify:
        storm_value, consistent = verify_with_storm(problem, colored_mdp, task, decisions)
        logger.info(f"Storm: value of the policy {storm_value:.10g}, the policy is {'' if consistent else 'NOT '}color consistent")
        if abs(storm_value - value) > 10 * precision * max(1.0, abs(value)) or not consistent:
            logger.warning("WARNING: Storm does not confirm the policy")
    policy = describe(problem, colored_mdp, decisions) if show_policy or export_policy else []
    if show_policy:
        for step, assignment in enumerate(policy):
            logger.info(f"step {step + 1}: " + ", ".join(f"{name}={option}" for name, option in assignment.items()))
    if export_policy:
        with open(export_policy, "w") as file:
            json.dump(exported | {"policy": policy}, file, indent=2, default=float)
        logger.info(f"policy written to {export_policy}")
    logger.info(f"done in {time.time() - started:.2f}s")


if __name__ == "__main__":
    main()
