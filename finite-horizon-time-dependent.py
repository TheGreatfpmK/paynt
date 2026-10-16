#!/usr/bin/env python3
"""Time-dependent color consistency over a finite horizon.

Given a model, a property and a horizon H, the script builds the colored MDP C = (M, V, kappa) of the model using PAYNT (for a POMDP, the parameters are the
observations and the options are the actions, i.e. a color consistent policy is an observation-based one) and looks for a policy that is color consistent
*within each time step*, but is free to use a different parameter assignment in every step.

Usage (MODEL and PROPERTIES are loaded like by the paynt command line; see --help for the options):

    python finite-horizon-time-dependent.py models/tests/pomdp-maze/sketch.templ models/tests/pomdp-maze/sketch.props 10 --show-policy

Time unfolding. The MDP M is unfolded into states (s,t), t in [0,H], where (s,t) --a--> (s',t+1) with the probabilities of s --a--> s'. Only the pairs (s,t)
that are reachable from the initial state are built (on large models these are a small fraction of all (H+1)*|S| pairs). The coloring is reconstructed so that
colors of different time steps never conflict: the parameter p is replaced with the parameters p@1,...,p@H (the parameter t with the domain [1..H] is added to
each color) and the color (p,o) of (s,a) becomes the color (p@t,o) of ((s,t),a), so an inconsistency can only occur within a step.

Semantics. The horizon truncates the property:
  * P[F target]: the probability of reaching the target within H steps,
  * R[F target]: the expected reward collected until the target is reached or the horizon is exhausted, whichever happens first.

Algorithm.
  1. The unfolded MDP is solved as if there were no colors (backward induction over t). This gives the best value that any policy can achieve and a policy
     that is in general not color consistent.
  2. Repeat until the value stops improving: compute the probability rho(s,t) of reaching (s,t) under the current policy (initially the policy of 1.), then go
     backwards over the steps t = H-1,...,0: given the values V(.,t+1) of the new policy in the next step, pick the color consistent assignment of step t
     maximizing (minimizing) sum_s rho(s,t) * Q(s,t,a(s)), where Q(s,t,a) = r(s,a) + sum_s' P(s,a,s') V(s',t+1). The first pass yields the first color
     consistent policy, each following pass improves it.
  Since rho(s,t) is exactly the distribution of the policy that is used before the step t, the assignment of step t maximizes the value of the policy that is
  old before t and new from t on. The old assignment of step t is among the candidates (if the old policy is consistent), hence the value cannot decrease.

Conditional reachability. For a model that is unfolded in time already, i.e. acyclic with the time step as a part of its states and observations (e.g. the
product of a hidden Markov chain and a monitor that has been unrolled, see models/luko-models), a conditional property P[F "goal" || F "condition"] asks for
the policy that maximizes or minimizes the probability of reaching the goal given that the condition is reached. HORIZON is not given then, the layers of the
model are its time steps, and the policy is color consistent (e.g. observation-based) in the whole model. The goal states have to be condition states, and both
sets have to be absorbing. Since the probability is a ratio, the Bellman equations do not hold for it, but they do hold for the threshold question: Pr(F goal |
F condition) >= lambda iff Pr(F goal) - lambda*Pr(F condition) >= 0, and the latter is an ordinary expected reward (the goal states are worth 1-lambda, the
other condition states -lambda). So the iterations above are used with these terminal values, where lambda is the conditional probability of the current policy
(Dinkelbach's method); the conditional probability does not decrease. The unconstrained optimum (computed this way as well, and a bound for every color
consistent policy) is not a good first policy then, since it can be much higher than what a color consistent policy gets: the iterations start from the policy
that is uniformly random in every state as well, and the better result is kept. The result is checked by Storm.

Exact search (--dp). The iterations are a heuristic, in particular the policies that are a sequence of observations to be chosen layer by layer are hard for
them: improving a single layer does not get from one sequence to a very different one. With --dp the script also searches all color consistent policies, by
dynamic programming over the reach vectors of the layers (see ReachVectorDp, where the idea is explained). It finds the optimal policy and proves that no color
consistent policy satisfies a property that the heuristic does not satisfy, as long as the number of distinct reach vectors is not too large (the log shows it;
it is the number of beliefs for the models of monitor verification). Without --dp nothing of this is done.

  python finite-horizon-time-dependent.py models/luko-models/01-airportB7-h9-fn/model.drn models/luko-models/01-airportB7-h9-fn/property.props --dp

Windows (--window N, --window-changes K). The backward pass changes one step (a layer, for a conditional property) at a time, so it stops in a policy in which
no single step can be improved given the others. That is not necessarily optimal: two steps (or two parameters of a step) may have to change together, e.g. a
step makes a state reachable that only pays off if the next step treats it in a special way, and each change alone makes the policy worse. With --window N, when
the backward passes stop improving, windows of 2, 3, ..., N consecutive steps are tried, one size after the other. A window is a start step and a size. It is
explored from a *seed*: the parameter of a component is forced to take an option that it does not have in the policy, at the first step of the window (with
--window-changes K a seed has up to K such changes: the others are in components that the earlier changes of the seed make reachable, in the first step or in a
later one). The other components of the window respond to the changes: they are optimized by the backward pass over the window, in which the weights of the
steps after the first one are the reachability probabilities that the changed beginning of the window induces (the steps after the window keep their policy).
The weights depend on the responses, so this is repeated a few times (WINDOW_ROUNDS). The steps before the window are not changed, so the value of the window is
exact: it is the difference of the values of the policy with the changed window and of the old policy, weighted with the probabilities of entering the window
(for conditional properties the objective N - lambda*D of the iterations above, with lambda the conditional probability of the policy, which is zero for the old
policy). The windows are tried from the last to the first, and the best seed of the first window with an improvement is applied. After that the backward passes
run again (and the windows start from the size 2 again).

The value of the policy never decreases and every window that is applied improves it, so the procedure terminates (the policies are finitely many); the result
is a policy that no window of up to N steps, with up to K changes at its beginning, improves. A window is not solved exactly (this would be as expensive as the
whole problem is for N the horizon): the number of seeds does not depend on N but grows with the K-th power of the number of parameter-option pairs that can be
reached, so the cost of a window grows linearly with N and fast with K. A larger N continues the search of a smaller one (the windows of the smaller sizes are
tried first), so the value does not decrease with N. With the default --window 1 only the backward passes are run; --timeout limits the search of windows as
well.

Experience. On 99 benchmark POMDPs (horizons 4 and 8) in which AR finds the optimum, the backward passes alone are optimal on 83 of them. Windows of up to 3
steps that change a parameter (K=1) are on 92, with up to two parameters (K=2) on 98 (the remaining one is 1.1% from the optimum, and neither windows of more
steps nor K=3 change that); all of them within 3 s. The windows that were applied had 2 or 3 steps (17 and 6 of them) and 1 or 2 changes (8 and 15), so a window
does not need to be long, it needs the right pair of changes: --window 3 --window-changes 2 is a reasonable setting. After a window the backward passes did not
improve the policy any further. On large models (e.g. rocks, network) the windows rarely help and cost much more, and on the conditional models of runtime
monitoring (see models/luko-models) they improve the heuristic (on one of them to the optimum) but not much further: the objective is flat there, since a
changed observation makes the evidence of the policy vanish, so the optimal observation word cannot be found by changing a few positions of the current one, and
--dp is the way to the optimum.

Why the backward passes get stuck can be seen in finite-horizon-landscape.py: for small models it evaluates every policy, and draws the values of the policies
as an interactive landscape in which two policies are neighbours if they differ in a single option. The fixed points of the backward passes are exactly the
policies without a better neighbour (the local maxima), because a pass optimizes the options of a step exactly given the others and the cells of a step are
independent. The traps that it shows are a policy that the unconstrained policy lures into (it uses what the real agent cannot see), cells that no policy of the
iterations visits and that are therefore not optimized, and needles in a flat landscape; the windows above and the value vectors below are moves that leave such
a trap.

Value vectors (--alpha-vectors N). The backward pass keeps a single value vector per step: the values that the states of the step have under the suffix of the
policy (the policy from the step on). The suffix is the best one for the reach vector of the policy that is being improved, i.e. for the distribution over the
states of the step that the steps before it lead to, and the options of a step are chosen for this one suffix. That is where the pass is stuck: the steps before
could lead to other states, for which another suffix would be better, and the options of the steps before would be better with it, but neither is better alone.
The value of a policy is the inner product of the reach vector of a step and the value vector of the step (plus what the steps before it have collected), as the
value of a policy of a POMDP is the inner product of a belief and an alpha vector. The backward pass is value iteration with a single belief in each step, and
the remedy is that of point based value iteration: several beliefs, with a value vector for each. With --alpha-vectors N, when the backward passes stop
improving (before the windows are tried), a *pooled* backward pass is run for N policies, the *pool*: the policy, the unconstrained optimum (the policy that the
first backward pass is based on, which leads to the states that a policy that is not color consistent would use), and N-2 changes of the policy: first every
option that a component that can be reached in a step does not have, from the first step to the last, then the changes of two components, and so on up to
--alpha-changes K components (2 by default; as in the seeds of the windows the other components are the ones that the earlier changes reach, in the same step or
a later one). Their reach vectors in a step are the *anchors* of the step (the ones that are the same up to a factor are one anchor, since they weight the
options in the same way). From the last step to the first, the pass computes a value vector for each anchor of the step: the value of the best suffix for the
anchor that the pool of the next step has, with the options of the step that are the best for it (as in the backward pass, the choices of a component are
weighted by the reach vector of the anchor). The suffixes are searched among *all* the vectors of the next step, not only the one that was computed for the same
policy, and that is what the pass does that the backward passes do not: a step can use the suffix of another policy, so the beginning of a policy and the end of
another one are put together. A vector is the value of a concrete color consistent suffix policy, so the vector of the initial state is the exact value of the
policy that the pass ends with. If it is better than the policy, it is the result and the backward passes continue from it. Otherwise the policies of the
changes respond: each of them keeps its steps up to its first change and takes the suffix of the vector of its anchor in the next step, and the pass is run
again (at most ALPHA_ROUNDS times, as long as some of them change). Example: in the policy that the backward passes get stuck in on models/fh-counterexample, a
better first step leads to a state for which the better last step is bad, and the other way around. The unconstrained optimum and the policy lead to the same
states in the second step, so the pool of the second step has one suffix; the change of the first step leads to different states and adds the suffix that is
good for them, and the first step can then use it.

The policy is in the pool and is preferred where it is as good as another one, so the value does not decrease, and it does not decrease with more anchors either
(there are more suffixes to choose from). With a single anchor in each step the pass is the backward pass, and if all the reach vectors of the model are anchors
it is exact: that is the dynamic programming of --dp, in which the beginnings that leave the same reach vector are the same state. So N is a dial between the
heuristic and the exact search; what it buys is the reach vectors that the changes make available, and the pool is a sample of them (the number of components
that a change of the policy has is limited, and the problem is NP-hard, so for any pool of a polynomial size there are models that have the optimum only behind
a change that it does not have: e.g. a choice in the first step in front of two copies of a trap, which pays off only if both traps are left, three changes that
the pool of pairs does not have however large N is, and the pool of the changes of three does, --alpha-changes 3). The cost of a pass is that of all the pairs
of an anchor and a suffix in a step, so it grows with the square of N and linearly with the model (the reach vectors of all the policies are computed together,
a step with one sparse matrix product, and the pairs are matrix products per component). The memory is that of the N value vectors of a step and of the values
of their choices: a pool that would take more than ALPHA_MEMORY is reduced (the log says so). The windows are tried after the pool (if --window is given), and
the value vectors need no conditional property (--alpha-vectors is not supported for them).

Experience (the benchmarks of the windows above). On the 99 benchmark POMDPs the backward passes alone are not optimal on 16. The windows of up to 3 steps with
2 changes make 15 of them optimal in 14 s together, 64 value vectors 12 of them in 0.3 s, 256 vectors 13 in under 1 s, and 1024 vectors 15 in about 3 s (the
remaining one is the same as for the windows, 1.1% from the optimum); the windows after the vectors (--alpha-vectors 256 --window 3 --window-changes 2) are 15
as well, so the vectors do not take anything from the windows. On the random colored MDPs of the tests (cyclic, with states without colors, in which the
backward passes are not optimal: 31 of 3000) 64 vectors find the optimum of all of them, the windows of 26, and on random layered POMDPs (84 stuck models) 256
vectors find the optimum of all, the windows of 79. A model in which the optimum needs the change of one parameter that makes the other parameters respond is
not the only trap that the pool finds, but it is where it is strongest: it considers all the single changes at once, and the windows evaluate them one by one
with their own response. On large models (e.g. network-prio-2-8-20 with 19k states, in which the windows of 2 steps find nothing) 16 vectors improve the policy
almost as much as the windows of 3 steps with 2 changes do (24.59 to 24.6487, against 24.6489), in 4 s instead of 93 s, but where a step has many states per
component the cost grows quickly with N (maze-10: 8 s with 64 vectors, 51 s with 256), so N is of the order of 16-64 for large models and of 256-1024 for small
ones.

Colorings. Every state of the colored MDP has to be either free (no colors, i.e. the choice is not restricted by any parameter) or colored, where all choices
of the state mention the same parameters and cover all their combinations of options. This is how the colorings of POMDPs (also with memory) look like, but
the colorings of e.g. decision trees (which are not a set of parameter-option pairs) or of multi-agent problems (whose states depend on different sets of
parameters) are not supported.

The model checking needed for the time unfolding is done with numpy on the layered structure (the time-unfolded MDP is just H copies of the MDP), while the
unfolded colored MDP itself is built by payntbind. It is used to double check the results with Storm and, with --ar, to run the standard PAYNT AR synthesis on
it. AR searches the whole parameter space V x [1..H], so it finds the optimal time-dependent policy (or proves that no policy satisfies a constraint) to compare
the heuristic with, but it only scales to small models: the full coloring that it needs takes memory quadratic in the horizon.

Warm started AR (--ar-warm-start). AR discards a family of assignments when the best value that any of its policies can have (the optimum of the MDP of the
family, in which the colors are not enforced) does not beat the best value known so far, and it starts without one. With --ar-warm-start the value of the policy
of the backward passes (with the windows and the value vectors, if --window or --alpha-vectors is given: a better policy prunes more) is the optimum so far from
the beginning, so AR prunes the families that cannot beat it right away and only looks for better policies. If there are none, which AR then proves (the log
says so), the policy of the heuristic is optimal and it is the result of AR; otherwise AR finds the optimum, as it does without the option. Only a property to
optimize has an optimum to start with. Experience. On the 100 small benchmark POMDPs (horizons 4 and 8) in which AR finishes with and without the option, it
finds the same optimum and never checks more families (the log shows their number). In 45 of them the heuristic is as good as the unconstrained optimum, which
bounds all policies, so AR proves it with a single family (it needed up to 47). In the other 55 the number of families is 9% smaller in total (12% with --window
3 --window-changes 2; the time is 14% smaller), by 8% in the median case, and by 47% in the best one (network, H=8). Most of the work of AR is the proof that
nothing beats the optimum: every family whose bound is better than the optimum has to be refined whatever the optimum so far is, so the option cannot shorten
it, and AR finds good policies early anyway. It matters most where AR would not finish: it does with the option on the two hallway H=8 models (in 55 s, while it
does not in 60 s without it).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import psutil
import rich_click as click
import scipy.sparse
import scipy.sparse.csgraph
import stormpy

import payntbind

import paynt.cli
import paynt.colored_mdp
import paynt.model.model
import paynt.parameter_space.parameter_space
import paynt.parser.sketch
import paynt.specification.property
import paynt.synthesizer.synthesizer
import paynt.task
import paynt.utils.version_check

if not hasattr(payntbind.synthesis, "FiniteHorizonUnfolder"):
    raise ImportError("the installed payntbind does not support finite horizons, rebuild it with: pip install . --no-build-isolation")

logger = logging.getLogger("finite-horizon")

# relative tolerance within which two candidate assignments are considered equally good
TIE_TOLERANCE = 1e-13

# the number of times that the steps of a window adapt to each other, see FiniteHorizonSolver.window_gain
WINDOW_ROUNDS = 3

# the pooled backward pass (--alpha-vectors, see FiniteHorizonSolver.search_alpha): the default number of changes of a policy of the pool (--alpha-changes; the
# smaller changes come first), the number of rounds in which the policies of the pool respond to each other, the number of anchors that are handled at once, and
# the memory (bytes) that the value vectors may take
ALPHA_CHANGES = 2
ALPHA_ROUNDS = 6
ALPHA_BLOCK = 256
ALPHA_MEMORY = 512 * 2**20


class UnsupportedProblemError(Exception):
    """The model or the property is outside of what this script can solve."""


# ----------------------------------------------------------------------------------------------------
# the problem: property and the arrays describing the colored MDP
# ----------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Objective:
    """The property to optimize over the horizon."""

    prop: Any
    maximize: bool
    is_reward: bool
    target_label: str
    reward_name: str | None

    @property
    def target_value(self) -> float:
        """Value of a state once the target is reached."""
        return 0.0 if self.is_reward else 1.0

    @classmethod
    def from_task(cls, task: Any) -> Objective:
        specification = task.specification
        if specification.num_properties != 1:
            raise UnsupportedProblemError(f"expecting exactly one property, found {specification.num_properties}: {specification}")
        prop = specification.all_properties()[0]
        if prop.has_game_formula or not isinstance(prop.formula.subformula, stormpy.logic.EventuallyFormula):
            raise UnsupportedProblemError(f"unsupported property {prop}: only P[F target] and R[F target] are supported (the horizon is a separate argument)")
        try:
            target_label = prop.get_target_label()
        except ValueError as e:
            raise UnsupportedProblemError(f"unsupported property {prop}: {e}") from e
        return cls(prop, prop.maximizing, prop.reward, target_label, prop.get_reward_name() if prop.reward else None)


@dataclasses.dataclass
class ConditionalObjective:
    """The conditional reachability probability Pr(F goal | F condition) of the property P[F "goal" || F "condition"], to be maximized or minimized.

    It is the ratio of two probabilities, so the optimal action of a state is not determined by the state alone (what is best depends on how likely the
    condition is overall), which is why the Bellman equations do not apply to it directly. ConditionalSolver gets around that. The goal states have to be
    condition states and both sets have to be absorbing, which LayeredMdp checks.
    """

    prop: Any
    maximize: bool
    goal_label: str
    condition_label: str

    # what MdpArrays.from_mdp expects of an objective: a probability, whose target states are the goal states
    is_reward = False
    reward_name = None

    @property
    def target_label(self) -> str:
        return self.goal_label

    @staticmethod
    def is_conditional(task: Any) -> bool:
        """Whether the task is to solve a single conditional property."""
        specification = task.specification
        if specification.num_properties != 1:
            return False
        prop = specification.all_properties()[0]
        return not prop.has_game_formula and isinstance(prop.formula.subformula, stormpy.logic.ConditionalFormula)

    @classmethod
    def from_task(cls, task: Any) -> ConditionalObjective:
        specification = task.specification
        if specification.num_properties != 1:
            raise UnsupportedProblemError(f"expecting exactly one property, found {specification.num_properties}: {specification}")
        prop = specification.all_properties()[0]
        if not cls.is_conditional(task):
            raise UnsupportedProblemError(f'expecting a conditional property P[F "goal" || F "condition"], found {prop}')
        if prop.reward:
            raise UnsupportedProblemError(f"unsupported property {prop}: conditional expected rewards are not supported")
        conditional = prop.formula.subformula
        labels = []
        for part in (conditional.main_subformula, conditional.conditional_subformula):
            if not isinstance(part, stormpy.logic.EventuallyFormula) or not isinstance(part.subformula, stormpy.logic.AtomicLabelFormula):
                raise UnsupportedProblemError(f'unsupported property {prop}: only P[F "goal" || F "condition"] with labels of states is supported')
            labels.append(part.subformula.label)
        return cls(prop, prop.maximizing, labels[0], labels[1])


def label_mask(mdp: Any, label: str, what: str = "states") -> np.ndarray:
    """Mask of the states with the label."""
    if not mdp.labeling.contains_label(label):
        raise UnsupportedProblemError(f"the model has no label '{label}' of the {what}")
    mask = np.zeros(mdp.nr_states, dtype=bool)
    mask[np.fromiter(mdp.labeling.get_states(label), dtype=np.int64)] = True
    return mask


@dataclasses.dataclass
class MdpArrays:
    """The MDP, as arrays: choices (rows) of a state s are row_groups[s]..row_groups[s+1]-1, the successor distribution of a choice is a row of the matrix."""

    row_groups: np.ndarray
    choice_state: np.ndarray
    matrix: scipy.sparse.csr_matrix
    reward: np.ndarray
    target: np.ndarray
    initial_state: int

    @property
    def num_states(self) -> int:
        return len(self.row_groups) - 1

    @property
    def num_choices(self) -> int:
        return self.matrix.shape[0]

    @classmethod
    def from_mdp(cls, mdp: Any, objective: Objective) -> MdpArrays:
        num_states, num_choices = mdp.nr_states, mdp.nr_choices
        row_pointers, columns, values = payntbind.synthesis.sparseMatrixToCsr(mdp.transition_matrix)
        if np.any(np.diff(row_pointers) == 0):
            raise UnsupportedProblemError("the MDP has a choice without successors")
        matrix = scipy.sparse.csr_matrix((values, columns, row_pointers), shape=(num_choices, num_states))
        row_groups = np.asarray(mdp.nondeterministic_choice_indices, dtype=np.int64)
        choice_state = np.repeat(np.arange(num_states, dtype=np.int64), np.diff(row_groups))

        reward = np.zeros(num_choices)
        if objective.is_reward:
            name = objective.reward_name
            reward_models = mdp.reward_models
            if name not in reward_models and not name and len(reward_models) == 1:
                name = next(iter(reward_models))
            if name not in reward_models:
                raise UnsupportedProblemError(f"the model has no reward model '{name}'")
            reward_model = reward_models[name]
            if reward_model.has_state_rewards or reward_model.has_transition_rewards:
                raise UnsupportedProblemError(f"reward model '{name}' is not a state-action reward model")
            if reward_model.has_state_action_rewards:
                reward = np.asarray(reward_model.state_action_rewards, dtype=np.float64)

        target = label_mask(mdp, objective.target_label, "target states")

        initial_states = list(mdp.initial_states)
        if len(initial_states) != 1:
            logger.warning(f"WARNING: the model has {len(initial_states)} initial states, using the first one")
        return cls(row_groups, choice_state, matrix, reward, target, initial_states[0])


@dataclasses.dataclass
class ColorStructure:
    """Structure of a coloring that makes the best color consistent assignment of a time step easy to find.

    Parameters used together by some state form a component (e.g. A(obs,mem) and M(obs,mem) of a POMDP unfolded with memory). All states of a component depend
    on all of its parameters, so their choices correspond one-to-one to the joint options of the component. The joint options of all components are numbered
    consecutively: the joint option j of the component g has the global index comp_offset[g]+j.
    """

    num_holes: int
    hole_num_options: np.ndarray
    # for each parameter, its component (-1 if no choice depends on it) and the factor of its option in the joint option of the component
    hole_component: np.ndarray
    hole_stride: np.ndarray
    comp_size: np.ndarray
    comp_offset: np.ndarray
    # colored choices, the component they depend on and their global joint option
    colored_choices: np.ndarray
    choice_component: np.ndarray
    choice_joint: np.ndarray

    @property
    def num_components(self) -> int:
        return len(self.comp_size)

    @property
    def num_joint_options(self) -> int:
        return int(self.comp_offset[-1])

    @classmethod
    def analyze(cls, mdp: MdpArrays, coloring: Any, hole_num_options: np.ndarray) -> ColorStructure:
        """Check that the coloring has the structure described at the top of this file and index its joint options."""
        num_choices = mdp.num_choices
        num_holes = len(hole_num_options)
        pair_pointers, pair_hole, pair_option = payntbind.synthesis.coloringToCsr(coloring)
        choice_num_pairs = np.diff(pair_pointers)
        choice_colored = choice_num_pairs > 0
        state_first = mdp.row_groups[:-1]
        state_num_choices = np.diff(mdp.row_groups)

        def state_count(choice_mask: np.ndarray) -> np.ndarray:
            return np.add.reduceat(choice_mask.astype(np.int64), state_first)

        num_colored = state_count(choice_colored)
        state_colored = num_colored > 0
        if not state_colored.any():
            return cls(
                num_holes, hole_num_options, np.full(num_holes, -1, dtype=np.int64), np.ones(num_holes, dtype=np.int64),
                np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64), *(np.zeros(0, dtype=np.int64) for _ in range(3)),
            )  # fmt: skip
        mixed = state_colored & (num_colored != state_num_choices)
        if mixed.any():
            raise UnsupportedProblemError(f"state {int(np.flatnonzero(mixed)[0])} (and {int(mixed.sum()) - 1} more) has both colored and uncolored choices")

        # order the pairs of each choice by hole
        pair_choice = np.repeat(np.arange(num_choices), choice_num_pairs)
        order = np.lexsort((pair_hole, pair_choice))
        pair_hole, pair_option = pair_hole[order], pair_option[order]
        if np.any((pair_choice[1:] == pair_choice[:-1]) & (pair_hole[1:] == pair_hole[:-1])):
            raise UnsupportedProblemError("a choice assigns more than one option to a parameter")
        if np.any(pair_option >= hole_num_options[pair_hole]):
            raise UnsupportedProblemError("the coloring uses an option outside of the domain of a parameter")

        # all choices of a state have to depend on the same parameters
        choice_ref = mdp.row_groups[mdp.choice_state]
        pair_position = np.arange(len(pair_hole)) - pair_pointers[pair_choice]
        pair_ref = pair_pointers[choice_ref[pair_choice]] + pair_position
        pair_same_hole = (pair_position < choice_num_pairs[choice_ref[pair_choice]]) & (pair_hole[np.minimum(pair_ref, len(pair_hole) - 1)] == pair_hole)
        choice_different = (choice_num_pairs != choice_num_pairs[choice_ref]) | (np.bincount(pair_choice, weights=~pair_same_hole, minlength=num_choices) > 0)
        different = state_count(choice_different & choice_colored) > 0
        if different.any():
            raise UnsupportedProblemError(f"choices of state {int(np.flatnonzero(different)[0])} depend on different sets of parameters")

        # parameters used together by a state form a component
        used = np.zeros(num_holes, dtype=bool)
        used[pair_hole] = True
        same_choice = pair_choice[1:] == pair_choice[:-1]
        edges = scipy.sparse.coo_matrix(
            (np.ones(int(same_choice.sum())), (pair_hole[:-1][same_choice], pair_hole[1:][same_choice])), shape=(num_holes, num_holes)
        )
        _, hole_label = scipy.sparse.csgraph.connected_components(edges, directed=False)
        component_labels, hole_component_used = np.unique(hole_label[used], return_inverse=True)
        num_components = len(component_labels)
        hole_component = np.full(num_holes, -1, dtype=np.int64)
        hole_component[used] = hole_component_used
        comp_num_holes = np.bincount(hole_component[used], minlength=num_components)

        colored_choices = np.flatnonzero(choice_colored)
        choice_component = hole_component[pair_hole[pair_pointers[colored_choices]]]
        states_first_choice = state_first[state_colored]
        state_component = hole_component[pair_hole[pair_pointers[states_first_choice]]]
        if np.any(choice_num_pairs[states_first_choice] != comp_num_holes[state_component]):
            state = int(states_first_choice[np.flatnonzero(choice_num_pairs[states_first_choice] != comp_num_holes[state_component])[0]])
            raise UnsupportedProblemError(f"state {state} depends on only some of the parameters that other states of its component depend on")

        # index of the joint option of each choice, as a number in the mixed radix given by the domains of the parameters of the choice
        domain = hole_num_options[pair_hole]
        choice_joint_size = np.multiply.reduceat(domain.astype(np.float64), pair_pointers[colored_choices])
        if np.any(choice_joint_size > 2.0**52):
            raise UnsupportedProblemError("a state depends on too many parameters")
        pair_stride = np.ones(len(pair_hole), dtype=np.int64)
        for position in range(int(choice_num_pairs.max()) - 2, -1, -1):
            start = pair_pointers[np.flatnonzero(choice_num_pairs > position + 1)] + position
            pair_stride[start] = pair_stride[start + 1] * domain[start + 1]
        choice_cell = np.add.reduceat(pair_option * pair_stride, pair_pointers[colored_choices])

        # every joint option of the parameters has to be covered by exactly one choice of each state
        state_joint_size = choice_joint_size[np.searchsorted(colored_choices, states_first_choice)].astype(np.int64)
        incomplete = state_num_choices[state_colored] != state_joint_size
        if incomplete.any():
            raise UnsupportedProblemError(
                f"choices of state {int(np.flatnonzero(state_colored)[np.flatnonzero(incomplete)[0]])} do not cover all options of its parameters"
            )
        choice_order = np.lexsort((choice_cell, mdp.choice_state[colored_choices]))
        sorted_state, sorted_cell = mdp.choice_state[colored_choices][choice_order], choice_cell[choice_order]
        if np.any((sorted_state[1:] == sorted_state[:-1]) & (sorted_cell[1:] == sorted_cell[:-1])):
            raise UnsupportedProblemError("two choices of a state have the same color")

        comp_size = np.zeros(num_components, dtype=np.int64)
        comp_size[state_component] = state_joint_size
        comp_offset = np.concatenate(([0], np.cumsum(comp_size)))
        hole_stride = np.ones(num_holes, dtype=np.int64)
        hole_stride[pair_hole] = pair_stride
        return cls(
            num_holes,
            hole_num_options,
            hole_component,
            hole_stride,
            comp_size,
            comp_offset,
            colored_choices,
            choice_component,
            comp_offset[choice_component] + choice_cell,
        )

    def decode(self, joint_options: np.ndarray) -> np.ndarray:
        """Convert the best joint option of each component (as in the solver's output, with shape (steps,components)) to the option of each parameter."""
        options = np.zeros((joint_options.shape[0], self.num_holes), dtype=np.int64)
        used = self.hole_component >= 0
        options[:, used] = (joint_options[:, self.hole_component[used]] // self.hole_stride[used]) % self.hole_num_options[used]
        return options

    def best_joint_options(self, weighted_values: np.ndarray, previous: np.ndarray | None, scale: np.ndarray | None = None) -> np.ndarray:
        """For each component, find the joint option with the highest weighted value (weighted_values has an entry for each joint option).

        Joint options whose values are within a tolerance of the best one are equally good. Ties are broken in favor of the previous option (which keeps the
        iterations from switching between equally good policies), then by the lowest index. The tolerance is relative to the scale of the values of a component,
        which defaults to the best value; if the values are sums of terms with different signs, the scale has to be given because the best value can be much
        smaller than the terms (the floating-point noise is as big as the terms).
        """
        offsets = self.comp_offset[:-1]
        best = np.maximum.reduceat(weighted_values, offsets)
        tolerance = TIE_TOLERANCE * (np.abs(best) if scale is None else scale)
        candidates = weighted_values >= np.repeat(best - tolerance, self.comp_size)
        if previous is not None and np.count_nonzero(candidates) > self.num_components:
            previous_mask = np.zeros_like(candidates)
            previous_mask[offsets + previous] = True
            previous_mask &= candidates
            candidates = np.where(np.repeat(np.logical_or.reduceat(previous_mask, offsets), self.comp_size), previous_mask, candidates)
        return np.minimum.reduceat(np.where(candidates, np.arange(self.num_joint_options), self.num_joint_options), offsets) - offsets


# ----------------------------------------------------------------------------------------------------
# solver
# ----------------------------------------------------------------------------------------------------


class SegmentArgmax:
    """Pick the first best item in each contiguous segment of items, i.e. a vectorized argmax over the choices of states."""

    def __init__(self, segment_starts: np.ndarray, item_segment: np.ndarray, items: np.ndarray | None = None):
        self.segment_starts = segment_starts
        self.item_segment = item_segment
        self.items = items

    def __call__(self, values: np.ndarray) -> np.ndarray:
        best = np.maximum.reduceat(values, self.segment_starts)
        position = np.flatnonzero(values == best[self.item_segment])
        segment = self.item_segment[position]
        first = np.empty(len(position), dtype=bool)
        first[:1] = True
        first[1:] = segment[1:] != segment[:-1]
        result = np.empty(len(self.segment_starts), dtype=np.int64)
        result[segment[first]] = position[first] if self.items is None else self.items[position[first]]
        return result


@dataclasses.dataclass
class Solution:
    """A policy: policy[t,s] is the choice executed in the state s in the step t (0-based) and value is its value.

    A color consistent policy additionally has the parameter assignment options[t,p] and the equivalent joint_options[t,g] (the joint option of each component).
    """

    policy: np.ndarray
    value: float
    options: np.ndarray | None = None
    joint_options: np.ndarray | None = None


# a change that a window makes: the index of the step (layer) in the window, the component and the option that it is forced to execute
Cell = tuple[int, int, int]


def window_seeds(first: Sequence[Cell], later: Callable[[tuple[Cell, ...]], Sequence[Cell]], changes: int) -> Iterator[tuple[Cell, ...]]:
    """The seeds of a window, i.e. the changes that are forced at its beginning (depth first): a seed is a change in the first step (or layer) of the window and
    at most changes-1 more, each of them in a part of the policy that the changes before it reach.

    :param first: the changes that a seed starts with
    :param later: for a seed, the changes that can be added to it
    """
    stack = [(cell,) for cell in reversed(first)]
    while stack:
        seed = stack.pop()
        yield seed
        if len(seed) < changes:
            stack.extend(seed + (cell,) for cell in reversed(later(seed)))


@dataclasses.dataclass
class Anchors:
    """The reach vectors of some policies, i.e. the distributions over the states that they induce in each step (see FiniteHorizonSolver.search_alpha).

    rows[t] has a row for each of the distinct distributions in the step t (a sparse matrix), and index[p][t] is the row of the policy p in the step t (-1 if
    the policy reaches no state that has a choice to make in the step).
    """

    rows: list[scipy.sparse.csr_matrix]
    index: list[list[int]]


@dataclasses.dataclass
class AlphaPool:
    """The result of a pooled backward pass: the value vectors of the steps, as the suffix policies that they are the values of.

    The vector j of the step t is the value of the suffix policy that executes the joint options joints[t][j] in the step t and then the suffix policy of the
    vector parents[t][j] of the step t+1. values are the value vectors of the step 0 (the value of the vector of the initial state is the value of the policy).
    """

    parents: list[np.ndarray]
    joints: list[np.ndarray]
    values: np.ndarray

    def suffix(self, step: int, vector: int) -> np.ndarray:
        """The joint options of the steps step..H-1 of the suffix policy of the vector of the step (as rows)."""
        rows = []
        for t in range(step, len(self.parents)):
            rows.append(self.joints[t][vector])
            vector = int(self.parents[t][vector])
        return np.array(rows)


@dataclasses.dataclass
class ColoredLayout:
    """The colored states grouped by component: the states states[start[g]:start[g+1]] depend on the component g, and choices[i,o] is the choice that states[i]
    executes if its component takes the joint option o (-1 if the component has fewer options). The states without colors and with a single choice are listed
    as well, since a step executes their choice whatever the options are."""

    states: np.ndarray
    component: np.ndarray
    start: np.ndarray
    choices: np.ndarray
    single_states: np.ndarray
    single_choices: np.ndarray


class FiniteHorizonSolver:
    """Backward induction and backward iteration on the layered structure of the time-unfolded colored MDP.

    All time steps share the same MDP, so the arrays describe a single layer and the unfolding is the loop over the steps t = 0..H-1. A policy is the choice of
    each state in each step. The reachability probabilities and values are always the ones of the first H steps, i.e. the layers 0..H. The unfolded model is
    never materialized here, so a step handles all states at once with vectorized operations, including those that are not reachable in it (they have no weight
    and nothing reachable depends on their values); this is cheaper than gathering the reachable ones and needs memory only for a layer of the MDP.
    """

    def __init__(self, mdp: MdpArrays, colors: ColorStructure, horizon: int, objective: Objective):
        self.mdp = mdp
        self.colors = colors
        self.horizon = horizon
        self.objective = objective
        self.sign = 1.0 if objective.maximize else -1.0
        self.matrix_transposed = mdp.matrix.T.tocsr()
        self.nontarget = ~mdp.target
        self.terminal = np.where(mdp.target, objective.target_value, 0.0)
        self.policy_dtype = np.int32 if mdp.num_choices < 2**31 else np.int64

        self.argmax_all = SegmentArgmax(mdp.row_groups[:-1], mdp.choice_state)
        # states without any restriction on their choices that still have a choice to make
        state_colored = np.zeros(mdp.num_states, dtype=bool)
        state_colored[mdp.choice_state[colors.colored_choices]] = True
        free_nondeterministic = ~state_colored & (np.diff(mdp.row_groups) > 1)
        self.free_states = np.flatnonzero(free_nondeterministic)
        self.free_choices = np.flatnonzero(free_nondeterministic[mdp.choice_state])
        free_sizes = np.diff(mdp.row_groups)[self.free_states]
        self.argmax_free = SegmentArgmax(np.cumsum(free_sizes) - free_sizes, np.repeat(np.arange(len(self.free_states)), free_sizes), self.free_choices)
        # the choice made by default (and in states with a single choice)
        self.default_policy = mdp.row_groups[:-1].astype(self.policy_dtype)

        self.colored_state = mdp.choice_state[colors.colored_choices]
        # what the search with windows has done: the improving windows, and for each size the candidates tried and the time spent
        self.window_moves: list[dict[str, Any]] = []
        self.window_stats: dict[int, dict[str, float]] = {}
        # what the pooled backward passes have done (see search_alpha): the improving ones, and the number of attempts, passes, policies and the time spent
        self.alpha_moves: list[dict[str, Any]] = []
        self.alpha_stats: dict[str, float] = {}
        self._layout: ColoredLayout | None = None
        self._alpha_reduced = False

    # --- values ---

    def _choice_values(self, values: np.ndarray) -> np.ndarray:
        return self.mdp.reward + self.mdp.matrix @ values

    def solve_unconstrained(self) -> Solution:
        """Optimal policy of the unfolded MDP, ignoring the colors."""
        policy = np.empty((self.horizon, self.mdp.num_states), dtype=self.policy_dtype)
        values = self.terminal.copy()
        for step in range(self.horizon - 1, -1, -1):
            choice_values = self._choice_values(values)
            policy[step] = self.argmax_all(self.sign * choice_values)
            values = choice_values[policy[step]]
            values[self.mdp.target] = self.terminal[self.mdp.target]
        return Solution(policy, float(values[self.mdp.initial_state]))

    def _propagate(self, reach_row: np.ndarray, step_policy: np.ndarray) -> np.ndarray:
        """The reachability probabilities of the states in the next step, when the choices of the step are executed."""
        executed = np.zeros(self.mdp.num_choices)
        executed[step_policy] = reach_row * self.nontarget
        return self.matrix_transposed @ executed

    def reach_probabilities(self, policy: np.ndarray) -> np.ndarray:
        """Probability rho[t,s] of reaching the state s in the step t (for the first time if s is a target, which is absorbing) under the policy."""
        reach = np.zeros((self.horizon + 1, self.mdp.num_states))
        reach[0, self.mdp.initial_state] = 1.0
        for step in range(self.horizon):
            reach[step + 1] = self._propagate(reach[step], policy[step])
        return reach

    def values_of(self, policy: np.ndarray) -> np.ndarray:
        """The values[t,s] of the states s in the steps t = 0..H under the policy; the values of the last step are the terminal values."""
        values = np.empty((self.horizon + 1, self.mdp.num_states))
        values[self.horizon] = self.terminal
        for step in range(self.horizon - 1, -1, -1):
            values[step] = self._choice_values(values[step + 1])[policy[step]]
            values[step][self.mdp.target] = self.terminal[self.mdp.target]
        return values

    def evaluate(self, policy: np.ndarray) -> float:
        """Value of the policy, computed forward from the reachability probabilities (as opposed to the backward computation of the solver)."""
        reach = self.reach_probabilities(policy)
        if not self.objective.is_reward:
            return float(reach[:, self.mdp.target].sum())
        return float(sum(np.dot(reach[step] * self.nontarget, self.mdp.reward[policy[step]]) for step in range(self.horizon)))

    # --- color consistent assignments ---

    def _decide_step(
        self,
        values: np.ndarray,
        reach_row: np.ndarray | None,
        previous_joint: np.ndarray | None,
        fixed_joint: np.ndarray | None = None,
        forced: Sequence[tuple[int, int]] = (),
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """One step of the backward pass: the best choices of the states of the step given the values of the states in the next step.

        :param reach_row: the weights of the states in the step, i.e. their reachability probabilities (unused if fixed_joint is given)
        :param previous_joint: the joint option of each component in the previous solution (used to prefer it over the equally good ones)
        :param fixed_joint: the joint options to execute, if they are not to be optimized (the choices of the free states are optimized nevertheless)
        :param forced: components and their joint options to execute whatever their weighted values are
        :returns: the choice of each state, the joint option of each component (None if there are no components) and the values of the states of the step
        """
        mdp, colors = self.mdp, self.colors
        choice_values = self._choice_values(values)
        signed_values = self.sign * choice_values
        step_policy = self.default_policy.copy()
        if len(self.free_states):
            step_policy[self.free_states] = self.argmax_free(signed_values[self.free_choices])
        best = None
        if colors.num_components:
            if fixed_joint is not None:
                best = fixed_joint
            else:
                # value of each joint option: the values of its choices weighted by the reachability of their states
                weighted_values = np.bincount(
                    colors.choice_joint,
                    weights=(reach_row * self.nontarget)[self.colored_state] * signed_values[colors.colored_choices],
                    minlength=colors.num_joint_options,
                )
                best = colors.best_joint_options(weighted_values, previous_joint)
                for component, option in forced:
                    best[component] = option
            self._execute_joint_options(step_policy, best)
        new_values = choice_values[step_policy]
        new_values[mdp.target] = self.terminal[mdp.target]
        return step_policy, best, new_values

    def _execute_joint_options(self, step_policy: np.ndarray, joint_row: np.ndarray) -> None:
        """Let the colored states execute the joint options of their components."""
        colors = self.colors
        chosen = colors.choice_joint == (colors.comp_offset[:-1] + joint_row)[colors.choice_component]
        step_policy[self.colored_state[chosen]] = colors.colored_choices[chosen]

    def improve(self, reach: np.ndarray | None, previous: Solution | None = None, fixed: np.ndarray | None = None) -> Solution:
        """Backward pass: the best color consistent policy of each step given the values of the new policy in the next step and the weights rho of the states.

        :param reach: weights of the states in each step, i.e. their reachability probabilities under the policy being improved (unused if fixed is given)
        :param previous: the previous color consistent solution (used to prefer its assignments over the equally good ones)
        :param fixed: the joint options to execute in each step, if they are not to be optimized (the choices of the free states are optimized nevertheless)
        """
        mdp, colors = self.mdp, self.colors
        policy = np.empty((self.horizon, mdp.num_states), dtype=self.policy_dtype)
        joint_options = np.empty((self.horizon, colors.num_components), dtype=np.int64)
        previous_joint = None if previous is None else previous.joint_options
        values = self.terminal.copy()
        for step in range(self.horizon - 1, -1, -1):
            policy[step], best, values = self._decide_step(
                values,
                None if fixed is not None else reach[step],
                None if previous_joint is None else previous_joint[step],
                None if fixed is None else fixed[step],
            )
            if colors.num_components:
                joint_options[step] = best
        return Solution(policy, float(values[mdp.initial_state]), colors.decode(joint_options), joint_options)

    def solution_of_options(self, options: np.ndarray) -> Solution:
        """The solution that executes the parameter assignment options[t,p] in each step t (the choices of the free states are optimized)."""
        colors = self.colors
        used = np.flatnonzero(colors.hole_component >= 0)
        incidence = scipy.sparse.csr_matrix((np.ones(len(used)), (used, colors.hole_component[used])), shape=(colors.num_holes, colors.num_components))
        joint_options = np.rint((incidence.T @ (options * colors.hole_stride).T).T).astype(np.int64)
        return self.improve(None, fixed=joint_options)

    # --- windows: several steps that are changed at the same time ---

    def _reaching_components(self, reach_row: np.ndarray) -> np.ndarray:
        """The components that have a state that can be reached in the step."""
        colors = self.colors
        weights = np.bincount(colors.choice_component, weights=(reach_row * self.nontarget)[self.colored_state], minlength=colors.num_components)
        return np.flatnonzero(weights > 1e-12)

    def _seeded_window(
        self, solution: Solution, start: int, size: int, seed: Sequence[Cell]
    ) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[np.ndarray]]:
        """The window of the steps start..start+size-1 when the changes of the seed are made and nothing else is: the joint options of its steps (as rows), the
        changes of each step as pairs of a component and its option, and the choices of the states in each step."""
        assert solution.joint_options is not None
        joint = solution.joint_options[start : start + size].copy()
        forced: list[list[tuple[int, int]]] = [[] for _ in range(size)]
        for index, component, option in seed:
            joint[index, component] = option
            forced[index].append((component, option))
        policies = []
        for index in range(size):
            step_policy = solution.policy[start + index].copy()
            self._execute_joint_options(step_policy, joint[index])
            policies.append(step_policy)
        return joint, forced, policies

    def _window_cells(self, solution: Solution, start: int, reaches: Sequence[np.ndarray], first_index: int = 0) -> list[Cell]:
        """The changes that can be forced in the steps of a window, from the step first_index on: each option that a component does not have in the solution,
        for each component that can be reached in the step (reaches[index] are the probabilities of reaching the states in the step of the window)."""
        assert solution.joint_options is not None
        return [
            (index, int(component), option)
            for index in range(first_index, len(reaches))
            for component in self._reaching_components(reaches[index])
            for option in range(int(self.colors.comp_size[component]))
            if option != solution.joint_options[start + index, component]
        ]

    def _later_cells(self, solution: Solution, reach: np.ndarray, start: int, size: int, seed: Sequence[Cell]) -> list[Cell]:
        """The changes that can be added to a seed of a window: what the changes of the seed reach in the window, where the other steps keep their policy (the
        added change is after the last one of the seed, and not in a component that it changes already)."""
        policies = self._seeded_window(solution, start, size, seed)[2]
        reaches = [reach[start]]
        for index in range(size - 1):
            reaches.append(self._propagate(reaches[index], policies[index]))
        taken = {cell[:2] for cell in seed}
        return [cell for cell in self._window_cells(solution, start, reaches, seed[-1][0]) if cell > seed[-1] and cell[:2] not in taken]

    def window_gain(self, solution: Solution, reach: np.ndarray, values: np.ndarray, start: int, size: int, seed: Sequence[Cell]) -> tuple[float, np.ndarray]:
        """The gain of the value of the solution from changing components at the beginning of a window, and the joint options of the window then.

        The backward pass changes one step, taking the weights of the states in it from the old policy: this is exact for the step, but the steps after it are
        optimized for the old weights as well, they cannot respond to the change of the step. A window of several steps changes them together: the options of
        some components (the seed: the step of the window, the component and its option; the first one is in the first step) are forced in the steps
        start..start+size-1, and the other components of the window respond to the changes. The weights of the states in the first step are those of the
        solution, which does not change before the window, but the weights in the next steps are the ones that the changed beginning of the window induces. The
        responses are found with the backward pass restricted to the window, and since the induced weights depend on the responses, this is repeated a few
        times. The steps after the window keep the policy of the solution. The gain is the exact difference of the values, since the part of the value that the
        steps before the window contribute does not change.

        :param reach: the reachability probabilities of the states under the solution
        :param values: the values of the states under the solution
        :returns: the gain (positive if the value improves) and the joint options of the steps of the window, each of them a row
        """
        assert solution.joint_options is not None
        joint, forced, policies = self._seeded_window(solution, start, size, seed)
        previous = solution.joint_options[start : start + size]
        new_values = values[start]
        for _ in range(WINDOW_ROUNDS):
            reaches = [reach[start]]
            for index in range(size - 1):
                reaches.append(self._propagate(reaches[index], policies[index]))
            new_values = values[start + size]
            new_joint = joint.copy()
            for index in reversed(range(size)):
                policies[index], new_joint[index], new_values = self._decide_step(new_values, reaches[index], previous[index], forced=forced[index])
            changed = bool(np.any(new_joint != joint))
            joint = new_joint
            if not changed:
                break
        return self.sign * float(np.dot(reach[start] * self.nontarget, new_values - values[start])), joint

    def search_windows(self, solution: Solution, size: int, tolerance: float, deadline: float | None = None, changes: int = 1) -> Solution | None:
        """Look for a window of the given size that improves the solution (see window_gain): every component that can be reached in the first step of a window,
        with every option that it does not have. The windows are tried from the last to the first, and the best of the first window that has an improvement is
        applied.

        :param changes: the maximum number of changes (a component with an option that it does not have) that are forced at the same time in the window: the
            first one in the first step, the others in the components that the changes before them reach (in the first step or later). Changing a single
            component is not enough if two components have to change together to improve the value (e.g. two observation classes that have to be told apart in
            the same step, or a state that is made reachable by a step and only pays off if the next step treats it in a special way), and a window of several
            steps does not help in this case.
        :returns: the improved solution, or None if no window of this size improves it (or the time is up)
        """
        colors = self.colors
        if size > self.horizon or not colors.num_components:
            return None
        statistics = self.window_stats.setdefault(size, {"candidates": 0, "moves": 0, "seconds": 0.0})
        started = time.time()
        reach, values = self.reach_probabilities(solution.policy), self.values_of(solution.policy)
        threshold = tolerance * max(1.0, abs(solution.value))
        assert solution.joint_options is not None
        try:
            for start in range(self.horizon - size, -1, -1):

                def later(seed: tuple[Cell, ...], start: int = start) -> list[Cell]:
                    return self._later_cells(solution, reach, start, size, seed)

                best = None
                for seed in window_seeds(self._window_cells(solution, start, [reach[start]]), later, changes):
                    if deadline is not None and time.time() > deadline:
                        return None
                    statistics["candidates"] += 1
                    gain, joint = self.window_gain(solution, reach, values, start, size, seed)
                    if gain > threshold and (best is None or gain > best[0]):
                        best = (gain, joint, seed)
                if best is None:
                    continue
                joint_options = solution.joint_options.copy()
                joint_options[start : start + size] = best[1]
                improved = self.improve(None, fixed=joint_options)
                gain = self.sign * (improved.value - solution.value)
                if gain > threshold:
                    statistics["moves"] += 1
                    self.window_moves.append(
                        {"window": size, "step": start + 1, "cells": [list(cell) for cell in best[2]], "gain": gain, "value": improved.value}
                    )
                    return improved
                logger.debug(f"the window of {size} steps at the step {start + 1} did not improve the value, the gain was estimated as {best[0]:.3g}")
            return None
        finally:
            statistics["seconds"] += time.time() - started

    def widen(self, solution: Solution, max_window: int, tolerance: float, deadline: float | None = None, changes: int = 1) -> Solution | None:
        """The solution improved by the smallest window (of at least two steps and at most max_window steps) that has an improvement, or None."""
        for size in range(2, max_window + 1):
            improved = self.search_windows(solution, size, tolerance, deadline, changes)
            if improved is not None or (deadline is not None and time.time() > deadline):
                return improved
        return None

    # --- value vectors: the suffixes of several policies are pooled ---

    def _colored_layout(self) -> ColoredLayout:
        """The colored states grouped by component (built when it is first needed)."""
        if self._layout is None:
            colors, mdp = self.colors, self.mdp
            option = colors.choice_joint - colors.comp_offset[colors.choice_component]
            order = np.lexsort((option, self.colored_state, colors.choice_component))
            component, state, option, choice = colors.choice_component[order], self.colored_state[order], option[order], colors.colored_choices[order]
            # every state of a component has a choice for each of its joint options (see ColorStructure.analyze): the choices of a state are consecutive
            first = np.ones(len(state), dtype=bool)
            first[1:] = (state[1:] != state[:-1]) | (component[1:] != component[:-1])
            choices = np.full((int(first.sum()), int(colors.comp_size.max()) if colors.num_components else 0), -1, dtype=np.int64)
            choices[np.cumsum(first) - 1, option] = choice
            states, state_component = state[first], component[first]
            colored = np.zeros(mdp.num_states, dtype=bool)
            colored[states] = True
            free = np.zeros(mdp.num_states, dtype=bool)
            free[self.free_states] = True
            single = np.flatnonzero(~colored & ~free)
            start = np.searchsorted(state_component, np.arange(colors.num_components + 1))
            self._layout = ColoredLayout(states, state_component, start, choices, single, mdp.row_groups[single])
        return self._layout

    def _pool_backup(self, vectors: np.ndarray, anchors: np.ndarray, previous_joint: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One step of the pooled backward pass: for each anchor, the best suffix among the value vectors of the next step and the best options for it.

        The weights of an anchor are the probabilities of reaching the states in the step. For a given suffix the best options are chosen as in _decide_step:
        the choices of a component are weighted by the reachability of their states and the best option is taken; the suffix for which this value is the
        highest is the best one. All the suffixes are evaluated for all the anchors at once (the loops are over the components that the anchors reach, the
        anchors and the suffixes are the dimensions of matrix products), the values of the choices of a suffix are shared by all the anchors.

        :param vectors: the value vectors of the states in the next step, one for each suffix (rows)
        :param anchors: the reach vectors in the step, one for each anchor (rows)
        :param previous_joint: the joint options of the step in the solution that is being improved (preferred if they are among the best ones)
        :returns: the index of the best suffix of each anchor, its joint options (rows) and the value vector of the step that the anchor gets (rows)
        """
        mdp, colors, layout = self.mdp, self.colors, self._colored_layout()
        count, width = len(anchors), len(vectors)
        choice_values = mdp.matrix @ vectors.T
        choice_values += mdp.reward[:, None]
        signed = choice_values if self.sign > 0 else -choice_values
        weights = anchors * self.nontarget
        # the value of the best choices of the states that every suffix gets with the weights of each anchor
        total = np.zeros((count, width))
        free_best = None
        if len(self.free_states):
            free_best = np.maximum.reduceat(signed[self.free_choices], self.argmax_free.segment_starts, axis=0)
            total += weights[:, self.free_states] @ free_best
        if len(layout.single_states):
            total += weights[:, layout.single_states] @ signed[layout.single_choices]
        colored_weight = weights[:, layout.states]
        reached: np.ndarray = np.zeros(0, dtype=np.int64)
        if colors.num_components:
            reached = np.flatnonzero((np.add.reduceat(colored_weight, layout.start[:-1], axis=1) > 0).any(axis=0))
        live = {}
        for component in reached:
            low, high = layout.start[component], layout.start[component + 1]
            live[component] = low + np.flatnonzero((colored_weight[:, low:high] > 0).any(axis=0))
            size = int(colors.comp_size[component])
            suffix_values = signed[layout.choices[live[component], :size]]
            for first in range(0, count, ALPHA_BLOCK):
                block = colored_weight[first : first + ALPHA_BLOCK][:, live[component]] @ suffix_values.reshape(len(live[component]), size * width)
                total[first : first + ALPHA_BLOCK] += block.reshape(-1, size, width).max(axis=1)
        top = total.max(axis=1)
        parent = (total >= (top - TIE_TOLERANCE * np.abs(top))[:, None]).argmax(axis=1)
        # the best options for the chosen suffix of each anchor (the components that no anchor reaches keep their previous option)
        joint = np.zeros((count, colors.num_components), dtype=np.int64)
        if previous_joint is not None:
            joint[:] = previous_joint
        for component in reached:
            size = int(colors.comp_size[component])
            suffix_values = signed[layout.choices[live[component], :size][:, :, None], parent[None, None, :]]
            score = np.einsum("al,loa->ao", colored_weight[:, live[component]], suffix_values)
            best = score.max(axis=1)
            equally_good = score >= (best - TIE_TOLERANCE * np.abs(best))[:, None]
            option = equally_good.argmax(axis=1)
            if previous_joint is not None:
                option = np.where(equally_good[:, previous_joint[component]], previous_joint[component], option)
            joint[:, component] = option
        new_vectors = np.empty((count, mdp.num_states))
        if free_best is not None:
            new_vectors[:, self.free_states] = (self.sign * free_best[:, parent]).T
        if len(layout.single_states):
            new_vectors[:, layout.single_states] = choice_values[layout.single_choices][:, parent].T
        if len(layout.states):
            executed = layout.choices[np.arange(len(layout.states))[None, :], joint[:, layout.component]]
            new_vectors[:, layout.states] = choice_values[executed, parent[:, None]]
        new_vectors[:, mdp.target] = self.terminal[mdp.target]
        return parent, joint, new_vectors

    def alpha_sweep(self, anchors: Anchors, previous_joint: np.ndarray | None = None, deadline: float | None = None) -> AlphaPool | None:
        """The pooled backward pass: in each step, a value vector for every anchor (the value of the best suffix policy for it that the pool of the step below
        has, with the best options of the step), from the last step to the first. None if the time is up."""
        vectors = self.terminal[None, :]
        parents: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * self.horizon
        joints: list[np.ndarray] = [np.empty((0, 0), dtype=np.int64)] * self.horizon
        for step in range(self.horizon - 1, -1, -1):
            if deadline is not None and time.time() > deadline:
                return None
            parents[step], joints[step], vectors = self._pool_backup(
                vectors, anchors.rows[step].toarray(), None if previous_joint is None else previous_joint[step]
            )
        return AlphaPool(parents, joints, vectors)

    def _alpha_anchors(self, solution: Solution, joints: Sequence[np.ndarray], unconstrained: Solution) -> Anchors:
        """The anchors of the policies of the pool: the distinct reach vectors in each step (those that are the same up to a factor are one anchor, since they
        weight the options in the same way). The policies are those that execute the joint options of the members (the last member is the unconstrained one);
        the reach vectors of all of them are computed together, a step with a single sparse matrix product.

        The states without colors that have several choices choose the best one given the options after them, so the policy of a member needs a backward pass
        then; otherwise the choices of these states are those of the solution (they are not a choice), and the policy is given by the options.
        """
        layout, count = self._colored_layout(), len(joints)
        policies = np.stack([self.improve(None, fixed=joint).policy for joint in joints]) if len(self.free_states) else None
        reach = np.zeros((count + 1, self.mdp.num_states))
        reach[:, self.mdp.initial_state] = 1.0
        rows: list[scipy.sparse.csr_matrix] = []
        index: list[list[int]] = [[] for _ in range(count + 1)]
        for step in range(self.horizon):
            weighted = reach * self.nontarget
            mass = weighted.sum(axis=1)
            reached = np.flatnonzero(mass > 1e-14)
            positions = np.full(count + 1, -1, dtype=np.int64)
            if len(reached):
                _, first, inverse = np.unique(np.round(weighted[reached] / mass[reached, None], 10), axis=0, return_index=True, return_inverse=True)
                order = np.argsort(first)
                rank = np.empty(len(order), dtype=np.int64)
                rank[order] = np.arange(len(order))
                positions[reached] = rank[inverse.ravel()]
                rows.append(scipy.sparse.csr_matrix(weighted[reached[first[order]]]))
            else:
                rows.append(scipy.sparse.csr_matrix((1, self.mdp.num_states)))  # nothing is reached, but the step needs a vector for the steps before it
            for member, position in enumerate(positions):
                index[member].append(int(position))
            # the choices of the states in the step: those of the policies of the members, and the unconstrained policy
            if policies is not None:
                chosen = np.vstack([policies[:, step], unconstrained.policy[step]])
            else:
                chosen = np.tile(solution.policy[step], (count + 1, 1))
                chosen[:count, layout.states] = layout.choices[
                    np.arange(len(layout.states))[None, :], np.array([joint[step] for joint in joints])[:, layout.component]
                ]
                chosen[count] = unconstrained.policy[step]
            executed = np.zeros((self.mdp.num_choices, count + 1))
            executed[chosen, np.arange(count + 1)[:, None]] = weighted
            reach = (self.matrix_transposed @ executed).T
        return Anchors(rows, index)

    def _alpha_changes(self, solution: Solution, limit: int, deadline: float | None = None, changes: int = ALPHA_CHANGES) -> list[tuple[np.ndarray, int]]:
        """The changes of the solution that the pool is made of, as (joint options, step of the first change): first the single changes (every option that a
        component that can be reached in a step does not have there), from the first step to the last, then the changes of two components, and so on up to
        `changes` components, until there are `limit` of them. The first change of a larger one is in a step, the others are in the components that the changes
        before them reach (in that step or a later one), as the seeds of the windows that run to the horizon.

        :param changes: the maximum number of components that a change changes
        """
        assert solution.joint_options is not None
        if limit <= 0:
            return []
        options: np.ndarray = solution.joint_options
        reach = self.reach_probabilities(solution.policy)
        found: list[tuple[np.ndarray, int]] = []

        def change(start: int, seed: Sequence[Cell]) -> tuple[np.ndarray, int]:
            joint = options.copy()
            for index, component, option in seed:
                joint[start + index, component] = option
            return joint, start

        for number in range(1, changes + 1):
            for start in range(self.horizon):
                window = self.horizon - start
                first = self._window_cells(solution, start, [reach[start]])

                def later(seed: tuple[Cell, ...], start: int = start, window: int = window) -> list[Cell]:
                    return self._later_cells(solution, reach, start, window, seed)

                for seed in window_seeds(first, later, number) if number > 1 else ((cell,) for cell in first):
                    if len(seed) < number:
                        continue
                    if deadline is not None and number > 1 and time.time() > deadline:
                        return found
                    found.append(change(start, seed))
                    if len(found) >= limit:
                        return found
        return found

    def _alpha_budget(self, vectors: int) -> int:
        """The number of policies of the pool: vectors, or fewer if their value vectors (and the values of the choices) would not fit in ALPHA_MEMORY."""
        fits = max(2, ALPHA_MEMORY // (8 * (self.mdp.num_choices + 3 * self.mdp.num_states)))
        if vectors > fits and not self._alpha_reduced:
            logger.warning(f"WARNING: the model is large, so the pool is reduced from {vectors} to {fits} policies to fit the value vectors in the memory")
            self._alpha_reduced = True
        return min(vectors, fits)

    def _alpha_respond(self, members: list[list[Any]], anchors: Anchors, pool: AlphaPool) -> bool:
        """Let the members of the pool that are changes of the solution respond to the pooled backward pass: a member keeps its steps up to its first change and
        gets the suffix of the vector of its anchor in the next step (changing the joint options of the member, which are rows of the list [joint, step]).

        :returns: whether some member changed
        """
        responded = False
        for index, (joint, step) in enumerate(members):
            vector = anchors.index[index][step + 1] if 0 <= step < self.horizon - 1 else -1
            if vector >= 0:
                response = pool.suffix(step + 1, vector)
                if np.any(response != joint[step + 1 :]):
                    joint[step + 1 :] = response
                    responded = True
        return responded

    def search_alpha(
        self,
        solution: Solution,
        vectors: int,
        tolerance: float,
        deadline: float | None = None,
        unconstrained: Solution | None = None,
        changes: int = ALPHA_CHANGES,
    ) -> Solution | None:
        """Try to improve a solution that the backward passes cannot improve, by pooling the value vectors of several policies (see the top of this file).

        The pool consists of the solution, the unconstrained optimum (the policy that the first backward pass is based on), and changes of the solution (see
        _alpha_changes), `vectors` policies in all (without the unconstrained one). A pooled backward pass is run for the reach vectors of the policies of the
        pool as the anchors (see alpha_sweep). The policy that it ends with executes the best options for the initial state, whose suffix is the best one that
        the pool of the step below has, and so on; it is the result if it is better than the solution. Otherwise the policies that are changes of the solution
        respond: each of them keeps its steps up to its first change and gets the suffix of the vector of its anchor in the next step, and the pass is
        repeated, at most ALPHA_ROUNDS times and as long as some policy changes.

        The solution is in the pool and it is preferred where it is as good as the others, so the value of the result is never below its value. The vector of
        the initial state is the value of a concrete color consistent policy, which is computed again exactly (with the choices of the states without colors
        optimized) for the result.

        :param unconstrained: the unconstrained optimum, if it has been computed already
        :param changes: the maximum number of components that a change of the solution that is a policy of the pool changes, see _alpha_changes
        :returns: the improved solution, or None if the pooled backward passes do not improve the solution (or the time is up)
        """
        if vectors <= 1 or not self.colors.num_components or solution.joint_options is None:
            return None
        statistics = self.alpha_stats
        for key in ("attempts", "sweeps", "policies", "moves", "seconds"):
            statistics.setdefault(key, 0)
        statistics["attempts"] += 1
        started = time.time()
        try:
            pool_changes = self._alpha_changes(solution, self._alpha_budget(vectors) - 1, deadline, changes)
            if not pool_changes:
                return None
            if unconstrained is None:
                unconstrained = self.solve_unconstrained()
            members = [[solution.joint_options.copy(), -1]] + [[joint, step] for joint, step in pool_changes]
            statistics["policies"] += len(members)
            threshold = tolerance * max(1.0, abs(solution.value))
            for round_number in range(1, ALPHA_ROUNDS + 1):
                anchors = self._alpha_anchors(solution, [joint for joint, _ in members], unconstrained)
                pool = self.alpha_sweep(anchors, solution.joint_options, deadline)
                if pool is None:
                    return None
                statistics["sweeps"] += 1
                candidate = self.improve(None, fixed=pool.suffix(0, 0))
                gain = self.sign * (candidate.value - solution.value)
                if gain > threshold:
                    statistics["moves"] += 1
                    self.alpha_moves.append({"policies": len(members), "rounds": round_number, "gain": gain, "value": candidate.value})
                    return candidate
                if not self._alpha_respond(members, anchors, pool):
                    break
            return None
        finally:
            statistics["seconds"] += time.time() - started

    def solve(
        self,
        max_iterations: int,
        tolerance: float,
        time_limit: float | None = None,
        max_window: int = 1,
        changes: int = 1,
        alpha_vectors: int = 1,
        alpha_changes: int = ALPHA_CHANGES,
    ) -> tuple[Solution, Solution, list[float]]:
        """Run the whole procedure; returns the best color consistent solution, the unconstrained one and the values of the color consistent iterations.

        :param max_window: if the backward passes (which change a single step) stop improving, windows of 2, 3, ..., max_window steps are tried, one size after
            the other; when a window improves the solution, the backward passes continue (and the windows start from 2 again, if they stop improving again).
            This terminates since the value of every window increases the value and the policies are finitely many. The result is a policy that no window of up
            to max_window steps improves (in the sense of window_gain); with 1 only single steps are changed.
        :param alpha_vectors: if the backward passes stop improving, the value vectors of this many policies are pooled (see search_alpha) before the windows
            are tried; a policy that it finds is improved by the backward passes again. 1 does not pool anything.
        :param alpha_changes: the maximum number of components that a change of the policy that is a policy of the pool changes at the same time
        """
        started = time.time()
        deadline = None if time_limit is None else started + time_limit
        self.window_moves, self.window_stats = [], {}
        self.alpha_moves, self.alpha_stats = [], {}
        unconstrained = self.solve_unconstrained()
        logger.info(f"unconstrained optimum of the unfolded MDP: {unconstrained.value:.8g} [{time.time() - started:.2f}s]")

        reach = self.reach_probabilities(unconstrained.policy)
        best: Solution | None = None
        history: list[float] = []
        for iteration in range(1, max_iterations + 1):
            iteration_started = time.time()
            solution = self.improve(reach, best)
            history.append(solution.value)
            gain = None if best is None else self.sign * (solution.value - best.value) + 0.0  # (+ 0.0 turns -0.0 into 0.0)
            logger.info(
                f"iteration {iteration}: value {solution.value:.8g}"
                + ("" if gain is None else f" (improvement {gain:+.3g})")
                + f" [{time.time() - iteration_started:.2f}s]"
            )
            if best is not None and gain < -1e-9 * max(1.0, abs(best.value)):
                logger.warning("WARNING: the value has decreased, this should not happen")
            improving = best is None or gain > tolerance * max(1.0, abs(best.value))
            if best is None or gain > 0:
                best = solution
            if not improving:
                escaped = self.search_alpha(best, alpha_vectors, tolerance, deadline, unconstrained, alpha_changes) if alpha_vectors > 1 else None
                if escaped is not None:
                    move = self.alpha_moves[-1]
                    logger.info(
                        f"pooling the value vectors of {move['policies']} policies improves the value to {escaped.value:.8g} ({move['gain']:+.3g}, "
                        f"{move['rounds']} {'round' if move['rounds'] == 1 else 'rounds'})"
                    )
                elif max_window > 1:
                    escaped = self.widen(best, max_window, tolerance, deadline, changes)
                    if escaped is not None:
                        move = self.window_moves[-1]
                        logger.info(
                            f"a window of {move['window']} steps from the step {move['step']} improves the value to {escaped.value:.8g} ({move['gain']:+.3g})"
                        )
                if escaped is None:
                    if deadline is not None and time.time() > deadline:
                        logger.info("time limit reached, aborting...")
                    else:
                        also = ([f"with {alpha_vectors} value vectors"] if alpha_vectors > 1 else []) + (
                            [f"with windows of up to {max_window} steps"] if max_window > 1 else []
                        )
                        logger.info("the value has stopped improving" + (", also " + " and ".join(also) if also else ""))
                    break
                best = escaped
                history.append(escaped.value)
                reach = self.reach_probabilities(escaped.policy)
                continue
            if deadline is not None and time.time() > deadline:
                logger.info("time limit reached, aborting...")
                break
            reach = self.reach_probabilities(solution.policy)
        else:
            logger.info("the maximum number of iterations has been reached")
        assert best is not None
        return best, unconstrained, history


# ----------------------------------------------------------------------------------------------------
# conditional reachability on unfolded models
# ----------------------------------------------------------------------------------------------------

# the evidence of a policy that is less likely than this is considered impossible: the conditional probability of the policy is undefined
DEGENERATE = 1e-300


@dataclasses.dataclass
class Layer:
    """The states of one layer of an unfolded MDP (without the absorbing ones) and their choices, in arrays that are indexed by positions within the layer.

    The choices of the state at the position i are choices[first[i]:first[i]+counts[i]]. The matrix has a row for each of the choices (and a column for each
    state of the MDP). A colored choice is described by its position, the global index of its joint option and its component; state_component is the component
    that a state depends on (-1 for a state without colors).
    """

    states: np.ndarray
    choices: np.ndarray
    first: np.ndarray
    counts: np.ndarray
    state_of_choice: np.ndarray
    matrix: scipy.sparse.csr_matrix
    colored: np.ndarray
    colored_joint: np.ndarray
    colored_component: np.ndarray
    state_component: np.ndarray

    @property
    def components(self) -> np.ndarray:
        return np.unique(self.colored_component)

    def best_choices(self, scores: np.ndarray, tolerance: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        """Position of the best choice of each state, given the score and the tolerance of each choice.

        Choices whose scores are within the tolerance of the best one are equally good; among them the previous choice (given by its position) is preferred, and
        then the first one.
        """
        best = np.maximum.reduceat(scores, self.first)
        candidates = scores >= best[self.state_of_choice] - tolerance
        if previous is not None:
            preferred = np.zeros(len(scores), dtype=bool)
            preferred[previous] = True
            preferred &= candidates
            has_preferred = np.zeros(len(self.states), dtype=bool)
            has_preferred[self.state_of_choice[preferred]] = True
            candidates = np.where(has_preferred[self.state_of_choice], preferred, candidates)
        positions = np.flatnonzero(candidates)
        owner = self.state_of_choice[positions]
        is_first = np.empty(len(positions), dtype=bool)
        is_first[:1] = True
        is_first[1:] = owner[1:] != owner[:-1]
        result = np.empty(len(self.states), dtype=np.int64)
        result[owner[is_first]] = positions[is_first]
        return result


class LayeredMdp:
    """An MDP that is already unfolded in time: it is acyclic (except for the self-loops of absorbing states) and the states are organized in layers.

    The layer of a state is the length of the longest path from the initial state to it, so every transition leads to a later layer, and the states of a layer
    can be handled together. Only the states that are reachable from the initial state are put in layers, and the absorbing ones (the states of the goal and
    condition sets and sinks) are not, since they do not make any decisions. This is the structure that the models of monitor verification have: the time step
    is a part of their states and observations. Since the parameters of a policy (e.g. the observations) have to be assigned layer by layer, all states that
    depend on the same component of parameters have to be in the same layer.

    The conditional probability is only supported for goal states that are condition states and for absorbing condition states (hence goal states): then the
    probabilities of reaching the goal and the condition are all that the paths contribute, a path ends in one of the absorbing states.
    """

    def __init__(self, mdp: MdpArrays, colors: ColorStructure, goal: np.ndarray, condition: np.ndarray):
        self.mdp = mdp
        self.colors = colors
        self.goal = goal
        self.condition = condition
        num_states, num_choices = mdp.num_states, mdp.num_choices
        matrix = mdp.matrix
        self.policy_dtype = np.int32 if num_choices < 2**31 else np.int64
        self.default_policy = mdp.row_groups[:-1].astype(self.policy_dtype)

        # a state is absorbing if each of its choices is a self-loop
        starts = matrix.indptr[:-1]
        self_loop = (np.diff(matrix.indptr) == 1) & (matrix.indices[starts] == mdp.choice_state) & (matrix.data[starts] >= 1 - 1e-12)
        self.absorbing = np.add.reduceat(self_loop.astype(np.int64), mdp.row_groups[:-1]) == np.diff(mdp.row_groups)

        # the states that can be reached from the initial state; the graph of non-absorbing states does not have the self-loops
        source = np.repeat(mdp.choice_state, np.diff(matrix.indptr))
        keep = ~self.absorbing[source]
        graph = scipy.sparse.csr_matrix((np.ones(int(keep.sum())), (source[keep], matrix.indices[keep])), shape=(num_states, num_states))
        reached = np.zeros(num_states, dtype=bool)
        reached[mdp.initial_state] = True
        frontier = np.array([mdp.initial_state])
        while len(frontier):
            targets = np.unique(graph[frontier].indices)
            frontier = targets[~reached[targets]]
            reached[frontier] = True
        self.reached = reached
        active = reached & ~self.absorbing
        if np.any(goal & reached & ~condition):
            raise UnsupportedProblemError(
                f"{int((goal & reached & ~condition).sum())} reachable goal states are not condition states, but the goal has to be a subset of the condition"
            )
        if np.any(condition & active):
            raise UnsupportedProblemError(
                f"{int((condition & active).sum())} reachable condition states are not absorbing, but the condition states have to be absorbing"
            )

        # layers: peel off the states without predecessors again and again
        edges = graph.tocoo()
        inside = active[edges.row] & active[edges.col]
        indegree = np.bincount(edges.col[inside], minlength=num_states).astype(np.int64)
        level = np.full(num_states, -1, dtype=np.int64)
        frontier = np.flatnonzero(active & (indegree == 0))
        depth = 0
        while len(frontier):
            level[frontier] = depth
            targets = graph[frontier].indices
            targets = targets[active[targets]]
            successors, counts = np.unique(targets, return_counts=True)
            indegree[successors] -= counts
            frontier = successors[indegree[successors] == 0]
            depth += 1
        if np.any(active & (level < 0)):
            raise UnsupportedProblemError("the MDP is not acyclic (all its cycles have to be self-loops of absorbing states), it is not unfolded in time")
        self.level = level

        self.choice_component = np.full(num_choices, -1, dtype=np.int64)
        self.choice_component[colors.colored_choices] = colors.choice_component
        self.choice_joint = np.full(num_choices, -1, dtype=np.int64)
        self.choice_joint[colors.colored_choices] = colors.choice_joint
        self.state_component = self.choice_component[mdp.row_groups[:-1]]
        self._check_components(active)

        active_states = np.flatnonzero(active)
        order = np.argsort(level[active_states], kind="stable")
        sizes = np.bincount(level[active_states]) if len(active_states) else np.zeros(0, dtype=np.int64)
        layer_states = np.split(active_states[order], np.cumsum(sizes)[:-1]) if len(active_states) else []
        self.layers = [self._layer(states) for states in layer_states]

    def _check_components(self, active: np.ndarray) -> None:
        """The states of a component have to be in the same layer."""
        depended = active & (self.state_component >= 0)
        components, levels = self.state_component[depended], self.level[depended]
        lowest = np.full(self.colors.num_components, np.iinfo(np.int64).max)
        highest = np.full(self.colors.num_components, -1)
        np.minimum.at(lowest, components, levels)
        np.maximum.at(highest, components, levels)
        self.component_level = np.where(highest >= 0, lowest, -1)
        spanning = np.flatnonzero((highest >= 0) & (lowest != highest))
        if len(spanning):
            raise UnsupportedProblemError(
                f"{len(spanning)} components of parameters are used by states of different layers (e.g. the layers {int(lowest[spanning[0]])} and "
                f"{int(highest[spanning[0]])}): the parameters have to depend on the time step, i.e. the observations of an unfolded model have to include it"
            )

    def _layer(self, states: np.ndarray) -> Layer:
        mdp = self.mdp
        counts = np.diff(mdp.row_groups)[states]
        first = np.cumsum(counts) - counts
        choices = np.repeat(mdp.row_groups[states] - first, counts) + np.arange(int(counts.sum()))
        component = self.choice_component[choices]
        colored = np.flatnonzero(component >= 0)
        return Layer(
            states, choices, first, counts, np.repeat(np.arange(len(states)), counts), mdp.matrix[choices],
            colored, self.choice_joint[choices][colored], component[colored], self.state_component[states],
        )  # fmt: skip


@dataclasses.dataclass
class ConditionalSolution:
    """A policy of a layered MDP: policy[s] is the choice of the state s. The conditional probability of the policy is value = numerator / evidence, where the
    numerator is the probability of reaching the goal and the evidence is the probability of reaching the condition.

    A color consistent policy additionally has the joint option of each component and the parameter assignment options[p].
    """

    policy: np.ndarray
    value: float
    numerator: float
    evidence: float
    joint_options: np.ndarray | None = None
    options: np.ndarray | None = None


class ConditionalSolver:
    """Backward iteration for the conditional reachability probability Pr(F goal | F condition) on a layered MDP.

    The probability is the ratio N/D of N = Pr(F goal) and D = Pr(F condition) (the goal states are condition states). A state does not have a value that could
    be used to compare its actions, but for a number lambda the *threshold* question does: Pr(F goal | F condition) >= lambda iff N - lambda*D >= 0. And N -
    lambda*D is an ordinary expected reward, in which the goal states are worth 1-lambda, the other condition states are worth -lambda and everything else is
    worth nothing (all paths end in an absorbing state). The Bellman equations hold for it, so a backward pass over the layers finds the best action of each
    state; on acyclic MDPs this is the linear time algorithm for the threshold problem.

    The number lambda is taken from the policy itself (Dinkelbach's method). If lambda is the conditional probability of the current policy, the value N -
    lambda*D of the policy is zero, so a policy of a positive value has a larger conditional probability. The iterations are thus those of the unconstrained
    solution of a reachability problem, with one difference: the terminal values depend on lambda, which is updated in every iteration to the conditional
    probability of the new policy. Without colors, lambda converges after a few iterations (this is how the optimum of the MDP is found, which is memoryless
    since the goal and condition states are absorbing). For color consistent policies, the backward pass picks in each layer the best assignment of the
    parameters given the reachability probabilities of the previous policy (as in the unconditional case: the parameters are not a memory, the same action is
    taken regardless of the path). It cannot decrease the value N - lambda*D, and thus not the conditional probability.

    The unconstrained optimum is an upper bound (lower bound for minimization) of the conditional probability of every color consistent policy, but it can be
    very far from it: the best policy can choose the action according to the state, which lets it condition on the paths that reach the goal. It is a bad
    reference for the reachability probabilities then, so the iterations are also started from the probabilities of the uniformly random policy, and the better
    result is kept.
    """

    def __init__(self, layered: LayeredMdp, objective: ConditionalObjective):
        self.layered = layered
        self.mdp = layered.mdp
        self.colors = layered.colors
        self.objective = objective
        self.sign = 1.0 if objective.maximize else -1.0
        # what the search with windows has done: the improving windows, and for each size the candidates tried and the time spent
        self.window_moves: list[dict[str, Any]] = []
        self.window_stats: dict[int, dict[str, float]] = {}

    # --- passes over the layers ---

    def _flow(self, layer: Layer, policy: np.ndarray | None, weights: np.ndarray) -> np.ndarray:
        """The probability that the states of the layer, which have the weights (e.g. the probabilities of reaching them), lead to each state of the MDP when
        the policy is executed; None stands for the policy that picks a choice uniformly at random in each state."""
        if policy is None:
            executed = np.repeat(weights / layer.counts, layer.counts)
        else:
            executed = np.zeros(len(layer.choices))
            executed[policy[layer.states] - self.mdp.row_groups[layer.states] + layer.first] = weights
        return layer.matrix.T @ executed

    def reach_probabilities(self, policy: np.ndarray | None = None) -> np.ndarray:
        """Probability rho[s] of reaching the state s under the policy; None stands for the policy that picks a choice uniformly at random in each state."""
        reach = np.zeros(self.mdp.num_states)
        reach[self.mdp.initial_state] = 1.0
        for layer in self.layered.layers:
            reach += self._flow(layer, policy, reach[layer.states])
        return reach

    def values_of(self, policy: np.ndarray, goal_value: float, stop_value: float) -> np.ndarray:
        """The expected reward of a path from each state under the policy: goal_value in the goal states, stop_value in the other condition states."""
        values = self._terminal_values(goal_value, stop_value)
        for layer in reversed(self.layered.layers):
            values[layer.states] = (layer.matrix @ values)[policy[layer.states] - self.mdp.row_groups[layer.states] + layer.first]
        return values

    def evaluate(self, policy: np.ndarray) -> tuple[float, float]:
        """The probabilities of reaching the goal and the condition under the policy."""
        reach = self.reach_probabilities(policy)
        return float(reach[self.layered.goal].sum()), float(reach[self.layered.condition].sum())

    def _sweep(
        self,
        goal_value: float,
        stop_value: float,
        reach: np.ndarray | None = None,
        previous: ConditionalSolution | None = None,
        sign: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Backward pass over the layers that maximizes (minimizes) the expected reward of a path: goal_value in the goal states, stop_value in the other
        condition states, zero in other absorbing states. The values of the states of the last layer are known first, those of the previous layer follow, and so
        on.

        :param reach: weights of the states (the probabilities of reaching them) to decide the parameters, or None to let every state pick its best choice
            (ignoring the colors)
        :param previous: the previous solution, whose choices are preferred over equally good ones
        :param sign: 1 to maximize the reward and -1 to minimize it (by default as the objective)
        :returns: the policy and the joint option of each component (None if the colors have been ignored)
        """
        layered, colors = self.layered, self.colors
        sign = self.sign if sign is None else sign
        values = self._terminal_values(goal_value, stop_value)
        policy = layered.default_policy.copy()
        previous_joint = None if previous is None else previous.joint_options
        joint = None
        if reach is not None:
            joint = np.zeros(colors.num_components, dtype=np.int64) if previous_joint is None else previous_joint.copy()
        for layer in reversed(layered.layers):
            weights = np.ones(len(layer.states)) if reach is None else reach[layer.states]
            positions, choice_values = self._decide_layer(layer, values, weights, None if previous is None else previous.policy, sign, previous_joint, joint)
            policy[layer.states] = layer.choices[positions]
            values[layer.states] = choice_values[positions]
        if joint is not None:
            self._execute_joint_options(policy, joint)
        return policy, joint

    def _terminal_values(self, goal_value: float, stop_value: float) -> np.ndarray:
        """The values of the states: goal_value in the goal states, stop_value in the other condition states and zero in the others."""
        values = np.zeros(self.mdp.num_states)
        values[self.layered.goal] = goal_value
        values[self.layered.condition & ~self.layered.goal] = stop_value
        return values

    def _decide_layer(
        self,
        layer: Layer,
        values: np.ndarray,
        weights: np.ndarray,
        previous_policy: np.ndarray | None,
        sign: float,
        previous_joint: np.ndarray | None,
        joint: np.ndarray | None,
        forced: Sequence[tuple[int, int]] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        """One layer of the backward pass: the best choices of its states given the values of the successors and the weights of the states.

        :param previous_policy: the choices of the previous solution, which are preferred over equally good ones
        :param previous_joint: the joint options of the previous solution (preferred as well)
        :param joint: the joint options to be set for the components of the layer, or None to ignore the colors (the choices of the colored states are not
            restricted then)
        :param forced: components and their joint options to be executed whatever their weighted values are
        :returns: the position of the choice of each state of the layer and the values of all choices of the layer
        """
        choice_values = layer.matrix @ values
        weight_of_choice = weights[layer.state_of_choice]
        previous_positions = None if previous_policy is None else previous_policy[layer.states] - self.mdp.row_groups[layer.states] + layer.first
        scores = sign * choice_values * weight_of_choice
        # the noise of floating-point numbers is proportional to the weights that are summed, not to the (possibly cancelling) scores
        positions = layer.best_choices(scores, TIE_TOLERANCE * weight_of_choice, previous_positions)
        if joint is not None and len(layer.colored):
            self._decide_components(layer, scores, weights, positions, previous_joint, joint, forced)
        return positions, choice_values

    def _execute_joint_options(self, policy: np.ndarray, joint_options: np.ndarray) -> None:
        """Let all states that depend on a component execute its joint option, also those that are not reachable."""
        colors = self.colors
        chosen = colors.choice_joint == (colors.comp_offset[:-1] + joint_options)[colors.choice_component]
        policy[self.mdp.choice_state[colors.colored_choices[chosen]]] = colors.colored_choices[chosen]

    def _decide_components(
        self,
        layer: Layer,
        scores: np.ndarray,
        weights: np.ndarray,
        positions: np.ndarray,
        previous_joint: np.ndarray | None,
        joint: np.ndarray,
        forced: Sequence[tuple[int, int]] = (),
    ) -> None:
        """Choose the joint option of each component of the layer: the one with the best sum of the scores of its choices (the scores have the weights of the
        states in them), and let the colored states execute it.

        The joint options of the components of the layer are set in joint, and the positions of the choices of the colored states in positions.
        """
        colors = self.colors
        weighted_values = np.bincount(layer.colored_joint, weights=scores[layer.colored], minlength=colors.num_joint_options)
        in_component = layer.state_component >= 0
        scale = np.bincount(layer.state_component[in_component], weights=weights[in_component], minlength=colors.num_components)
        best = colors.best_joint_options(weighted_values, previous_joint, scale)
        for component, option in forced:
            best[component] = option
        joint[layer.components] = best[layer.components]
        chosen = layer.colored_joint == colors.comp_offset[layer.colored_component] + best[layer.colored_component]
        positions[layer.state_of_choice[layer.colored[chosen]]] = layer.colored[chosen]

    def solution_of_joint_options(self, joint_options: np.ndarray) -> ConditionalSolution:
        """The color consistent solution that executes the joint option of each component."""
        policy = self.layered.default_policy.copy()
        self._execute_joint_options(policy, joint_options)
        return self.solution_of_policy(policy, joint_options)

    def solution_of_policy(self, policy: np.ndarray, joint_options: np.ndarray | None = None) -> ConditionalSolution:
        numerator, evidence = self.evaluate(policy)
        options = None if joint_options is None else self.colors.decode(joint_options[None, :])[0]
        return ConditionalSolution(policy, numerator / evidence if evidence > DEGENERATE else float("nan"), numerator, evidence, joint_options, options)

    # --- windows: several layers that are changed at the same time ---

    def _reaching_components(self, layer: Layer, reach: np.ndarray) -> np.ndarray:
        """The components of the layer that have a state that can be reached."""
        in_component = layer.state_component >= 0
        weights = np.bincount(layer.state_component[in_component], weights=reach[layer.states][in_component], minlength=self.colors.num_components)
        return np.flatnonzero(weights > 1e-12)

    def inflow(self, policy: np.ndarray, reach: np.ndarray, start: int) -> np.ndarray:
        """The probability that the paths of the policy enter each state coming from the layers before the layer start (from the initial state if start is
        0)."""
        flow = np.zeros(self.mdp.num_states)
        if start == 0:
            flow[self.mdp.initial_state] = 1.0
        for layer in self.layered.layers[:start]:
            flow += self._flow(layer, policy, reach[layer.states])
        return flow

    def _seeded_window(
        self, solution: ConditionalSolution, start: int, size: int, seed: Sequence[Cell]
    ) -> tuple[np.ndarray, list[list[tuple[int, int]]], np.ndarray]:
        """The window of the layers start..start+size-1 when the changes of the seed are made and nothing else is: the joint option of every component, the
        changes of each layer of the window as pairs of a component and its option, and the choices of the states of the MDP."""
        assert solution.joint_options is not None
        joint = solution.joint_options.copy()
        forced: list[list[tuple[int, int]]] = [[] for _ in range(size)]
        for index, component, option in seed:
            joint[component] = option
            forced[index].append((component, option))
        policy = solution.policy.copy()
        for layer in self.layered.layers[start : start + size]:
            self._execute_layer(layer, policy, joint)
        return joint, forced, policy

    def _window_cells(
        self, solution: ConditionalSolution, start: int, size: int, flow: np.ndarray, first_index: int = 0, last_index: int | None = None
    ) -> list[Cell]:
        """The changes that can be forced in the layers of a window, from the layer first_index to last_index: each option that a component does not have in the
        solution, for each component that can be reached in the layer (flow[s] is the probability of reaching the state s)."""
        assert solution.joint_options is not None
        layers = self.layered.layers[start : start + size]
        last_index = size - 1 if last_index is None else last_index
        return [
            (index, int(component), option)
            for index in range(first_index, last_index + 1)
            for component in self._reaching_components(layers[index], flow)
            for option in range(int(self.colors.comp_size[component]))
            if option != solution.joint_options[component]
        ]

    def window_gain(
        self,
        solution: ConditionalSolution,
        values: np.ndarray,
        inflow: np.ndarray,
        start: int,
        size: int,
        seed: Sequence[Cell],
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """The gain of the objective N - lambda*D from changing components at the beginning of a window of layers, and the policy of the window then.

        The backward pass changes one layer, taking the weights of the states in it from the old policy: this is exact for the layer, but the layers after it
        are optimized for the old weights as well, they cannot respond to the change of the layer. A window changes several layers together: the options of
        some components (the seed: the layer of the window, the component and its option; the first one is in the first layer) are forced in the layers
        start..start+size-1, and the other components of the window respond to the changes. The weights of the states in the first layer are those of the
        solution (nothing changes before the window), but the weights in the next layers are the ones that the changed beginning of the window induces. The
        responses are found with the backward pass restricted to the window, and since the induced weights depend on the responses, this is repeated a few
        times. The layers after the window keep the policy of the solution. The paths that enter the window from the layers before it (the inflow) are not
        changed, and the paths that leave it continue as before, so the gain is the exact difference of the objectives, in which lambda is the conditional
        probability of the solution (so that the objective of the solution is zero).

        :param values: the values of the states under the solution, with the terminal values for lambda (see values_of)
        :param inflow: the probabilities of entering the states from the layers before the window (see inflow)
        :returns: the gain (positive if the conditional probability improves: it increases for the maximization and decreases for the minimization), the joint
            option of every component, and the policy of the whole MDP
        """
        assert solution.joint_options is not None
        layers = self.layered.layers[start : start + size]
        joint, forced, policy = self._seeded_window(solution, start, size, seed)
        new_values = values.copy()
        for _ in range(WINDOW_ROUNDS):
            flow = inflow.copy()
            for layer in layers[:-1]:
                flow += self._flow(layer, policy, flow[layer.states])
            new_joint = joint.copy()
            for index in reversed(range(size)):
                layer = layers[index]
                positions, choice_values = self._decide_layer(
                    layer, new_values, flow[layer.states], solution.policy, self.sign, solution.joint_options, new_joint, forced[index]
                )
                policy[layer.states] = layer.choices[positions]
                new_values[layer.states] = choice_values[positions]
            changed = bool(np.any(new_joint != joint))
            joint = new_joint
            if not changed:
                break
        states = np.concatenate([layer.states for layer in layers])
        return self.sign * float(np.dot(inflow[states], new_values[states] - values[states])), joint, policy

    def _execute_layer(self, layer: Layer, policy: np.ndarray, joint_options: np.ndarray) -> None:
        """Let the colored states of the layer execute the joint options of their components."""
        colors = self.colors
        chosen = layer.colored[layer.colored_joint == colors.comp_offset[layer.colored_component] + joint_options[layer.colored_component]]
        policy[layer.states[layer.state_of_choice[chosen]]] = layer.choices[chosen]

    def search_windows(
        self, solution: ConditionalSolution, size: int, tolerance: float, deadline: float | None = None, changes: int = 1, reference: str = ""
    ) -> ConditionalSolution | None:
        """Look for a window of the given size that improves the solution (see window_gain): every component that can be reached in the first layer of a window,
        with every option that it does not have. The windows are tried from the last to the first, and the best of the first window that has an improvement is
        applied.

        :param changes: the maximum number of changes (a component with an option that it does not have) that are forced at the same time in the window: the
            first one in the first layer, the others in the components that the changes before them reach (in the first layer or later). One change is not
            enough if the options of several components have to change together, e.g. two consecutive symbols of an observation word.
        :param reference: the reference that the solution comes from (for the record of the moves)
        :returns: the improved solution, or None if no window of this size improves it (or the time is up)
        """
        layers, colors = self.layered.layers, self.colors
        if size > len(layers) or not colors.num_components or solution.joint_options is None:
            return None
        statistics = self.window_stats.setdefault(size, {"candidates": 0, "moves": 0, "seconds": 0.0})
        started = time.time()
        lam = solution.value
        reach = self.reach_probabilities(solution.policy)
        values = self.values_of(solution.policy, 1.0 - lam, -lam)
        try:
            for start in range(len(layers) - size, -1, -1):
                first = self._window_cells(solution, start, size, reach, 0, 0)
                if not first:
                    continue
                inflow = self.inflow(solution.policy, reach, start)
                states = np.concatenate([layer.states for layer in layers[start : start + size]])
                # the noise of the gain is proportional to the weights that are summed
                threshold = TIE_TOLERANCE * float(inflow[states].sum())

                def later(seed: tuple[Cell, ...], start: int = start, inflow: np.ndarray = inflow) -> list[Cell]:
                    # what the changes of the seed reach in the window, where the other layers keep their policy
                    policy = self._seeded_window(solution, start, size, seed)[2]
                    flow = inflow.copy()
                    for layer in layers[start : start + size - 1]:
                        flow += self._flow(layer, policy, flow[layer.states])
                    taken = {cell[1] for cell in seed}
                    return [cell for cell in self._window_cells(solution, start, size, flow, seed[-1][0]) if cell > seed[-1] and cell[1] not in taken]

                best = None
                for seed in window_seeds(first, later, changes):
                    if deadline is not None and time.time() > deadline:
                        return None
                    statistics["candidates"] += 1
                    gain, joint, policy = self.window_gain(solution, values, inflow, start, size, seed)
                    if gain > threshold and (best is None or gain > best[0]):
                        best = (gain, joint, policy, seed)
                if best is None:
                    continue
                improved = self.solution_of_policy(best[2], best[1])
                gain = self.sign * (improved.value - solution.value)
                if gain > tolerance * max(1.0, abs(solution.value)):
                    statistics["moves"] += 1
                    self.window_moves.append(
                        {
                            "reference": reference,
                            "window": size,
                            "layer": start + 1,
                            "cells": [list(cell) for cell in best[3]],
                            "gain": gain,
                            "value": improved.value,
                        }
                    )
                    return improved
                logger.debug(f"the window of {size} layers at the layer {start + 1} did not improve the value enough, the estimated gain was {best[0]:.3g}")
            return None
        finally:
            statistics["seconds"] += time.time() - started

    def widen(
        self, solution: ConditionalSolution, max_window: int, tolerance: float, deadline: float | None = None, changes: int = 1, reference: str = ""
    ) -> ConditionalSolution | None:
        """The solution improved by the smallest window (of at least two layers and at most max_window layers) that has an improvement, or None."""
        for size in range(2, max_window + 1):
            improved = self.search_windows(solution, size, tolerance, deadline, changes, reference)
            if improved is not None or (deadline is not None and time.time() > deadline):
                return improved
        return None

    # --- optimization ---

    def _policy_of_lambda(self, lam: float, reach: np.ndarray | None, previous: ConditionalSolution | None) -> ConditionalSolution | None:
        """The best policy for the objective N - lambda*D (color consistent for the weights reach, or unconstrained if reach is None), or None if the evidence
        cannot be reached by it, which happens if the objective is flat and the first of the equally good policies is bad."""
        policy, joint = self._sweep(1.0 - lam, -lam, reach, previous)
        solution = self.solution_of_policy(policy, joint)
        if solution.evidence <= DEGENERATE and previous is None:
            # start from a policy that has a conditional probability: one that makes the condition as likely as possible
            policy, joint = self._sweep(1.0, 1.0, reach, previous, sign=1.0)
            solution = self.solution_of_policy(policy, joint)
        return solution if solution.evidence > DEGENERATE else None

    def _improving(self, new: float, old: float | None, tolerance: float) -> bool:
        return old is None or self.sign * (new - old) > tolerance * max(1.0, abs(old))

    def solve_unconstrained(self, max_iterations: int = 1000, tolerance: float = 1e-13) -> ConditionalSolution:
        """The optimal policy of the MDP without colors, and its conditional probability. The policy is memoryless since the goal and condition states are
        absorbing.

        The iterations start from lambda = 0 (maximization) or 1 (minimization). The objective N - lambda*D is then the probability of the goal, or minus the
        probability of the condition states that are not goal states. Afterwards, lambda is the conditional probability of the policy that was found.
        """
        lam = 0.0 if self.objective.maximize else 1.0
        best: ConditionalSolution | None = None
        for _ in range(max_iterations):
            solution = self._policy_of_lambda(lam, None, best)
            if solution is None:
                break
            if not self._improving(solution.value, None if best is None else best.value, tolerance):
                break
            best, lam = solution, solution.value
        if best is None:
            raise UnsupportedProblemError("the condition cannot be reached: the conditional probability is undefined")
        return best

    def iterate(
        self,
        reach: np.ndarray,
        max_iterations: int,
        tolerance: float,
        deadline: float | None,
        reference: str = "",
        max_window: int = 1,
        changes: int = 1,
    ) -> tuple[ConditionalSolution | None, list[float]]:
        """Improve color consistent policies until the conditional probability stops improving, starting from the probabilities of reaching the states reach.

        :param max_window: if the backward passes (which change a single layer) stop improving, windows of 2, 3, ..., max_window layers are tried, one size
            after the other; when a window improves the solution, the backward passes continue (and the windows start from 2 again, if they stop improving
            again). This terminates since every window increases the conditional probability and the policies are finitely many. The result is a policy that no
            window of up to max_window layers improves (in the sense of window_gain); with 1 only single layers are changed.
        :param changes: the number of components that a window changes at the same time at its beginning, see search_windows
        """
        name = f"[from the {reference} policy] " if reference else ""
        lam = 0.0 if self.objective.maximize else 1.0
        best: ConditionalSolution | None = None
        history: list[float] = []
        for iteration in range(1, max_iterations + 1):
            started = time.time()
            solution = self._policy_of_lambda(lam, reach, best)
            if solution is None:
                logger.info(f"{name}iteration {iteration}: the policy cannot reach the condition")
                break
            history.append(solution.value)
            gain = None if best is None else self.sign * (solution.value - best.value) + 0.0  # (+ 0.0 turns -0.0 into 0.0)
            logger.info(
                f"{name}iteration {iteration}: conditional probability {solution.value:.8g}"
                + ("" if gain is None else f" (improvement {gain:+.3g})")
                + f" [{time.time() - started:.2f}s]"
            )
            if best is not None and gain < -1e-9 * max(1.0, abs(best.value)):
                logger.warning("WARNING: the conditional probability has decreased, this should not happen")
            improving = self._improving(solution.value, None if best is None else best.value, tolerance)
            if best is None or gain > 0:
                best = solution
            if not improving:
                assert best is not None
                widened = self.widen(best, max_window, tolerance, deadline, changes, reference) if max_window > 1 else None
                if widened is None:
                    if max_window > 1:
                        if deadline is not None and time.time() > deadline:
                            logger.info(f"{name}time limit reached, aborting...")
                        else:
                            logger.info(f"{name}no window of up to {max_window} layers improves the conditional probability")
                    break
                move = self.window_moves[-1]
                logger.info(
                    f"{name}a window of {move['window']} layers from the layer {move['layer']} improves the conditional probability to {widened.value:.8g} "
                    f"({move['gain']:+.3g})"
                )
                best = widened
                history.append(widened.value)
                lam, reach = widened.value, self.reach_probabilities(widened.policy)
                continue
            if deadline is not None and time.time() > deadline:
                logger.info("time limit reached, aborting...")
                break
            lam, reach = solution.value, self.reach_probabilities(solution.policy)
        return best, history

    def solve(
        self, max_iterations: int, tolerance: float, time_limit: float | None = None, max_window: int = 1, changes: int = 1
    ) -> tuple[ConditionalSolution, ConditionalSolution, list[float], dict[str, float]]:
        """Run the whole procedure; returns the best color consistent solution, the unconstrained one, the conditional probabilities of the iterations that led
        to the best solution and the best conditional probability found from each of the references (see iterate for max_window and changes)."""
        started = time.time()
        self.window_moves, self.window_stats = [], {}
        unconstrained = self.solve_unconstrained()
        logger.info(f"unconstrained optimum of the unfolded MDP: {unconstrained.value:.8g} [{time.time() - started:.2f}s]")
        deadline = None if time_limit is None else started + time_limit
        references = {"unconstrained": self.reach_probabilities(unconstrained.policy), "uniform": self.reach_probabilities(None)}
        best: ConditionalSolution | None = None
        best_history: list[float] = []
        values: dict[str, float] = {}
        for name, reach in references.items():
            solution, history = self.iterate(reach, max_iterations, tolerance, deadline, name, max_window, changes)
            if solution is None:
                continue
            values[name] = solution.value
            if best is None or self.sign * (solution.value - best.value) > 0:
                best, best_history = solution, history
        if best is None:
            raise UnsupportedProblemError("no color consistent policy reaches the condition: the conditional probability is undefined")
        return best, unconstrained, best_history, values


@dataclasses.dataclass
class DpResult:
    """The outcome of the exact search: the best color consistent solution (None if the search did not get to find any), whether the search considered all color
    consistent policies (rather than being stopped by a limit), the number of distinct states in each layer and the time it took."""

    solution: ConditionalSolution | None
    complete: bool
    sizes: list[int]
    seconds: float


class ReachVectorDp:
    """Exact search for the best color consistent policy, by dynamic programming over the reach vectors of the layers (option --dp).

    A color consistent policy chooses an option for each component of parameters, e.g. an action for each observation, and since the observations of an
    unfolded model include the time, a component belongs to one layer. So the number of policies is exponential in the number of layers. The search constructs
    the policies layer by layer, but it does not tell apart the beginnings of policies that leave the same *reach vector* behind.

    The reach vector x of a layer is the distribution over the states of the layer, i.e. x[s] is the probability of being in the state s when the layer is
    reached (the paths that have ended in an absorbing state before do not count). The paths continue the same way whatever the beginning of the policy was,
    and what they collect is linear in x: if the rest of the policy collects N' = <x,a> and D' = <x,c> (a[s] and c[s] are the probabilities to reach the goal
    and the condition from s), the conditional probability of the policy is (N + N') / (D + D'), where N and D are the probabilities that the beginning has
    collected already, i.e. that the paths have ended in the goal and in the condition before the layer. Hence the dynamic programming state is (x, N, D), and
    two beginnings with the same state are equally good for every continuation: keep one.

    Even more, multiplying (x, N, D) by a constant does not change the ratio, so the state is rescaled to sum(x) + D = 1 (the paths lost in sinks do not
    matter). In the typical case that no path has ended yet (N = D = 0) the state is the normalized reach vector, and beginnings that differ only in how likely
    they are, but not in what they lead to, are the same state.

    Example: a policy that chooses a sequence of symbols that a hidden Markov chain has to emit one by one (a mismatch leads to a sink, and the policy ends in
    the goal with the probability of the hidden state at the end of the sequence). The conditional probability is the probability of the bad event given the
    sequence, and the reach vector after a sequence is the belief about the hidden state, up to a factor. With 18 symbols and 9 steps there are 18^9 = 2*10^11
    sequences (policies), but only about 3000 beliefs, and the search looks at them all: it takes a fraction of a second, and the result is the optimum, not a
    heuristic. Whether this works depends on the number of distinct states, which the log of the search shows.

    The search is limited to the case that a policy chooses a single thing in every layer: at most one component of the layer has some probability of being
    reached by the beginning of the policy (otherwise the options of several components would have to be combined, which multiplies the number of states);
    states without colors may have only one choice. The transitions have to lead to the next layer. States are merged if their (rescaled) vectors, N and D
    agree in about 10 digits, so the value is exact up to about 10^-9.
    """

    DIGITS = 10
    MIX = np.uint64(0x9E3779B97F4A7C15)

    def __init__(self, solver: ConditionalSolver):
        self.solver = solver
        self.layered = solver.layered
        self.colors = solver.colors
        self.mdp = solver.mdp
        for layer in self.layered.layers:
            free = (layer.state_component < 0) & (layer.counts > 1)
            if free.any():
                state = int(layer.states[np.flatnonzero(free)[0]])
                raise UnsupportedProblemError(f"state {state} has several choices that no parameter restricts, which the exact search does not support")
        self._columns: dict[int, np.ndarray] = {}
        self._transitions: dict[tuple[int, int | None, int | None], scipy.sparse.csr_matrix] = {}

    # --- transitions between the states of the dynamic program ---

    def _column_of_state(self, depth: int) -> np.ndarray:
        """Column of each state of the MDP in the output of the layer: 0 for the goal states, 1 for the other condition states, 2+i for the i-th state of the
        next layer, -1 for the other absorbing states (they are lost) and -2 for the states that cannot be reached from the layer."""
        if depth not in self._columns:
            layered = self.layered
            next_states = layered.layers[depth + 1].states if depth + 1 < len(layered.layers) else np.zeros(0, dtype=np.int64)
            column = np.full(self.mdp.num_states, -2, dtype=np.int64)
            column[layered.absorbing] = -1
            column[layered.condition & ~layered.goal] = 1
            column[layered.goal] = 0
            column[next_states] = 2 + np.arange(len(next_states))
            self._columns[depth] = column
        return self._columns[depth]

    def _transition(self, depth: int, component: int | None, option: int | None) -> scipy.sparse.csr_matrix:
        """The matrix (states of the layer) x (goal, other condition states, states of the next layer) of the probabilities of the transitions of the choices
        that the option of the component selects, and of the only choices of the states without colors (the other rows are zero)."""
        key = (depth, component, option)
        if key not in self._transitions:
            layer = self.layered.layers[depth]
            if component is None:
                selected = layer.first[layer.state_component < 0]
            else:
                chosen = (layer.colored_component == component) & (layer.colored_joint - self.colors.comp_offset[component] == option)
                selected = np.concatenate([layer.colored[chosen], layer.first[layer.state_component < 0]])
            block = layer.matrix[selected]
            columns = self._column_of_state(depth)[block.indices]
            if np.any(columns == -2):
                raise UnsupportedProblemError("the exact search needs every transition to lead to the next layer")
            rows = np.repeat(layer.state_of_choice[selected], np.diff(block.indptr))
            kept = columns >= 0
            width = 2 + (len(self.layered.layers[depth + 1].states) if depth + 1 < len(self.layered.layers) else 0)
            self._transitions[key] = scipy.sparse.coo_matrix((block.data[kept], (rows[kept], columns[kept])), shape=(len(layer.states), width)).tocsr()
        return self._transitions[key]

    def _keys(self, states: np.ndarray, numerators: np.ndarray, evidences: np.ndarray) -> np.ndarray:
        """A key of each state of the program (a 53-bit number), equal for states that agree in about DIGITS digits: two random projections of the state,
        rounded and mixed."""
        width = states.shape[1]
        projection = np.random.default_rng(width).random((width + 2, 2)) + 0.5
        projected = states @ projection[:width] + numerators[:, None] * projection[width] + evidences[:, None] * projection[width + 1]
        rounded = np.round(projected * 10**self.DIGITS).astype(np.int64).astype(np.uint64)
        with np.errstate(over="ignore"):
            mixed = rounded[:, 0] * self.MIX ^ rounded[:, 1]
        return (mixed >> np.uint64(11)).astype(np.float64)

    @staticmethod
    def _new_keys(seen: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The positions of the keys (sorted and unique) that are not in the sorted array seen, and the array seen with these keys added."""
        position = np.searchsorted(seen, keys)
        known = (position < len(seen)) & (seen[np.minimum(position, max(len(seen) - 1, 0))] == keys) if len(seen) else np.zeros(len(keys), dtype=bool)
        fresh = np.flatnonzero(~known)
        return fresh, np.insert(seen, position[fresh], keys[fresh])

    # --- the search ---

    def run(self, max_states: int, time_limit: float | None = None) -> DpResult:
        """Search all color consistent policies.

        :param max_states: stop (the result is incomplete) if a layer has more distinct states than this, or if they would not fit in the memory
        :param time_limit: time limit of the search (s)
        """
        started = time.time()
        solver, layers, colors = self.solver, self.layered.layers, self.colors
        if not layers:
            # the initial state is absorbing: there is nothing to decide
            return DpResult(solver.solution_of_joint_options(np.zeros(colors.num_components, dtype=np.int64)), True, [], 0.0)
        vectors = np.zeros((1, len(layers[0].states)))
        vectors[0, 0] = 1.0  # the initial state is the only state of the first layer
        numerators, evidences = np.zeros(1), np.zeros(1)
        # how the states of each layer have been constructed: the state of the previous layer, and the component and option that were chosen there
        parents, chosen_components, chosen_options = [np.zeros(1, dtype=np.int64)], [np.full(1, -1)], [np.full(1, -1)]
        best_value, best_leaf = -np.inf, None
        sizes: list[int] = []
        complete = True
        for depth, layer in enumerate(layers):
            count = len(vectors)
            sizes.append(count)
            logger.debug(f"exact search: layer {depth} has {count} distinct states [{time.time() - started:.2f}s]")
            if count > max_states or (time_limit is not None and time.time() - started > time_limit):
                complete = False
                break
            if count * (vectors.shape[1] + 2) * 8 * 4 > psutil.virtual_memory().available / 2:
                logger.warning(f"WARNING: the {count} states of the layer {depth} would not fit in the memory")
                complete = False
                break

            masses = {int(component): vectors[:, layer.state_component == component].sum(axis=1) for component in layer.components}
            forced = layer.state_component < 0
            forced_mass = vectors[:, forced].sum(axis=1) if forced.any() else np.zeros(count)
            num_active = sum((mass > 0).astype(np.int64) for mass in masses.values()) if masses else np.zeros(count, dtype=np.int64)
            if np.any(num_active > 1):
                raise UnsupportedProblemError(
                    f"in the layer {depth} the beginnings of policies reach states of several components of parameters, but the exact search is only for "
                    "policies that choose a single thing in every layer"
                )
            groups = [(component, np.flatnonzero(mass > 0)) for component, mass in masses.items()]
            groups.append((None, np.flatnonzero((num_active == 0) & (forced_mass > 0))))  # states without colors only
            children: list[tuple[np.ndarray, ...]] = []
            seen = np.zeros(0)
            stopped = False
            width = 2 + (len(layers[depth + 1].states) if depth + 1 < len(layers) else 0)
            # large layers are done in pieces to limit the memory of the temporary arrays
            piece = max(1, int(psutil.virtual_memory().available / 4 / (width * 8 * 6)))
            for component, rows in groups:
                for start in range(0, len(rows), piece):
                    chunk = rows[start : start + piece]
                    sparse_chunk = scipy.sparse.csr_matrix(vectors[chunk])
                    for option in range(int(colors.comp_size[component])) if component is not None else [None]:
                        if time_limit is not None and time.time() - started > time_limit:
                            stopped = True
                            break
                        reached = (sparse_chunk @ self._transition(depth, component, option)).toarray()
                        goal_mass, stop_mass, onward = reached[:, 0], reached[:, 1], reached[:, 2:]
                        numerator = numerators[chunk] + goal_mass
                        evidence = evidences[chunk] + goal_mass + stop_mass
                        onward_mass = onward.sum(axis=1)
                        # nothing continues: the policy ends here (or is lost in a sink), and this is a complete policy
                        ended = (onward_mass <= 1e-15 * np.maximum(evidence, DEGENERATE)) & (evidence > DEGENERATE)
                        if ended.any():
                            values = np.where(ended, solver.sign * numerator / np.where(ended, evidence, 1.0), -np.inf)
                            best_row = int(np.argmax(values))
                            if values[best_row] > best_value:
                                best_value, best_leaf = float(values[best_row]), (depth, int(chunk[best_row]), component, option)
                        alive = np.flatnonzero(onward_mass > 0)
                        if len(alive) == 0:
                            continue
                        scale = onward_mass[alive] + evidence[alive]  # rescale to sum(x) + D = 1
                        new_states, new_numerators, new_evidences = onward[alive] / scale[:, None], numerator[alive] / scale, evidence[alive] / scale
                        # merge the states that are equal: keep the first one, also of the states constructed from other pieces and options
                        keys, first = np.unique(self._keys(new_states, new_numerators, new_evidences), return_index=True)
                        fresh, seen = self._new_keys(seen, keys)
                        first = first[fresh]
                        parents_of_new, component_of_new = chunk[alive][first], np.full(len(first), -1 if component is None else component)
                        children.append(
                            (
                                new_states[first],
                                new_numerators[first],
                                new_evidences[first],
                                parents_of_new,
                                component_of_new,
                                np.full(len(first), -1 if option is None else option),
                            )
                        )
                    if stopped:
                        break
                if stopped:
                    break
            if stopped:
                complete = False
                break
            if not children:
                break
            vectors = np.vstack([child[0] for child in children])
            numerators = np.concatenate([child[1] for child in children])
            evidences = np.concatenate([child[2] for child in children])
            parents.append(np.concatenate([child[3] for child in children]))
            chosen_components.append(np.concatenate([child[4] for child in children]))
            chosen_options.append(np.concatenate([child[5] for child in children]))

        solution = None
        if best_leaf is not None:
            # the options that were chosen on the way to the best policy; the other components do not matter and keep their first option
            depth, row, component, option = best_leaf
            joint = np.zeros(colors.num_components, dtype=np.int64)
            while True:
                if component is not None and component >= 0:
                    joint[component] = option
                if depth == 0:
                    break
                component, option, row = chosen_components[depth][row], chosen_options[depth][row], parents[depth][row]
                depth -= 1
            solution = solver.solution_of_joint_options(joint)
            if abs(solver.sign * solution.value - best_value) > 1e-8 * max(1.0, abs(best_value)):
                logger.warning(f"WARNING: the value of the policy of the exact search computed again ({solution.value:.8g}) differs from the one it found")
        return DpResult(solution, complete, sizes, time.time() - started)


# ----------------------------------------------------------------------------------------------------
# unfolded colored MDP
# ----------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class UnfoldedColoredMdp:
    """The time unfolding of a colored MDP: M_H, the parameter space V x [1..H] and the coloring kappa_H as described at the top of this file.

    Only the states (s,t) that are reachable from the initial state in t steps are built, layer by layer (by t, then by s). The maps state_step/state_base and
    choice_step/choice_base give for each state and choice of M_H its step t and the state s or choice c of M it copies (the self-loops of the last layer, t=H,
    copy no choice: -1). The parameter p@t has index t*K+p, where K is the number of parameters of V. Time steps are numbered from 0 here, but from 1 in the
    names of the parameters.
    """

    horizon: int
    base: Any
    mdp: Any
    parameter_space: paynt.parameter_space.parameter_space.ParameterSpace
    coloring: Any
    state_step: np.ndarray
    state_base: np.ndarray
    choice_step: np.ndarray
    choice_base: np.ndarray

    @property
    def num_base_states(self) -> int:
        return self.base.underlying_mdp.nr_states

    @property
    def num_base_choices(self) -> int:
        return self.base.underlying_mdp.nr_choices

    @staticmethod
    def make_unfolder(colored_mdp: Any, horizon: int) -> Any:
        """Find the states of the unfolding, which tells its size; the unfolding itself is constructed by build()."""
        return payntbind.synthesis.FiniteHorizonUnfolder(colored_mdp.underlying_mdp, horizon)

    @classmethod
    def build(cls, colored_mdp: Any, horizon: int, final_layer_labels: list[str], unfolder: Any = None) -> UnfoldedColoredMdp:
        """Unfold the colored MDP; the states of the last layer are added to the labels final_layer_labels (see payntbind.synthesis.FiniteHorizonUnfolder).

        :param unfolder: the result of make_unfolder, if the size of the unfolding has been looked at already
        """
        base_space = colored_mdp.parameter_space
        if unfolder is None:
            unfolder = cls.make_unfolder(colored_mdp, horizon)
        mdp = unfolder.construct_unfolded_model(final_layer_labels)
        coloring = payntbind.synthesis.TimeUnfoldedColoring(colored_mdp.coloring, base_space.native, unfolder)
        space = paynt.parameter_space.parameter_space.ParameterSpace()
        space.native = payntbind.synthesis.unfoldFamilyFiniteHorizon(base_space.native, horizon)
        space.parameter_to_name = [f"{name}@{step + 1}" for step in range(horizon) for name in base_space.parameter_to_name]
        space.parameter_to_option_labels = [labels for _ in range(horizon) for labels in base_space.parameter_to_option_labels]
        return cls(horizon, colored_mdp, mdp, space, coloring, unfolder.state_step, unfolder.state_base, unfolder.choice_step, unfolder.choice_base)

    def to_colored_mdp(self, full_coloring: bool = False) -> paynt.colored_mdp.ColoredMdp:
        """The unfolding as a regular PAYNT colored MDP.

        Constructing a ColoredMdp computes the destinations of all choices, which is expensive for large unfoldings. The coloring of this class is lighter than
        payntbind's Coloring, but implements only a part of its interface, which suffices to build the MDP of a parameter (sub)space or the Markov chain of a
        parameter assignment. The synthesizers need the whole interface, which is what full_coloring gives, at the price of a memory requirement that is
        quadratic in the horizon: use it only for small models, e.g. to find the optimal time-dependent policy by AR synthesis and compare it to the solver's.
        """
        coloring = self.coloring
        if full_coloring:
            k = self.base.parameter_space.num_parameters
            base_colors = self.base.coloring.getChoiceToAssignment()
            colors = [
                [(hole + step * k, option) for hole, option in base_colors[choice]] if choice >= 0 else []
                for step, choice in zip(self.choice_step.tolist(), self.choice_base.tolist(), strict=True)
            ]
            coloring = payntbind.synthesis.Coloring(self.parameter_space.native, self.mdp.nondeterministic_choice_indices, colors)
        # a parameter assignment induces a Markov chain only if it fixes the choice in every state, otherwise the induced model is an MDP (a "family")
        feature_kind = "family" if self.has_free_choices() else "generic"
        return paynt.colored_mdp.ColoredMdp(self.mdp, self.parameter_space, coloring, use_exact=False, feature_kind=feature_kind)

    def has_free_choices(self) -> bool:
        """Whether some state has several choices that no parameter restricts, so that fixing the parameters does not fix the policy."""
        pointers, _, _ = payntbind.synthesis.coloringToCsr(self.base.coloring)
        row_groups = np.asarray(self.base.underlying_mdp.nondeterministic_choice_indices, dtype=np.int64)
        num_uncolored = np.add.reduceat((np.diff(pointers) == 0).astype(np.int64), row_groups[:-1])
        return bool(np.any((num_uncolored > 0) & (np.diff(row_groups) > 1)))

    def assignment(self, options: np.ndarray) -> paynt.parameter_space.parameter_space.ParameterSpace:
        """The parameter assignment that sets p@t to options[t,p]."""
        return self.parameter_space.assume_options_copy([[int(option)] for option in options.ravel()])

    def policy_choices(self, policy: np.ndarray) -> Any:
        """Bit vector of the choices of the unfolded MDP that execute the policy (policy[t,s] is the choice of the state s in the step t)."""
        first_choice = np.asarray(self.mdp.nondeterministic_choice_indices, dtype=np.int64)[:-1]
        base_first_choice = np.asarray(self.base.underlying_mdp.nondeterministic_choice_indices, dtype=np.int64)
        # the choices of a state are those of the state it copies, in the same order; the states of the last layer have a single choice
        inner = self.state_step < self.horizon
        step, base = self.state_step[inner], self.state_base[inner]
        chosen = np.concatenate((first_choice[inner] + policy[step, base] - base_first_choice[base], first_choice[~inner]))
        return payntbind.synthesis.bitVectorFromIndices(self.mdp.nr_choices, chosen)

    def check_consistent(self, choices: Any, options: np.ndarray) -> bool:
        """Whether the choices are compatible with the parameter assignment, i.e. whether the policy is color consistent in each step."""
        return choices.is_subset_of(self.coloring.selectCompatibleChoices(self.assignment(options).native))

    def value_of_policy(self, choices: Any, prop: Any) -> float:
        """Value of the property in the Markov chain induced by the policy, computed by Storm."""
        submdp = paynt.model.model.SubmodelBuilder.build_submdp(self.mdp, choices, paynt.model.model.SubmodelBuilder.default_builder_options())
        dtmc = paynt.model.model.SubmodelBuilder.mdp_to_dtmc(submdp.model)
        return float(paynt.specification.property.Property.model_check(dtmc, prop.formula).at(dtmc.initial_states[0]))

    def optimum(self, prop: Any) -> float:
        """The optimal value of the property in the unfolded MDP, computed by Storm."""
        return float(paynt.specification.property.Property.model_check(self.mdp, prop.formula).at(self.mdp.initial_states[0]))


# ----------------------------------------------------------------------------------------------------
# AR synthesis
# ----------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class ArResult:
    """The outcome of the AR synthesis on the unfolded colored MDP.

    options is the parameter assignment found (None if there is none: for an optimality property that means the search did not get to find one, for a constraint
    it means no assignment satisfies it unless the search was cut short) and value is the value of the property in the model it induces. complete tells whether
    the search finished (rather than being stopped by the time limit), solution is the assignment as executed by the solver.

    A search that is warm started (initial is the value of the policy that it started from, the optimum so far) only looks for policies that are better than it,
    and prunes the families that cannot have one. If it finds none, found is False and options and value are those of the policy that it started from, which is
    optimal if the search is complete. families is the number of families (sets of assignments) whose MDP the search checked, i.e. its size.
    """

    options: np.ndarray | None
    value: float | None
    seconds: float
    complete: bool
    solution: Solution | None = None
    initial: float | None = None
    found: bool = False
    families: int = 0


def estimate_ar_memory(unfolded: UnfoldedColoredMdp) -> int:
    """Rough number of bytes needed to build the colored MDP with the full coloring of the unfolding.

    It takes about 400 bytes per choice plus two bit masks over all parameters for each choice and state, which makes it quadratic in the horizon. Measured on
    several models, this is within 20% of the peak memory; a margin of 25% is added.
    """
    words = (unfolded.parameter_space.num_parameters + 63) // 64
    return int(1.25 * (unfolded.mdp.nr_choices * (420 + 8 * words) + unfolded.mdp.nr_states * (48 + 8 * words)))


def run_ar(unfolded: UnfoldedColoredMdp, task: Any, timeout: float | None, start: Solution | None = None) -> ArResult:
    """Run the standard PAYNT AR synthesis on the unfolded colored MDP, for the specification of the task.

    :param timeout: time limit of the search (s); when it hits, the best assignment found so far is returned
    :param start: the policy to start from, for an optimality property (e.g. the one that the backward passes found). Its value is the optimum so far, so AR
        prunes the families that have no policy better than it (by more than the model checking precision) right away, and only looks for better policies
    """
    needed, available = estimate_ar_memory(unfolded), psutil.virtual_memory().available
    if needed > available / 2:
        raise UnsupportedProblemError(
            f"AR needs the full coloring of the unfolded colored MDP, which would take about {needed / 2**30:.1f} GiB "
            f"({available / 2**30:.1f} GiB are available), try a smaller horizon"
        )
    started = time.time()
    colored_mdp = unfolded.to_colored_mdp(full_coloring=True)
    logger.info(f"built the colored MDP with the full coloring for AR [{time.time() - started:.2f}s]")
    # the synthesis changes the optimum of the specification, so it gets its own copy
    ar_task = paynt.task.SynthesisTask.from_specification(task.specification.copy())
    if start is not None:
        assert start.options is not None and ar_task.specification.optimality is not None, "only a property to optimize has an optimum to start from"
        logger.info(f"AR starts with the value {start.value:.8g} of the policy of the heuristic as the optimum so far: it only looks for better policies")
    initial = None if start is None else start.value
    synthesizer = paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, ar_task, "ar")
    limit_reached = synthesizer.resource_limit_reached
    stopped_by_limit = []

    def resource_limit_reached() -> bool:
        stopped_by_limit.append(limit_reached())
        return stopped_by_limit[-1]

    synthesizer.resource_limit_reached = resource_limit_reached  # type: ignore[method-assign]

    started = time.time()
    assignment = synthesizer.synthesize(timeout=timeout, optimum_threshold=initial, keep_optimum=True)
    seconds = time.time() - started
    complete = not any(stopped_by_limit)
    assert synthesizer.stat is not None
    families = synthesizer.stat.iterations_mdp or 0
    if assignment is None:
        if start is None:
            return ArResult(None, None, seconds, complete, families=families)
        # nothing is better than the policy that the search started from, which is the optimum if the search is complete
        assert start.options is not None
        return ArResult(start.options.copy(), float(start.value), seconds, complete, initial=initial, families=families)
    options = np.array([assignment.parameter_options(parameter)[0] for parameter in range(assignment.num_parameters)], dtype=np.int64)
    options = options.reshape(unfolded.horizon, unfolded.base.parameter_space.num_parameters)
    value = synthesizer.best_assignment_value
    if value is None:
        # AR does not record the value of the assignment that satisfies a constraint
        result = colored_mdp.build_assignment(assignment).check_specification(ar_task.specification)
        value = (result.optimality_result if ar_task.specification.has_optimality else result.constraints_result.results[0]).value
    return ArResult(options, float(value), seconds, complete, initial=initial, found=True, families=families)


def report_ar(ar: ArResult, solver: FiniteHorizonSolver, heuristic: Solution, prop: Any, tolerance: float) -> None:
    """Log the outcome of AR and compare it with the value of the heuristic."""
    status = "finished" if ar.complete else "was stopped by the time limit"
    effort = f"{ar.seconds:.2f}s, {ar.families} families checked"
    if ar.initial is not None and not ar.found:
        # AR only looked for policies that are better than the one of the heuristic
        if ar.complete:
            logger.info(f"AR finished: no policy is better than the one of the heuristic (up to the model checking precision), so it is optimal [{effort}]")
        else:
            logger.info(f"AR was stopped by the time limit without finding a policy that is better than the one of the heuristic [{effort}]")
        return
    if ar.options is None or ar.value is None:
        logger.info(f"AR {status} without finding a policy [{effort}]")
        if ar.complete and type(prop) is paynt.specification.property.Property:
            logger.info("AR proved that no time-dependent color consistent policy satisfies the property")
        return
    assert ar.solution is not None
    logger.info(f"AR {status}: value {ar.value:.8g} [{effort}]")
    if abs(ar.solution.value - ar.value) > tolerance:
        logger.warning(f"WARNING: the value of the policy of AR computed by the solver ({ar.solution.value:.8g}) differs from the one computed by AR")
    if type(prop) is paynt.specification.property.Property:
        logger.info(f"AR found a policy that {'satisfies' if prop.satisfies_threshold(ar.value) else 'does not satisfy'} the property {prop}")
        if not prop.satisfies_threshold(heuristic.value):
            logger.info("the heuristic did not find a policy that satisfies the property")
        return
    # positive if AR found a better policy than the heuristic
    gap = solver.sign * (ar.value - heuristic.value)
    relative = f" ({gap / abs(ar.value):.2%})" if abs(ar.value) > 1e-9 else ""
    if abs(gap) <= tolerance:
        logger.info("the heuristic is optimal: its value is the one of AR" if ar.complete else "the policy of AR found so far has the value of the heuristic")
    elif gap > 0 and ar.complete:
        logger.info(f"the heuristic is suboptimal: its value is {gap:.3g}{relative} away from the optimum found by AR")
    elif gap > 0:
        logger.info(f"AR has found a policy that is {gap:.3g}{relative} better than the one of the heuristic")
    elif ar.complete:
        logger.warning(f"WARNING: the heuristic is better than AR by {-gap:.3g}, even though the search of AR finished")
    else:
        logger.info(f"AR has not found a policy as good as the one of the heuristic yet ({-gap:.3g} worse)")


# ----------------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------------


def relevant_parameters(solver: FiniteHorizonSolver, solution: Solution) -> np.ndarray:
    """Mask (steps,parameters) of the parameters that matter to the policy: some state that depends on them is reachable in the step."""
    colors = solver.colors
    reach = solver.reach_probabilities(solution.policy)[: solver.horizon] * solver.nontarget
    relevant = np.zeros((solver.horizon, colors.num_holes), dtype=bool)
    used = colors.hole_component >= 0
    for step in range(solver.horizon):
        reachable = np.bincount(colors.choice_component, weights=reach[step][solver.colored_state] > 1e-12, minlength=colors.num_components) > 0
        relevant[step, used] = reachable[colors.hole_component[used]]
    return relevant


def describe_policy(colored_mdp: Any, solver: FiniteHorizonSolver, solution: Solution) -> list[dict[str, str]]:
    """For each step, the options of the parameters that matter to the policy."""
    assert solution.options is not None
    space = colored_mdp.parameter_space
    relevant = relevant_parameters(solver, solution)
    return [
        {
            space.parameter_name(parameter): str(space.parameter_to_option_labels[parameter][solution.options[step, parameter]])
            for parameter in np.flatnonzero(relevant[step])
        }
        for step in range(solver.horizon)
    ]


def check_supported_colored_mdp(colored_mdp: Any) -> None:
    """The colorings that are supported are sets of parameter-option pairs (see the top of this file)."""
    if colored_mdp.feature_kind in ("dt", "posmg", "decpomdp") or not hasattr(colored_mdp.coloring, "getChoiceToAssignment"):
        hint = " (the decision-tree coloring that PAYNT builds for MDPs without parameters)" if colored_mdp.feature_kind == "dt" else ""
        raise UnsupportedProblemError(
            f"colored MDPs of kind '{colored_mdp.feature_kind}'{hint} are not supported, the coloring has to be a set of parameter-option pairs, e.g. a POMDP"
        )


def solve_time_dependent(
    colored_mdp: Any,
    task: Any,
    horizon: int,
    max_iterations: int = 100,
    tolerance: float = 1e-9,
    time_limit: float | None = None,
    verify: bool = True,
    max_unfolded_states: int = 5_000_000,
    ar: bool = False,
    ar_timeout: float | None = None,
    max_window: int = 1,
    changes: int = 1,
    ar_warm_start: bool = False,
    alpha_vectors: int = 1,
    alpha_changes: int = ALPHA_CHANGES,
) -> dict[str, Any]:
    """Solve the time-dependent color consistency query of the colored MDP for the (single) property of the task over the given horizon.

    :param max_window: the maximum number of steps that are changed at the same time when the backward passes get stuck, see FiniteHorizonSolver.solve
    :param changes: the number of components that a window changes at the same time at its beginning, see FiniteHorizonSolver.search_windows
    :param alpha_vectors: the number of policies whose value vectors are pooled when the backward passes get stuck, see FiniteHorizonSolver.search_alpha
    :param alpha_changes: the number of components that a policy of the pool changes at the same time, see FiniteHorizonSolver.search_alpha
    :param verify: whether to build the unfolded colored MDP and double check the results with Storm on it
    :param ar: whether to run the AR synthesis on the unfolded colored MDP, which finds the optimal policy (for small models) to compare the result with
    :param ar_timeout: time limit of the AR synthesis (s)
    :param ar_warm_start: whether AR starts with the value of the policy that the backward passes found as the optimum so far (for a property to optimize)
    """
    check_supported_colored_mdp(colored_mdp)
    objective = Objective.from_task(task)
    mdp = MdpArrays.from_mdp(colored_mdp.underlying_mdp, objective)
    space = colored_mdp.parameter_space
    hole_num_options = np.array([len(labels) for labels in space.parameter_to_option_labels], dtype=np.int64)
    colors = ColorStructure.analyze(mdp, colored_mdp.coloring, hole_num_options)
    logger.info(
        f"colored MDP ({colored_mdp.feature_kind}): {mdp.num_states} states, {mdp.num_choices} choices, {space.num_parameters} parameters in "
        f"{colors.num_components} components, {colors.num_joint_options} joint options"
    )

    unfolded = None
    if verify or ar:
        # only the reachable states (s,t) are built; looking for them is cheap and tells the size of the unfolding
        unfolder = UnfoldedColoredMdp.make_unfolder(colored_mdp, horizon)
        if unfolder.num_states > max_unfolded_states:
            message = f"the unfolded colored MDP would have {unfolder.num_states} states, above the limit of {max_unfolded_states} (--max-unfolded-states)"
            if ar:
                raise UnsupportedProblemError("AR needs the unfolded colored MDP, but " + message)
            logger.warning(f"WARNING: {message}: skipping the Storm checks")
        else:
            started = time.time()
            unfolded = UnfoldedColoredMdp.build(colored_mdp, horizon, [objective.target_label] if objective.is_reward else [], unfolder)
            reachable = unfolded.mdp.nr_states / ((horizon + 1) * mdp.num_states)
            logger.info(
                f"unfolded colored MDP for horizon {horizon}: {unfolded.mdp.nr_states} reachable states ({reachable:.1%} of all pairs (s,t)), "
                f"{unfolded.mdp.nr_choices} choices, {unfolded.mdp.nr_transitions} transitions, {unfolded.parameter_space.num_parameters} parameters "
                f"[{time.time() - started:.2f}s]"
            )

    solver = FiniteHorizonSolver(mdp, colors, horizon, objective)
    solution, unconstrained, history = solver.solve(max_iterations, tolerance, time_limit, max_window, changes, alpha_vectors, alpha_changes)
    result: dict[str, Any] = {
        "horizon": horizon,
        "objective": str(objective.prop),
        "unconstrained_value": unconstrained.value,
        "value": solution.value,
        "iterations": len(history),
        "history": history,
        "solution": solution,
        "solver": solver,
        "unfolded": unfolded,
    }
    gap = abs(unconstrained.value - solution.value)
    logger.info(f"final value {solution.value:.8g}, the unconstrained optimum is {unconstrained.value:.8g} (gap {gap:.3g})")
    prop = objective.prop
    # values computed by model checking are only precise up to the precision
    value_tolerance = 10 * paynt.specification.property.Property.model_checking_precision * max(1.0, abs(solution.value))
    if type(prop) is paynt.specification.property.Property:
        logger.info(f"the property {prop} is {'satisfied' if prop.satisfies_threshold(solution.value) else 'not satisfied'} by the policy")

    forward = solver.evaluate(solution.policy)
    result["forward_value"] = forward
    if abs(forward - solution.value) > 1e-8 * max(1.0, abs(solution.value)):
        logger.warning(f"WARNING: the value of the policy computed forward ({forward:.8g}) differs from the one computed backward")

    if verify and unfolded is not None:
        assert solution.options is not None
        started = time.time()
        storm_optimum = unfolded.optimum(prop)
        choices = unfolded.policy_choices(solution.policy)
        consistent = unfolded.check_consistent(choices, solution.options)
        storm_value = unfolded.value_of_policy(choices, prop)
        result.update(storm_unconstrained_value=storm_optimum, storm_value=storm_value, consistent=consistent)
        logger.info(
            f"Storm on the unfolded colored MDP: unconstrained optimum {storm_optimum:.8g}, value of the policy {storm_value:.8g}, "
            f"the policy is {'' if consistent else 'NOT '}color consistent in every step [{time.time() - started:.2f}s]"
        )
        if not consistent:
            logger.warning("WARNING: the policy is not color consistent")
        if abs(storm_value - solution.value) > value_tolerance or abs(storm_optimum - unconstrained.value) > value_tolerance:
            logger.warning("WARNING: the values computed by Storm differ from the ones computed by the solver")

    if ar:
        assert unfolded is not None
        start = None
        if ar_warm_start:
            if type(prop) is paynt.specification.property.Property:
                logger.warning("WARNING: a property that is a constraint has no optimum to start AR with, so AR is not warm started")
            else:
                start = solution
        ar_result = run_ar(unfolded, task, ar_timeout, start)
        if ar_result.options is not None:
            ar_result.solution = solver.solution_of_options(ar_result.options)
        report_ar(ar_result, solver, solution, prop, value_tolerance)
        result["ar"] = ar_result
    return result


def conditional_probability_by_storm(mdp: Any, objective: ConditionalObjective, choices: Any = None) -> float:
    """The conditional probability computed by Storm: the optimal one of the MDP, or the one of the Markov chain induced by the policy that executes the
    choices."""
    environment = stormpy.Environment()
    environment.model_checker_environment.conditional.algorithm = stormpy.ConditionalAlgorithmSetting.bisection_pt
    event = f'F "{objective.goal_label}" || F "{objective.condition_label}"'
    if choices is None:
        model, text = mdp, f"P{'max' if objective.maximize else 'min'}=? [{event}]"
    else:
        submdp = paynt.model.model.SubmodelBuilder.build_submdp(mdp, choices, paynt.model.model.SubmodelBuilder.default_builder_options())
        model, text = paynt.model.model.SubmodelBuilder.mdp_to_dtmc(submdp.model), f"P=? [{event}]"
    result = stormpy.model_checking(model, stormpy.parse_properties_without_context(text)[0], only_initial_states=True, environment=environment)
    return float(result.at(model.initial_states[0]))


def is_color_consistent(colored_mdp: Any, choices: Any, options: np.ndarray) -> bool:
    """Whether the choices are compatible with the parameter assignment, i.e. whether the policy is color consistent."""
    assignment = colored_mdp.parameter_space.assume_options_copy([[int(option)] for option in options])
    return choices.is_subset_of(colored_mdp.coloring.selectCompatibleChoices(assignment.native))


def report_dp(dp: DpResult, solver: ConditionalSolver, heuristic: ConditionalSolution, prop: Any, tolerance: float) -> None:
    """Log the outcome of the exact search and compare it with the value of the heuristic."""
    status = "finished" if dp.complete else "was stopped by a limit"
    layers = len(dp.sizes)
    summary = f"{sum(dp.sizes)} distinct states in {layers} layers, at most {max(dp.sizes, default=0)} in one" if dp.sizes else "no layers"
    if dp.solution is None:
        logger.info(f"the exact search {status} without finding a policy ({summary}) [{dp.seconds:.2f}s]")
        return
    logger.info(f"the exact search {status}: conditional probability {dp.solution.value:.8g} ({summary}) [{dp.seconds:.2f}s]")
    # positive if the exact search found a better policy than the heuristic
    gap = solver.sign * (dp.solution.value - heuristic.value)
    if type(prop) is paynt.specification.property.Property:
        found = prop.satisfies_threshold(dp.solution.value)
        if found or dp.complete:
            logger.info(
                "the exact search "
                + (f"found a policy that satisfies the property {prop}" if found else f"proved that no color consistent policy satisfies the property {prop}")
            )
    if abs(gap) <= tolerance:
        logger.info(
            "the heuristic is optimal: its value is the one of the exact search" if dp.complete else "the exact search found so far the value of the heuristic"
        )
    elif gap > 0 and dp.complete:
        relative = f" ({gap / abs(dp.solution.value):.2%})" if abs(dp.solution.value) > 1e-9 else ""
        logger.info(f"the heuristic is suboptimal: its value is {gap:.3g}{relative} away from the optimum")
    elif gap > 0:
        logger.info(f"the exact search has found a policy that is {gap:.3g} better than the one of the heuristic")
    elif dp.complete:
        logger.warning(f"WARNING: the heuristic is better than the exact search by {-gap:.3g}, even though the search finished")
    else:
        logger.info(f"the exact search has not found a policy as good as the one of the heuristic yet ({-gap:.3g} worse)")


def solve_conditional(
    colored_mdp: Any,
    task: Any,
    max_iterations: int = 100,
    tolerance: float = 1e-9,
    time_limit: float | None = None,
    verify: bool = True,
    dp: bool = False,
    dp_max_states: int = 2_000_000,
    dp_time_limit: float | None = None,
    max_window: int = 1,
    changes: int = 1,
) -> dict[str, Any]:
    """Solve the color consistency query of the colored MDP, which is already unfolded in time, for the conditional reachability property of the task.

    :param max_window: the maximum number of layers that are changed at the same time when the backward passes get stuck, see ConditionalSolver.iterate
    :param changes: the number of components that a window changes at the same time at its beginning, see ConditionalSolver.search_windows
    :param verify: whether to double check the results with Storm: the unconstrained optimum, and the value of the policy and that it is color consistent
    :param dp: whether to also search all color consistent policies exactly (see ReachVectorDp), which finds the optimal policy if there are not too many states
    :param dp_max_states: limit of the exact search on the number of distinct states in a layer
    :param dp_time_limit: time limit of the exact search (s)
    """
    check_supported_colored_mdp(colored_mdp)
    objective = ConditionalObjective.from_task(task)
    mdp = MdpArrays.from_mdp(colored_mdp.underlying_mdp, objective)
    condition = label_mask(colored_mdp.underlying_mdp, objective.condition_label, "condition states")
    space = colored_mdp.parameter_space
    hole_num_options = np.array([len(labels) for labels in space.parameter_to_option_labels], dtype=np.int64)
    colors = ColorStructure.analyze(mdp, colored_mdp.coloring, hole_num_options)
    logger.info(
        f"colored MDP ({colored_mdp.feature_kind}): {mdp.num_states} states, {mdp.num_choices} choices, {space.num_parameters} parameters in "
        f"{colors.num_components} components, {colors.num_joint_options} joint options"
    )
    started = time.time()
    layered = LayeredMdp(mdp, colors, mdp.target, condition)
    logger.info(
        f"the MDP is unfolded in {len(layered.layers)} layers of {int(layered.reached.sum())} reachable states in total, "
        f"{int((layered.reached & layered.absorbing).sum())} of them absorbing [{time.time() - started:.2f}s]"
    )

    solver = ConditionalSolver(layered, objective)
    solution, unconstrained, history, references = solver.solve(max_iterations, tolerance, time_limit, max_window, changes)
    heuristic = solution
    result: dict[str, Any] = {
        "objective": str(objective.prop),
        "unconstrained_value": unconstrained.value,
        "heuristic_value": heuristic.value,
        "iterations": len(history),
        "history": history,
        "references": references,
        "solver": solver,
        "layered": layered,
    }
    gap = abs(unconstrained.value - heuristic.value)
    logger.info(
        f"conditional probability {heuristic.value:.8g} (the probability of the condition is {heuristic.evidence:.4g}), "
        f"the unconstrained optimum is {unconstrained.value:.8g} (gap {gap:.3g})"
    )
    prop = objective.prop
    # values computed by model checking are only precise up to the precision
    value_tolerance = 10 * paynt.specification.property.Property.model_checking_precision * max(1.0, abs(heuristic.value))
    if type(prop) is paynt.specification.property.Property:
        logger.info(f"the property {prop} is {'satisfied' if prop.satisfies_threshold(heuristic.value) else 'not satisfied'} by the policy")

    if dp:
        dp_result = ReachVectorDp(solver).run(dp_max_states, dp_time_limit)
        result["dp"] = dp_result
        report_dp(dp_result, solver, heuristic, prop, value_tolerance)
        if dp_result.solution is not None and solver.sign * (dp_result.solution.value - solution.value) > 0:
            solution = dp_result.solution
            logger.info(f"final conditional probability {solution.value:.8g}, the policy of the exact search")
        result["optimal"] = dp_result.complete
    result.update(value=solution.value, evidence=solution.evidence, solution=solution)

    if verify:
        assert solution.options is not None
        started = time.time()
        storm_unconstrained = conditional_probability_by_storm(colored_mdp.underlying_mdp, objective)
        choices = payntbind.synthesis.bitVectorFromIndices(mdp.num_choices, solution.policy)
        consistent = is_color_consistent(colored_mdp, choices, solution.options)
        storm_value = conditional_probability_by_storm(colored_mdp.underlying_mdp, objective, choices)
        result.update(storm_unconstrained_value=storm_unconstrained, storm_value=storm_value, consistent=consistent)
        logger.info(
            f"Storm: unconstrained optimum {storm_unconstrained:.8g}, value of the policy {storm_value:.8g}, "
            f"the policy is {'' if consistent else 'NOT '}color consistent [{time.time() - started:.2f}s]"
        )
        if not consistent:
            logger.warning("WARNING: the policy is not color consistent")
        if abs(storm_value - solution.value) > value_tolerance or abs(storm_unconstrained - unconstrained.value) > value_tolerance:
            logger.warning("WARNING: the values computed by Storm differ from the ones computed by the solver")
    return result


def describe_conditional_policy(colored_mdp: Any, solver: ConditionalSolver, solution: ConditionalSolution) -> list[dict[str, str]]:
    """For each layer, the options of the parameters that matter to the policy: some state that depends on them is reachable."""
    assert solution.options is not None
    space, colors = colored_mdp.parameter_space, solver.colors
    reach = solver.reach_probabilities(solution.policy)
    used = np.flatnonzero(colors.hole_component >= 0)
    policy = []
    for layer in solver.layered.layers:
        reachable = np.zeros(colors.num_components, dtype=bool)
        reachable[solver._reaching_components(layer, reach)] = True
        policy.append(
            {
                space.parameter_name(parameter): str(space.parameter_to_option_labels[parameter][solution.options[parameter]])
                for parameter in used[reachable[colors.hole_component[used]]]
            }
        )
    return policy


def show_assignments(policy: list[dict[str, str]], prefix: str = "") -> None:
    for step, assignment in enumerate(policy, start=1):
        logger.info(f"{prefix}step {step}: " + ", ".join(f"{name}={option}" for name, option in assignment.items()))


def window_summary(solver: FiniteHorizonSolver | ConditionalSolver, max_window: int, changes: int) -> dict[str, Any]:
    """What the search with windows has done (for the export): the limits, the improving windows, and the candidates tried and the time spent per size."""
    return {
        "max_window": max_window,
        "changes": changes,
        "moves": solver.window_moves,
        "sizes": {str(size): statistics for size, statistics in sorted(solver.window_stats.items())},
    }


def alpha_summary(solver: FiniteHorizonSolver, alpha_vectors: int, alpha_changes: int) -> dict[str, Any]:
    """What the pooled backward passes have done (for the export): the limits, the improving passes, and the attempts, passes, policies and the time spent."""
    return {"alpha_vectors": alpha_vectors, "alpha_changes": alpha_changes, "moves": solver.alpha_moves, "statistics": solver.alpha_stats}


def run_time_dependent(
    colored_mdp: Any,
    task: Any,
    horizon: int,
    max_iterations: int,
    tolerance: float,
    timeout: float | None,
    verify: bool,
    max_unfolded_states: int,
    ar: bool,
    ar_timeout: float | None,
    show_policy: bool,
    export_policy: str | None,
    max_window: int = 1,
    changes: int = 1,
    ar_warm_start: bool = False,
    alpha_vectors: int = 1,
    alpha_changes: int = ALPHA_CHANGES,
) -> None:
    """Solve the query over the finite horizon, and show and export the policy."""
    result = solve_time_dependent(
        colored_mdp, task, horizon, max_iterations, tolerance, timeout, verify, max_unfolded_states, ar=ar, ar_timeout=ar_timeout, max_window=max_window,
        changes=changes, ar_warm_start=ar_warm_start, alpha_vectors=alpha_vectors, alpha_changes=alpha_changes,
    )  # fmt: skip
    policies_wanted = show_policy or export_policy is not None
    policy = describe_policy(colored_mdp, result["solver"], result["solution"]) if policies_wanted else []
    ar_result = result.get("ar")
    ar_policy = None
    # a warm started AR that finds nothing better has no policy of its own, the one of the heuristic is the optimum
    if policies_wanted and ar_result is not None and ar_result.found and ar_result.solution is not None:
        ar_policy = describe_policy(colored_mdp, result["solver"], ar_result.solution)
    if show_policy:
        show_assignments(policy)
        show_assignments(ar_policy or [], "AR ")
    if export_policy is not None:
        exported = {k: result[k] for k in ("horizon", "objective", "value", "unconstrained_value", "iterations", "history")} | {"policy": policy}
        if max_window > 1:
            exported["windows"] = window_summary(result["solver"], max_window, changes)
        if alpha_vectors > 1:
            exported["alpha"] = alpha_summary(result["solver"], alpha_vectors, alpha_changes)
        if ar_result is not None:
            exported["ar"] = {
                "value": ar_result.value,
                "seconds": ar_result.seconds,
                "complete": ar_result.complete,
                "policy": ar_policy,
                "initial": ar_result.initial,
                "found": ar_result.found,
                "families": ar_result.families,
            }
        with open(export_policy, "w") as file:
            json.dump(exported, file, indent=2)
        logger.info(f"policy written to {export_policy}")


def run_conditional(
    colored_mdp: Any,
    task: Any,
    max_iterations: int,
    tolerance: float,
    timeout: float | None,
    verify: bool,
    dp: bool,
    dp_max_states: int,
    dp_timeout: float | None,
    show_policy: bool,
    export_policy: str | None,
    max_window: int = 1,
    changes: int = 1,
) -> None:
    """Solve the query for a conditional property of a model that is unfolded already, and show and export the policy."""
    result = solve_conditional(colored_mdp, task, max_iterations, tolerance, timeout, verify, dp, dp_max_states, dp_timeout, max_window, changes)
    policy = describe_conditional_policy(colored_mdp, result["solver"], result["solution"]) if show_policy or export_policy is not None else []
    if show_policy:
        show_assignments(policy)
    if export_policy is not None:
        keys = ("objective", "value", "evidence", "unconstrained_value", "heuristic_value", "iterations", "history", "references")
        exported = {key: result[key] for key in keys} | {"policy": policy}
        if max_window > 1:
            exported["windows"] = window_summary(result["solver"], max_window, changes)
        dp_result = result.get("dp")
        if dp_result is not None:
            exported["dp"] = {"complete": dp_result.complete, "seconds": dp_result.seconds, "states": dp_result.sizes}
            exported["optimal"] = result["optimal"]
        with open(export_policy, "w") as file:
            json.dump(exported, file, indent=2)
        logger.info(f"policy written to {export_policy}")


@click.command()
@click.argument("model", type=click.Path(exists=True, dir_okay=False))
@click.argument("properties", type=click.Path(exists=True, dir_okay=False))
@click.argument("horizon", type=click.IntRange(min=1), required=False)
@click.option("--fsc-memory-size", type=click.IntRange(min=1), default=1, show_default=True, help="implicit memory size for POMDP policies")
@click.option("--max-iterations", type=click.IntRange(min=1), default=100, show_default=True, help="maximum number of backward iterations")
@click.option("--tolerance", type=click.FLOAT, default=1e-9, show_default=True, help="stop when the (relative) improvement of the value is not above this")
@click.option("--timeout", type=click.FLOAT, help="time limit for the iterations (s)")
@click.option("--precision", type=click.FLOAT, default=1e-6, show_default=True, help="model checking precision of Storm")
@click.option(
    "--no-unfold",
    is_flag=True,
    default=False,
    help="do not build the unfolded colored MDP and do not double check the results with Storm (a conditional property: do not double check them)",
)
@click.option("--max-unfolded-states", type=click.INT, default=5_000_000, show_default=True, help="do not build the unfolded MDP if it would have more states")
@click.option(
    "--ar",
    is_flag=True,
    default=False,
    help="also run the standard PAYNT AR synthesis on the unfolded colored MDP, which finds the optimal policy to compare the result with (small models)",
)
@click.option("--ar-timeout", type=click.FLOAT, help="time limit for the AR synthesis (s)")
@click.option(
    "--ar-warm-start",
    is_flag=True,
    default=False,
    help="start the AR synthesis with the value of the policy that the backward passes found (with the windows and the value vectors, if any) as the optimum "
    "so far, so that it "
    "prunes the families that cannot beat it from the beginning and only looks for better policies (a property to optimize); requires --ar",
)
@click.option(
    "--dp",
    is_flag=True,
    default=False,
    help="also search all color consistent policies exactly (conditional properties), by dynamic programming over the distinct reach vectors of the layers, "
    "which finds the optimal policy if there are not too many of them (see the top of this file)",
)
@click.option(
    "--dp-max-states", type=click.IntRange(min=1), default=2_000_000, show_default=True, help="stop the exact search if a layer has more distinct states"
)
@click.option("--dp-timeout", type=click.FLOAT, help="time limit for the exact search (s)")
@click.option(
    "--window",
    "max_window",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="when the backward passes (which change one step, or layer, at a time) stop improving, try to improve the policy by changing windows of 2, 3, ..., N "
    "consecutive steps at the same time, see the top of this file; 1 only runs the backward passes",
)
@click.option(
    "--window-changes",
    "changes",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="the number of parameters whose options a window changes at the same time at its beginning (2 finds the improvements that need two parameters of a "
    "step to change together, the number of candidates grows with a power of it); requires --window",
)
@click.option(
    "--alpha-vectors",
    "alpha_vectors",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="when the backward passes stop improving (before the windows are tried), run a backward pass that pools the value vectors of N policies: the policy, "
    "the unconstrained optimum and N-2 changes of the policy, whose suffixes the others can use, see the top of this file; 1 does not pool anything "
    "(not for conditional properties)",
)
@click.option(
    "--alpha-changes",
    "alpha_changes",
    type=click.IntRange(min=1),
    default=ALPHA_CHANGES,
    show_default=True,
    help="the number of parameters whose options a policy of the pool of --alpha-vectors changes at the same time (single changes come first, then the pairs, "
    "and so on; a larger number finds the improvements that need more parameters to change together, the number of candidates grows with a power of it); "
    "requires --alpha-vectors",
)
@click.option("--show-policy", is_flag=True, default=False, help="print the parameter assignment of each step")
@click.option("--export-policy", type=click.Path(), help="write the parameter assignment of each step to this JSON file")
@click.option("--debug", is_flag=True, default=False, help="print debug messages")
def main(
    model: str,
    properties: str,
    horizon: int | None,
    fsc_memory_size: int,
    max_iterations: int,
    tolerance: float,
    timeout: float | None,
    precision: float,
    no_unfold: bool,
    max_unfolded_states: int,
    ar: bool,
    ar_timeout: float | None,
    ar_warm_start: bool,
    dp: bool,
    dp_max_states: int,
    dp_timeout: float | None,
    max_window: int,
    changes: int,
    alpha_vectors: int,
    alpha_changes: int,
    show_policy: bool,
    export_policy: str | None,
    debug: bool,
) -> None:
    """Solve a color consistency query of MODEL (a PAYNT sketch, DRN or UMB file) and PROPERTIES.

    For a reachability probability or reward, the policy is color consistent within each step of the finite HORIZON but may change from step to step.

    For a conditional property P[F "goal" || F "condition"], the model has to be unfolded in time already (acyclic, the time is a part of its states and
    observations) and HORIZON is not given.
    """
    if ar and no_unfold:
        raise click.UsageError("--ar needs the unfolded colored MDP, it cannot be combined with --no-unfold")
    if ar_timeout is not None and not ar:
        raise click.UsageError("--ar-timeout requires --ar")
    if ar_warm_start and not ar:
        raise click.UsageError("--ar-warm-start requires --ar")
    paynt.cli.setup_logger()
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    paynt.utils.version_check.check_stormpy_compatibility()

    started = time.time()
    factory, task = paynt.parser.sketch.Sketch.load_sketch(model, properties, precision=precision, task_kwargs={"memory_size": fsc_memory_size})
    conditional = ConditionalObjective.is_conditional(task)
    if conditional and horizon is not None:
        raise click.UsageError("a conditional property is solved on a model that is already unfolded in time, do not give the HORIZON")
    if conditional and ar:
        raise click.UsageError("--ar does not support conditional properties")
    if not conditional and horizon is None:
        raise click.UsageError("the HORIZON is required (it is only omitted for conditional properties)")
    if dp and not conditional:
        raise click.UsageError("--dp is only for conditional properties")
    if conditional and alpha_vectors > 1:
        raise click.UsageError("--alpha-vectors does not support conditional properties")
    context = click.get_current_context()
    if not dp and any(context.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE for name in ("dp_max_states", "dp_timeout")):
        raise click.UsageError("--dp-max-states and --dp-timeout require --dp")
    if max_window == 1 and context.get_parameter_source("changes") == click.core.ParameterSource.COMMANDLINE:
        raise click.UsageError("--window-changes requires --window of at least 2")
    if alpha_vectors == 1 and context.get_parameter_source("alpha_changes") == click.core.ParameterSource.COMMANDLINE:
        raise click.UsageError("--alpha-changes requires --alpha-vectors of at least 2")

    colored_mdp = factory.build()
    try:
        if conditional:
            run_conditional(
                colored_mdp,
                task,
                max_iterations,
                tolerance,
                timeout,
                not no_unfold,
                dp,
                dp_max_states,
                dp_timeout,
                show_policy,
                export_policy,
                max_window,
                changes,
            )
        else:
            assert horizon is not None
            run_time_dependent(
                colored_mdp, task, horizon, max_iterations, tolerance, timeout, not no_unfold, max_unfolded_states, ar, ar_timeout, show_policy, export_policy,
                max_window, changes, ar_warm_start, alpha_vectors, alpha_changes,
            )  # fmt: skip
    except UnsupportedProblemError as e:
        raise click.ClickException(str(e)) from e
    logger.info(f"done in {time.time() - started:.2f}s")


if __name__ == "__main__":
    main()
