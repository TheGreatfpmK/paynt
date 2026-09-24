"""Algorithm 1 "CheckMDPs" (arXiv:2511.08078): decide whether a `viable(...)`/`not viable(...)` literal
over a partial parameter assignment is refuted, by model-checking the induced sub-MDP with Storm.

Ported from molehill's Mole.partial_model_consistent + counterexamples.check
(https://github.com/linusheck/molehill, GPL-3.0), but built directly on PAYNT's own primitives instead
of molehill's bespoke fastmole/MatrixGenerator and modelchecker.py:
  - the induced sub-MDP C[eta] is ColoredMdp.build(eta) (paynt/colored_mdp.py), not a separate C++
    extension -- PAYNT's own coloring.selectCompatibleChoices already does this;
  - V^max/V^min of C[eta] is SubMdp.model_check_property(prop, alt=...) (paynt/model/model.py), which
    already gives both optimization directions via prop.formula/prop.formula_alt -- the same mechanism
    AR already uses for its own primary/secondary bounds;
  - the conflict-minimization step (paper's Theorem 6: drop parameters not reachable in C[eta]) uses
    coloring.getStateToHoles() unioned over the induced sub-MDP's reachable states, rather than porting
    molehill's own (partly dead-code) binary-search minimizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import payntbind.synthesis
import stormpy

import paynt.colored_mdp
import paynt.specification.property
import paynt.specification.property_result
import paynt.synthesizer.smpmc.cache
import paynt.synthesizer.statistic

import logging

logger = logging.getLogger(__name__)

#: relative-error threshold above which a value-iteration result is treated as non-converged rather than
#: as ordinary floating-point disagreement between two different (but both legitimate) numerical methods.
#: The failure mode this guards against is not subtle -- a capped, non-converged VI result differs from
#: the true (policy-iteration-computed) value by orders of magnitude, not by a few ULPs -- so this can
#: afford to be generous without risking false positives.
_REWARD_CONVERGENCE_RELATIVE_TOLERANCE = 0.01

#: sub-MDP state-count above which a policy-iteration verification call is skipped entirely, rather than
#: attempted with an iteration cap. Confirmed empirically that capping iterations (both the outer minmax
#: loop and the inner linear-equation solve each outer iteration depends on) does not help here: on a
#: 1,652,566-state model, even a single outer iteration capped to just 10 inner iterations took over 40
#: seconds, and every larger cap tried was worse -- the cost is dominated by the sheer size of the sparse
#: system a single call has to set up and touch, not by how many iterations run. Below this threshold,
#: verification is cheap and valuable (the feature's own motivating case, generic-maze, is 183 states);
#: at or above it, an unconditional per-direction PI cross-check would make SMPMC impractical on
#: realistically large models, for a safety net AR/CEGIS/Hybrid already go without everywhere. Calibrated
#: from only two real data points (183 states: fine; 1.65M states: catastrophic), so treat this as a
#: coarse heuristic, not a precisely-tuned value -- revisit if a real model lands in between and disagrees
#: with it either way.
_PI_VERIFICATION_MAX_STATES = 50_000


@dataclass
class TheoryResult:
    """A refutation: the Theorem-6-minimized set of parameters responsible, and the value that triggered
    it (kept only for logging/stats)."""

    conflict_parameters: list[int]
    value: Any


class ColoredMdpTheory:
    """Owns the Storm-calling half of the theory solver. Stateless across polarities/literals except for
    the epoch counter (bumped once per optimum update, see the optimality loop in synthesizer.py) and the
    result cache -- one instance is shared by every SmpmcPropagator produced via Z3's fresh() during MBQI,
    so the cache (and epoch) stay consistent across quantifier-instantiation sub-contexts.
    """

    def __init__(
        self,
        colored_mdp: paynt.colored_mdp.ColoredMdp,
        prop: paynt.specification.property.Property,
        stat: paynt.synthesizer.statistic.Statistic | None = None,
    ):
        self.colored_mdp = colored_mdp
        self.prop = prop
        # optional: not needed by the propagator tests (test_propagator.py -- a fake theory that never
        # touches this class at all) or the pure checker.check() tests (test_theory.py -- real Storm calls
        # but no interest in progress reporting), only by a real synthesis run (synthesizer.py always
        # passes one). When present, every model check feeds paynt.synthesizer.statistic.Statistic's
        # existing DTMC/MDP iteration counters and throttled status-line logging (the same mechanism
        # AR/CEGIS already use via self.stat.iteration(...)), and every fresh refutation accumulates an
        # "explored" estimate onto the synthesizer, mirroring SmtSolver.exclude_conflict's
        # pruning_estimate for CEGIS -- see check()'s use of it below.
        self.stat = stat
        # kappa restricted to "which parameters are relevant at this state", unioned over an induced
        # sub-MDP's reachable states to minimize a conflict (Theorem 6) -- one BitVector of parameter
        # indices per underlying-MDP state
        self.state_to_parameters: list[Any] = colored_mdp.coloring.getStateToHoles()
        self.cache = paynt.synthesizer.smpmc.cache.PartialModelCache()
        # bumped by the optimality loop (synthesizer.py) each time the threshold tightens, so cached
        # "inconclusive" verdicts computed under a looser threshold aren't wrongly reused -- see
        # PartialModelCache's docstring. Refutations need no such tagging: they stay valid as the
        # threshold only ever tightens.
        self.epoch = 0
        self.mc_calls = 0

        # Set once the first fully-fixed parameter assignment has been checked (via check()), regardless
        # of its verdict -- see theory.py's _analyse(), which skips every *partial* query until this is
        # True, mirroring molehill's own "check a DTMC first" heuristic. Lives here (not on the
        # propagator) because it's a property of the search as a whole, shared across every
        # SmpmcPropagator instance fresh() creates during MBQI, exactly like epoch/cache.
        self.first_full_assignment_checked = False

        # Reward properties only (see _model_check): PAYNT's default model-checking environment
        # (paynt.specification.property.Property.environment) caps value-iteration at a fixed number of
        # iterations to avoid hanging forever (see the plan's Phase 0), but on some models one direction
        # of a reward property converges pathologically slowly -- capped VI then silently returns whatever
        # partial snapshot it reached, which can be wrong by orders of magnitude and is unsound to use for
        # a refutation decision. Verified once per direction (alt=False/True), lazily, against policy
        # iteration (which solves the exact fixed-point system directly, so it isn't affected by the same
        # slow-convergence failure mode as VI -- see _pi_environment for its own, separate iteration cap);
        # whichever method actually works for that direction is then used for the rest of this
        # ColoredMdpTheory's lifetime, so the comparison cost is paid at most twice per run, not once per
        # query. Scoped locally to SMPMC rather than changing paynt.specification.property's shared
        # default, since AR/CEGIS/Hybrid already tolerate this (their bounds only affect pruning, and any
        # answer they report gets independently re-verified as a concrete DTMC before being trusted) in a
        # way SMPMC's Z3-learned refutations do not.
        self._reward_pi_environment: Any = None
        self._reward_convergence_verified: dict[bool, bool] = {}
        self._reward_use_policy_iteration: dict[bool, bool] = {}
        # throttles the "skipping policy-iteration verification" log line (see _model_check) to once per
        # direction -- purely a logging concern, unlike _reward_convergence_verified this never suppresses
        # a retry of the actual (cheap) size check on a later, possibly smaller sub-MDP.
        self._reward_verification_skip_warned: dict[bool, bool] = {}

    def check(self, fixed: dict[int, int], polarity: bool) -> TheoryResult | None:
        """:param fixed: {parameter_index: option} for every parameter currently decided in the partial
            model Z3 is exploring.
        :param polarity: True to check the `viable(...)` literal, False for `not viable(...)`.
        :returns: None if inconclusive (this partial assignment could still go either way), else a
            TheoryResult describing the refutation to hand back to Z3 as a learned conflict.
        """
        # The BitVec encoding's width has spare headroom beyond what any parameter's option count needs
        # (see paynt/parameter_space/bitvec.py), so Z3 can transiently fix a variable to a value outside
        # its actual option range while still exploring whether the plain Boolean-level in_range()
        # constraint holds -- that constraint alone will reject such a branch without any help from this
        # theory, so the correct (and, empirically, necessary -- an out-of-range option previously reached
        # ColoredMdp.build() and crashed stormpy's submodel construction with a deadlock-state error) move
        # here is to treat it as inconclusive and let ordinary constraint solving handle it.
        for parameter, option in fixed.items():
            if option not in self.colored_mdp.parameter_space.parameter_options(parameter):
                return None

        cached = self.cache.lookup(fixed, polarity, self.epoch)
        if cached is not paynt.synthesizer.smpmc.cache.MISS:
            return TheoryResult(cached, None) if cached is not None else None

        eta = self.colored_mdp.parameter_space.copy()
        for parameter, option in fixed.items():
            eta.parameter_set_options(parameter, [option])

        if eta.size == 1:
            # Every parameter is fixed. ColoredMdp.build() would still hand back an MDP-typed model here
            # (even though it has exactly one choice per state) -- and solving that via minmax/policy
            # iteration can disagree with the exact DTMC computation build_assignment() gives for the very
            # same deterministic process, by an amount just past model_checking_precision (confirmed
            # empirically: two different underlying linear-equation solves of a mathematically identical
            # system, not a convergence failure -- policy iteration is exact either way). That tiny
            # disagreement is enough to make a non-improving witness look viable to the theory when it
            # shouldn't. build_assignment() sidesteps this by computing the exact value directly (and is
            # cheaper besides). Only "family"/"pomdp_family" ColoredMdps keep genuine nondeterminism even
            # once every declared parameter is fixed (the agent's policy isn't itself a declared
            # parameter there) -- build_assignment() already knows this and returns an MDP for those, so
            # this shortcut is correct unconditionally, not just for the common case.
            sub_mdp = self.colored_mdp.build_assignment(eta)
        else:
            # deliberately no parent_selected_choices reuse hint here: that optimization assumes
            # monotonically-nested parameter spaces (true for AR's splitting), which does not hold for
            # Z3's arbitrary CDCL backtracking -- see the plan's design-decisions section.
            sub_mdp, _selected_choices = self.colored_mdp.build(eta)
        self.mc_calls += 1
        if self.stat is not None:
            # classifies DTMC vs MDP by inspecting sub_mdp.model's stormpy type and throttles the actual
            # log line to once every Statistic.status_period_seconds -- exactly what AR/CEGIS get from the
            # same call
            self.stat.iteration(sub_mdp)

        # alt=(polarity is False): prop.formula is already set (via minimizing/optimality_type) to
        # compute the value in *this property's own success direction* -- so alt=False always gives the
        # best-case value for the `viable` check, and alt=True (formula_alt, the opposite direction)
        # always gives the worst-case value for the `not viable` check, regardless of whether prop is a
        # minimizing or maximizing property. See Algorithm 1: (viable -> max, "<"), (not viable -> min, ">=").
        result = self._model_check(sub_mdp, alt=(polarity is False))

        if result.sat is polarity:
            # best case (viable) still meets the threshold, or worst case (not viable) still misses it --
            # this partial assignment does not yet decide the literal either way
            self.cache.insert_inconclusive(fixed, polarity, self.epoch)
            return None

        # refuted: minimize the conflict by dropping every fixed parameter that isn't reachable (hence
        # irrelevant) in the induced sub-MDP
        relevant_parameters: set[int] = set()
        for state in sub_mdp.underlying_mdp_state_map:
            relevant_parameters.update(self.state_to_parameters[state])
        conflict_parameters = [parameter for parameter in fixed if parameter in relevant_parameters]

        if self.stat is not None:
            # "explored" estimate for this refutation: the number of complete assignments it rules out is
            # the product of every *other* parameter's option count, since conflict_parameters is exactly
            # the (Theorem-6-minimized) set that had to be fixed to reach this refutation -- any value of
            # every parameter *not* in it is still covered. Mirrors SmtSolver.exclude_conflict's own
            # pruning_estimate for CEGIS (paynt/parameter_space/smt.py), including its same caveat: since
            # nothing guarantees learned conflicts are pairwise disjoint, this is an estimate that can
            # double-count overlapping regions, not an exact count -- CEGIS's existing "explored" figure
            # has always carried the identical caveat, so this isn't a new source of imprecision, just the
            # same one on a new engine. Only charged once per fresh refutation (this line is unreachable
            # from the cache-hit path above), never on a cache replay of the same conflict.
            pruning_estimate = math.prod(
                self.colored_mdp.parameter_space.parameter_num_options(parameter)
                for parameter in range(self.colored_mdp.parameter_space.num_parameters)
                if parameter not in conflict_parameters
            )
            self.stat.synthesizer.explored += pruning_estimate

        self.cache.insert_refuted(fixed, polarity, conflict_parameters)
        return TheoryResult(conflict_parameters, result.value)

    def _model_check(self, sub_mdp: Any, alt: bool) -> paynt.specification.property_result.PropertyResult:
        """Like sub_mdp.model_check_property(self.prop, alt=alt), but for reward properties verifies (once
        per direction) that PAYNT's default capped value-iteration environment actually converged, falling
        back to policy iteration for this direction's remaining queries if it didn't. See __init__."""
        if not self.prop.reward:
            return sub_mdp.model_check_property(self.prop, alt=alt)

        if self._reward_use_policy_iteration.get(alt, False):
            return self._model_check_with_environment(sub_mdp, alt, self._pi_environment())

        vi_result = sub_mdp.model_check_property(self.prop, alt=alt)
        if self._reward_convergence_verified.get(alt, False):
            return vi_result

        if sub_mdp.model.nr_states >= _PI_VERIFICATION_MAX_STATES:
            # See _PI_VERIFICATION_MAX_STATES: a single PI call is not affordable on a sub-MDP this large,
            # regardless of iteration caps, so there is nothing to verify against here -- trust VI for
            # *this* query, but deliberately do NOT set _reward_convergence_verified[alt]: a later query
            # for this same direction can land on a much smaller sub-MDP (a full assignment is exactly
            # such a case, and skips straight to build_assignment()'s DTMC), where verification is cheap
            # again and still worth actually doing. Marking this direction "verified" here (as an earlier
            # version of this method did) would have permanently disabled the safety net for the rest of
            # the run the moment the very first query happened to be large -- confirmed as a real bug: it
            # let a catastrophically wrong VI value on a full assignment (a DTMC, ~10^180 vs the true
            # ~10^-13) through unquestioned, because "verified" had already been latched on an earlier,
            # skipped, much larger query.
            if not self._reward_verification_skip_warned.get(alt, False):
                self._reward_verification_skip_warned[alt] = True
                logger.warning(
                    f"skipping policy-iteration verification for {'the alt' if alt else 'the primary'} "
                    f"direction of {self.prop}: the induced sub-MDP has {sub_mdp.model.nr_states} states, "
                    "too large to verify affordably -- trusting value iteration's own result until a "
                    "smaller sub-MDP makes verification affordable (this warning will not repeat)"
                )
            return vi_result

        self._reward_convergence_verified[alt] = True
        pi_result = self._model_check_with_environment(sub_mdp, alt, self._pi_environment())
        vi_value, pi_value = vi_result.value, pi_result.value
        relative_error = abs(vi_value - pi_value) / max(abs(pi_value), 1e-12)
        if relative_error <= _REWARD_CONVERGENCE_RELATIVE_TOLERANCE:
            return vi_result

        logger.warning(
            f"value iteration did not converge for {'the alt' if alt else 'the primary'} direction of "
            f"{self.prop}: got {vi_value}, policy iteration gives {pi_value}; using policy iteration for "
            "the rest of this run"
        )
        self._reward_use_policy_iteration[alt] = True
        return pi_result

    def _model_check_with_environment(self, sub_mdp: Any, alt: bool, environment: Any) -> paynt.specification.property_result.PropertyResult:
        formula = self.prop.formula if not alt else self.prop.formula_alt
        result = stormpy.model_checking(sub_mdp.model, formula, extract_scheduler=True, environment=environment)
        value = result.at(sub_mdp.initial_state)
        return paynt.specification.property_result.PropertyResult(self.prop, result, value)

    def _pi_environment(self) -> Any:
        if self._reward_pi_environment is None:
            env = stormpy.Environment()
            env.solver_environment.minmax_solver_environment.method = stormpy.MinMaxMethod.policy_iteration
            # Policy iteration solves the exact fixed-point system directly, so in principle it shouldn't
            # need an iteration cap at all -- but on some MDP structures (confirmed on a 165-state
            # POMDP-derived model) Storm's PI solver doesn't satisfy its own convergence check even after
            # 10000 iterations, so cap it too rather than trust it to always self-terminate. Empirically
            # (5 consecutive trials on that same model) the *value* it reports is identical to 7 significant
            # figures at this cap and takes well under a second, matching VI's own capped value on the same
            # model -- so a capped-but-technically-"non-converged" PI result is still trustworthy here.
            payntbind.synthesis.set_max_iterations_minmax(
                env.solver_environment.minmax_solver_environment, self.prop.max_minmax_iterations
            )
            self._reward_pi_environment = env
        return self._reward_pi_environment
