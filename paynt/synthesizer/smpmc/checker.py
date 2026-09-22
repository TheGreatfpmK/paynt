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

from dataclasses import dataclass
from typing import Any

import stormpy

import paynt.colored_mdp
import paynt.specification.property
import paynt.specification.property_result
import paynt.synthesizer.smpmc.cache

import logging

logger = logging.getLogger(__name__)

#: relative-error threshold above which a value-iteration result is treated as non-converged rather than
#: as ordinary floating-point disagreement between two different (but both legitimate) numerical methods.
#: The failure mode this guards against is not subtle -- a capped, non-converged VI result differs from
#: the true (policy-iteration-computed) value by orders of magnitude, not by a few ULPs -- so this can
#: afford to be generous without risking false positives.
_REWARD_CONVERGENCE_RELATIVE_TOLERANCE = 0.01


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

    def __init__(self, colored_mdp: paynt.colored_mdp.ColoredMdp, prop: paynt.specification.property.Property):
        self.colored_mdp = colored_mdp
        self.prop = prop
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

        # Reward properties only (see _model_check): PAYNT's default model-checking environment
        # (paynt.specification.property.Property.environment) caps value-iteration at a fixed number of
        # iterations to avoid hanging forever (see the plan's Phase 0), but on some models one direction
        # of a reward property converges pathologically slowly -- capped VI then silently returns whatever
        # partial snapshot it reached, which can be wrong by orders of magnitude and is unsound to use for
        # a refutation decision. Verified once per direction (alt=False/True), lazily, against policy
        # iteration (which solves the exact fixed-point system directly, so it isn't affected by any
        # iteration cap); whichever method actually works for that direction is then used for the rest of
        # this ColoredMdpTheory's lifetime, so the comparison cost is paid at most twice per run, not once
        # per query. Scoped locally to SMPMC rather than changing paynt.specification.property's shared
        # default, since AR/CEGIS/Hybrid already tolerate this (their bounds only affect pruning, and any
        # answer they report gets independently re-verified as a concrete DTMC before being trusted) in a
        # way SMPMC's Z3-learned refutations do not.
        self._reward_pi_environment: Any = None
        self._reward_convergence_verified: dict[bool, bool] = {}
        self._reward_use_policy_iteration: dict[bool, bool] = {}

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
            self._reward_pi_environment = env
        return self._reward_pi_environment
