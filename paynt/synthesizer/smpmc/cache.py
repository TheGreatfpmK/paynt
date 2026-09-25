"""Memo for ColoredMdpTheory.check() results -- Algorithm 1's per-literal conflict cache
(arXiv:2511.08078). Ported from molehill's Mole.all_violated_models/inconclusive_models
(https://github.com/linusheck/molehill, GPL-3.0), using the same mercury-settrie subset/superset
subsumption the reference implementation uses, rather than the exact-match memo this module started with
(see git history) -- one SetTrie per (refuted/inconclusive, polarity) combination, four total, matching
molehill's own [SetTrie(), SetTrie()] x2 structure.

Subsumption rules (derived directly from MDP choice-removal monotonicity -- fixing more parameters can
only narrow the induced sub-MDP's choices, which can only lower Vmax and raise Vmin):
  - a REFUTED entry for polarity p, keyed by a *subset* of the current query, refutes the query too: the
    same "every completion fails" conclusion only gets stronger as more parameters are pinned down.
  - an INCONCLUSIVE entry for polarity p, keyed by a *superset* of the current query, means the query is
    inconclusive too: if a more-constrained region wasn't already refuted, a less-constrained one (whose
    completions are a superset of the more-constrained region's) can't be either.
  - a REFUTED entry for the *opposite* polarity (1-p), whether a subset or a superset of the query, can
    only ever be used to conclude "inconclusive" for polarity p, never a refutation -- so both directions
    are safe to check without the risk of manufacturing a false conflict (see the conversation this was
    ported in for the full derivation). Ported faithfully since molehill's Mole.partial_model_consistent
    does the same cross-polarity check, but molehill's *additional* step of mining a fresh cross-polarity
    refutation out of a witness scheduler on every inconclusive result is not ported here -- that needs
    scheduler-to-parameter mapping PAYNT's model-checking wrapper doesn't currently expose, and is a
    separate, larger piece of work from "port the subsumption cache" if wanted later.

One deliberate correctness addition beyond a literal port: molehill has no notion of a live-tightening
threshold (its own TODO confirms optimality search was never implemented), so its inconclusive tries never
need invalidating. PAYNT's SMPMC supports optimality objectives, where a threshold tightening can turn a
previously-inconclusive verdict wrong (see the epoch mechanism this module already had before the
mercury-settrie port). Reusing an inconclusive entry across a subset/superset match must stay bound by the
exact same rule, so instead of tagging every entry with the epoch it was computed under, the two
inconclusive tries are simply replaced with fresh, empty ones whenever the epoch advances -- cheaper than
per-entry filtering, and correct because every entry computed under an old epoch becomes simultaneously
unusable the moment the epoch changes. Refuted entries need no such handling: they stay valid forever, as
the feasible region only ever shrinks while a threshold tightens.

Two other deliberate deviations from a byte-for-byte port, both behavior-preserving: mercury-settrie's
`id` must be a string, so an entry's payload needs the same stringify/parse round-trip molehill uses --
but ast.literal_eval (safe, stdlib) replaces molehill's bare eval() for parsing it back. And trie elements
are f"{parameter}={option}" strings, matching molehill's own f"{name}={option}" encoding exactly (adapted
from a parameter *name* to PAYNT's integer parameter index) rather than plain (parameter, option) tuples --
confirmed empirically that mercury-settrie's subset/superset comparison does not treat tuple elements
correctly (silently ignores everything but the first component, so e.g. (1, 0) and (1, 1) collide),
so this isn't just fidelity to the reference implementation, it's a required workaround: molehill's own
choice of string-encoded elements over a more natural tuple encoding was very likely for this exact reason.
"""

from __future__ import annotations

import ast
import enum
from typing import Final

import settrie


class _Miss(enum.Enum):
    MISS = enum.auto()


# a single-member enum rather than object(), so that `is MISS` narrows lookup()'s result for mypy
MISS: Final = _Miss.MISS


def _key(fixed: dict[int, int]) -> set[str]:
    return {f"{parameter}={option}" for parameter, option in fixed.items()}


class PartialModelCache:
    """Caches ColoredMdpTheory.check(fixed, polarity) verdicts, with subset/superset subsumption so a single refutation (or inconclusive verdict) can answer
    many future queries without another Storm call.

    See module docstring for the subsumption rules and the epoch-driven reset used for inconclusive entries.
    """

    def __init__(self) -> None:
        # index 0 -> polarity False, index 1 -> polarity True, matching molehill's int(invert)/1-int(invert)
        # indexing translated to PAYNT's polarity (no negated-spec "invert" concept here, just direct
        # polarity indexing).
        self._refuted: list[settrie.SetTrie] = [settrie.SetTrie(), settrie.SetTrie()]
        self._inconclusive: list[settrie.SetTrie] = [settrie.SetTrie(), settrie.SetTrie()]
        self._inconclusive_epoch: int | None = None

    def _sync_epoch(self, epoch: int) -> None:
        if epoch != self._inconclusive_epoch:
            self._inconclusive = [settrie.SetTrie(), settrie.SetTrie()]
            self._inconclusive_epoch = epoch

    def lookup(self, fixed: dict[int, int], polarity: bool, epoch: int) -> list[int] | None | _Miss:
        """:returns: the cached conflict-parameter list on a cached (or subsumed) refutation, None on a
        still-valid cached (or subsumed) inconclusive verdict, or the MISS sentinel if nothing usable
        is cached."""
        self._sync_epoch(epoch)
        key = _key(fixed)
        p = int(polarity)

        conflicts = list(self._refuted[p].subsets(key))
        if conflicts:
            return min((ast.literal_eval(c) for c in conflicts), key=len)

        if any(self._inconclusive[p].supersets(key)):
            return None

        # cross-polarity: a subset or superset already proven refuted for the *opposite* polarity means
        # this region is entirely one thing or the other -- either way, that can only make the *current*
        # polarity's literal inconclusive, never refuted (see module docstring).
        other = 1 - p
        if any(self._refuted[other].subsets(key)) or any(self._refuted[other].supersets(key)):
            return None

        return MISS

    def insert_refuted(self, fixed: dict[int, int], polarity: bool, conflict_parameters: list[int]) -> None:
        key = {f"{parameter}={fixed[parameter]}" for parameter in conflict_parameters}
        p = int(polarity)
        # this new (already Theorem-6-minimized) conflict subsumes any existing cached entry that's a
        # superset of it -- anything the old, larger entry could answer, this smaller one answers too, so
        # drop the redundant entry to keep the trie lean (mirrors molehill's own insert-time compaction).
        for stale in list(self._refuted[p].supersets(key)):
            self._refuted[p].remove(stale)
        self._refuted[p].insert(key, str(sorted(conflict_parameters)))

    def insert_inconclusive(self, fixed: dict[int, int], polarity: bool, epoch: int) -> None:
        self._sync_epoch(epoch)
        key = _key(fixed)
        p = int(polarity)
        if self._inconclusive[p].find(key) == "":
            self._inconclusive[p].insert(key, str(len(self._inconclusive[p])))
