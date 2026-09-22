"""Memo for ColoredMdpTheory.check() results -- Algorithm 1's per-literal conflict cache
(arXiv:2511.08078), scoped down from molehill's SetTrie-based subset/superset subsumption to an
exact-match memo for Phase 1 (see the plan: upgrading to a small in-repo set-trie, without taking on
mercury-settrie as a dependency, is Phase 2 work once this facade is proven)."""

from __future__ import annotations

MISS = object()


class PartialModelCache:
    """Caches ColoredMdpTheory.check(fixed, polarity) verdicts, keyed on the exact {parameter: option}
    assignment and polarity queried.

    Refutations are always safe to reuse regardless of how many times the optimality threshold has since
    tightened: the feasible region only shrinks as the threshold tightens, so anything already proven
    infeasible stays infeasible. Inconclusive verdicts are not safe to reuse across a tightening -- a set
    that looked fine under a looser threshold may no longer be good enough -- so each inconclusive entry
    is tagged with the epoch it was computed under and only reused within that same epoch.
    """

    def __init__(self) -> None:
        self._refuted: dict[tuple[frozenset[tuple[int, int]], bool], list[int]] = {}
        self._inconclusive: dict[tuple[frozenset[tuple[int, int]], bool], int] = {}

    @staticmethod
    def _key(fixed: dict[int, int], polarity: bool) -> tuple[frozenset[tuple[int, int]], bool]:
        return frozenset(fixed.items()), polarity

    def lookup(self, fixed: dict[int, int], polarity: bool, epoch: int) -> list[int] | None | object:
        """:returns: the cached conflict-parameter list on a cached refutation, None on a still-valid
        cached inconclusive verdict, or the MISS sentinel if nothing usable is cached."""
        key = self._key(fixed, polarity)
        if key in self._refuted:
            return self._refuted[key]
        if self._inconclusive.get(key) == epoch:
            return None
        return MISS

    def insert_refuted(self, fixed: dict[int, int], polarity: bool, conflict_parameters: list[int]) -> None:
        self._refuted[self._key(fixed, polarity)] = conflict_parameters

    def insert_inconclusive(self, fixed: dict[int, int], polarity: bool, epoch: int) -> None:
        self._inconclusive[self._key(fixed, polarity)] = epoch
