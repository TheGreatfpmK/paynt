"""Pure unit tests for PartialModelCache's mercury-settrie-backed subset/superset subsumption -- no Storm, no ColoredMdp, just the cache's own {parameter:
option} dict contract.

See cache.py's module docstring for the subsumption rules these tests exercise.
"""

from __future__ import annotations

import paynt.synthesizer.smpmc.cache as cache_module
from paynt.synthesizer.smpmc.cache import MISS, PartialModelCache


class TestExactMatch:
    """Baseline behavior a subset/superset-aware cache must still get right: an unrelated query misses, and an identical query hits."""

    def test_lookup_on_empty_cache_misses(self):
        cache = PartialModelCache()
        assert cache.lookup({0: 1, 1: 2}, True, epoch=0) is MISS

    def test_exact_match_refuted_lookup_hits(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1, 1: 2}, True, conflict_parameters=[0, 1])
        assert cache.lookup({0: 1, 1: 2}, True, epoch=0) == [0, 1]

    def test_exact_match_inconclusive_lookup_hits(self):
        cache = PartialModelCache()
        cache.insert_inconclusive({0: 1, 1: 2}, True, epoch=0)
        assert cache.lookup({0: 1, 1: 2}, True, epoch=0) is None

    def test_different_option_on_the_same_parameter_misses(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, True, conflict_parameters=[0])
        assert cache.lookup({0: 2}, True, epoch=0) is MISS


class TestRefutedSubsetSubsumption:
    """A refutation keyed by a *subset* of the query refutes the query too -- fixing more parameters can only narrow the induced sub-MDP further, so an already-
    infeasible region stays infeasible."""

    def test_a_superset_query_reuses_a_smaller_cached_refutation(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, True, conflict_parameters=[0])
        assert cache.lookup({0: 1, 1: 2, 2: 3}, True, epoch=0) == [0]

    def test_a_subset_query_does_not_reuse_a_larger_cached_refutation(self):
        """The opposite direction is not a valid subsumption: a larger fixed set being refuted says nothing about a smaller (less constrained) one."""
        cache = PartialModelCache()
        cache.insert_refuted({0: 1, 1: 2}, True, conflict_parameters=[0, 1])
        assert cache.lookup({0: 1}, True, epoch=0) is MISS

    def test_the_smallest_matching_conflict_is_returned(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1, 1: 2, 2: 3}, True, conflict_parameters=[0, 1, 2])
        cache.insert_refuted({0: 1}, True, conflict_parameters=[0])
        assert cache.lookup({0: 1, 1: 2, 2: 3, 3: 4}, True, epoch=0) == [0]

    def test_a_non_matching_option_on_a_shared_parameter_does_not_subsume(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, True, conflict_parameters=[0])
        assert cache.lookup({0: 2, 1: 2}, True, epoch=0) is MISS

    def test_refuted_entries_are_never_invalidated_by_an_epoch_change(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, True, conflict_parameters=[0])
        assert cache.lookup({0: 1}, True, epoch=5) == [0]


class TestRefutedCompaction:
    """Inserting a smaller (more general) refutation should retire any existing larger, now-redundant cached entry it subsumes -- mirrors molehill's own insert-
    time trie compaction."""

    def test_inserting_a_subset_conflict_retires_a_previously_cached_superset(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1, 1: 2}, True, conflict_parameters=[0, 1])
        assert len(cache._refuted[1]) == 1
        cache.insert_refuted({0: 1}, True, conflict_parameters=[0])
        assert len(cache._refuted[1]) == 1, "the now-redundant {0:1,1:2} entry should have been removed"
        assert cache.lookup({0: 1, 1: 2}, True, epoch=0) == [0], "the surviving entry must still answer the old query"

    def test_inserting_a_superset_conflict_does_not_disturb_an_existing_subset(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, True, conflict_parameters=[0])
        cache.insert_refuted({0: 1, 1: 2}, True, conflict_parameters=[0, 1])
        assert len(cache._refuted[1]) == 2, "the new, strictly-larger entry is not redundant and should be kept"


class TestInconclusiveSupersetSubsumption:
    """An inconclusive verdict keyed by a *superset* of the query means the query is inconclusive too: if a more-constrained region wasn't refuted, a less-
    constrained one (whose completions are a superset) can't be either."""

    def test_a_subset_query_reuses_a_larger_cached_inconclusive_verdict(self):
        cache = PartialModelCache()
        cache.insert_inconclusive({0: 1, 1: 2}, True, epoch=0)
        assert cache.lookup({0: 1}, True, epoch=0) is None

    def test_a_superset_query_does_not_reuse_a_smaller_cached_inconclusive_verdict(self):
        cache = PartialModelCache()
        cache.insert_inconclusive({0: 1}, True, epoch=0)
        assert cache.lookup({0: 1, 1: 2}, True, epoch=0) is MISS

    def test_inconclusive_entries_are_invalidated_by_an_epoch_change(self):
        cache = PartialModelCache()
        cache.insert_inconclusive({0: 1, 1: 2}, True, epoch=0)
        assert cache.lookup({0: 1}, True, epoch=0) is None
        assert cache.lookup({0: 1}, True, epoch=1) is MISS, "a tightened threshold must invalidate a stale inconclusive verdict"

    def test_a_fresh_insert_after_an_epoch_change_is_usable_again(self):
        cache = PartialModelCache()
        cache.insert_inconclusive({0: 1}, True, epoch=0)
        cache.insert_inconclusive({0: 2}, True, epoch=1)
        assert cache.lookup({0: 2}, True, epoch=1) is None
        assert cache.lookup({0: 1}, True, epoch=1) is MISS


class TestCrossPolaritySubsumption:
    """A refutation for the *opposite* polarity, whether a subset or superset of the query, can only ever make the current polarity's query inconclusive --
    never a false refutation.

    See cache.py's module docstring for the derivation.
    """

    def test_an_opposite_polarity_subset_refutation_makes_the_query_inconclusive(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, False, conflict_parameters=[0])
        assert cache.lookup({0: 1, 1: 2}, True, epoch=0) is None

    def test_an_opposite_polarity_superset_refutation_makes_the_query_inconclusive(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1, 1: 2}, False, conflict_parameters=[0, 1])
        assert cache.lookup({0: 1}, True, epoch=0) is None

    def test_an_exact_opposite_polarity_refutation_makes_the_query_inconclusive(self):
        """The most common real case: a full assignment's viable(fixed) is refuted, so the subsequent not_viable(fixed) query on the exact same fixed set must
        come back inconclusive, never refuted -- the two literals are logical negations of each other on a fully-fixed assignment."""
        cache = PartialModelCache()
        cache.insert_refuted({0: 1, 1: 2}, True, conflict_parameters=[0, 1])
        assert cache.lookup({0: 1, 1: 2}, False, epoch=0) is None

    def test_cross_polarity_reuse_never_produces_a_false_refutation(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, False, conflict_parameters=[0])
        result = cache.lookup({0: 1, 1: 2}, True, epoch=0)
        assert result is None, "cross-polarity information must never manufacture a positive refutation"

    def test_same_polarity_refutation_still_takes_priority_and_returns_a_conflict(self):
        """If the query is refuted for its *own* polarity too, that (more useful) answer must win over the weaker cross-polarity "just inconclusive"
        shortcut."""
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, True, conflict_parameters=[0])
        cache.insert_refuted({1: 2}, False, conflict_parameters=[1])
        assert cache.lookup({0: 1, 1: 2}, True, epoch=0) == [0]


class TestSetTrieBackedStructure:
    """Confirms the cache is actually backed by mercury-settrie, not a reimplementation -- the whole point of this port was fidelity to molehill's own caching
    mechanism."""

    def test_refuted_tries_are_mercury_settrie_instances(self):
        cache = PartialModelCache()
        import settrie

        assert all(isinstance(trie, settrie.SetTrie) for trie in cache._refuted)
        assert all(isinstance(trie, settrie.SetTrie) for trie in cache._inconclusive)

    def test_module_imports_settrie_directly(self):
        assert hasattr(cache_module, "settrie")
