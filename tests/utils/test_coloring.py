"""paynt.utils.coloring: telling a ColoringGeneral from a coloring given by explicit (parameter, option) pairs, and refusing the former where the pairs are
needed.

Both functions take the raw payntbind coloring, so they are exercised on the two lifted fixtures' colorings the way the consumers use them.
"""

from __future__ import annotations

import pytest

import paynt.utils.coloring

from helpers.helper import general_colored_mdp, load_colored_mdp

FIXTURES = ["tests/smpmc-tiny", "tests/mdp-family-avoid-8-2-easy"]


@pytest.mark.parametrize("project", FIXTURES)
class TestIsGeneralColoring:
    def test_false_for_a_standard_coloring_and_true_once_lifted(self, project):
        colored_mdp, _task = load_colored_mdp(project)
        assert paynt.utils.coloring.is_general_coloring(colored_mdp.coloring) is False
        assert paynt.utils.coloring.is_general_coloring(general_colored_mdp(colored_mdp).coloring) is True

    def test_is_what_the_colored_mdp_property_reports(self, project):
        colored_mdp, _task = load_colored_mdp(project)
        for candidate in (colored_mdp, general_colored_mdp(colored_mdp)):
            assert candidate.has_general_coloring is paynt.utils.coloring.is_general_coloring(candidate.coloring)


@pytest.mark.parametrize("project", FIXTURES)
class TestRequirePairListColoring:
    def test_accepts_a_standard_coloring(self, project):
        colored_mdp, _task = load_colored_mdp(project)
        paynt.utils.coloring.require_pair_list_coloring(colored_mdp.coloring, "anything")
        paynt.utils.coloring.require_pair_list_coloring(colored_mdp.coloring, "anything", "do something else")

    def test_rejects_a_general_coloring_naming_what_needed_the_pairs(self, project):
        colored_mdp, _task = load_colored_mdp(project)
        with pytest.raises(NotImplementedError) as error:
            paynt.utils.coloring.require_pair_list_coloring(general_colored_mdp(colored_mdp).coloring, "the policy tree")
        assert str(error.value) == "cannot use a general coloring for the policy tree: it has no (parameter, option) pairs to read back"

    def test_appends_the_alternative_when_there_is_one(self, project):
        colored_mdp, _task = load_colored_mdp(project)
        with pytest.raises(NotImplementedError) as error:
            paynt.utils.coloring.require_pair_list_coloring(general_colored_mdp(colored_mdp).coloring, "the policy tree", "use another method")
        assert str(error.value).endswith("no (parameter, option) pairs to read back; use another method")
