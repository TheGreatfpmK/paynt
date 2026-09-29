"""Telling apart the two ways a colored MDP's choices can be colored: by explicit (parameter, option) pairs, as in a payntbind Coloring, or by an arbitrary
formula, as in a ColoringGeneral (see paynt.utils.coloring_builder), which has no pairs to read back."""

from __future__ import annotations

from typing import Any

import payntbind.synthesis


def is_general_coloring(coloring: Any) -> bool:
    """Whether coloring is a payntbind ColoringGeneral rather than a coloring given by explicit (parameter, option) pairs."""
    return isinstance(coloring, payntbind.synthesis.ColoringGeneral)


def require_pair_list_coloring(coloring: Any, what: str, alternative: str | None = None) -> None:
    """Raise NotImplementedError if coloring is a general coloring.

    For code that reads the coloring's (parameter, option) pairs back (getChoiceToAssignment, collectHoleOptions, ...): a general coloring is an arbitrary
    formula per choice, so it has none.

    :param what: what needs the pairs, as a noun phrase, e.g. "the policy tree"
    :param alternative: what to do instead, if anything
    """
    if is_general_coloring(coloring):
        message = f"cannot use a general coloring for {what}: it has no (parameter, option) pairs to read back"
        raise NotImplementedError(message if alternative is None else f"{message}; {alternative}")
