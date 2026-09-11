from __future__ import annotations

from dataclasses import dataclass

import paynt.result
from .policy_tree import PolicyTree


@dataclass
class PolicyTreeResult(paynt.result.Result):
    """
    value/assignment (inherited from Result) are always left None here: a policy tree is a set of
    region-specific policies covering the whole family, not a single answer the way a normal synthesizer
    result is -- see paynt.result.Result's own docstring for this convention. A future family-wide
    synthesizer (searching for one policy that works across every family member) would return a plain Result
    with a real value/assignment instead, once built.
    """

    policy_tree: PolicyTree | None = None
