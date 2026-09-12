from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from paynt.dt.task import DtTask


@dataclass(kw_only=True)
class DtNestTask(DtTask):
    """
    Still inherits from DtTask (unlike the SynthesisTask/feature-task split, which has no inheritance at
    all): dtnest's build concern is a strict superset of plain DT's (it re-unfolds a fresh DtColoredMdp per
    subtree exactly like plain DT does per depth), so this is a build-to-build relationship, not a
    build/synthesis one. Field assignment (including forwarding shared fields like tree_depth up to DtTask)
    is entirely handled by dataclass inheritance now -- no hand-written super().__init__(...) call needed.
    """

    tree_depth: int = 7  # overrides DtTask's own default of 0, keeping its original field position
    error_threshold: float = 0.05
    initial_tree: Any = None
    max_subtree_depth: int = 7
