from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from paynt.dt.task import DtTask


@dataclass(kw_only=True)
class DtNestTask(DtTask):
    tree_depth: int = 7  # overrides DtTask's own default of 0, keeping its original field position
    error_threshold: float = 0.05
    initial_tree: Any = None
    max_subtree_depth: int = 7
