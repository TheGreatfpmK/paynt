from __future__ import annotations

from dataclasses import dataclass

import paynt.result

from .decision_tree import DecisionTree

import logging

logger = logging.getLogger(__name__)


@dataclass
class DtResult(paynt.result.Result):
    tree: DecisionTree | None = None
