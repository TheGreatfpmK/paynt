"""Internal support for the Dec-POMDP feature."""

from __future__ import annotations

from typing import Any

import paynt.parameter_space.parameter_space

import logging

logger = logging.getLogger(__name__)


def assignment_to_fscs(info: Any, assignment: paynt.parameter_space.parameter_space.ParameterSpace) -> None:
    """
    TODO
    Stub: per-agent FSC extraction from a synthesized assignment is not yet implemented for Dec-POMDP -- see the JAIR-refactor plan's "Result-class steps
    (d)/(e)" note for why (no existing payntbind accessor cleanly maps a choice back to a per-agent action; reconciling row_joint_action/joint_actions/
    agent_row_action_option would need its own scoped investigation).

    Takes (info, assignment) to match paynt.pomdp._utils.assignment_to_fsc's calling convention for when this gets implemented for real -- note that will also
    require a real DecPomdpInfo dataclass, since feature_info is None for decpomdp today.
    """
    logger.info("Dec-POMDP per-agent FSC extraction from a synthesized assignment is not yet implemented.")
    return
