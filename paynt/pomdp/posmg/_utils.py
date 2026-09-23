"""Internal support for the POSMG feature: the PosmgInfo companion dataclass (attached as ColoredMdp.feature_info for feature_kind "posmg") and functions that
interpret it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import stormpy

import paynt.model.model
import paynt.parameter_space.parameter_space

import logging

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class PosmgInfo:
    # needed by create_smg_from_mdp to recover each state's player index -- the coloring/parameter_space
    # alone don't carry this
    posmg_manager: Any


def assignment_to_fsc(info: PosmgInfo, assignment: paynt.parameter_space.parameter_space.ParameterSpace) -> None:
    """
    TODO
    Stub: FSC extraction for the optimizing player from a synthesized assignment is not yet implemented for
    POSMG -- see the JAIR-refactor plan's "Result-class steps (d)/(e)" note (POSMG's result shape was never
    decided). Same future-facing calling convention as paynt.pomdp._utils.assignment_to_fsc.
    """
    logger.info("POSMG FSC extraction from a synthesized assignment is not yet implemented.")
    return


def create_smg_from_mdp(info: PosmgInfo, mdp: paynt.model.model.SubMdp) -> paynt.model.model.Smg:
    """Re-attach game (player-indication) structure to a restricted sub-MDP so it can be verified as a game rather than as a plain MDP."""
    underlying_player_indications = info.posmg_manager.get_state_player_indications()

    transition_matrix = mdp.model.transition_matrix
    state_labeling = mdp.model.labeling
    components = stormpy.SparseModelComponents(transition_matrix=transition_matrix, state_labeling=state_labeling)

    if mdp.model.has_choice_labeling():
        components.choice_labeling = mdp.model.choice_labeling

    state_player_indications = []
    for state in range(mdp.states):
        underlying_mdp_state = mdp.underlying_mdp_state_map[state]
        player = underlying_player_indications[underlying_mdp_state]
        state_player_indications.append(player)
    components.state_player_indications = state_player_indications

    return paynt.model.model.Smg(stormpy.storage.SparseSmg(components))
