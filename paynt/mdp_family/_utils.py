"""
Internal support for the "family" feature: the MdpFamilyInfo companion dataclass (attached as
ColoredMdp.feature_info for feature_kind "family") and functions that interpret it -- policy
construction/export logic used only by already-family-typed driver code
(paynt.mdp_family.policy_tree/.policy_tree_synthesizer). Not part of the public API. build_assignment's
"family"/"pomdp_family" branch (see paynt.colored_mdp.ColoredMdp) is the one representation-level
difference this feature needs -- fixing an environment does not also fix the agent's policy, so the result
can stay nondeterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import payntbind

import paynt.colored_mdp
import paynt.parameter_space.parameter_space
import paynt.model.model
import paynt.specification.property

import json

import logging

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MdpFamilyInfo:
    # number of distinct actions in the underlying MDP
    num_actions: int
    # a list of action labels
    action_labels: list[str]
    # for each choice of the underlying MDP, the executed action
    choice_to_action: list[int]
    # for each state of the underlying MDP and for each action, a list of choices that execute this action
    state_action_choices: list[list[list[int]]]
    # for each state of the underlying MDP, a list of available actions
    state_to_actions: list[list[int]]


def empty_policy(colored_mdp: paynt.colored_mdp.ColoredMdp) -> list[int | None]:
    return paynt.model.model.ModelIndex.empty_scheduler(colored_mdp.underlying_mdp)


def scheduler_to_policy(colored_mdp: paynt.colored_mdp.ColoredMdp, scheduler: Any, mdp: paynt.model.model.SubMdp) -> list[int | None]:
    info = cast(MdpFamilyInfo, colored_mdp.feature_info)
    state_to_choice = paynt.model.model.ModelIndex.scheduler_to_state_to_choice(colored_mdp.underlying_mdp, colored_mdp.choice_destinations, mdp, scheduler)
    policy = empty_policy(colored_mdp)
    for state in range(colored_mdp.underlying_mdp.nr_states):
        choice = state_to_choice[state]
        if choice is not None:
            policy[state] = info.choice_to_action[choice]
    return policy


def policy_to_state_valuation_actions(
    colored_mdp: paynt.colored_mdp.ColoredMdp, policy: tuple[list[int | None], list[int]]
) -> list[tuple[dict[str, Any], str]]:
    """
    Create a representation for a policy that associates action labels with state valuations. States with only
    one available action are omitted.
    """
    info = cast(MdpFamilyInfo, colored_mdp.feature_info)
    policy_actions, _ = policy
    sv = colored_mdp.underlying_mdp.state_valuations
    state_valuation_to_action = []
    for state, action in enumerate(policy_actions):
        if action is None:
            continue
        if len(info.state_to_actions[state]) == 1:
            continue
        # get action label
        action_label = info.action_labels[action]
        if action_label == "empty_label":
            continue

        # get state valuation
        valuation_jani = json.loads(str(sv.get_json(state)))
        valuation = {}
        for variable, value in valuation_jani.items():
            if "_loc_prism2jani_" in variable:
                continue
            valuation[variable] = value

        state_valuation_to_action.append((valuation, action_label))

    # omit variables that are assigned to the same value
    default_valuation, _ = state_valuation_to_action[0]
    irrelevant_variables = set(default_valuation)
    for valuation, _ in state_valuation_to_action[1:]:
        for variable in list(irrelevant_variables):
            if valuation[variable] != default_valuation[variable]:
                irrelevant_variables.remove(variable)
    return [
        ({variable: value for variable, value in valuation.items() if variable not in irrelevant_variables}, action)
        for valuation, action in state_valuation_to_action
    ]


def policy_to_json(state_valuation_to_action: list[tuple[dict[str, Any], str]], dt_control: bool = False) -> list[Any]:
    """
    :param state_valuation_to_action: a list of tuples (valuation,action) where valuation is a dictionary of variable
    :param dt_control: if True, outputs JSON in the format expected by the DT control tool,
            otherwise simpler format is used
    """
    json_whole: list[Any] = []
    for _index, valuation_action in enumerate(state_valuation_to_action):
        if dt_control:
            json_unit: dict[str, Any] = {}
            valuation, action = valuation_action
            json_unit["c"] = [{"origin": {"action-label": action}}]
            json_unit["s"] = valuation
            json_whole.append(json_unit)
        else:
            json_whole.append(valuation_action)

    return json_whole


def fix_and_apply_policy_to_parameter_space(
    colored_mdp: paynt.colored_mdp.ColoredMdp, selected_choices: Any, policy: list[int | None]
) -> tuple[tuple[list[int | None], list[int]], paynt.model.model.SubMdp]:
    """
    Apply policy to the underlying MDP restricted to selected_choices. Every undefined action in a policy
    is set to an arbitrary one. Upon constructing the MDP, reset unused actions in a policy to None.
    :param selected_choices the compatible-choices bitmask to restrict the policy to -- passed explicitly
        (not a parameter_space) since callers may want to verify against a snapshot taken at an earlier
        point (e.g. before postprocessing widened a node's parameter_space via parameter_set_options)
    :returns fixed policy
    :returns the resulting MDP
    """
    info = cast(MdpFamilyInfo, colored_mdp.feature_info)
    policy = [action if action is not None else info.state_to_actions[state][0] for state, action in enumerate(policy)]
    policy_choices = []
    for state, action in enumerate(policy):
        assert action is not None
        policy_choices += info.state_action_choices[state][action]
    choices = payntbind.synthesis.policyToChoicesForFamily(policy_choices, selected_choices)

    # build MDP and keep only reachable states in policy
    mdp = paynt.model.model.SubmodelBuilder.build_submdp(colored_mdp.underlying_mdp, choices, colored_mdp.subsystem_builder_options)
    policy_fixed = empty_policy(colored_mdp)
    for state in mdp.underlying_mdp_state_map:
        policy_fixed[state] = policy[state]

    mask = [state for state, action in enumerate(policy_fixed) if action is not None]
    return (policy_fixed, mask), mdp


def assert_mdp_is_deterministic(
    colored_mdp: paynt.colored_mdp.ColoredMdp, mdp: paynt.model.model.SubMdp, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace
) -> None:
    if mdp.is_deterministic:
        return

    info = cast(MdpFamilyInfo, colored_mdp.feature_info)
    logger.error(f"applied policy to a singleton parameter assignment {parameter_space} and obtained MDP with nondeterminism")
    for state in range(mdp.model.nr_states):

        choices = mdp.model.transition_matrix.get_rows_for_group(state)
        if len(choices) > 1:
            underlying_mdp_state = mdp.underlying_mdp_state_map[state]
            underlying_mdp_choices = [mdp.underlying_mdp_choice_map[choice] for choice in choices]
            state_str = colored_mdp.underlying_mdp.state_valuations.get_string(underlying_mdp_state)
            state_str = state_str.replace(" ", "")
            state_str = state_str.replace("\t", "")
            actions_str = [info.action_labels[info.choice_to_action[choice]] for choice in underlying_mdp_choices]
            logger.error(f"the following state {state_str} has multiple actions {actions_str}")
    logger.error("aborting...")
    exit(1)


def build_game_abstraction_solver(colored_mdp: paynt.colored_mdp.ColoredMdp, prop: paynt.specification.property.Property) -> Any:
    info = cast(MdpFamilyInfo, colored_mdp.feature_info)
    target_label = prop.get_target_label()
    precision = paynt.specification.property.Property.model_checking_precision
    return payntbind.synthesis.GameAbstractionSolver(
        colored_mdp.underlying_mdp, len(info.action_labels), info.choice_to_action, prop.formula, prop.maximizing, target_label, precision
    )
