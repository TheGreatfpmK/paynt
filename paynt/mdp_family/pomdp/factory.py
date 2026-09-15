"""
Constructs a ColoredMdp (feature_kind "pomdp_family"): like MdpFamilyColoredMdpFactory, but tracks observation classes so that
policy decisions can be tied together across environment variants that look the same to the agent.
"""

from __future__ import annotations

from typing import Any, cast

import paynt.colored_mdp
import paynt.mdp_family.task
import paynt.parameter_space.parameter_space
from paynt.mdp_family.factory import MdpFamilyColoredMdpFactory
from paynt.mdp_family.pomdp._utils import PomdpFamilyInfo

import logging

logger = logging.getLogger(__name__)


class PomdpFamilyColoredMdpFactory(MdpFamilyColoredMdpFactory):

    feature_kind = "pomdp_family"

    def __init__(
        self,
        underlying_mdp: Any,
        parameter_space: paynt.parameter_space.parameter_space.ParameterSpace,
        coloring: Any,
        build_task: paynt.mdp_family.task.MdpFamilyTask,
        obs_evaluator: Any,
        use_exact: bool = False,
    ):
        self.obs_evaluator = obs_evaluator
        super().__init__(underlying_mdp, parameter_space, coloring, build_task, use_exact=use_exact)

    def unfold_scheduler_memory(
        self, underlying_mdp: Any, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace, coloring: Any
    ) -> tuple[Any, paynt.parameter_space.parameter_space.ParameterSpace, Any]:
        unfolded_mdp, parameter_space, new_coloring = super().unfold_scheduler_memory(underlying_mdp, parameter_space, coloring)
        # memory was unfolded, so obs_evaluator must be updated to match the unfolded state space
        prototype_states = list(self.memory_unfolder.state_prototype)
        state_to_obs_class = list(self.obs_evaluator.state_to_obs_class)
        new_obs_classes_map = [state_to_obs_class[prototype_states[state]] for state in range(unfolded_mdp.nr_states)]
        self.obs_evaluator.state_to_obs_class = new_obs_classes_map
        return unfolded_mdp, parameter_space, new_coloring

    def build(self) -> paynt.colored_mdp.ColoredMdp:
        # identify actions available at each observation
        observation_to_actions: list[list[int] | None] = [None] * self.obs_evaluator.num_obs_classes
        state_to_observation = self.obs_evaluator.state_to_obs_class
        for state, available_actions in enumerate(self.state_to_actions):
            obs = state_to_observation[state]
            if observation_to_actions[obs] is not None:
                assert observation_to_actions[obs] == available_actions, f"two states in observation class {obs} differ in available actions"
                continue
            observation_to_actions[obs] = available_actions

        # every observation class is populated by at least one state above, so no entry is left None
        assert all(actions is not None for actions in observation_to_actions)

        colored_mdp = paynt.colored_mdp.ColoredMdp(self.underlying_mdp, self.parameter_space, self.coloring, self.use_exact, feature_kind="pomdp_family")
        colored_mdp.feature_info = PomdpFamilyInfo(
            num_actions=self.num_actions,
            action_labels=self.action_labels,
            choice_to_action=self.choice_to_action,
            state_action_choices=self.state_action_choices,
            state_to_actions=self.state_to_actions,
            obs_evaluator=self.obs_evaluator,
            observation_to_actions=cast("list[list[int]]", observation_to_actions),
        )
        return colored_mdp
