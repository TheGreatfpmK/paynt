from __future__ import annotations

import paynt.colored_mdp
import paynt.pomdp.factory
import paynt.task
import paynt.pomdp._utils
import paynt.parameter_space.parameter_space
import paynt.pomdp.result

from paynt.pomdp import synthesizer_iterative_memory

import logging

logger = logging.getLogger(__name__)


class PomdpSynthesizer(synthesizer_iterative_memory.IterativeMemorySynthesizer):
    def __init__(self, colored_mdp_factory: paynt.pomdp.factory.PomdpColoredMdpFactory, task: paynt.task.SynthesisTask, method: str = "ar") -> None:
        super().__init__(colored_mdp_factory, task, method)
        # the specific colored_mdp (memory-size unfolding) that produced best_assignment -- needed because
        # parameter indices are tied to one specific unfolding, and self.colored_mdp gets reassigned to a
        # fresh (larger) unfolding on every later iteration, so it can't be relied on to still match
        # best_assignment by the time synthesis finishes
        self.best_colored_mdp: paynt.colored_mdp.ColoredMdp | None = None

    def synthesize(
        self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace, print_stats: bool = True
    ) -> paynt.parameter_space.parameter_space.ParameterSpace | None:
        assignment = super().synthesize(parameter_space, print_stats)
        if assignment is not None:
            self.best_colored_mdp = self.colored_mdp
        return assignment

    def build_result(self) -> paynt.pomdp.result.PomdpResult:
        fsc = None
        if self.best_assignment is not None:
            assert self.best_colored_mdp is not None
            fsc = paynt.pomdp._utils.assignment_to_fsc(self.best_colored_mdp.feature_info, self.best_assignment)
        return paynt.pomdp.result.PomdpResult(
            success=self.best_assignment is not None, value=self.best_assignment_value, assignment=self.best_assignment, fsc=fsc
        )
