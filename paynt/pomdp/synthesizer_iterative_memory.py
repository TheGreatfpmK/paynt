"""
Shared driver shape for FSC synthesis features that repeatedly re-unfold at increasing memory sizes and run
an AR-family engine (AR or Hybrid) against each unfolding, keeping the best assignment found so far across
memory sizes. Used as-is by DecPomdpSynthesizer; PosmgSynthesizer overrides stat_iterations_field;
PomdpSynthesizer additionally overrides __init__/synthesize/build_result to track which specific
memory-size unfolding produced the best assignment (needed for FSC extraction).
"""

from __future__ import annotations

from typing import Any

import paynt.task
import paynt.parameter_space.parameter_space
import paynt.synthesizer.synthesizer_ar
import paynt.synthesizer.synthesizer_hybrid
import paynt.utils.timer
import paynt.result

import logging

logger = logging.getLogger(__name__)


class IterativeMemorySynthesizer:

    # overridden by PosmgSynthesizer: its induced model is verified as a game (see
    # SynthesizerAR.check_specification's "posmg" branch), so Statistic accumulates iterations_game instead
    stat_iterations_field = "iterations_mdp"

    def __init__(self, colored_mdp_factory: Any, task: paynt.task.SynthesisTask, method: str = "ar") -> None:
        self.colored_mdp_factory = colored_mdp_factory
        self.colored_mdp = colored_mdp_factory.colored_mdp
        self.task = task
        # TODO add support for cegis/onebyone
        if method == "ar":
            self.synthesizer: type[paynt.synthesizer.synthesizer_ar.SynthesizerAR] = paynt.synthesizer.synthesizer_ar.SynthesizerAR
        elif method == "hybrid":
            self.synthesizer = paynt.synthesizer.synthesizer_hybrid.SynthesizerHybrid
        else:
            raise ValueError(f"unsupported method for FSC synthesis: {method!r}")
        self.total_iters = 0
        # best assignment/value found so far across memory-size iterations -- strategy_iterative constructs a
        # fresh inner synthesizer per iteration and discards it, so this is the only place these survive once
        # a later, larger-memory iteration doesn't improve on an earlier one
        self.best_assignment: paynt.parameter_space.parameter_space.ParameterSpace | None = None
        self.best_assignment_value: Any = None

    def synthesize(
        self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace, print_stats: bool = True
    ) -> paynt.parameter_space.parameter_space.ParameterSpace | None:
        synthesizer = self.synthesizer(self.colored_mdp, self.task)
        assignment = synthesizer.synthesize(parameter_space, keep_optimum=True, print_stats=print_stats)
        if assignment is not None:
            # keep_optimum=True means this only fires when the assignment genuinely improves on
            # self.task.specification.optimality's current (cross-iteration) optimum
            self.best_assignment = assignment
            self.best_assignment_value = synthesizer.best_assignment_value
        assert synthesizer.stat is not None
        iters = getattr(synthesizer.stat, self.stat_iterations_field)
        if iters is not None:
            self.total_iters += iters
        return assignment

    def strategy_iterative(self) -> None:
        mem_size = self.colored_mdp_factory.build_task.memory_size
        while True:
            if paynt.utils.timer.GlobalTimer.time_limit_reached():
                break
            logger.info(f"Synthesizing optimal k={mem_size} controller ...")

            assert self.colored_mdp_factory.current_memory_size is not None
            if mem_size > self.colored_mdp_factory.current_memory_size:
                self.colored_mdp = self.colored_mdp_factory.set_imperfect_memory_size(mem_size)

            self.synthesize(self.colored_mdp.parameter_space)

            mem_size += 1

    def run(self, optimum_threshold: Any = None) -> paynt.result.Result:
        self.strategy_iterative()
        if self.task.export_synthesis_filename_base is not None:
            logger.info(f"--export-synthesis is not yet supported for {type(self).__name__} (only for SAYNT)")
        return self.build_result()

    def build_result(self) -> paynt.result.Result:
        return paynt.result.Result(success=self.best_assignment is not None, value=self.best_assignment_value, assignment=self.best_assignment)
