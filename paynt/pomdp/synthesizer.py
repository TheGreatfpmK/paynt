"""
Driver for FSC synthesis over a POMDP: repeatedly re-unfolds the agent's imperfect-information strategy at
increasing memory sizes and runs AR or Hybrid against each unfolding, keeping the best assignment found so
far across memory sizes. See paynt.pomdp.synthesizer_iterative_memory.IterativeMemorySynthesizer for the
shared driver shape. SAYNT (Storm-guided synthesis) is a separate driver,
paynt.pomdp.saynt.SayntSynthesizer, since it needs a fundamentally different (interactive, threaded)
control flow -- see that module instead if you're looking for --storm-pomdp.
"""

from __future__ import annotations

import paynt.colored_mdp
import paynt.pomdp.factory
import paynt.task
import paynt.pomdp._utils
import paynt.parameter_space.parameter_space
import paynt.pomdp.result

# `from paynt.pomdp import synthesizer_iterative_memory`, not `import paynt.pomdp.synthesizer_iterative_memory`:
# this module is loaded from paynt/pomdp/__init__.py's own `from .synthesizer import PomdpSynthesizer` line,
# before paynt.pomdp has finished initializing -- a bare dotted-chain reference in this class's base-list
# (paynt.pomdp.synthesizer_iterative_memory.X) would need `pomdp` to already be a resolved attribute of
# `paynt`, which it isn't yet at this point, and fails with "cannot access submodule 'pomdp' of module
# 'paynt' (most likely due to a circular import)". The `from` form resolves the submodule directly via
# sys.modules instead, sidestepping that. See the identical fix in decpomdp/synthesizer.py and
# posmg/synthesizer.py.
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
