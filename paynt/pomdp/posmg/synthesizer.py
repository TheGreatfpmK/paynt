"""
Driver for FSC synthesis over a POSMG: repeatedly re-unfolds the optimizing player's imperfect-information
strategy at increasing memory sizes. See paynt.pomdp.synthesizer_iterative_memory.IterativeMemorySynthesizer
for the shared driver shape.
"""

from __future__ import annotations

# `from paynt.pomdp import synthesizer_iterative_memory`, not `import paynt.pomdp.synthesizer_iterative_memory`:
# this module is loaded from paynt/pomdp/__init__.py's own `from . import posmg` line, before paynt.pomdp has
# finished initializing -- a bare dotted-chain reference (paynt.pomdp.synthesizer_iterative_memory.X) would
# need `pomdp` to already be a resolved attribute of `paynt`, which it isn't yet at this point, and fails with
# "cannot access submodule 'pomdp' of module 'paynt' (most likely due to a circular import)". The `from` form
# resolves the submodule directly via sys.modules instead, sidestepping that.
from paynt.pomdp import synthesizer_iterative_memory
import paynt.pomdp.posmg._utils
import paynt.result


class PosmgSynthesizer(synthesizer_iterative_memory.IterativeMemorySynthesizer):

    # the induced model is verified as a game (see SynthesizerAR.check_specification's "posmg" branch), so
    # Statistic accumulates iterations_game rather than iterations_mdp
    stat_iterations_field = "iterations_game"

    def build_result(self) -> paynt.result.Result:
        if self.best_assignment is not None:
            paynt.pomdp.posmg._utils.assignment_to_fsc(self.colored_mdp.feature_info, self.best_assignment)
        return super().build_result()
