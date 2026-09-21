from __future__ import annotations

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
