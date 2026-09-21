from __future__ import annotations

from paynt.pomdp import synthesizer_iterative_memory
import paynt.pomdp.decpomdp._utils
import paynt.result


class DecPomdpSynthesizer(synthesizer_iterative_memory.IterativeMemorySynthesizer):
    def build_result(self) -> paynt.result.Result:
        if self.best_assignment is not None:
            paynt.pomdp.decpomdp._utils.assignment_to_fscs(self.colored_mdp.feature_info, self.best_assignment)
        return super().build_result()
