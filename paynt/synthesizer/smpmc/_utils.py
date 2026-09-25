from __future__ import annotations

import paynt.colored_mdp
import paynt.parameter_space.parameter_space
import paynt.synthesizer.synthesizer_ar
import paynt.task

import logging

logger = logging.getLogger(__name__)


def verify_robust(
    colored_mdp: paynt.colored_mdp.ColoredMdp,
    task: paynt.task.SynthesisTask,
    robust_assignment: paynt.parameter_space.parameter_space.ParameterSpace,
) -> bool:
    """Check a robust result independently of Z3, as molehill's tests/test_robust.py does: robust_assignment is robust iff AR finds none of its members to
    satisfy the negated specification."""
    assert not task.specification.has_optimality, "robustness is only defined for threshold specifications"
    negated_task = paynt.task.SynthesisTask.from_specification(task.specification.negate(), use_exact=task.use_exact)
    synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(colored_mdp, negated_task)
    counterexample = synthesizer.synthesize(robust_assignment, print_stats=False)
    if counterexample is not None:
        logger.error(f"robustness check failed, this member violates the specification: {counterexample}")
    return counterexample is None
