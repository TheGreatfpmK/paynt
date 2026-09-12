from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True)
class PomdpTask:
    """
    Feature-specific build knobs for POMDP (and Dec-POMDP, which reuses this same class -- see
    DecPomdpColoredMdpFactory) FSC-unfolding synthesis, owned by the colored-MDP factory (factory.build_task)
    -- deliberately not a subclass of paynt.task.SynthesisTask, since these fields are read only by
    the factory that unfolds a PomdpColoredMdp/DecPomdpColoredMdp, never by the generic AR/CEGIS/Hybrid
    algorithms. A plain dataclass (see paynt.dt.task.DtTask's docstring for why, and why kw_only) -- no
    **kwargs catch-all; see sketch.py's _dataclass_task_kwargs for how Sketch.load_sketch copes with that.
    """

    # implicit initial size for FSC memory unfolding, consumed by PomdpColoredMdpFactory/
    # DecPomdpColoredMdpFactory at construction time
    memory_size: int = 1
    # if True, posterior-aware unfolding is applied (POMDP only, ignored by Dec-POMDP)
    posterior_aware: bool = False
