from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True)
class PosmgTask:
    """
    Feature-specific build knobs for POSMG FSC-unfolding synthesis, owned by PosmgColoredMdpFactory
    (factory.build_task) -- deliberately not a subclass of paynt.task.SynthesisTask, since this
    field is read only by the factory that unfolds a PosmgColoredMdp, never by the generic AR/CEGIS/Hybrid
    algorithms. A plain dataclass (see paynt.dt.task.DtTask's docstring for why, and why kw_only) -- no
    **kwargs catch-all; see sketch.py's _dataclass_task_kwargs for how Sketch.load_sketch copes with that.
    """

    # implicit initial size for FSC memory unfolding, consumed by PosmgColoredMdpFactory at
    # construction time
    memory_size: int = 1
