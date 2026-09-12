from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True)
class FamilyTask:
    """
    Feature-specific build knobs for family-of-models synthesis, owned by FamilyColoredMdpFactory (and
    PomdpFamilyColoredMdpFactory) as factory.build_task -- deliberately not a subclass of
    paynt.task.SynthesisTask, since this field is read only by the factory that unfolds scheduler
    memory, never by the generic AR/CEGIS/Hybrid algorithms or PolicyTreeSynthesizer. A plain dataclass (see
    paynt.dt.task.DtTask's docstring for why, and why kw_only) -- no **kwargs catch-all; see sketch.py's
    _dataclass_task_kwargs for how Sketch.load_sketch copes with that.
    """

    # implicit initial size for scheduler-memory unfolding, consumed by FamilyColoredMdpFactory (and
    # PomdpFamilyColoredMdpFactory) at construction time
    memory_size: int = 1
