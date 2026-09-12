from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(kw_only=True)
class DtTask:
    """
    Feature-specific build knobs for decision-tree synthesis, owned by DtColoredMdpFactory
    (factory.build_task) -- deliberately not a subclass of paynt.task.SynthesisTask, since these
    fields are read only by the factory that builds a DtColoredMdp, never by the generic AR/CEGIS/Hybrid
    algorithms (those read a separate, plain SynthesisTask instead; see paynt.dt.synthesizer.DtSynthesizer,
    which holds both).

    A plain dataclass (pure field assignment, no computed values) -- unlike SynthesisTask, which still needs
    a hand-written __init__ to construct its Specification. kw_only=True is not just a style choice: DtTask
    is a base class (see DtNestTask below), and dataclass inheritance always places a subclass's own new
    fields after every inherited one in the generated __init__'s positional order -- silently different from
    what a hand-written subclass __init__ might have declared. Since every real call site already constructs
    these with keyword arguments, kw_only=True makes that the only way, turning what would otherwise be a
    silent "wrong field gets the value" risk into a loud TypeError instead.

    Sketch.load_sketch funnels one shared kwargs dict (every feature's CLI options at once, since it doesn't
    know the sketch's feature until it has parsed it) into whichever feature task ends up being constructed;
    unlike the pre-dataclass version, this class has no **kwargs catch-all of its own to swallow the
    irrelevant keys -- being a plain dataclass is exactly what buys it a real, auto-generated __init__
    instead of hand-written boilerplate. See sketch.py's _dataclass_task_kwargs instead.
    """

    tree_depth: int = 0
    tree_enumeration: bool = False
    # path to a JSON scheduler file to be mapped to a decision tree (CLI's --tree-map-scheduler)
    scheduler_path: str | None = None
    # if true, an explicit action executing a random choice of an available action will be added to
    # each state (consumed by DtColoredMdpFactory at construction time)
    add_dont_care_action: bool = True
    # an already-in-memory scheduler to map (distinct from scheduler_path above, which is a file path
    # loaded by the CLI's own run loop) -- the library-facing entry point used by paynt.dt.api.synthesize.
    # init=False: never a real constructor argument, only ever set later via set_scheduler_to_map.
    scheduler_to_map: Any = field(init=False, default=None)

    def set_scheduler_to_map(self, scheduler: Any) -> None:
        self.scheduler_to_map = scheduler

    @property
    def has_scheduler_to_map(self) -> bool:
        return self.scheduler_to_map is not None
