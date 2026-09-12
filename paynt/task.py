"""
A SynthesisTask pairs an underlying PCTL specification with the generic synthesis knobs the AR/CEGIS/Hybrid/
OneByOne algorithms read (as opposed to feature-specific build knobs like DT's tree_depth or POMDP's
memory_size, which live on a separate, non-inheriting per-feature task class owned by that feature's
colored-MDP factory instead -- see e.g. paynt.dt.task.DtTask).

Connection to old PAYNT implementation and explanation of new structure (TODO remove in the future):
The three fields below (export_synthesis_filename_base, conflict_generator_type, disable_expected_visits)
are generic synthesis knobs that apply regardless of feature, so they live on this class rather than being
duplicated onto every feature. Before this class carried them, they were CLI-set mutable class attributes
on Synthesizer/SynthesizerCEGIS/ModelIndex -- meaning two syntheses running in the same process (e.g. a
library caller doing two Sketch.load_sketch + synthesize calls back to back) would silently step on each
other's settings, since a class attribute is shared by every instance. Being fields on a per-instance
SynthesisTask fixes that: each synthesis run gets its own SynthesisTask, so its settings can no longer leak
into another run's.

Constructing a SynthesisTask is the one canonical way to turn a list of raw stormpy properties into a
Specification; every parser should funnel through this (or paynt.specification.property.construct_specification
directly) rather than constructing a Specification by hand.
"""

from __future__ import annotations

from typing import Any

import paynt.specification.property


class SynthesisTask:

    def __init__(
        self,
        properties: list[Any],
        timeout: int | None = None,
        use_exact: bool = False,
        relative_error: float = 0,
        export_synthesis_filename_base: str | None = None,
        conflict_generator_type: str | None = None,
        disable_expected_visits: bool = False,
        discard_unreachable_choices: bool = False,
    ):
        self.specification = paynt.specification.property.construct_specification(properties, relative_error, use_exact)
        self.timeout = timeout
        self.use_exact = use_exact
        self.export_synthesis_filename_base = export_synthesis_filename_base
        self.conflict_generator_type = conflict_generator_type
        self.disable_expected_visits = disable_expected_visits
        # PolicyTreeSynthesizer: if set, unreachable choices are discarded from the splitting scheduler
        self.discard_unreachable_choices = discard_unreachable_choices

    @classmethod
    def from_specification(
        cls,
        specification: paynt.specification.property.Specification,
        timeout: int | None = None,
        use_exact: bool = False,
        export_synthesis_filename_base: str | None = None,
        conflict_generator_type: str | None = None,
        disable_expected_visits: bool = False,
        discard_unreachable_choices: bool = False,
        **kwargs: Any,
    ) -> SynthesisTask:
        """
        Wrap an already-constructed Specification directly, bypassing property parsing. Used when a caller
        already has a (e.g. copied/negated) Specification in hand rather than a fresh list of raw properties,
        and by Sketch.load_sketch, which builds the Specification itself while parsing the sketch.

        Unlike __init__, silently accepts and ignores unrecognized keyword arguments (**kwargs, unused here):
        Sketch.load_sketch hands the same task_kwargs dict -- every task-relevant CLI option, regardless of
        feature -- to both this constructor and whichever feature-specific task ends up being constructed,
        so this call necessarily receives keys irrelevant to the generic SynthesisTask. They are dropped here.

        Uses cls.__new__(cls) rather than cls(...) purely to avoid constructing-then-discarding an empty
        Specification (paynt.specification.property.construct_specification([]) followed immediately by
        overwriting self.specification) -- not, as in a subclassing design, to preserve a caller's subclass.
        """
        task = cls.__new__(cls)
        task.specification = specification
        task.timeout = timeout
        task.use_exact = use_exact
        task.export_synthesis_filename_base = export_synthesis_filename_base
        task.conflict_generator_type = conflict_generator_type
        task.disable_expected_visits = disable_expected_visits
        task.discard_unreachable_choices = discard_unreachable_choices
        return task

    def get_property(self) -> paynt.specification.property.Property:
        assert self.specification.num_properties == 1, "expecting a single property"
        return self.specification.all_properties()[0]
