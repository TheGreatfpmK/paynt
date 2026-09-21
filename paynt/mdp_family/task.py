from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True)
class MdpFamilyTask:
    # implicit initial size for scheduler-memory unfolding, consumed by MdpFamilyColoredMdpFactory (and
    # PomdpFamilyColoredMdpFactory) at construction time
    memory_size: int = 1
