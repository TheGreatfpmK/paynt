from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True)
class PosmgTask:
    """Feature-specific build knobs for POSMG FSC-unfolding synthesis, owned by PosmgColoredMdpFactory (factory.build_task)."""

    # implicit initial size for FSC memory unfolding, consumed by PosmgColoredMdpFactory at
    # construction time
    memory_size: int = 1
