from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True)
class PomdpTask:
    # implicit initial size for FSC memory unfolding, consumed by PomdpColoredMdpFactory/
    # DecPomdpColoredMdpFactory at construction time
    memory_size: int = 1
    # if True, posterior-aware unfolding is applied (POMDP only, ignored by Dec-POMDP)
    posterior_aware: bool = False
