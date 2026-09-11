from __future__ import annotations

from dataclasses import dataclass

import paynt.result
import paynt.pomdp.fsc


@dataclass
class PomdpResult(paynt.result.Result):
    # the synthesized FSC (paynt.pomdp.fsc.FscFactored), or None if no assignment was found
    fsc: paynt.pomdp.fsc.FscFactored | None = None
