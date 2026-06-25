"""Model factory — dispatch on `arch` so trainer/eval/export are architecture-agnostic.

    arch: gssm  -> ClearVADModel  (selective state-space core, the current default)
    arch: cfc   -> LiquidVADModel (closed-form continuous-time core, the novel hybrid student)
"""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    arch = (cfg or {}).get("arch", "gssm").lower()
    if arch in ("gssm", "clearvad", "ssm"):
        from clearvad.model.clearvad_model import ClearVADModel
        return ClearVADModel.from_config(cfg)
    if arch in ("cfc", "liquid", "liquidvad"):
        from clearvad.model.liquid_vad import LiquidVADModel
        return LiquidVADModel.from_config(cfg)
    raise ValueError(f"unknown arch {arch!r} (expected 'gssm' or 'cfc')")
