"""Track A 물리 기반 날씨 합성(atmospheric scattering)."""

from .atmosphere import FOG_BETA, SMOKE_BETA, SMOKE_SIGMA, apply_fog, apply_smoke, synthesize_frame

__all__ = [
    "FOG_BETA",
    "SMOKE_BETA",
    "SMOKE_SIGMA",
    "apply_fog",
    "apply_smoke",
    "synthesize_frame",
]
