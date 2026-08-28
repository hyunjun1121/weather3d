"""Atmospheric scattering 기반 안개/연기 합성 (Track A).

모델: I = J * t + A * (1 - t),  t = exp(-beta * d)

- J: clean 영상 픽셀(radiance), A: 대기광(airlight), t: transmission
- 균질 안개는 상수 beta, 연기는 비균질 장으로 beta + sigma * noise(x, y, t).
- depth가 무효(<=0)인 픽셀은 t=1로 두어 원본 픽셀을 유지한다(기하 보존).
- 변환은 외관만 바꾸고 카메라 pose/기하 GT는 그대로 유효하다는 것이 본
  실험의 전제다(Foggy Cityscapes/KITTI-fog 계열의 표준 방법론).

실내 벤치마크(7-Scenes/Neural-RGBD, 유효 깊이 ~10m) 기준 프리셋:
beta=0.08이면 t(5m)=0.67, t(10m)=0.45로 중간 강도 안개에 해당한다.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

FOG_BETA: Mapping[str, float] = {"light": 0.04, "mid": 0.08, "heavy": 0.16}
SMOKE_BETA: Mapping[str, float] = {"light": 0.03, "mid": 0.06, "heavy": 0.12}
SMOKE_SIGMA: Mapping[str, float] = {"light": 0.05, "mid": 0.10, "heavy": 0.20}


def transmission(depth_m: np.ndarray, beta: np.ndarray | float) -> np.ndarray:
    """t = exp(-beta * d). 무효 깊이(d <= 0)는 t = 1(원본 유지)."""
    beta_arr = np.asarray(beta, dtype=np.float32)
    t = np.exp(-beta_arr * np.maximum(depth_m, 0.0))
    t = np.where(depth_m > 0, t, 1.0).astype(np.float32)
    return t


def _scatter(img: np.ndarray, t: np.ndarray, airlight: np.ndarray) -> np.ndarray:
    """I = J * t + A * (1 - t). img (H,W,3) [0,1], t (H,W), airlight (3,)."""
    if img.shape[:2] != t.shape:
        raise ValueError(f"shape mismatch: img {img.shape}, t {t.shape}")
    airlight = np.asarray(airlight, dtype=np.float32).reshape(1, 1, 3)
    out = img * t[..., None] + airlight * (1.0 - t[..., None])
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_fog(
    img: np.ndarray,
    depth_m: np.ndarray,
    beta: float,
    airlight: tuple[float, float, float] = (0.85, 0.85, 0.85),
) -> np.ndarray:
    """균질 안개. beta는 1/m 단위 산란 계수."""
    return _scatter(img, transmission(depth_m, beta), airlight)


def apply_smoke(
    img: np.ndarray,
    depth_m: np.ndarray,
    beta: float,
    sigma: float,
    noise01: np.ndarray,
    airlight: tuple[float, float, float] = (0.82, 0.80, 0.78),
) -> np.ndarray:
    """비균질 연기. noise01은 [0,1] 밸류 노이즈의 해당 프레임 슬라이스."""
    if noise01.shape != depth_m.shape:
        raise ValueError(f"noise shape {noise01.shape} != depth shape {depth_m.shape}")
    beta_map = beta + sigma * noise01
    return _scatter(img, transmission(depth_m, beta_map), airlight)


def synthesize_frame(
    img: np.ndarray,
    depth_m: np.ndarray,
    kind: str,
    level: str,
    noise01: np.ndarray | None = None,
    rng: "np.random.Generator | None" = None,
    *,
    fog_beta: Mapping[str, float] | None = None,
    smoke_beta: Mapping[str, float] | None = None,
    smoke_sigma: Mapping[str, float] | None = None,
    rain_beta: Mapping[str, float] | None = None,
    rain_density: Mapping[str, int] | None = None,
    rain_length: Mapping[str, int] | None = None,
    lowlight_gamma: Mapping[str, float] | None = None,
    lowlight_gain: Mapping[str, float] | None = None,
    lowlight_sigma: Mapping[str, float] | None = None,
    fog_airlight: tuple[float, float, float] = (0.85, 0.85, 0.85),
    smoke_airlight: tuple[float, float, float] = (0.82, 0.80, 0.78),
    rain_airlight: tuple[float, float, float] = (0.72, 0.74, 0.78),
    lowlight_tint: tuple[float, float, float] = (0.96, 0.99, 1.06),
) -> np.ndarray:
    """프리셋 조회를 포함한 1프레임 합성 진입점.

    kind는 "fog" | "smoke" | "rain" | "lowlight", level은 "light" | "mid" |
    "heavy"(fog는 "xheavy"/"extreme" 추가). smoke는 noise01 슬라이스,
    rain/lowlight는 프레임별 결정론 rng를 받는다. config에서 프리셋을
    덮어쓴 경우 각 *_beta/*_sigma/*_gamma 맵을 넘긴다.
    """
    if kind == "fog":
        beta_map = fog_beta if fog_beta is not None else FOG_BETA
        return apply_fog(img, depth_m, float(beta_map[level]), fog_airlight)
    if kind == "smoke":
        if noise01 is None:
            raise ValueError("smoke requires noise01 slice")
        beta_map = smoke_beta if smoke_beta is not None else SMOKE_BETA
        sig_map = smoke_sigma if smoke_sigma is not None else SMOKE_SIGMA
        return apply_smoke(img, depth_m, float(beta_map[level]), float(sig_map[level]), noise01, smoke_airlight)
    if kind in ("rain", "lowlight"):
        # 순환 import 방지(atmosphere <-> weather_ext): 지연 import.
        from .weather_ext import RAIN_BETA, RAIN_DENSITY, RAIN_LENGTH, apply_lowlight, apply_rain
        from .weather_ext import LOWLIGHT_GAIN, LOWLIGHT_GAMMA, LOWLIGHT_SIGMA

        if rng is None:
            raise ValueError(f"{kind} requires rng")
        if kind == "rain":
            beta_map = rain_beta if rain_beta is not None else RAIN_BETA
            den_map = rain_density if rain_density is not None else RAIN_DENSITY
            len_map = rain_length if rain_length is not None else RAIN_LENGTH
            return apply_rain(
                img, depth_m, float(beta_map[level]), int(den_map[level]), int(len_map[level]),
                rng, airlight=rain_airlight,
            )
        gam_map = lowlight_gamma if lowlight_gamma is not None else LOWLIGHT_GAMMA
        gain_map = lowlight_gain if lowlight_gain is not None else LOWLIGHT_GAIN
        sig_map = lowlight_sigma if lowlight_sigma is not None else LOWLIGHT_SIGMA
        return apply_lowlight(
            img, float(gam_map[level]), float(gain_map[level]), float(sig_map[level]),
            rng, tint=lowlight_tint,
        )
    raise ValueError(f"unknown weather kind: {kind}")
