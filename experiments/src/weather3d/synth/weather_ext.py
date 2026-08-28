"""Track A v2 확장 날씨 합성: 비(veiling + streaks)와 저조도.

fog/smoke(atmosphere.py)와 같은 설계 원칙을 따른다: 외관만 바꾸고
카메라 pose/기하 GT는 그대로 유효하다.

비(비 완전 물리 시뮬레이션은 아님 - 문서화된 근사 합성 모델):
1. veiling 층: 젖은 대기의 산란 감쇠. atmosphere.transmission을 재사용해
   I_v = J*t + A*(1-t), t = exp(-beta*d). 실외 원거리까지 고려한 beta 프리셋.
2. streak 층: 낙하 물방울의 모션블러 궤적. 프레임별 결정론적 rng에서
   위치/밝기/길이를 뽑아 얇은 사선으로 누적 가산한다. 빗줄기는 프레임
   사이 독립(빠른 낙하)이므로 smoke와 달리 시간축 fBm을 쓰지 않는다.
   Garg & Nayar 계열 분석을 근사한 streak+veiling 합성이 벤치마크
   증강(GTA-rain, RainMix 계열)에서 표준적으로 쓰는 구성과 같다.

저조도:
I_e = clip((J^gamma)*gain*tint + noise). gamma가 밝기 응답을 비선형
압축하고(gain과 함께 조도 감쇠), noise는 어두운 신호의 우세한
가우시안 근사 read/shot noise다. tint는 야간 색온도 편향(약한 청색).
"""

from __future__ import annotations

import numpy as np

from .atmosphere import transmission

RAIN_BETA: dict[str, float] = {"light": 0.015, "mid": 0.04, "heavy": 0.09}
RAIN_DENSITY: dict[str, int] = {"light": 150, "mid": 400, "heavy": 900}  # 640x480 기준 방울 수/프레임
RAIN_LENGTH: dict[str, int] = {"light": 12, "mid": 18, "heavy": 26}      # streak 길이(px)
RAIN_AIRLIGHT: tuple[float, float, float] = (0.72, 0.74, 0.78)
RAIN_SLANT: float = 0.15          # y 1px당 x 이동(약 8.5도 기울기)
RAIN_STREAK_GAIN: float = 0.30    # streak층 최대 가산 강도

LOWLIGHT_GAMMA: dict[str, float] = {"light": 1.6, "mid": 2.2, "heavy": 3.0}
LOWLIGHT_GAIN: dict[str, float] = {"light": 0.55, "mid": 0.32, "heavy": 0.16}
LOWLIGHT_SIGMA: dict[str, float] = {"light": 0.004, "mid": 0.008, "heavy": 0.015}
LOWLIGHT_TINT: tuple[float, float, float] = (0.96, 0.99, 1.06)


def streak_layer(h: int, w: int, density: int, length: int, rng: np.random.Generator) -> np.ndarray:
    """(H, W) [0,1] 빗줄기 레이어. density=0이면 0 행렬(멱등성 테스트용)."""
    layer = np.zeros((h, w), dtype=np.float32)
    for _ in range(density):
        x0 = rng.uniform(-RAIN_SLANT * length, w - 1.0)
        y0 = rng.uniform(-length, h - 1.0)
        bright = rng.uniform(0.35, 1.0)
        for s in range(length):
            y = int(y0 + s)
            if not 0 <= y < h:
                continue
            x = int(x0 + RAIN_SLANT * s)
            v = bright * (1.0 - 0.7 * s / length)
            for dx, scale in ((0, 1.0), (1, 0.6)):
                xi = x + dx
                if 0 <= xi < w:
                    # 같은 픽셀에 겹친 방울은 더 밝은 쪽을 유지한다.
                    if v * scale > layer[y, xi]:
                        layer[y, xi] = v * scale
    return layer


def apply_rain(
    img: np.ndarray,
    depth_m: np.ndarray,
    beta: float,
    density: int,
    length: int,
    rng: np.random.Generator,
    airlight: tuple[float, float, float] = RAIN_AIRLIGHT,
    streak_gain: float = RAIN_STREAK_GAIN,
) -> np.ndarray:
    """비 합성. img (H,W,3) [0,1], depth_m (H,W) m(0=무효), rng는 프레임별 결정론."""
    if img.shape[:2] != depth_m.shape:
        raise ValueError(f"shape mismatch: img {img.shape}, depth {depth_m.shape}")
    t = transmission(depth_m, beta)
    airlight = np.asarray(airlight, dtype=np.float32).reshape(1, 1, 3)
    veiled = img * t[..., None] + airlight * (1.0 - t[..., None])
    h, w = img.shape[:2]
    streaks = streak_layer(h, w, density, length, rng)[..., None]
    tint = np.asarray((0.85, 0.90, 1.00), dtype=np.float32).reshape(1, 1, 3)
    out = veiled + streak_gain * streaks * tint
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_lowlight(
    img: np.ndarray,
    gamma: float,
    gain: float,
    sigma: float,
    rng: np.random.Generator,
    tint: tuple[float, float, float] = LOWLIGHT_TINT,
) -> np.ndarray:
    """저조도 합성. depth 불문 균일 적용(조도는 장면 전역 광원 감쇠)."""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"img must be (H,W,3): {img.shape}")
    dark = np.clip(img, 1e-6, 1.0).astype(np.float32) ** gamma * gain
    dark = dark * np.asarray(tint, dtype=np.float32).reshape(1, 1, 3)
    if sigma > 0:
        dark = dark + rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(dark, 0.0, 1.0).astype(np.float32)
