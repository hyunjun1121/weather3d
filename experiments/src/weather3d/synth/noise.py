"""시간 축을 포함한 fBm 밸류 노이즈(연기의 비균질 density field용).

연기는 프레임마다 독립적인 노이즈를 쓰면 깜빡이는 인공물이 생긴다. 대신
(x, y, t) 3차원 coarse 격자에 난수를 뿌리고 trilinear 보간으로 필드를
만들면 공간·시간이 함께 부드러워진다. 여기에 octave를 쌓아(fBm) 자기
유사성을 준다. 결정론 재현을 위해 seed는 호출부에서 지정한다.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def _rand_field(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.random(shape).astype(np.float32)


def fbm_noise_3d(
    spatial_hw: tuple[int, int],
    num_frames: int,
    base_res: tuple[int, int, int] = (8, 6, 4),
    octaves: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """(H, W, T) 프랙탈 밸류 노이즈를 [0, 1] 범위로 반환한다.

    base_res는 (nx, ny, nt) 기본 격자 셀 수. octave마다 해상도와 진폭이
    2배/0.5배가 된다. 시간 해상도가 1이면 시간 변화가 없는 정적 필드가 된다.
    """
    h, w = spatial_hw
    rng = np.random.default_rng(seed)
    total = np.zeros((h, w, num_frames), dtype=np.float32)
    amp_sum = 0.0
    amp = 1.0
    for octv in range(octaves):
        nx, ny, nt = [max(1, int(round(r * (2**octv)))) for r in base_res]
        field = _rand_field(rng, (nx + 1, ny + 1, nt + 1))

        # 연속 좌표계에서 격자점 사이를 trilinear 보간
        gx = np.linspace(0.0, nx, w, dtype=np.float32)
        gy = np.linspace(0.0, ny, h, dtype=np.float32)
        gt = np.linspace(0.0, nt, num_frames, dtype=np.float32)
        xg, yg, tg = np.meshgrid(gx, gy, gt, indexing="xy")  # (H, W, T)
        coords = np.stack([yg, xg, tg], 0)
        layer = map_coordinates(field, coords, order=1, mode="nearest")
        total += amp * layer
        amp_sum += amp
        amp *= 0.5
    total /= max(amp_sum, 1e-8)
    return np.clip(total, 0.0, 1.0)


def derive_seed(*parts) -> int:
    """문자열 조합에서 결정론적 seed를 만든다(시퀀스별 재현성)."""
    import hashlib

    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")
