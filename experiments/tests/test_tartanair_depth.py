"""TartanAir V2 depth 디코더 단위테스트(4ch float32 패킹 + 레거시 16bit).

2026-08-26 11b 실측으로 확정된 V2 규격 - "H x W float32를 4채널 8비트
PNG로 무손실 패킹"(tartanair.org/modalities.html) - 을 디코더가 지키는지
검증한다. 패킹은 공식 디코더(cv2.imread 결과 메모리를 .view("<f4")로
재해석)와 같은 cv2 BGRA 메모리 규약으로 만든다.
"""

from __future__ import annotations

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weather3d.gt import (  # noqa: E402
    TARTANAIR_DEPTH_DIV,
    TARTANAIR_INTRINSICS,
    TARTANAIR_MAX_DEPTH_M,
    _read_tartanair_depth_png,
)


@contextmanager
def _tmp():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _pack_f4(depth: np.ndarray) -> np.ndarray:
    """(H, W) float32 -> cv2 BGRA 메모리 규약 (H, W, 4) uint8."""
    return (
        np.frombuffer(depth.astype("<f4").tobytes(), dtype=np.uint8)
        .reshape(depth.shape[0], depth.shape[1], 4)
        .copy()
    )


def test_tartanair_depth_f4_roundtrip_and_masking():
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 50.0, size=(24, 32)).astype(np.float32)
    depth[0, 0] = 1e9            # 하늘(거대 값) -> 상한 마스킹
    depth[1, 1] = float("nan")   # 비유한 -> 마스킹
    depth[2, 2] = 1e-4           # 하한 미만 -> 마스킹
    with _tmp() as d:
        p = d / "000000_lcam_front_depth.png"
        assert cv2.imwrite(str(p), _pack_f4(depth))
        out = _read_tartanair_depth_png(p)
        assert out.shape == depth.shape
        assert out.dtype == np.float32
        expect = depth.copy()
        expect[0, 0] = 0.0
        expect[1, 1] = 0.0
        expect[2, 2] = 0.0
        assert np.allclose(out, expect, rtol=0.0, atol=1e-6)


def test_tartanair_depth_legacy_16bit_path():
    rng = np.random.default_rng(1)
    raw = rng.integers(500, 9000, size=(16, 16)).astype(np.uint16)
    raw[0, 0] = 65535
    with _tmp() as d:
        p = d / "legacy_depth.png"
        assert cv2.imwrite(str(p), raw)
        out = _read_tartanair_depth_png(p)
        expect = raw.astype(np.float32) / TARTANAIR_DEPTH_DIV
        expect[0, 0] = 0.0
        assert np.allclose(out, expect, rtol=0.0, atol=1e-6)


def test_tartanair_depth_rejects_3channel():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    with _tmp() as d:
        p = d / "bad_depth.png"
        assert cv2.imwrite(str(p), img)
        try:
            _read_tartanair_depth_png(p)
        except ValueError:
            return
        raise AssertionError("3채널 png가 ValueError 없이 통과됨")


def test_tartanair_v2_intrinsics_square_principal_point():
    # V2 규격: 640x640, f=320, 주점 (320,320) - V1 cy=240 회귀 방지.
    assert TARTANAIR_INTRINSICS == (320.0, 320.0, 320.0, 320.0)
    # V2 환경(야외형 포함)은 실내 10m 한정 규격이 아니다.
    assert TARTANAIR_MAX_DEPTH_M > 10.0
