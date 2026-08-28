"""Track A v2 합성(비/저조도) 단위 테스트."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weather3d.synth.atmosphere import synthesize_frame
from weather3d.synth.weather_ext import apply_lowlight, apply_rain


def _img(h=48, w=64, seed=1):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.9, (h, w, 3)).astype(np.float32)


def _depth(h=48, w=64, val=4.0):
    return np.full((h, w), val, dtype=np.float32)


def test_rain_shape_and_range():
    out = apply_rain(_img(), _depth(), 0.04, 100, 12, np.random.default_rng(0))
    assert out.shape == (48, 64, 3) and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_rain_identity_when_disabled():
    img = _img()
    depth = _depth()
    out = apply_rain(img, depth, 0.0, 0, 12, np.random.default_rng(0))
    assert np.allclose(out, img, atol=1e-6)


def test_rain_deterministic_same_rng():
    img, depth = _img(), _depth()
    a = apply_rain(img, depth, 0.04, 200, 14, np.random.default_rng(7))
    b = apply_rain(img, depth, 0.04, 200, 14, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_rain_streaks_change_image():
    img, depth = _img(), _depth()
    calm = apply_rain(img, depth, 0.0, 0, 12, np.random.default_rng(0))
    rainy = apply_rain(img, depth, 0.0, 300, 14, np.random.default_rng(0))
    assert np.abs(rainy - calm).max() > 0.05


def test_rain_depth_invalid_pixels_keep_veiling_identity():
    # depth<=0 픽셀은 veiling이 원본을 유지한다(streak 가산은 그대로).
    img = _img()
    depth = _depth()
    depth[0, :8] = 0.0
    out = apply_rain(img, depth, 0.10, 0, 12, np.random.default_rng(0))
    assert np.allclose(out[0, :8], img[0, :8], atol=1e-6)


def test_lowlight_darker():
    out = apply_lowlight(_img(), 2.2, 0.32, 0.0, np.random.default_rng(0))
    assert out.mean() < _img().mean()


def test_lowlight_shape_range_deterministic():
    img = _img(seed=3)
    a = apply_lowlight(img, 2.2, 0.32, 0.008, np.random.default_rng(5))
    b = apply_lowlight(img, 2.2, 0.32, 0.008, np.random.default_rng(5))
    assert a.shape == (48, 64, 3)
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert np.array_equal(a, b)


def test_synthesize_frame_dispatch_rain_lowlight():
    img, depth = _img(), _depth()
    rain = synthesize_frame(img, depth, "rain", "heavy", rng=np.random.default_rng(0))
    low = synthesize_frame(img, depth, "lowlight", "heavy", rng=np.random.default_rng(0))
    assert rain.shape == img.shape and low.shape == img.shape
    assert rain.max() > 0.0 and low.mean() < img.mean()
    try:
        synthesize_frame(img, depth, "rain", "heavy")  # rng 없으면 거부
        raise AssertionError("rain without rng must raise")
    except ValueError:
        pass
