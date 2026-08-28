"""atmospheric scattering 합성 수학 검증."""

from __future__ import annotations

import numpy as np

from weather3d.synth.atmosphere import (
    FOG_BETA,
    SMOKE_BETA,
    SMOKE_SIGMA,
    apply_fog,
    apply_smoke,
    synthesize_frame,
    transmission,
)


def _img(h=6, w=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((h, w, 3)).astype(np.float32)


def _depth(h=6, w=8, fill=3.0):
    return np.full((h, w), fill, dtype=np.float32)


def test_transmission_identity_and_monotonic():
    d = np.array([[0.0, 1.0, 5.0, 10.0]], dtype=np.float32)
    t = transmission(d, beta=0.1)
    assert t[0, 0] == 1.0, "깊이 0에서는 t=1이어야 한다"
    assert np.all(np.diff(t) < 0), "t는 깊이에 대해 단조 감소"
    assert np.isclose(t[0, 1], np.exp(-0.1))


def test_invalid_depth_keeps_original():
    img = _img()
    depth = _depth()
    depth[0, 0] = 0.0  # 무효
    out = apply_fog(img, depth, beta=1.0)
    assert np.allclose(out[0, 0], img[0, 0], atol=1e-6), "무효 깊이 픽셀은 원본 유지"


def test_zero_beta_identity():
    img = _img()
    out = apply_fog(img, _depth(fill=7.0), beta=0.0)
    assert np.allclose(out, img, atol=1e-6), "beta=0이면 항등 변환"


def test_deep_pixel_approaches_airlight():
    img = _img()
    out = apply_fog(img, _depth(fill=1e6), beta=1.0, airlight=(0.9, 0.8, 0.7))
    assert np.allclose(out, np.array([0.9, 0.8, 0.7], dtype=np.float32), atol=1e-3), \
        "깊이->inf에서 픽셀은 airlight로 수렴"


def test_output_range_and_formula():
    img = _img()
    depth = _depth(fill=4.0)
    beta, airlight = 0.25, (0.85, 0.85, 0.85)
    out = apply_fog(img, depth, beta=beta, airlight=airlight)
    assert out.min() >= 0.0 and out.max() <= 1.0
    t = np.exp(-beta * 4.0)
    expected = np.clip(img * t + np.array(airlight) * (1 - t), 0, 1)
    assert np.allclose(out, expected, atol=1e-5)


def test_smoke_zero_noise_equals_fog():
    img = _img()
    depth = _depth()
    noise = np.zeros_like(depth)
    out_smoke = apply_smoke(img, depth, beta=0.1, sigma=0.5, noise01=noise, airlight=(0.8, 0.8, 0.8))
    out_fog = apply_fog(img, depth, beta=0.1, airlight=(0.8, 0.8, 0.8))
    assert np.allclose(out_smoke, out_fog, atol=1e-6)


def test_smoke_noise_increases_haze_locally():
    img = _img(seed=1)
    depth = _depth()
    noise = np.zeros_like(depth)
    noise[:, : depth.shape[1] // 2] = 1.0  # 왼쪽 절반에 진한 연기
    out = apply_smoke(img, depth, beta=0.05, sigma=0.2, noise01=noise)
    left = slice(None, depth.shape[1] // 2)
    right = slice(depth.shape[1] // 2, None)
    left_err = np.abs(out[:, left] - img[:, left]).mean()
    right_err = np.abs(out[:, right] - img[:, right]).mean()
    assert left_err > right_err * 2, "noise가 높은 영역이 더 많은 안개를 받아야 한다"


def test_presets_and_entrypoint():
    for level in ("light", "mid", "heavy"):
        assert 0 < FOG_BETA[level] < SMOKE_BETA[level] + SMOKE_SIGMA[level] + 1
    out = synthesize_frame(_img(), _depth(), "fog", "mid")
    assert out.shape == (6, 8, 3)
    out = synthesize_frame(_img(), _depth(), "smoke", "mid", noise01=np.zeros((6, 8), np.float32))
    assert out.shape == (6, 8, 3)
    try:
        synthesize_frame(_img(), _depth(), "smoke", "mid")
        raise AssertionError("noise 없는 smoke는 에러여야 한다")
    except ValueError:
        pass
