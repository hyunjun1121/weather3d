"""프랙탈 노이즈 필드 검증(범위, 결정론, 시간 일관성)."""

from __future__ import annotations

import numpy as np

from weather3d.synth.noise import derive_seed, fbm_noise_3d


def test_range_and_shape():
    field = fbm_noise_3d((48, 64), 10, base_res=(4, 3, 3), octaves=3, seed=1)
    assert field.shape == (48, 64, 10)
    assert field.min() >= 0.0 and field.max() <= 1.0
    assert 0.05 < field.std() < 0.5, "완전 평평하거나(상수) 너무 거친 필드는 비정상"


def test_determinism():
    a = fbm_noise_3d((24, 24), 6, seed=42)
    b = fbm_noise_3d((24, 24), 6, seed=42)
    c = fbm_noise_3d((24, 24), 6, seed=43)
    assert np.array_equal(a, b), "같은 seed는 같은 필드"
    assert not np.array_equal(a, c), "다른 seed는 다른 필드"


def test_temporal_smoothness():
    """연속 프레임 차이는 독립 난수장 차이보다 훨씬 작아야 한다(깜빡임 방지)."""
    field = fbm_noise_3d((32, 32), 20, base_res=(4, 3, 4), octaves=3, seed=7)
    step = np.abs(np.diff(field, axis=2)).mean()
    rng = np.random.default_rng(0)
    independent = np.abs(rng.random((32, 32, 20)) - rng.random((32, 32, 20))).mean()
    assert step < 0.2 * independent, (
        f"프레임 간 변화가 지나치게 급격하다: temporal step {step:.4f} vs independent {independent:.4f}"
    )


def test_seed_derivation_stable():
    assert derive_seed("a", "b") == derive_seed("a", "b")
    assert derive_seed("a", "b") != derive_seed("a", "c")
    assert 0 <= derive_seed("x") < 2**32
