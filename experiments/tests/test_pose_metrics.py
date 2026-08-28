"""pose 지표 검증: Umeyama 복원과 ATE/RPE known-answer."""

from __future__ import annotations

import numpy as np

from weather3d.eval.pose import evaluate_pose, extri_to_c2w, umeyama_alignment
from weather3d.gt import unproject_depth


def _trajectory(n=20, seed=0):
    """부드러운 나선 이동 c2w trajectory (n, 4, 4)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n)
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = 2.0 * np.cos(t)
    poses[:, 1, 3] = 2.0 * np.sin(t)
    poses[:, 2, 3] = 0.5 * t
    # 작은 회전 변화를 더한다
    for i in range(n):
        ang = 0.05 * i
        c, s = np.cos(ang), np.sin(ang)
        poses[i, :3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return poses.astype(np.float64)


def test_umeyama_recovers_transform():
    rng = np.random.default_rng(1)
    src = rng.random((500, 3)) * 10.0
    scale = 2.5
    ang = np.deg2rad(37.0)
    rot = np.array(
        [
            [np.cos(ang), -np.sin(ang), 0],
            [np.sin(ang), np.cos(ang), 0],
            [0, 0, 1],
        ]
    )
    shift = np.array([1.0, -2.0, 3.0])
    dst = (scale * (rot @ src.T)).T + shift

    s_est, r_est, t_est = umeyama_alignment(src, dst, with_scale=True)
    assert np.isclose(s_est, scale, rtol=1e-6)
    assert np.allclose(r_est, rot, atol=1e-8)
    assert np.allclose(t_est, shift, atol=1e-7)


def test_ate_zero_for_similar_trajectories():
    gt = _trajectory()
    # GT에 Sim(3) 변환을 적용한 예측(모델 좌표계 차이 모사) -> 정렬 후 ATE ~ 0
    s = 3.7
    rot = np.array([[0, 0, 1.0], [0, 1, 0], [-1.0, 0, 0]])
    pred = gt.copy()
    pred[:, :3, 3] = (s * (rot @ gt[:, :3, 3].T)).T + np.array([5, -3, 2])
    pred[:, :3, :3] = rot @ gt[:, :3, :3]

    res = evaluate_pose(gt, pred, align="sim3")
    assert res["ate_rmse"] < 1e-8, f"정합된 trajectory의 ATE는 0이어야 한다: {res['ate_rmse']}"
    assert res["rpe_trans_mean"] < 1e-8
    assert res["rpe_rot_deg_mean"] < 1e-6
    assert np.isclose(res["alignment_scale"], 1.0 / s)


def test_ate_nonzero_for_perturbed():
    gt = _trajectory()
    rng = np.random.default_rng(2)
    pred = gt.copy()
    pred[:, :3, 3] += rng.normal(0, 0.1, size=(len(gt), 3))
    res = evaluate_pose(gt, pred, align="se3")
    assert 0 < res["ate_rmse"] < 0.25


def test_invalid_pose_frames_excluded():
    gt = _trajectory()
    valid = np.ones(len(gt), dtype=bool)
    valid[5:8] = False
    pred = gt.copy()
    pred[5:8, :3, 3] = np.nan
    res = evaluate_pose(gt, pred, pose_valid=valid, align="sim3")
    assert res["num_valid_frames"] == len(gt) - 3
    assert res["ate_rmse"] < 1e-8


def test_extri_to_c2w_roundtrip():
    rng = np.random.default_rng(3)
    c2w = _trajectory(5)
    w2c = np.stack([np.linalg.inv(p) for p in c2w])
    extri = w2c[:, :3, :]
    assert np.allclose(extri_to_c2w(extri), c2w, atol=1e-9)


def test_unproject_depth_geometry():
    h, w, d_val = 48, 64, 2.5
    k = np.array([[100.0, 0, w / 2], [0, 100.0, h / 2], [0, 0, 1.0]], dtype=np.float32)
    depth = np.full((h, w), d_val, dtype=np.float32)
    pts = unproject_depth(depth, k)
    assert pts.shape == (h, w, 3)
    assert np.allclose(pts[..., 2], d_val), "모든 점의 z는 깊이와 일치"
    cy, cx = h // 2, w // 2
    assert np.allclose(pts[cy, cx], [0, 0, d_val], atol=1e-5), "주점 픽셀은 광선축상"
    assert np.allclose(pts[0, cx, 0], 0.0, atol=1e-5)
    assert np.isclose(pts[cy, cx + 1, 0], d_val / 100.0, atol=1e-5), "x = (u-cx)/fx * depth"
