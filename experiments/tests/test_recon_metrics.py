"""재구성 지표(recon_metrics) 회귀 테스트.

일관된 sim3 게이지(점: s·R·p + c, pose: [R·Ri | s·R·ti + c])로 만든
합성 장면에서 스케일 정규화 + ICP가 게이지 차이를 흡수해 acc/comp/NC가
노이즈 수준으로 수렴하는지 확인한다. 실수하기 쉬운 함정: 점을 s·(R·p + c)로
만들면 이동량이 s배가 되어 pose와 어긋난다(게이지 불일치).
"""

from __future__ import annotations

import numpy as np

from weather3d.eval.recon import recon_metrics


def _scene(n=8, h=48, w=64):
    def surface(x, y):
        z = 3.0 + 0.6 * np.sin(1.4 * x) * np.cos(1.1 * y) + 0.25 * np.sin(3.1 * x + 1.7 * y)
        return np.stack([x, y, z], -1)

    gt_pts = np.zeros((n, h, w, 3))
    for i in range(n):
        du, dv = 0.9 * i / n, 0.5 * i / n
        u = (np.arange(w) / w + du)[None, :].repeat(h, 0) * 2 - 1
        v = (np.arange(h) / h + dv)[:, None].repeat(w, 1) * 1.5 - 0.75
        gt_pts[i] = surface(u, v)

    poses = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    for i in range(n):
        a = 0.05 * i
        poses[i, :3, :3] = np.array(
            [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]]
        )
        poses[i, 0, 3], poses[i, 1, 3], poses[i, 2, 3] = 0.2 * i, 0.1 * np.sin(0.7 * i), 0.05 * i
    return gt_pts, poses


def test_sim3_gauge_absorbed_by_alignment():
    gt_pts, poses = _scene()
    s, ang = 2.0, np.deg2rad(30.0)
    rot = np.array(
        [[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]]
    )
    shift = np.array([1.5, -0.7, 0.3])
    noise = np.random.default_rng(0).normal(0, 0.01, gt_pts.shape)
    pred_pts = s * (gt_pts @ rot.T) + shift + noise
    pred_c2w = poses.copy()
    pred_c2w[:, :3, :3] = rot @ poses[:, :3, :3]
    pred_c2w[:, :3, 3] = s * (rot @ poses[:, :3, 3].T).T + shift
    w2c = np.stack([np.linalg.inv(p) for p in pred_c2w])[:, :3, :]

    n, h, w = gt_pts.shape[:3]
    res = recon_metrics(
        pred_pts, w2c, gt_pts, poses, np.ones((n, h, w), dtype=bool), max_points=100_000
    )
    assert res["acc"] < 0.03, f"acc {res['acc']:.4f}가 노이즈 수준을 벗어남"
    assert res["comp"] < 0.03, f"comp {res['comp']:.4f}가 노이즈 수준을 벗어남"
    assert res["nc"] > 0.9, f"NC {res['nc']:.4f} 낮음"


def test_corrupted_cloud_reports_bad_metrics():
    gt_pts, poses = _scene()
    rng = np.random.default_rng(5)
    pred_pts = gt_pts + rng.normal(0, 0.5, gt_pts.shape)  # 큰 노이즈
    n, h, w = gt_pts.shape[:3]
    res = recon_metrics(
        pred_pts,
        np.stack([np.linalg.inv(p) for p in poses])[:, :3, :],
        gt_pts,
        poses,
        np.ones((n, h, w), dtype=bool),
        max_points=100_000,
    )
    assert res["acc"] > 0.1, "손상된 점군은 높은 오류를 보고해야 함"
