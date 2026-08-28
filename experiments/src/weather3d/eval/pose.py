"""카메라 pose 평가: ATE(Umeyama 정렬 후 RMSE)와 RPE.

모델 pose는 자체 scale/gauge를 가지므로 Sim(3) 정렬을 기본으로 하고
SE(3) 결과도 함께 보고한다. RPE는 연속 프레임 쌍의 상대 pose 오차
(TUM RPE 관례: mean translation[m], mean rotation[deg]).
"""

from __future__ import annotations

import numpy as np


def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """src -> dst 로의 최적 s, R, t 반환 (Umeyama). 입력은 (N, 3)."""
    assert src.shape == dst.shape
    n, dim = src.shape
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    sigma = dst_c.T @ src_c / n
    u, d, vt = np.linalg.svd(sigma)
    s_mat = np.eye(dim)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[-1, -1] = -1
    r = u @ s_mat @ vt
    if with_scale:
        var_src = (src_c**2).sum() / n
        scale = (d * s_mat.diagonal()).sum() / var_src
    else:
        scale = 1.0
    t = mu_dst - scale * r @ mu_src
    return float(scale), r, t


def _rot_angle_deg(r: np.ndarray) -> float:
    cos = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def extri_to_c2w(extri: np.ndarray) -> np.ndarray:
    """(N, 3, 4) w2c extrinsic -> (N, 4, 4) c2w."""
    n = extri.shape[0]
    c2w = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    c2w[:, :3, :3] = extri[:, :, :3]
    c2w[:, :3, 3] = extri[:, :, 3]
    return np.linalg.inv(c2w)


def evaluate_pose(
    gt_c2w: np.ndarray,
    pred_c2w: np.ndarray,
    pose_valid: np.ndarray | None = None,
    align: str = "sim3",
) -> dict:
    """GT/pred c2w (N, 4, 4) -> ATE/RPE 지표 dict.

    pose_valid가 False인 프레임은 평가에서 제외한다(nan pose 등).
    """
    gt = np.asarray(gt_c2w, dtype=np.float64)
    pr = np.asarray(pred_c2w, dtype=np.float64)
    if gt.shape != pr.shape or gt.ndim != 3:
        raise ValueError(f"pose shape mismatch: gt {gt.shape}, pred {pr.shape}")
    if pose_valid is None:
        pose_valid = np.ones(len(gt), dtype=bool)
    pose_valid = np.asarray(pose_valid, dtype=bool)

    gt = gt[pose_valid]
    pr = pr[pose_valid]
    if len(gt) < 3:
        return {"num_valid_frames": int(len(gt)), "ate_rmse": float("nan"),
                "rpe_trans_mean": float("nan"), "rpe_rot_deg_mean": float("nan"),
                "alignment_scale": float("nan")}

    gt_pos = gt[:, :3, 3]
    pr_pos = pr[:, :3, 3]
    scale, r_opt, t_opt = umeyama_alignment(pr_pos, gt_pos, with_scale=(align == "sim3"))
    pr_pos_aligned = (scale * (r_opt @ pr_pos.T)).T + t_opt
    ate = float(np.sqrt(np.mean(np.sum((pr_pos_aligned - gt_pos) ** 2, axis=1))))

    # RPE는 게이지(임의 scale/원점/좌표계) 차이가 남지 않도록 정렬 변환을
    # pose 전체에 적용한 뒤 계산한다(모델 좌표계 스케일이 GT와 다르기 때문).
    pr_aligned = pr.copy()
    pr_aligned[:, :3, :3] = r_opt @ pr_aligned[:, :3, :3]
    pr_aligned[:, :3, 3] = pr_pos_aligned

    # RPE: 연속 쌍 (i, i+1)의 상대 변환 오차
    trans_errs, rot_errs = [], []
    for i in range(len(gt) - 1):
        gt_rel = np.linalg.inv(gt[i]) @ gt[i + 1]
        pr_rel = np.linalg.inv(pr_aligned[i]) @ pr_aligned[i + 1]
        err = np.linalg.inv(pr_rel) @ gt_rel
        trans_errs.append(np.linalg.norm(err[:3, 3]))
        rot_errs.append(_rot_angle_deg(err[:3, :3]))

    return {
        "num_valid_frames": int(len(gt)),
        "ate_rmse": ate,
        "rpe_trans_mean": float(np.mean(trans_errs)) if trans_errs else float("nan"),
        "rpe_rot_deg_mean": float(np.mean(rot_errs)) if rot_errs else float("nan"),
        "alignment_scale": scale,
    }
