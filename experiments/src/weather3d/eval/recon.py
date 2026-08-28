"""다중뷰 재구성 평가: Accuracy / Completeness / NC.

StreamVGGT의 mv_recon 평가(src/eval/mv_recon/launch.py + utils.py) 절차를
단일 GPU 단독 실행 형태로 재현한다:

1. pred point map과 GT point map을 각각 첫 프레임 카메라 좌표계로 변환
2. 중앙 224x224 crop
3. GT depth 유효 마스크 적용, 점 수 상한 샘플링(재현 가능 seed)
4. pred를 GT 장면 스케일로 정규화(중심거리 중앙값 기반)
5. open3d ICP(threshold 0.1m) 정합 후 estimate_normals
6. KDTree 기반 accuracy/completion + normal consistency

원본 launch.py와의 차이는 accelerate 다중 프로세스 분산 제거와 점 수 상한
샘플링뿐이다.
"""

from __future__ import annotations

import numpy as np

from .pose import extri_to_c2w


def _crop_center(arr: np.ndarray, crop: int) -> np.ndarray:
    """(N, H, W, ...) 배열의 마지막 공간 축 기준 중앙 crop. H or W < crop이면 통과."""
    h, w = arr.shape[1], arr.shape[2]
    if h < crop or w < crop:
        return arr
    cy, cx = h // 2, w // 2
    half = crop // 2
    return arr[:, cy - half : cy + half, cx - half : cx + half]


def _joint_center_scale(pts: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """전체 프레임 마스크 점의 중앙값 중심과 중심거리 중앙값(스케일)."""
    sel = pts[mask]
    center = np.median(sel, axis=0)
    scale = float(np.median(np.linalg.norm(sel - center, axis=-1)))
    return center, max(scale, 1e-8)


def recon_metrics(
    pred_pts: np.ndarray,
    pred_extri: np.ndarray,
    gt_pts: np.ndarray,
    gt_c2w: np.ndarray,
    valid_mask: np.ndarray,
    crop224: bool = True,
    icp_threshold: float = 0.1,
    max_points: int = 500_000,
    seed: int = 0,
) -> dict:
    """입력:
        pred_pts   (N, H, W, 3) 모델 좌표계 world point map
        pred_extri (N, 3, 4)    모델 좌표계 w2c extrinsic
        gt_pts     (N, H, W, 3) GT world point map(unproject로 생성)
        gt_c2w     (N, 4, 4)    GT c2w pose
        valid_mask (N, H, W)    GT depth 유효 마스크
    """
    import open3d as o3d
    from scipy.spatial import cKDTree as KDTree

    pred_pts = np.asarray(pred_pts, dtype=np.float64)
    gt_pts = np.asarray(gt_pts, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if crop224:
        pred_pts = _crop_center(pred_pts, 224)
        gt_pts = _crop_center(gt_pts, 224)
        valid_mask = _crop_center(valid_mask, 224)

    # 첫 프레임 카메라 좌표계로 변환
    gt_t = np.linalg.inv(np.asarray(gt_c2w, dtype=np.float64)[0])
    gt_pts = gt_pts @ gt_t[:3, :3].T + gt_t[:3, 3]
    pred_w2c0 = np.asarray(pred_extri, dtype=np.float64)[0]
    pred_pts = pred_pts @ pred_w2c0[:3, :3].T + pred_w2c0[:3, 3]

    mask = valid_mask & np.isfinite(gt_pts).all(-1) & np.isfinite(pred_pts).all(-1)
    n_pts = int(mask.sum())
    if n_pts < 1000:
        return {"num_points": n_pts, "acc": float("nan"), "comp": float("nan"), "nc": float("nan")}

    if n_pts > max_points:
        rng = np.random.default_rng(seed)
        flat_idx = np.flatnonzero(mask.reshape(-1))
        keep = rng.choice(flat_idx, size=max_points, replace=False)
        keep_mask = np.zeros(mask.size, dtype=bool)
        keep_mask[keep] = True
        keep_mask = keep_mask.reshape(mask.shape)
    else:
        keep_mask = mask

    pr = pred_pts[keep_mask]
    gt = gt_pts[keep_mask]

    # pred를 GT 스케일에 맞춤(중심거리 중앙값 비율)
    _, pr_scale = _joint_center_scale(pred_pts, mask)
    _, gt_scale = _joint_center_scale(gt_pts, mask)
    pr = pr * (gt_scale / pr_scale)

    pcd_pr = o3d.geometry.PointCloud()
    pcd_pr.points = o3d.utility.Vector3dVector(pr)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt)

    reg = o3d.pipelines.registration.registration_icp(
        pcd_pr,
        pcd_gt,
        icp_threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pcd_pr = pcd_pr.transform(reg.transformation)

    pcd_pr.estimate_normals()
    pcd_gt.estimate_normals()
    gt_normal = np.asarray(pcd_gt.normals)
    pr_normal = np.asarray(pcd_pr.normals)
    gt_points = np.asarray(pcd_gt.points)
    pr_points = np.asarray(pcd_pr.points)

    def _acc_comp(gt_p, pr_p, gt_n, pr_n):
        dist_gt, idx = KDTree(gt_p).query(pr_p, workers=-1)
        acc, acc_med = float(np.mean(dist_gt)), float(np.median(dist_gt))
        nc1 = float(np.mean(np.abs(np.sum(gt_n[idx] * pr_n, axis=-1))))
        nc1_med = float(np.median(np.abs(np.sum(gt_n[idx] * pr_n, axis=-1))))
        dist_pr, idx2 = KDTree(pr_p).query(gt_p, workers=-1)
        comp, comp_med = float(np.mean(dist_pr)), float(np.median(dist_pr))
        nc2 = float(np.mean(np.abs(np.sum(gt_n * pr_n[idx2], axis=-1))))
        nc2_med = float(np.median(np.abs(np.sum(gt_n * pr_n[idx2], axis=-1))))
        return acc, acc_med, comp, comp_med, nc1, nc1_med, nc2, nc2_med

    acc, acc_med, comp, comp_med, nc1, nc1_med, nc2, nc2_med = _acc_comp(
        gt_points, pr_points, gt_normal, pr_normal
    )
    return {
        "num_points": n_pts,
        "num_points_used": int(keep_mask.sum()),
        "acc": acc,
        "acc_med": acc_med,
        "comp": comp,
        "comp_med": comp_med,
        "nc1": nc1,
        "nc1_med": nc1_med,
        "nc2": nc2,
        "nc2_med": nc2_med,
        "nc": (nc1 + nc2) / 2.0,
        "nc_med": (nc1_med + nc2_med) / 2.0,
    }


def build_gt_points(depths: np.ndarray, intrinsics: np.ndarray, poses_c2w: np.ndarray, target_hw) -> np.ndarray:
    """GT depth를 target 해상도로 resize 후 unproject해 world point map 생성.

    depths (N, H0, W0) -> (N, h, w, 3). intrinsics는 (N, 3, 3) 원본 해상도 기준.
    """
    import cv2

    from ..gt import unproject_depth

    h, w = target_hw
    n = depths.shape[0]
    out = np.zeros((n, h, w, 3), dtype=np.float64)
    for i in range(n):
        d = depths[i]
        if d.shape != (h, w):
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
        k = intrinsics[i].copy().astype(np.float64)
        # 원본 (H0, W0) -> (h, w) 스케일 보정
        sy, sx = h / depths.shape[1], w / depths.shape[2]
        k[0] *= sx
        k[1] *= sy
        pts_cam = unproject_depth(d.astype(np.float32), k).astype(np.float64)
        out[i] = pts_cam @ poses_c2w[i, :3, :3].T + poses_c2w[i, :3, 3]
    return out
