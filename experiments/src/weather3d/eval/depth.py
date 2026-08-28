"""비디오 depth 평가.

StreamVGGT 자체 평가 함수(src/eval/video_depth/tools.py의 depth_evaluation)를
재사용한다. 정렬은 기본 scale&shift(LAD 손실 기반 absolute_value_scaling2)로,
프로토콜은 시퀀스 단위 스택 정렬 후 AbsRel/RMSE/delta 지표를 뽑는 것이다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def _ensure_src(src_dir: str | Path):
    src_dir_path = Path(src_dir).resolve()
    if not src_dir_path.is_dir():
        raise FileNotFoundError(
            f"StreamVGGT src directory not found: {src_dir_path}. "
            "config의 streamvggt.src_dir을 확인하세요."
        )
    src_dir = str(src_dir_path)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def video_depth_metrics(
    pred_depths: np.ndarray,
    gt_depths: np.ndarray,
    src_dir: str | Path,
    max_depth: float = 10.0,
    align: str = "scale&shift",
    use_gpu: bool = True,
) -> dict:
    """pred (N, h, w), GT (N, H, W) -> StreamVGGT 프로토콜 depth 지표 dict.

    pred를 GT 해상도로 resize(INTER_CUBIC)한 뒤 시퀀스 전체를 한 번에
    정렬·평가한다(공개 eval 코드와 동일).
    """
    _ensure_src(src_dir)
    from eval.video_depth.tools import depth_evaluation

    if pred_depths.shape[0] != gt_depths.shape[0]:
        raise ValueError(
            f"frame count mismatch: pred {pred_depths.shape[0]}, gt {gt_depths.shape[0]}"
        )
    gt_h, gt_w = gt_depths.shape[1:3]
    pr = np.stack(
        [cv2.resize(p, (gt_w, gt_h), interpolation=cv2.INTER_CUBIC) for p in pred_depths], 0
    ).astype(np.float32)
    gt = gt_depths.astype(np.float32)

    kwargs = dict(max_depth=max_depth, use_gpu=use_gpu)
    if align == "scale&shift":
        results, *_ = depth_evaluation(pr, gt, align_with_lad2=True, **kwargs)
    elif align == "scale":
        results, *_ = depth_evaluation(pr, gt, align_with_scale=True, **kwargs)
    elif align == "metric":
        results, *_ = depth_evaluation(pr, gt, metric_scale=True, **kwargs)
    else:
        raise ValueError(f"unknown align: {align}")
    return {k: float(v) for k, v in results.items()}
