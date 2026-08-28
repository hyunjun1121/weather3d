"""4단계: case별 정량 평가(video depth / pose / 재구성).

사용:
    python run_evaluate.py --config configs/core_v1.yaml
    python run_evaluate.py --config configs/core_v1.yaml --cases c1 --variants fog_mid

출력: outputs/<exp>/eval/<case_label>/<seq_id>.json

지표(실험 설계 v1, StreamVGGT 기존 프로토콜 준수):
- video depth: AbsRel / Sq Rel / RMSE / Log RMSE / delta (scale&shift 정렬)
- camera pose: ATE RMSE(Sim3 정렬), RPE translation/rotation
- multi-view recon: Acc / Comp / NC(ICP threshold 0.1m)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from weather3d.config import load_config, output_dir
from weather3d.eval import build_gt_points, evaluate_pose, extri_to_c2w, recon_metrics, video_depth_metrics
from weather3d.gt import load_gt
from weather3d.sequences import discover_sequences


def evaluate_one(preds_path: Path, seq, cfg: dict) -> dict:
    from weather3d.infer.io import load_predictions

    preds = load_predictions(preds_path)
    gt = load_gt(seq)
    ev = cfg["eval"]

    n = preds["depths"].shape[0]
    if n != gt.num_frames:
        raise ValueError(
            f"{seq.seq_id}: prediction frames {n} != GT frames {gt.num_frames}; "
            "합성/추론 프레임 목록이 GT와 어긋납니다."
        )

    results = {
        "seq_id": seq.seq_id,
        "dataset": seq.dataset,
        "num_frames": n,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    results["video_depth"] = video_depth_metrics(
        preds["depths"],
        gt.depths,
        src_dir=cfg["streamvggt"]["src_dir"],
        max_depth=float(ev["max_depth"]),
        align=ev["align"],
        use_gpu=bool(ev["use_gpu"]),
    )

    results["pose"] = evaluate_pose(
        gt.poses_c2w,
        extri_to_c2w(preds["extri"].astype(np.float64)),
        gt.pose_valid,
        align=ev.get("pose_align", "sim3"),
    )

    img_h, img_w = [int(x) for x in preds["img_size_hw"]]
    gt_pts = build_gt_points(gt.depths, gt.intrinsics, gt.poses_c2w, (img_h, img_w))
    results["recon"] = recon_metrics(
        preds["pts3d"].astype(np.float64),
        preds["extri"].astype(np.float64),
        gt_pts,
        gt.poses_c2w.astype(np.float64),
        gt.depths > 0,
        crop224=bool(ev["crop224"]),
        icp_threshold=float(ev["icp_threshold"]),
        max_points=int(ev["max_points"]),
        seed=0,
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "core_v1.yaml"))
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--sequences", nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = output_dir(cfg)
    preds_root = out / "preds"
    if not preds_root.is_dir():
        print(f"no predictions found under {preds_root}; run_infer.py 먼저 실행하세요.")
        return 2

    seqs = {s.seq_id: s for s in discover_sequences(cfg)}
    if args.sequences:
        seqs = {k: v for k, v in seqs.items() if k in args.sequences}

    case_dirs = sorted(d.name for d in preds_root.iterdir() if d.is_dir())
    if args.cases:
        case_dirs = [c for c in case_dirs if c.split("_")[0] in args.cases]
    if args.variants:
        case_dirs = [c for c in case_dirs if "_" not in c or c.split("_", 1)[1] in args.variants]

    for case_label in case_dirs:
        for npz in sorted((preds_root / case_label).glob("*.npz")):
            seq_id = npz.stem
            if seq_id not in seqs:
                print(f"[WARN] {case_label}/{seq_id}: 설정에 없는 시퀀스, 건너뜀")
                continue
            dst = out / "eval" / case_label / f"{seq_id}.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                results = evaluate_one(npz, seqs[seq_id], cfg)
            except Exception as e:
                results = {"seq_id": seq_id, "case": case_label, "error": str(e)}
                print(f"[EVAL ERROR] {case_label}/{seq_id}: {e}")
            results["case"] = case_label
            dst.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            if "error" not in results:
                d, p, r = results["video_depth"], results["pose"], results["recon"]
                print(
                    f"[EVAL] {case_label}/{seq_id}: "
                    f"AbsRel {d.get('Abs Rel', float('nan')):.4f} "
                    f"delta<1.25 {d.get('δ < 1.25', float('nan')):.3f} "
                    f"| ATE {p['ate_rmse']:.4f}m "
                    f"| Acc {r['acc']:.4f} Comp {r['comp']:.4f} NC {r['nc']:.3f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
