"""1단계: 데이터셋 존재 확인 + GT 무결성 점검 + manifest 작성.

사용:
    python run_prepare.py --config configs/core_v1.yaml

데이터가 없으면 각 시퀀스 경로와 준비 방법(experiments/README.md)을 안내하고
0이 아닌 exit code로 종료한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from weather3d.config import load_config, output_dir
from weather3d.gt import load_gt
from weather3d.sequences import discover_sequences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "core_v1.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = output_dir(cfg)

    try:
        seqs = discover_sequences(cfg)
    except FileNotFoundError as e:
        print(f"[MISSING DATA] {e}")
        print("experiments/README.md의 데이터 준비 절차를 먼저 실행하세요.")
        return 2

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg["_config_path"],
        "experiment": cfg["experiment"],
        "sequences": [],
    }
    ok = True
    for seq in seqs:
        entry = {
            "seq_id": seq.seq_id,
            "dataset": seq.dataset,
            "dir": str(seq.dir),
            "stride": seq.stride,
            "num_frames": len(seq.frames),
        }
        try:
            gt = load_gt(seq)
            valid_depth_ratio = float((gt.depths > 0).mean())
            entry.update(
                {
                    "gt_ok": True,
                    "depth_shape": list(gt.depths.shape[1:]),
                    "valid_depth_ratio": round(valid_depth_ratio, 4),
                    "valid_pose_count": int(gt.pose_valid.sum()),
                }
            )
            print(
                f"[OK] {seq.seq_id}: {len(seq.frames)} frames, "
                f"depth {gt.depths.shape[1]}x{gt.depths.shape[2]}, "
                f"valid depth {valid_depth_ratio:.1%}, "
                    f"valid pose {gt.pose_valid.sum()}/{gt.num_frames}"
            )
        except Exception as e:  # GT 문제는 manifest에 기록하고 계속 진행
            ok = False
            entry.update({"gt_ok": False, "error": str(e)})
            print(f"[GT ERROR] {seq.seq_id}: {e}")
        manifest["sequences"].append(entry)

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nmanifest: {manifest_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
