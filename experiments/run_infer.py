"""3단계: 실험 case별 StreamVGGT 추론.

사용:
    python run_infer.py --config configs/core_v1.yaml
    python run_infer.py --config configs/core_v1.yaml --cases c1 --variants fog_mid

Case 구성 (readme 실험 설계 v1):
    c0        clean 입력 -> base StreamVGGT (성능 상한 기준선)
    c1_<변형>  degraded 입력 -> base StreamVGGT (성능 붕괴 측정)
    c2_<변형>  날씨 제거 전처리(ViWS-Net) 복원 입력 -> base StreamVGGT
              (restored/<variant>/<seq_id>/ 에 프레임이 준비돼 있어야 함)

출력: outputs/<exp>/preds/<case>/<seq_id>.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from weather3d.config import load_config, output_dir, split_variant
from weather3d.infer.io import save_predictions
from weather3d.infer.model import StreamVGGTRunner
from weather3d.sequences import SUPPORTED_IMAGE_SUFFIXES, discover_sequences


def list_case_images(case: str, variant: str, seq, cfg: dict, out_dir: Path) -> list[Path]:
    if case == "c0":
        return seq.image_paths
    sub = "degraded" if case == "c1" else cfg.get("c2_input_dir", "restored")
    folder = out_dir / sub / variant / seq.seq_id
    if not folder.is_dir():
        raise FileNotFoundError(
            f"{case}_{variant} 입력이 없습니다: {folder} "
            + ("(run_synthesize.py 먼저 실행)" if case == "c1" else "(ViWS-Net 복원 결과를 이 위치에 배치)")
        )
    files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"no images in {folder}")
    return files


def case_runs(cfg: dict, out_dir: Path):
    """(case, variant, case_label) 목록. c0은 variant 없음."""
    runs = []
    for case in cfg["cases"]:
        if case == "c0":
            runs.append((case, None, "c0"))
        else:
            for variant in cfg["weather"]["variants"]:
                runs.append((case, variant, f"{case}_{variant}"))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "core_v1.yaml"))
    parser.add_argument("--cases", nargs="*", default=None, help="c0/c1/c2 필터")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = output_dir(cfg)
    sv = cfg["streamvggt"]

    runs = case_runs(cfg, out)
    if args.cases:
        runs = [r for r in runs if r[0] in args.cases]
    if args.variants:
        runs = [r for r in runs if r[1] is None or r[1] in args.variants]

    seqs = discover_sequences(cfg)
    if args.sequences:
        seqs = [s for s in seqs if s.seq_id in args.sequences]

    runner = StreamVGGTRunner(sv["src_dir"], sv["weights"], device=sv["device"])
    for case, variant, label in runs:
        for seq in seqs:
            dst = out / "preds" / label / f"{seq.seq_id}.npz"
            if dst.is_file() and not args.force:
                print(f"[SKIP] {label}/{seq.seq_id} (exists)")
                continue
            try:
                image_paths = list_case_images(case, variant, seq, cfg, out)
            except FileNotFoundError as e:
                print(f"[MISSING INPUT] {e}")
                continue
            print(f"[INFER] {label}/{seq.seq_id}: {len(image_paths)} frames")
            preds = runner.infer(
                image_paths,
                size=int(sv["size"]),
                crop=bool(sv["crop"]),
                square_ok=bool(sv.get("square_ok", False)),
            )
            save_predictions(dst, preds)
            meta = {
                "case": label,
                "variant": variant or "clean",
                "seq_id": seq.seq_id,
                "num_frames": len(image_paths),
                "input_dir": str(image_paths[0].parent),
                "seconds": preds["seconds"],
                "fps": len(image_paths) / max(preds["seconds"], 1e-9),
            }
            dst.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"        -> {dst} ({preds['seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
