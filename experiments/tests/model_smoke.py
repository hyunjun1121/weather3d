"""StreamVGGT 가중치 로드 + 더미 이미지 추론 스모크(실데이터 불필요).

사용:
    python tests/model_smoke.py [--config configs/core_v1.yaml]

가중치 다운로드 직후 환경 검증용. 3장의 랜덤 이미지로 추론 파이프라인
(뷰 구성 -> model.inference -> pose decoding -> npz 저장/로딩)을 돌리고
출력 shape를 확인한다. 품질 검증이 아니다.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from weather3d.config import load_config
from weather3d.infer.io import load_predictions, save_predictions
from weather3d.infer.model import StreamVGGTRunner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "core_v1.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    sv = cfg["streamvggt"]

    import cv2

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(3):
            img = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
            p = Path(td) / f"frame-{i:06d}.color.png"
            cv2.imwrite(str(p), img)
            paths.append(p)

        runner = StreamVGGTRunner(sv["src_dir"], sv["weights"], device=sv["device"])
        preds = runner.infer(paths, size=int(sv["size"]), crop=bool(sv["crop"]))

        n, h, w = preds["depths"].shape
        print(f"depths      {preds['depths'].shape}")
        print(f"pts3d       {preds['pts3d'].shape}")
        print(f"extri       {preds['extri'].shape}")
        print(f"intri       {preds['intri'].shape}")
        print(f"seconds     {preds['seconds']:.2f} ({n / preds['seconds']:.1f} FPS)")
        assert preds["depths"].shape == (3, h, w)
        assert np.isfinite(preds["depths"]).any()
        assert preds["pts3d"].shape == (3, h, w, 3)
        assert preds["extri"].shape == (3, 3, 4) and preds["intri"].shape == (3, 3, 3)
        assert (preds["depths"] > 0).any(), "depth는 양수 구간을 가져야 함"

        npz = Path(td) / "smoke.npz"
        save_predictions(npz, preds)
        loaded = load_predictions(npz)
        assert np.allclose(loaded["depths"], preds["depths"])

    print("MODEL SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
