"""2단계: Track A 물리 기반 날씨 합성(fog/smoke)으로 degraded 데이터 생성.

사용:
    python run_synthesize.py --config configs/core_v1.yaml
    python run_synthesize.py --config configs/core_v1.yaml --variants fog_mid --force

clean 프레임 + GT depth -> atmospheric scattering으로 degraded PNG를
outputs/<exp>/degraded/<variant>/<seq_id>/ 에 기록한다. 파일명은 clean과
동일하게 유지해 이후 단계에서 프레임 대응이 어긋나지 않게 한다.
연기 노이즈 seed는 (전역 seed, seq_id)에서 파생되어 재현 가능하다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from weather3d.config import load_config, output_dir, split_variant
from weather3d.gt import load_gt
from weather3d.sequences import discover_sequences
from weather3d.synth.atmosphere import synthesize_frame
from weather3d.synth.noise import derive_seed, fbm_noise_3d


def synthesize_sequence(seq, gt, variant: str, cfg: dict, out_dir: Path, force: bool) -> Path:
    kind, level = split_variant(variant)
    weather = cfg["weather"]
    target = out_dir / "degraded" / variant / seq.seq_id
    marker = target / "_synth_done.json"
    if marker.is_file() and not force:
        print(f"[SKIP] {seq.seq_id}/{variant} (exists)")
        return target
    target.mkdir(parents=True, exist_ok=True)

    noise = None
    if kind == "smoke":
        seed = derive_seed(weather.get("seed", 0), seq.seq_id, variant)
        h, w = gt.depths.shape[1:3]
        noise = fbm_noise_3d(
            (h, w),
            gt.num_frames,
            base_res=tuple(weather["smoke"].get("noise_res", [8, 6, 4])),
            octaves=int(weather["smoke"].get("octaves", 4)),
            seed=seed,
        )

    fog = weather["fog"]
    smoke = weather["smoke"]
    rain = weather.get("rain", {})
    lowlight = weather.get("lowlight", {})
    for i, fr in enumerate(seq.frames):
        bgr = cv2.imread(str(fr.image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"cannot read image: {fr.image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        noise_slice = noise[..., i] if noise is not None else None
        rng = None
        if kind in ("rain", "lowlight"):
            # 빗줄기/노이즈는 프레임별 결정론 rng(시드: 전역seed|seq|variant|frame).
            rng = np.random.default_rng(derive_seed(weather.get("seed", 0), seq.seq_id, variant, i))
        degraded = synthesize_frame(
            rgb,
            gt.depths[i],
            kind,
            level,
            noise01=noise_slice,
            rng=rng,
            fog_beta=fog.get("beta"),
            smoke_beta=smoke.get("beta"),
            smoke_sigma=smoke.get("sigma"),
            rain_beta=rain.get("beta"),
            rain_density=rain.get("density"),
            rain_length=rain.get("length"),
            lowlight_gamma=lowlight.get("gamma"),
            lowlight_gain=lowlight.get("gain"),
            lowlight_sigma=lowlight.get("sigma"),
            fog_airlight=tuple(fog.get("airlight", (0.85, 0.85, 0.85))),
            smoke_airlight=tuple(smoke.get("airlight", (0.82, 0.80, 0.78))),
            rain_airlight=tuple(rain.get("airlight", (0.72, 0.74, 0.78))),
            lowlight_tint=tuple(lowlight.get("tint", (0.96, 0.99, 1.06))),
        )
        out_png = target / f"{fr.name}.png"
        cv2.imwrite(str(out_png), cv2.cvtColor((degraded * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR))

    params = {
        "variant": variant,
        "kind": kind,
        "level": level,
        "seed": derive_seed(weather.get("seed", 0), seq.seq_id, variant) if kind in ("smoke", "rain", "lowlight") else None,
        "frames": len(seq.frames),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    marker.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"[DONE] {seq.seq_id}/{variant}: {len(seq.frames)} frames -> {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "core_v1.yaml"))
    parser.add_argument("--variants", nargs="*", default=None, help="설정의 weather.variants 중 일부만 실행")
    parser.add_argument("--sequences", nargs="*", default=None, help="seq id 필터")
    parser.add_argument("--force", action="store_true", help="기존 결과 무시하고 재생성")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = output_dir(cfg)
    variants = args.variants or cfg["weather"]["variants"]
    for v in variants:
        split_variant(v)  # 유효성 검사

    seqs = discover_sequences(cfg)
    if args.sequences:
        seqs = [s for s in seqs if s.seq_id in args.sequences]

    for seq in seqs:
        gt = load_gt(seq)
        for variant in variants:
            synthesize_sequence(seq, gt, variant, cfg, out, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
