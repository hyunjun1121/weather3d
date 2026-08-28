"""Case 2 전처리: ViWS-Net(ICCV 2023) 날씨 제거 복원 드라이버.

사용(서버, .venv-viws):
    cd experiments
    ../.venv-viws/bin/python -B scripts/restore_viws.py \
        --outputs ../outputs/core_v1 --variants fog_extreme --sequences 7scenes_chess_seq-01

outputs/<exp>/degraded/<variant>/<seq_id>/ 프레임 전체에 5프레임
sliding-window 복원을 적용해 outputs/<exp>/restored/<variant>/<seq_id>/ 에
같은 파일명으로 기록한다(run_infer.py --cases c2가 바로 소비).

공식 eval_derain.py와의 차이:
- 프레임 수를 유지한다. 공식 test loader는 5프레임 창이 온전한 내부
  프레임만 내보내지만, 평가 정합을 위해 경계 프레임은 edge replicate로
  채운다(경계 2프레임의 temporal 문맥이 약해지는 한계는 보고서에 명시).
- forward_crop(lq_size=512)은 480행 입력에서 crop 시작 인덱스가 음수가
  되는 인덱싱 버그가 있어 full-frame 직접 호출로 대체한다. 백본(shunted
  transformer)에 절대 positional embedding이 없어 해상도 제약이 없고,
  어텐션 토큰 수가 프레임당 약 300개(480x640 기준)라 48GB GPU에서
  안전하다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]  # spaceai-research/
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
N_FRAMES = 5  # 공개 checkpoint의 학습 윈도우 길이(평가 설정과 동일)


def find_weights(viws_root: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"weights not found: {p}")
        return p
    for rel in ("models/model_motion.pth", "model_motion.pth", "best.pth"):
        p = viws_root / rel
        if p.is_file():
            return p
    hits = sorted(viws_root.rglob("*.pth")) if viws_root.is_dir() else []
    hits = [h for h in hits if any(t in h.name for t in ("model", "best"))] or hits
    if hits:
        print(f"[WARN] 기본 위치에 checkpoint 없음, 발견된 후보 사용: {hits[0]}")
        return hits[0]
    raise FileNotFoundError(f"no .pth under {viws_root}")


def build_model(viws_root: Path, weights: Path, device: str):
    if str(viws_root) not in sys.path:
        sys.path.insert(0, str(viws_root))
    import modeling.model as vm  # shunted backbone의 timm 등록 포함

    # 생성자의 init_weight가 ckpt_T.pth(하드코딩 상대경로)/ckpt_S.pth를 읽는다.
    # 이후 전체 checkpoint로 모든 weight를 덮어쓰므로 파일 부재/로드 실패는
    # 경고 후 진행한다.
    def _guard(orig):
        def safe(self, *a, **k):
            try:
                orig(self, *a, **k)
            except (OSError, RuntimeError, KeyError, EOFError) as e:
                print(f"[WARN] init_weight ignored ({e})")
        return safe

    vm.ViWSNet.init_weight = _guard(vm.ViWSNet.init_weight)
    vm.RefineNet.init_weight = _guard(vm.RefineNet.init_weight)

    params = dict(
        finetune=str(viws_root / "models" / "ckpt_S.pth"),
        hidden_dim=512,
        dropout=0.1,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=6,
        num_queries=48 * N_FRAMES,
        num_types=3,
    )
    prev_cwd = Path.cwd()
    os.chdir(viws_root)  # './models/ckpt_T.pth' 상대경로 해결
    try:
        model = vm.ViWSNet(params)
    finally:
        os.chdir(prev_cwd)

    # torch>=2.6 기본값 weights_only=True 회피(신뢰하는 공식 release만 로드).
    raw = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and isinstance(raw.get("model"), dict):
        raw = raw["model"]
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in raw.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[WARN] state_dict mismatch: missing={len(missing)} unexpected={len(unexpected)}")
        for tag, keys in (("missing", missing[:3]), ("unexpected", unexpected[:3])):
            for k in keys:
                print(f"    {tag}: {k}")
        if missing:
            raise RuntimeError("checkpoint가 모델을 완전히 덮지 못함(로드 중단)")
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] ViWSNet loaded from {weights} ({n_params/1e6:.1f}M params)")
    return model


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def restore_sequence(model, src: Path, dst: Path, device: str, force: bool) -> tuple[int, float]:
    marker = dst / "_restore_done.json"
    if marker.is_file() and not force:
        print(f"[SKIP] {src.parent.name}/{src.name} (exists)")
        return 0, 0.0
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(f"no images in {src}")
    imgs = [load_rgb(p) for p in files]
    dst.mkdir(parents=True, exist_ok=True)
    half = N_FRAMES // 2
    t0 = time.time()
    with torch.no_grad():
        for i, img in enumerate(imgs):
            idxs = [min(max(i + d, 0), len(imgs) - 1) for d in range(-half, half + 1)]
            stack = np.stack([imgs[j] for j in idxs])  # (T, H, W, 3)
            x = torch.from_numpy(stack).permute(0, 3, 1, 2).unsqueeze(0).to(device)
            out = model(x, 1, N_FRAMES, phase="test")  # (T, 3, H, W)
            center = out[half].clamp(0.0, 1.0)
            rgb = (center.cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
            cv2.imwrite(str(dst / f"{files[i].stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    dt = time.time() - t0
    marker.write_text(
        json.dumps(
            {
                "frames": len(files),
                "n_frames_window": N_FRAMES,
                "seconds": round(dt, 2),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[DONE] {src.parent.name}/{src.name}: {len(files)} frames in {dt:.1f}s")
    return len(files), dt


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--outputs", default=str(REPO_ROOT / "outputs" / "core_v1"))
    parser.add_argument("--viws-root", default=str(REPO_ROOT / "third_party" / "ViWS-Net"))
    parser.add_argument("--weights", default=None, help="기본: <viws-root>/models/model_motion.pth")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.outputs).resolve()
    degraded = out_root / "degraded"
    if not degraded.is_dir():
        print(f"degraded root 없음: {degraded} (run_synthesize.py 먼저)")
        return 2
    viws_root = Path(args.viws_root).resolve()
    if not viws_root.is_dir():
        print(f"ViWS-Net repo 없음: {viws_root} (clone 먼저)")
        return 2

    jobs = []
    for var_dir in sorted(degraded.iterdir()):
        if not var_dir.is_dir() or (args.variants and var_dir.name not in args.variants):
            continue
        for seq_dir in sorted(p for p in var_dir.iterdir() if p.is_dir()):
            if args.sequences and seq_dir.name not in args.sequences:
                continue
            jobs.append((var_dir.name, seq_dir))
    if not jobs:
        print("복원 대상 없음(variants/sequences 필터 확인)")
        return 2
    print(f"jobs: {len(jobs)} ({len({v for v, _ in jobs})} variants x "
          f"{len({s.name for _, s in jobs})} sequences)")

    model = build_model(viws_root, find_weights(viws_root, args.weights), args.device)
    total_frames = 0
    for variant, seq_dir in jobs:
        n, _ = restore_sequence(
            model, seq_dir, out_root / "restored" / variant / seq_dir.name, args.device, args.force
        )
        total_frames += n
    print(f"restored frames: {total_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
