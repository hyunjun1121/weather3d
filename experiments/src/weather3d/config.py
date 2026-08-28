"""실험 설정(YAML) 로딩과 검증."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
"""spaceai-research/ 루트."""

VALID_DATASETS = ("seven_scenes", "neural_rgbd", "tartanair2")
VALID_ALIGN = ("scale&shift", "scale", "metric")


@functools.lru_cache(maxsize=None)
def load_config(path: str | Path) -> dict[str, Any]:
    """YAML 설정을 읽고 필수 항목을 검증한 뒤 캐시해 반환한다."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")

    for key in ("experiment", "streamvggt", "data", "weather", "eval"):
        if key not in cfg:
            raise ValueError(f"config missing required key: {key}")

    sv = cfg["streamvggt"]
    sv.setdefault("src_dir", str((REPO_ROOT / "third_party" / "StreamVGGT" / "src").resolve()))
    sv.setdefault("weights", str((REPO_ROOT / "third_party" / "StreamVGGT" / "ckpt" / "checkpoints.pth").resolve()))
    sv.setdefault("device", "cuda")
    sv.setdefault("size", 518)
    sv.setdefault("crop", False)

    data = cfg["data"]
    data.setdefault("seven_scenes_root", str(REPO_ROOT / "data" / "7scenes"))
    data.setdefault("neural_rgbd_root", str(REPO_ROOT / "data" / "neural_rgbd"))
    data.setdefault("tartanair2_root", str(REPO_ROOT / "data" / "tartanair2"))
    if not data.get("sequences"):
        raise ValueError("config.data.sequences must list at least one sequence")

    for seq in data["sequences"]:
        if seq.get("dataset") not in VALID_DATASETS:
            raise ValueError(f"sequence {seq.get('id')}: dataset must be one of {VALID_DATASETS}")
        required = ("id", "stride", "scene") if seq["dataset"] != "tartanair2" else ("id", "stride", "env", "difficulty", "traj")
        for key in required:
            if key not in seq:
                raise ValueError(f"sequence entry missing key: {key}")
        if "seq" not in seq and seq["dataset"] == "seven_scenes":
            raise ValueError(f"seven_scenes sequence {seq['id']} requires 'seq'")
        seq.setdefault("seq", "")

    cfg.setdefault("cases", ["c0", "c1"])
    for case in cfg["cases"]:
        if case not in ("c0", "c1", "c2"):
            raise ValueError(f"unknown case: {case}")

    weather = cfg["weather"]
    weather.setdefault("variants", ["fog_light", "fog_mid", "fog_heavy", "smoke_mid"])
    fog = weather.setdefault("fog", {})
    fog.setdefault("beta", {"light": 0.04, "mid": 0.08, "heavy": 0.16})
    fog.setdefault("airlight", [0.85, 0.85, 0.85])
    smoke = weather.setdefault("smoke", {})
    smoke.setdefault("beta", {"light": 0.03, "mid": 0.06, "heavy": 0.12})
    smoke.setdefault("sigma", {"light": 0.05, "mid": 0.10, "heavy": 0.20})
    smoke.setdefault("airlight", [0.82, 0.80, 0.78])
    smoke.setdefault("noise_res", [8, 6, 4])  # (nx, ny, nt) 기본 해상도
    smoke.setdefault("octaves", 4)
    rain = weather.setdefault("rain", {})
    rain.setdefault("beta", {"light": 0.015, "mid": 0.04, "heavy": 0.09})
    rain.setdefault("density", {"light": 150, "mid": 400, "heavy": 900})
    rain.setdefault("length", {"light": 12, "mid": 18, "heavy": 26})
    rain.setdefault("airlight", [0.72, 0.74, 0.78])
    lowlight = weather.setdefault("lowlight", {})
    lowlight.setdefault("gamma", {"light": 1.6, "mid": 2.2, "heavy": 3.0})
    lowlight.setdefault("gain", {"light": 0.55, "mid": 0.32, "heavy": 0.16})
    lowlight.setdefault("sigma", {"light": 0.004, "mid": 0.008, "heavy": 0.015})
    lowlight.setdefault("tint", [0.96, 0.99, 1.06])
    weather.setdefault("seed", 0)

    ev = cfg["eval"]
    ev.setdefault("max_depth", 10.0)
    ev.setdefault("align", "scale&shift")
    if ev["align"] not in VALID_ALIGN:
        raise ValueError(f"eval.align must be one of {VALID_ALIGN}")
    ev.setdefault("pose_align", "sim3")  # sim3 | se3
    ev.setdefault("crop224", True)
    ev.setdefault("icp_threshold", 0.1)
    ev.setdefault("max_points", 500_000)
    ev.setdefault("use_gpu", True)

    cfg.setdefault("output_dir", str(REPO_ROOT / "outputs" / cfg["experiment"]))

    # 상대 경로는 설정 파일 위치 기준으로 해석한다(실행 디렉터리와 무관하게).
    cfg_dir = path.parent

    def _abs(p: str) -> str:
        q = Path(p)
        return str(q if q.is_absolute() else (cfg_dir / q).resolve())

    sv["src_dir"] = _abs(sv["src_dir"])
    sv["weights"] = _abs(sv["weights"])
    data["seven_scenes_root"] = _abs(data["seven_scenes_root"])
    data["neural_rgbd_root"] = _abs(data["neural_rgbd_root"])
    data["tartanair2_root"] = _abs(data["tartanair2_root"])
    cfg["output_dir"] = _abs(cfg["output_dir"])
    cfg["c2_input_dir"] = cfg.get("c2_input_dir", "restored")

    cfg["_config_path"] = str(path)
    return cfg


def output_dir(cfg: dict[str, Any]) -> Path:
    out = Path(cfg["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def split_variant(variant: str) -> tuple[str, str]:
    """'fog_mid' -> ('fog', 'mid'). TartanAir native 축의 'native'는
    run_infer/c2에서만 쓰는 의사 variant라 합성 검증 대상이 아니다."""
    kind, _, level = variant.partition("_")
    if not level or kind not in ("fog", "smoke", "rain", "lowlight"):
        raise ValueError(f"bad weather variant: {variant}")
    return kind, level
