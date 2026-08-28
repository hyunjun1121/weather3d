"""7-Scenes / Neural-RGBD / TartanAir V2 시퀀스 발견과 프레임 목록 구성.

디렉터리 규격은 StreamVGGT eval 코드(src/eval/mv_recon/data.py)가 기대하는
Spann3R/MonST3R 전처리 형식을 따른다.

- 7-Scenes:   <root>/<scene>/<seq-XX>/frame-XXXXXX.color.png (+ .depth.proj.png, .pose.txt)
- Neural-RGBD: <root>/<sequence>/images/imgXXXXXX.png (+ depth/depthXXXXXX.png, poses.txt)
- TartanAir V2(HF theairlabcmu/tartanair2, 10b 서버 실측 구조):
    <root>/<Env>/Data_<easy|hard>/extracted/<Env>/Data_<easy|hard>/<P###>/
      image_lcam_front/NNNNNN_lcam_front.png
      depth_lcam_front/NNNNNN_lcam_front_depth.png
      pose_lcam_front.txt   (프레임당 1행, tx ty tz qx qy qz qw)
  easy/hard는 같은 시작점의 서로 다른 궤적 집합(난이도)이며 clean/degraded
  쌍이 아니다. 날씨 환경(GreatMarsh=안개, Supermarket=비, PolarSciFi=눈)은
  원본 영상 자체가 native weather인 평가 전용 축이다(합성 대상 아님).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class FrameRef:
    index: int          # 데이터셋 내 원본 프레임 번호
    name: str           # 파일명(확장자 제외, degraded 출력 파일명으로도 사용)
    image_path: Path    # clean RGB 경로


@dataclass(frozen=True)
class Sequence:
    seq_id: str         # 실험 내 고유 id, 예: "7scenes_chess_seq-01"
    dataset: str        # "seven_scenes" | "neural_rgbd" | "tartanair2"
    root: Path          # 데이터셋 루트
    rel_dir: str        # root 아래 상대 경로, 예: "chess/seq-01" | "copan"
    stride: int
    frames: tuple[FrameRef, ...] = field(default_factory=tuple)
    # tartanair2 전용 메타(rel_dir는 이미 env/difficulty/traj를 포함).
    env: str = ""
    difficulty: str = ""
    traj: str = ""

    @property
    def dir(self) -> Path:
        return self.root / self.rel_dir

    @property
    def image_paths(self) -> list[Path]:
        return [f.image_path for f in self.frames]


def _list_images(folder: Path, prefix: str = "", suffix_filter: str | None = None) -> list[tuple[int, str, Path]]:
    out: list[tuple[int, str, Path]] = []
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        stem = p.stem
        if prefix and not stem.startswith(prefix):
            continue
        digits = "".join(ch for ch in stem if ch.isdigit())
        out.append((int(digits) if digits else -1, stem, p))
    out.sort(key=lambda x: x[0])
    return out


def discover_sequence(seq_cfg: dict, data_roots: dict[str, Path]) -> Sequence:
    """설정 항목 하나에서 Sequence를 만든다. 프레임은 stride 간격으로 샘플링한다."""
    dataset = seq_cfg["dataset"]
    if dataset not in ("seven_scenes", "neural_rgbd", "tartanair2"):
        raise ValueError(f"unsupported dataset: {dataset}")
    root_key = {
        "seven_scenes": "seven_scenes_root",
        "neural_rgbd": "neural_rgbd_root",
        "tartanair2": "tartanair2_root",
    }[dataset]
    root = Path(data_roots[root_key])

    env = difficulty = traj = ""
    if dataset == "seven_scenes":
        rel_dir = f"{seq_cfg['scene']}/{seq_cfg['seq']}"
        entries = _list_images(root / rel_dir, prefix="frame-")
        # frame-XXXXXX.color.png만 RGB 프레임으로 취급(.depth.proj.png 등 제외).
        # name은 "frame-XXXXXX.color"까지 유지해 degraded 출력 파일명 형식을 보존한다.
        frames = [FrameRef(idx, name, path) for idx, name, path in entries if name.endswith(".color")]
    elif dataset == "tartanair2":
        env = str(seq_cfg["env"])
        difficulty = str(seq_cfg["difficulty"])
        traj = str(seq_cfg["traj"])
        rel_dir = f"{env}/Data_{difficulty}/extracted/{env}/Data_{difficulty}/{traj}"
        entries = _list_images(root / rel_dir / "image_lcam_front")
        # image_lcam_front/ 안에는 NNNNNN_lcam_front.png만 있지만 depth와의
        # 파일명 대응(FrRef.name 기반)을 위해 접미사 필터로 확정한다.
        frames = [FrameRef(idx, name, path) for idx, name, path in entries if name.endswith("_lcam_front")]
    else:
        rel_dir = seq_cfg["scene"]
        entries = _list_images(root / rel_dir / "images", prefix="img")
        frames = [FrameRef(idx, name, path) for idx, name, path in entries]

    if not frames:
        raise FileNotFoundError(
            f"no frames found for {seq_cfg['id']} at {root / rel_dir}. "
            "데이터 준비 방법은 experiments/README.md 참고."
        )

    stride = max(1, int(seq_cfg.get("stride", 1)))
    frames = frames[::stride]
    max_frames = int(seq_cfg.get("max_frames", 0) or 0)
    if max_frames > 0:
        frames = frames[:max_frames]
    return Sequence(
        seq_id=seq_cfg["id"],
        dataset=dataset,
        root=root,
        rel_dir=rel_dir,
        stride=stride,
        frames=tuple(frames),
        env=env,
        difficulty=difficulty,
        traj=traj,
    )


def discover_sequences(cfg: dict) -> list[Sequence]:
    data_roots = {
        "seven_scenes_root": Path(cfg["data"]["seven_scenes_root"]),
        "neural_rgbd_root": Path(cfg["data"]["neural_rgbd_root"]),
        # load_config가 항상 기본값을 채운다. 직접 구성한 cfg 테스트용 fallback.
        "tartanair2_root": Path(cfg["data"].get("tartanair2_root", "../../data/tartanair2")),
    }
    return [discover_sequence(s, data_roots) for s in cfg["data"]["sequences"]]
