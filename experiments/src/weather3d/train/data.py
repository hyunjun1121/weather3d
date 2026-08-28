"""C3 학습용 윈도우 데이터셋: 7-Scenes TrainSplit + NRGBD 비평가 장면.

설계(실험맥락 §8 + 13번 센서스, 2026-08-26):
- 학습 장면: 7-Scenes 6장면(chess/fire/office/pumpkin/redkitchen/stairs의
  TrainSplit.txt에 있고 디스크에 있는 seq) + NRGBD 5장면(kitchen,
  grey_white_room, complete_kitchen, morning_apartment, thin_geometry).
  평가 장면(heads, whiteroom, staircase, breakfast_room, green_room,
  TartanAir V2)은 제외. 13번 센서스 합계 26,536프레임.
- 샘플 = 한 seq에서 연속 윈도우(num_views x stride). 비디오 특성상
  순서를 유지한다(streaming 모델 입력).
- 열화는 Track A v2(synth/)를 프레임 단위 즉석 적용. 외관만 바꾸고
  pose/기하 GT는 그대로(Track A 전제).
- "β 연속 샘플링, 윈도우 내 시간 일관": 강도 severity s~U(0,1)을 윈도우
  단위로 한 번 뽑고 eval 프리셋(ext_v2.yaml)의 light..heavy(fog는
  extreme) 구간을 선형 보간한다. 연기 노이즈는 시간축 상관 3D 볼륨.
- clean 비율: R1=0.5(clean 50% + 열화 50% replay), R2=0.0(열화만).

뷰 dict 규약은 13b 서베이로 확정한 upstream 학습/추론 공용 형태를 따른다:
default collate(batch_size=1)를 통과한 [1,...] 선두 차원 텐서.
img는 [0,1](upstream은 [-1,1] 생성 후 loop에서 (x+1)/2 환산; 여기는
처음부터 [0,1]을 내놓는다 — 결과 값은 동일).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from ..gt import _load_nrgbd_poses, _read_depth_png
from ..synth.atmosphere import apply_fog, apply_smoke
from ..synth.weather_ext import apply_lowlight, apply_rain

# 학습 해상도(4:3, patch-14 배수: 518=37x14, 392=28x14).
TRAIN_W, TRAIN_H = 518, 392
SEVEN_SCENES_INTRINSICS = (525.0, 525.0, 320.0, 240.0)
NRGBD_INTRINSICS = (554.2562584220408, 554.2562584220408, 320.0, 240.0)

# 평가 변형 비율(fog x5, smoke x2, rain x3, lowlight x3)과 같은 노출.
WEATHER_MIX = {"fog": 5 / 13, "smoke": 2 / 13, "rain": 3 / 13, "lowlight": 3 / 13}
_KINDS = tuple(WEATHER_MIX)
_PROBS = tuple(WEATHER_MIX[k] for k in _KINDS)

# severity s~U(0,1) 선형 보간 구간. 끝점은 ext_v2.yaml eval 프리셋과
# 일치시킨다(fog는 light..extreme 전 범위).
SEVERITY_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "fog": {"beta": (0.04, 0.64)},
    "smoke": {"beta": (0.03, 0.12), "sigma": (0.05, 0.20)},
    "rain": {"beta": (0.015, 0.09), "density": (150.0, 900.0), "length": (12.0, 26.0)},
    "lowlight": {"gamma": (1.6, 3.0), "gain": (0.55, 0.16), "sigma": (0.004, 0.015)},
}


def lerp(a: float, b: float, s: float) -> float:
    return float(a + (b - a) * s)


def sample_weather_params(kind: str, severity: float) -> dict[str, float]:
    """kind + severity -> 합성 파라미터. gain은 severity와 반대 방향 보간."""
    if kind not in SEVERITY_RANGES:
        raise ValueError(f"unknown weather kind: {kind}")
    out = {}
    for name, (lo, hi) in SEVERITY_RANGES[kind].items():
        out[name] = lerp(lo, hi, severity)
    return out


def pick_kind(rng: np.random.Generator) -> str:
    return str(rng.choice(_KINDS, p=_PROBS))


def noise_volume(h: int, w: int, t: int, rng: np.random.Generator, cell: int = 48, tcell: int = 4) -> np.ndarray:
    """시간 상관 [0,1] 밸류 노이즈 (t, h, w).

    스모크의 비균질 beta장용. coarse 격자(cell, tcell 간격 키프레임)에서
    랜덤 값을 뽑아 삼선형(공간 이선형 x 시간 선형) 보간한다. 시간축
    상관이 있어 윈도우 내 연기가 뭉게 이동한다(eval 합성의 fBm 정신,
    학습용 경량 근사).
    """
    gh, gw = math.ceil(h / cell) + 2, math.ceil(w / cell) + 2
    n_keys = math.ceil(t / tcell) + 2
    grid = rng.random((n_keys, gh, gw), dtype=np.float32)

    # 공간 업샘플 좌표(전 프레임 공유). +0.5로 셀 중심을 잡는다.
    ys = np.arange(h, dtype=np.float32) / cell + 0.5
    xs = np.arange(w, dtype=np.float32) / cell + 0.5
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, gh - 2)
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, gw - 2)
    y1 = y0 + 1
    x1 = x0 + 1
    wy = (ys - y0).astype(np.float32)[:, None]
    wx = (xs - x0).astype(np.float32)[None, :]

    out = np.empty((t, h, w), dtype=np.float32)
    for i in range(t):
        ft = i / tcell
        k0 = min(int(ft), n_keys - 2)
        wtf = ft - k0
        gk = grid[k0] * (1.0 - wtf) + grid[k0 + 1] * wtf  # (gh, gw) 시간 보간
        top = gk[y0][:, x0] * (1 - wx) + gk[y0][:, x1] * wx
        bot = gk[y1][:, x0] * (1 - wx) + gk[y1][:, x1] * wx
        out[i] = top * (1 - wy) + bot * wy
    return out


@dataclass
class TrainSeq:
    key: str                      # "7scenes/office/seq-01" | "nrgbd/kitchen"
    dataset: str                  # "seven_scenes" | "neural_rgbd"
    seq_dir: Path
    frame_names: list[str]        # 이미지 stem(확장자 제외) 정렬 목록
    poses: np.ndarray | None = None        # (N,4,4) c2w(NRGBD: 장면 단위 1회 로드)
    pose_valid: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


def _train_split_seqs(scene_dir: Path) -> list[str] | None:
    """TrainSplit.txt -> ["seq-01", ...]. 파일이 없으면 None(전체 seq 사용)."""
    f = scene_dir / "TrainSplit.txt"
    if not f.is_file():
        return None
    seqs = []
    for m in re.finditer(r"seq(-|_)(\d+)", f.read_text(encoding="utf-8", errors="ignore")):
        seqs.append(f"seq-{int(m.group(2)):02d}")
    return sorted(set(seqs))


def _seven_scene_seqs(root: Path, scenes: list[str]) -> list[TrainSeq]:
    out = []
    for scene in scenes:
        scene_dir = root / scene
        if not scene_dir.is_dir():
            continue
        wanted = _train_split_seqs(scene_dir)
        for seq_dir in sorted(scene_dir.glob("seq-*")):
            if not seq_dir.is_dir():
                continue
            if wanted is not None and seq_dir.name not in wanted:
                continue
            frames = [
                p.stem
                for p in sorted(seq_dir.glob("frame-*.color.png"))
            ]
            if frames:
                out.append(
                    TrainSeq(
                        key=f"7scenes/{scene}/{seq_dir.name}",
                        dataset="seven_scenes",
                        seq_dir=seq_dir,
                        frame_names=frames,
                    )
                )
    return out


def _frame_number(name: str) -> int:
    digits = re.sub(r"\D", "", name)
    return int(digits) if digits else -1


def _nrgbd_seqs(root: Path, scenes: list[str]) -> list[TrainSeq]:
    out = []
    for scene in scenes:
        seq_dir = root / scene
        img_dir = seq_dir / "images"
        if not img_dir.is_dir():
            continue
        # zero-padding 여부와 무관하게 숫자 오름차순(sequences.py 규약).
        frames = sorted((p.stem for p in img_dir.glob("img*.png")), key=_frame_number)
        if not frames:
            continue
        poses, valid = _load_nrgbd_poses(seq_dir / "poses.txt", _frame_number(frames[-1]) + 1)
        # pose가 nan인 프레임은 학습 창에서 제외(모델 입력으로 쓸 수 없다).
        frames = [n for n in frames if valid[_frame_number(n)]]
        if not frames:
            continue
        out.append(
            TrainSeq(
                key=f"nrgbd/{scene}",
                dataset="neural_rgbd",
                seq_dir=seq_dir,
                frame_names=frames,
                poses=poses,
                pose_valid=valid,
            )
        )
    return out


def load_train_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_seqs(cfg: dict) -> list[TrainSeq]:
    seqs = _seven_scene_seqs(Path(cfg["seven_scenes_root"]), list(cfg["seven_scenes"]))
    seqs += _nrgbd_seqs(Path(cfg["neural_rgbd_root"]), list(cfg["neural_rgbd"]))
    if not seqs:
        raise FileNotFoundError(
            f"no training sequences under {cfg['seven_scenes_root']} / {cfg['neural_rgbd_root']}"
        )
    return seqs


class WeatherTrainDataset(torch.utils.data.Dataset):
    """__getitem__ -> (views_student, views_teacher, meta).

    views_teacher는 clean 원본, views_student는 clean 또는 열화본.
    두 리스트는 img만 다를 수 있고 나머지 키는 값 공유(같은 텐서 객체)라
    메모리 중복이 없다. collate는 trainer 쪽 list-통과 collate를 쓴다
    (batch_size=1 + 이미 [1,...] 형태로 내놓는다).
    """

    def __init__(
        self,
        cfg: dict,
        num_views: int = 8,
        stride: int = 4,
        clean_ratio: float = 0.5,
        seed: int = 0,
        limit_windows: int = 0,
        max_frames_per_seq: int = 0,
        window_hop: int = 0,
    ):
        self.cfg = cfg
        self.num_views = num_views
        self.stride = stride
        self.clean_ratio = float(clean_ratio)
        self.seed = int(seed)
        self.epoch = 0
        self.seqs: list[TrainSeq] = build_seqs(cfg)
        if max_frames_per_seq and max_frames_per_seq > 0:
            for s in self.seqs:
                s.frame_names = s.frame_names[:max_frames_per_seq]
        span = (num_views - 1) * stride
        self.hop = window_hop if window_hop and window_hop > 0 else max(1, span // 2)
        self.index: list[tuple[int, int]] = []
        for si, s in enumerate(self.seqs):
            n = len(s.frame_names)
            if n < span + 1:
                continue
            for start in range(0, n - span, self.hop):
                self.index.append((si, start))
        if not self.index:
            raise ValueError(
                f"no windows: num_views={num_views} stride={stride} over "
                f"{sum(len(s.frame_names) for s in self.seqs)} frames"
            )
        if limit_windows and limit_windows > 0 and limit_windows < len(self.index):
            # 앞쪽 몇 개 seq에 몰리지 않게 등간격 서브샘플(스모크/probe용).
            step = max(1, len(self.index) // limit_windows)
            self.index = self.index[::step][:limit_windows]

    def __len__(self) -> int:
        return len(self.index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, idx: int) -> np.random.Generator:
        seq = (self.seed * 1_000_003 + self.epoch * 9_176 + idx) % (2**63 - 1)
        return np.random.default_rng(seq)

    # --- 프레임 로딩 ----------------------------------------------

    def _load_7s(self, seq: TrainSeq, name: str):
        img = cv2.imread(str(seq.seq_dir / f"{name}.png"), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(str(seq.seq_dir / f"{name}.png"))
        stem = name  # "frame-XXXXXX.color"
        depth_path = seq.seq_dir / f"{stem.replace('.color', '.depth.proj')}.png"
        if not depth_path.is_file():
            depth_path = seq.seq_dir / f"{stem.replace('.color', '.depth')}.png"
        depth = _read_depth_png(depth_path)
        pose_path = seq.seq_dir / f"{stem.replace('.color', '.pose')}.txt"
        pose = np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
        fx, fy, cx, cy = SEVEN_SCENES_INTRINSICS
        return img, depth, pose, (fx, fy, cx, cy)

    def _load_nrgbd(self, seq: TrainSeq, name: str):
        img = cv2.imread(str(seq.seq_dir / "images" / f"{name}.png"), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(str(seq.seq_dir / "images" / f"{name}.png"))
        depth = _read_depth_png(seq.seq_dir / "depth" / f"depth{name.removeprefix('img')}.png")
        pose = seq.poses[_frame_number(name)]
        fx, fy, cx, cy = NRGBD_INTRINSICS
        return img, depth, pose, (fx, fy, cx, cy)

    @staticmethod
    def _to_train_res(img: np.ndarray, depth: np.ndarray, k4: tuple[float, float, float, float]):
        h, w = img.shape[:2]
        fx, fy, cx, cy = k4
        img_r = cv2.resize(img, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_AREA)
        dep_r = cv2.resize(depth, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_NEAREST)
        k = np.array(
            [
                [fx * TRAIN_W / w, 0.0, cx * TRAIN_W / w],
                [0.0, fy * TRAIN_H / h, cy * TRAIN_H / h],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return img_r, dep_r, k

    def __getitem__(self, idx: int):
        si, start = self.index[idx]
        seq = self.seqs[si]
        rng = self._rng(idx)
        degraded = float(rng.random()) >= self.clean_ratio
        kind = pick_kind(rng) if degraded else "clean"
        severity = float(rng.random()) if degraded else 0.0
        params = sample_weather_params(kind, severity) if degraded else {}
        smoke_vol = None
        if kind == "smoke":
            smoke_vol = noise_volume(TRAIN_H, TRAIN_W, self.num_views, rng)

        views_teacher: list[dict] = []
        views_student: list[dict] = []
        for v in range(self.num_views):
            fi = start + v * self.stride
            name = seq.frame_names[fi]
            if seq.dataset == "seven_scenes":
                img, depth, pose, k4 = self._load_7s(seq, name)
            else:
                img, depth, pose, k4 = self._load_nrgbd(seq, name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img, depth, k = self._to_train_res(img, depth, k4)

            img_deg = img
            if degraded:
                frng = np.random.default_rng((self.seed * 7919 + idx * 131 + v) % (2**63 - 1))
                if kind == "fog":
                    img_deg = apply_fog(img, depth, params["beta"])
                elif kind == "smoke":
                    img_deg = apply_smoke(img, depth, params["beta"], params["sigma"], smoke_vol[v])
                elif kind == "rain":
                    img_deg = apply_rain(
                        img, depth, params["beta"], int(round(params["density"])), int(round(params["length"])), frng
                    )
                elif kind == "lowlight":
                    img_deg = apply_lowlight(
                        img, params["gamma"], params["gain"], params["sigma"], frng
                    )

            depth_t = torch.from_numpy(np.ascontiguousarray(depth))
            pose_t = torch.from_numpy(np.ascontiguousarray(pose.astype(np.float32)))
            k_t = torch.from_numpy(k)
            common = dict(
                depthmap=depth_t[None],
                # DistillLoss.compute_loss가 gts 측 valid_mask를 요구한다.
                # forward/inference는 뷰 dict의 이 값을 ress에 그대로 통과
                # 시킨다(14 survey3 streamvggt.py 643행). [1,H,W] bool로
                # [B,H,W] 브로드캐스트. _read_depth_png는 홀/초과깊이를
                # 0으로 정규화하므로 depth>0이 유효 픽셀 마스크다.
                valid_mask=depth_t[None] > 0,
                camera_pose=pose_t[None],
                camera_intrinsics=k_t[None],
                true_shape=torch.tensor([[TRAIN_H // 14, TRAIN_W // 14]]),
                dataset=seq.dataset,
                label=f"{seq.key}/{name}",
                instance=f"{idx}_{v}",
                idx=v,
                is_metric=True,
                is_video=True,
                quantile=torch.tensor([0.98], dtype=torch.float32),
                img_mask=torch.tensor([True]),
                ray_mask=torch.tensor([False]),
                camera_only=False,
                depth_only=False,
                single_view=False,
                update=torch.tensor([True]),
                reset=torch.tensor([False]),
            )
            views_teacher.append(
                dict(img=torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).contiguous()[None], **common)
            )
            views_student.append(
                dict(img=torch.from_numpy(np.ascontiguousarray(img_deg)).permute(2, 0, 1).contiguous()[None], **common)
            )

        meta = dict(
            idx=idx,
            seq=seq.key,
            mode="clean" if not degraded else "degraded",
            kind=kind,
            severity=severity,
            params=params,
            start_frame=start,
            num_views=self.num_views,
            stride=self.stride,
        )
        return views_student, views_teacher, meta
