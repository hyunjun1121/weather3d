"""GT(depth, pose, intrinsics) 로딩.

변환 규칙은 StreamVGGT의 데이터 로더(src/eval/mv_recon/data.py,
SevenScenes/NRGBD)와 동일하게 유지한다.

- 7-Scenes: depth.proj.png 16bit mm, 65535->무효, 10m 초과/1e-3 미만->0.
  K = (525, 525, 320, 240). pose는 c2w.
- Neural-RGBD: depth png 16bit mm /1000, 동일 범위 처리.
  K = (554.2562584220408, 554.2562584220408, 320, 240). poses.txt는 프레임별
  4x4가 순서대로 쌓인 형식이며 GL->CV 변환(pose[:, 1:3] *= -1)을 적용한다.
- TartanAir V2: depth_lcam_front/NNNNNN_lcam_front_depth.png는
  "H x W float32를 4채널 8비트 PNG로 무손실 패킹"한 공식 규격이다
  (tartanair.org/modalities.html; 2026-08-26 11b 실측 확정 - V1의 16bit
  grayscale과 다름). 디코딩은 공식 코드와 동일하게 cv2.imread
  (IMREAD_UNCHANGED) 결과를 .view("<f4")로 재해석하고, 하늘 등 원거리
  픽셀은 '매우 큰 값'이므로 상한 마스킹이 무효 처리를 겸한다.
  pose_lcam_front.txt는 프레임당
  "tx ty tz qx qy qz qw"(NED, x전방/y우측/z하방)이며 카메라 c2w로 해석해
  CV 좌표계로 축을 바꾼다. V2 카메라는 640x640 핀홀, f=320, 주점
  (320, 320)(tartanair.org modalities 규격).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .sequences import Sequence

MAX_DEPTH_M = 10.0
SEVEN_SCENES_INTRINSICS = (525.0, 525.0, 320.0, 240.0)
NRGBD_INTRINSICS = (554.2562584220408, 554.2562584220408, 320.0, 240.0)
TARTANAIR_INTRINSICS = (320.0, 320.0, 320.0, 320.0)  # V2: 640x640, f=320, 주점 (320,320)
TARTANAIR_MAX_DEPTH_M = 200.0  # V2 환경은 실내보다 넓다. 하늘 = 거대 float -> 상한 마스킹
TARTANAIR_DEPTH_DIV = 1000.0   # (레거시) 16bit grayscale png 배포본용
TARTANAIR_POSE_SCALE = 1.0     # pose 평행이동 원소 -> m 환산(캘리브레이션으로 검증)

# NED(x전방,y우측,z하방) -> CV(x우측,y하방,z전방) / GL(x우측,y상방,z후방)
_NED_TO_CV = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
_NED_TO_GL = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]], dtype=np.float64)


@dataclass
class GTSequence:
    depths: np.ndarray        # (N, H, W) float32 m, 0=invalid
    poses_c2w: np.ndarray     # (N, 4, 4) float32
    intrinsics: np.ndarray    # (N, 3, 3) float32
    frame_indices: list[int]
    pose_valid: np.ndarray    # (N,) bool; False면 해당 프레임 pose 무효(nan 등)

    @property
    def num_frames(self) -> int:
        return len(self.frame_indices)


def _intrinsics_matrix(fx: float, fy: float, cx: float, cy: float, n: int) -> np.ndarray:
    k = np.tile(
        np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32),
        (n, 1, 1),
    )
    return k


def _read_depth_png(path: Path, div: float = 1000.0) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"cannot read depth: {path}")
    if raw.ndim == 3:
        raw = raw[..., 0]
    depth = raw.astype(np.float32)
    depth[raw == 65535] = 0.0
    depth = depth / div
    depth[depth > MAX_DEPTH_M] = 0.0
    depth[depth < 1e-3] = 0.0
    return depth


def _read_tartanair_depth_png(path: Path) -> np.ndarray:
    """TartanAir V2 depth png -> (H, W) float32 m, 0=무효.

    V2 배포 규격(tartanair.org/modalities.html)은 H x W float32를
    4채널 8비트 PNG로 무손실 패킹한 것이다. 공식 디코더와 동일하게
    cv2.imread(IMREAD_UNCHANGED) 결과(BGRA 메모리)를 .view("<f4")로
    재해석한다(11b 실측: 이 파일들은 8bit RGBA로, 단일 채널을 읽으면
    전부 무효가 된다). 하늘 등 원거리 픽셀은 '매우 큰 값'으로 저장되므로
    상한 마스킹이 무효 처리를 겸한다. 2차원(16bit grayscale) 배포본은
    기존 div 경로를 유지한다.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"cannot read depth: {path}")
    if raw.ndim == 2:
        ref = raw
        depth = raw.astype(np.float32)
        depth[ref == 65535] = 0.0
        depth /= TARTANAIR_DEPTH_DIV
    elif raw.ndim == 3 and raw.shape[-1] == 4 and raw.dtype == np.uint8:
        depth = raw.view("<f4")[..., 0].astype(np.float32)
    else:
        raise ValueError(
            f"unsupported tartanair depth png layout: {path} shape={raw.shape} dtype={raw.dtype}"
        )
    depth[~np.isfinite(depth)] = 0.0
    depth[depth > TARTANAIR_MAX_DEPTH_M] = 0.0
    depth[depth < 1e-3] = 0.0
    return depth


def _quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    """정규화된 quaternion (qx,qy,qz,qw) -> 3x3 회전(카메라->세계 축 변환)."""
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_tartanair_poses(
    path: Path,
    num_frames: int,
    conv: str = "cv",
    scale: float = TARTANAIR_POSE_SCALE,
    invert: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """pose_lcam_front.txt(N x 7, NED c2w 가정) -> (N,4,4) c2w CV/GL 좌표계.

    conv/invert는 calibrate_tartanair.py의 그리드 탐색과 동일한 축을
    제공한다(기본값이 실측으로 확정된 조합).
    """
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != 7:
        raise ValueError(f"pose file must have 7 columns: {path} (got {data.shape[1]})")
    if num_frames > len(data):
        raise ValueError(f"pose file has {len(data)} entries but {num_frames} frames: {path}")
    m = _NED_TO_CV if conv == "cv" else _NED_TO_GL
    poses = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
    valid = np.ones(num_frames, dtype=bool)
    for i, row in enumerate(data[:num_frames]):
        t_cv = m @ (row[:3] * scale)
        r_cv = m @ _quat_xyzw_to_rot(row[3:7]) @ m.T
        t44 = np.eye(4)
        t44[:3, :3] = r_cv
        t44[:3, 3] = t_cv
        if invert:
            t44 = np.linalg.inv(t44)
        if not np.isfinite(t44).all():
            valid[i] = False
            continue
        poses[i] = t44.astype(np.float32)
    return poses, valid


def _load_nrgbd_poses(path: Path, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """poses.txt: 프레임당 4x4 행렬이 순서대로 나열. 'nan' 행렬은 무효.

    두 배포 형식을 지원한다(2026-08-24 TUM 원본 zip 대응):
    - 행마다 4개 숫자, 4줄이 한 행렬(Spann3R 전처리 형식)
    - 한 줄에 16개 숫자(TUM 원본 형식)
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    widths = {len(ln.split()) for ln in lines}
    if widths == {16}:
        blocks = [ln.split() for ln in lines]
    elif widths == {4} and len(lines) % 4 == 0:
        blocks = [" ".join(lines[i * 4 : (i + 1) * 4]).split() for i in range(len(lines) // 4)]
    else:
        raise ValueError(
            f"poses.txt is not a sequence of 4x4 matrices: {path} (widths={sorted(widths)})"
        )

    poses = np.tile(np.eye(4, dtype=np.float32), (len(blocks), 1, 1))
    valid = np.ones(len(poses), dtype=bool)
    for i, tokens in enumerate(blocks):
        if any("nan" in tok.lower() for tok in tokens):
            valid[i] = False
            continue
        poses[i] = np.array([float(x) for x in tokens], dtype=np.float32).reshape(4, 4)
        # GL convention -> CV convention (StreamVGGT NRGBD 로더와 동일)
        poses[i][:, 1:3] *= -1.0
    if num_frames > len(poses):
        raise ValueError(
            f"poses.txt has {len(poses)} entries but {num_frames} frames found: {path}"
        )
    return poses[:num_frames], valid[:num_frames]


def load_gt(seq: Sequence) -> GTSequence:
    """Sequence 프레임 목록에 대응하는 GT를 로드한다."""
    if seq.dataset == "seven_scenes":
        fx, fy, cx, cy = SEVEN_SCENES_INTRINSICS
        depths, poses, indices, valid = [], [], [], []
        for fr in seq.frames:
            stem = fr.image_path.stem  # "frame-XXXXXX.color"
            # StreamVGGT 로더 규격은 .depth.proj.png(전처리 배포본). 공식
            # Microsoft 배포본은 .depth.png라 둘 다 수용한다(내용 형식은 동일).
            depth_path = fr.image_path.with_name(stem.replace(".color", ".depth.proj") + ".png")
            if not depth_path.is_file():
                depth_path = fr.image_path.with_name(stem.replace(".color", ".depth") + ".png")
            pose_path = fr.image_path.with_name(stem.replace(".color", ".pose.txt"))
            if not depth_path.is_file():
                raise FileNotFoundError(f"missing GT depth: {depth_path}")
            if not pose_path.is_file():
                raise FileNotFoundError(f"missing GT pose: {pose_path}")
            depths.append(_read_depth_png(depth_path))
            pose = np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
            poses.append(pose)
            indices.append(fr.index)
            valid.append(bool(np.isfinite(pose).all()))
    elif seq.dataset == "neural_rgbd":
        fx, fy, cx, cy = NRGBD_INTRINSICS
        seq_dir = seq.dir
        all_poses, all_valid = _load_nrgbd_poses(seq_dir / "poses.txt", max(fr.index for fr in seq.frames) + 1)
        depths, poses, indices, valid = [], [], [], []
        for fr in seq.frames:
            # 이미지 stem "img<번호>"에서 "depth<번호>.png"를 파생한다.
            # 배포본에 따라 zero-padding 여부가 다르므로 원본 이름을 그대로 따른다.
            depth_path = seq_dir / "depth" / f"depth{fr.name.removeprefix('img')}.png"
            if not depth_path.is_file():
                raise FileNotFoundError(f"missing GT depth: {depth_path}")
            depths.append(_read_depth_png(depth_path))
            poses.append(all_poses[fr.index])
            indices.append(fr.index)
            valid.append(bool(all_valid[fr.index]))
    elif seq.dataset == "tartanair2":
        fx, fy, cx, cy = TARTANAIR_INTRINSICS
        traj_dir = seq.dir
        max_idx = max(fr.index for fr in seq.frames) + 1
        all_poses, all_valid = load_tartanair_poses(traj_dir / "pose_lcam_front.txt", max_idx)
        depths, poses, indices, valid = [], [], [], []
        for fr in seq.frames:
            # "000000_lcam_front" -> "000000_lcam_front_depth.png"
            depth_path = traj_dir / "depth_lcam_front" / f"{fr.name}_depth.png"
            if not depth_path.is_file():
                raise FileNotFoundError(f"missing GT depth: {depth_path}")
            depths.append(_read_tartanair_depth_png(depth_path))
            poses.append(all_poses[fr.index])
            indices.append(fr.index)
            valid.append(bool(all_valid[fr.index]))
    else:
        raise ValueError(f"unsupported dataset: {seq.dataset}")

    return GTSequence(
        depths=np.stack(depths, 0),
        poses_c2w=np.stack(poses, 0).astype(np.float32),
        intrinsics=_intrinsics_matrix(fx, fy, cx, cy, len(depths)),
        frame_indices=indices,
        pose_valid=np.array(valid, dtype=bool),
    )


def unproject_depth(depth: np.ndarray, k: np.ndarray) -> np.ndarray:
    """(H, W) depth + (3, 3) K -> 카메라 좌표계 (H, W, 3) 포인트맵."""
    h, w = depth.shape
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    u = np.arange(w, dtype=np.float32)[None, :].repeat(h, 0)
    v = np.arange(h, dtype=np.float32)[:, None].repeat(w, 1)
    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    return np.stack([x, y, depth], axis=-1)


def transform_points(pts: np.ndarray, pose44: np.ndarray) -> np.ndarray:
    """(…, 3) 포인트에 4x4 변환 적용."""
    return pts @ pose44[:3, :3].T + pose44[:3, 3]
