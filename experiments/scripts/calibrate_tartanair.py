"""TartanAir V2 pose/depth 규격 캘리브레이션(깊이 교차 일관성 검증).

사용(experiments/ 기준):
    python -B scripts/calibrate_tartanair.py --root ../data/tartanair2 \
        --env GreatMarsh --difficulty easy --traj P000

배경: 잘못된 depth 디코딩이나 pose 축은 점군 GT를 조용히 망가뜨린다.
여러 (depth 디코딩, 단위, pose_scale, 좌표계, c2w/w2c, 쿼터니언 순서)
조합 중 연속 프레임 '깊이 교차 일관성'(i점군을 j로 변환·투영해 depth_j와
비교) 오차가 가장 작은 조합을 데이터에서 직접 고른다. 픽셀 위치 오차가
아닌 이유: 카메라가 움직이면 같은 점이 다른 픽셀에 맞는 것이 정상이다.

2026-08-26 규격 확정 경위:
- script 11: 16bit grayscale 가정(depth_div 그리드)에서 6대상 전 조합
  median_relerr=inf.
- script 11b 통계: depth png가 8bit RGBA (640x640), cv2 Blue 채널 전체
  0 -> 단일 채널 16bit 읽기가 무효였다. tartanair.org/modalities.html
  확인: V2 depth는 "H x W float32를 4채널 8비트 PNG로 무손실 패킹"이며
  공식 디코더는 cv2.imread(IMREAD_UNCHANGED).view("<f4")다. V2 카메라는
  640x640, f=320, 주점 (320,320)이라 intrinsics도 갱신됐다.
- 이 버전: 공식 float 디코딩(f4_cv)을 기본값으로 하되, 바이트 순서
  가설(f4_pil)과 16bit 정수 가설(R+256G / R*256+G x mm/cm/dm 단위)을
  함께 그리드에 넣어 데이터가 스스로 판별하게 한다. pose 규격(NED
  quaternion, 단일 traj 내 일관성)은 11b에서 이미 정상으로 확인됐다.

절대 스케일은 이 검증으로 가려지지 않는다(일관성은 depth/pose 단위
비율에만 민감). 모든 평가 지표(sim3, scale&shift, ICP 정합)는 스케일
불변이므로 이 설계로 충분하다.

PASS 조건: 최적 조합이 gt.py 기본값(decode=f4_cv, unit=1.0 m,
pose_scale=1.0, conv=cv, invert=False, qorder=xyzw)과 일치하고 상대
깊이 오차 중앙값 < 0.02이며 2위와 3배 이상 격차.

exit code: 0 PASS / 1 FAIL(오차 임계 초과) / 2 MISMATCH·AMBIGUOUS /
4 [DATA] depth·pose 원시 데이터 문제(모든 디코딩에서 유효 픽셀 0%).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from weather3d.gt import (  # noqa: E402
    TARTANAIR_INTRINSICS,
    TARTANAIR_MAX_DEPTH_M,
    TARTANAIR_POSE_SCALE,
    _NED_TO_CV,
    _NED_TO_GL,
    _quat_xyzw_to_rot,
)

# (이름, 설명) - decode 함수는 원시 float 값(단위 미적용)을 반환.
DECODERS = ("f4_cv", "f4_pil", "rg_le", "rg_be")
# 정수 계열 가설의 단위 후보(mm/cm/dm). float 계열은 문서상 m.
INT_UNITS = (0.001, 0.01, 0.1)
FLOAT_UNITS = (1.0,)
POSE_SCALES = (1.0, 0.01, 0.001, 100.0)
QORDERS = ("xyzw", "wxyz")
DEFAULTS = {
    "decode": "f4_cv",
    "unit": 1.0,
    "pose_scale": TARTANAIR_POSE_SCALE,
    "conv": "cv",
    "invert": False,
    "qorder": "xyzw",
}


def load_rgba(path: Path) -> np.ndarray:
    """depth png를 4채널 원시 uint8로 읽는다(cv2 BGRA 메모리 순서)."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"cannot read depth: {path}")
    if raw.ndim != 3 or raw.shape[-1] != 4:
        raise ValueError(f"expected 4-channel png, got shape={raw.shape}: {path}")
    return raw


def decode_depth(raw: np.ndarray, name: str) -> np.ndarray:
    """(H, W, 4) cv2 BGRA uint8 -> (H, W) float 원시 값(단위 미적용).

    - f4_cv:  공식 디코더. cv2 메모리(B,G,R,A)를 little-endian float32
              비트 패턴으로 재해석.
    - f4_pil: 파일 채널 순서(R,G,B,A)가 float 바이트 순서라는 가설(
              cv2의 R<->B 재배열이 없는 PIL 호환 대안).
    - rg_le / rg_be: 16bit 정수 가설. R + 256*G / R*256 + G.
    """
    if name == "f4_cv":
        return raw.view("<f4")[..., 0].astype(np.float64)
    if name == "f4_pil":
        rgba = np.ascontiguousarray(raw[..., [2, 1, 0, 3]])
        return rgba.view("<f4")[..., 0].astype(np.float64)
    r = raw[..., 2].astype(np.float64)  # cv2 BGRA의 R
    g = raw[..., 1].astype(np.float64)
    if name == "rg_le":
        return r + 256.0 * g
    if name == "rg_be":
        return r * 256.0 + g
    raise ValueError(f"unknown decoder: {name}")


def units_for(name: str) -> tuple[float, ...]:
    return FLOAT_UNITS if name.startswith("f4") else INT_UNITS


def scale_depth(raw: np.ndarray, unit: float) -> np.ndarray:
    d = raw * unit
    d[~np.isfinite(d)] = 0.0
    d[(d > TARTANAIR_MAX_DEPTH_M) | (d < 1e-3)] = 0.0
    return d


def _quat_as_xyzw(q: np.ndarray, order: str) -> np.ndarray:
    return q if order == "xyzw" else q[[3, 0, 1, 2]]


def poses_for(rows: np.ndarray, conv: str, scale: float, invert: bool, qorder: str) -> np.ndarray:
    m = _NED_TO_CV if conv == "cv" else _NED_TO_GL
    out = []
    for row in rows:
        t44 = np.eye(4)
        t44[:3, :3] = m @ _quat_xyzw_to_rot(_quat_as_xyzw(row[3:7], qorder)) @ m.T
        t44[:3, 3] = m @ (row[:3] * scale)
        if invert:
            t44 = np.linalg.inv(t44)
        out.append(t44)
    return np.stack(out)


def depth_consistency(depth_i, depth_j, t_i, t_j, k, step: int = 2) -> float:
    """i프레임 점군을 j 카메라로 변환·투영한 위치에서 depth_j와 z가
    일치하는지(상대 오차 중앙값) 검사. 카메라가 움직이면 같은 점이 다른
    픽셀에 맞는 것이 정상이므로 픽셀 위치 오차가 아니라 깊이 교차
    일관성으로 판별한다. 정답 조합은 양자화 수준(<<0.02)으로 떨어진다."""
    h, w = depth_i.shape
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    v, u = np.mgrid[0:h:step, 0:w:step]
    d = depth_i[v, u]
    valid = d > 0
    x = (u - cx) / fx * d
    y = (v - cy) / fy * d
    pts = np.stack([x[valid], y[valid], d[valid]], 1)
    rel = np.linalg.inv(t_j) @ t_i  # cam_i -> world(T_i) -> cam_j(T_j^{-1})
    pts_w = pts @ rel[:3, :3].T + rel[:3, 3]
    zj_all = pts_w[:, 2]
    ok = zj_all > 1e-3
    if ok.sum() < 100:
        return float("inf")
    zj = zj_all[ok]
    uj = fx * pts_w[ok, 0] / zj + cx
    vj = fy * pts_w[ok, 1] / zj + cy
    inb = (uj >= 0) & (uj < w - 1) & (vj >= 0) & (vj < h - 1)
    if inb.sum() < 100:
        return float("inf")
    ujr = uj[inb].round().astype(int)
    vjr = vj[inb].round().astype(int)
    dj = depth_j[vjr, ujr]
    good = dj > 0
    if good.sum() < 100:
        return float("inf")
    rel_err = np.abs(zj[inb][good] - dj[good]) / dj[good]
    return float(np.median(rel_err))


def print_data_stats(traj_dir: Path, ids: list[int], raws: list[np.ndarray], rows: np.ndarray) -> bool:
    """디코딩별 depth 통계 + pose 통계 출력. 모든 디코딩에서 유효 픽셀
    0%면 False(=[DATA])."""
    depth_dir = traj_dir / "depth_lcam_front"
    sizes = [(depth_dir / f"{i:06d}_lcam_front_depth.png").stat().st_size for i in ids[:3]]
    print(f"  [PNG] shape={raws[0].shape} file_sizes={sizes}")
    for name in DECODERS:
        vals = [decode_depth(r, name) for r in raws[:3]]
        v0 = vals[0]
        fin = np.isfinite(v0)
        line = f"  [DEPTH:{name}] "
        if fin.any():
            line += f"min/med/max(원시)={np.nanmin(v0[fin]):.4g}/{np.nanmedian(v0[fin]):.4g}/{np.nanmax(v0[fin]):.4g} "
        else:
            line += "유한값 없음 "
        fracs = [float((scale_depth(v, u) > 0).mean()) for u in units_for(name) for v in vals]
        line += f"유효%({len(units_for(name))}단위 평균)={100 * float(np.mean(fracs)):.1f}"
        print(line)
    any_valid = any(
        float((scale_depth(decode_depth(r, name), u) > 0).mean()) > 0
        for name in DECODERS
        for u in units_for(name)
        for r in raws
    )
    img = cv2.imread(str(traj_dir / "image_lcam_front" / f"{ids[0]:06d}_lcam_front.png"), cv2.IMREAD_UNCHANGED)
    print(f"  [RGB] shape={None if img is None else img.shape}")
    sub = rows[ids]
    q = np.linalg.norm(sub[:, 3:7], axis=1)
    print(f"  [POSE] nan행={int(np.isnan(sub).any(axis=1).sum())}/{len(sub)} "
          f"quat_norm min/max={q.min():.4f}/{q.max():.4f} "
          f"t축별 min={sub[:, :3].min(axis=0).round(3).tolist()} max={sub[:, :3].max(axis=0).round(3).tolist()}")
    print(f"  [POSE] 첫 행: {np.array2string(sub[0], precision=4, suppress_small=False)}")
    try:
        from PIL import Image

        with Image.open(depth_dir / f"{ids[0]:06d}_lcam_front_depth.png") as im:
            arr = np.array(im)
        print(f"  [DEPTH] PIL 교차검증: mode={im.mode} dtype={arr.dtype} "
              f"shape={arr.shape} min/max={arr.min()}/{arr.max()}")
    except Exception as e:  # noqa: BLE001 - 진단 보조라 실패해도 진행
        print(f"  [DEPTH] PIL 교차검증 실패: {e}")
    return any_valid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="../data/tartanair2")
    ap.add_argument("--env", default="GreatMarsh")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--traj", default="P000")
    ap.add_argument("--num-frames", type=int, default=10)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--err-thresh", type=float, default=0.02,
                    help="PASS 판정 상대 깊이 오차 중앙값 기준")
    args = ap.parse_args()

    traj_dir = Path(args.root).resolve() / args.env / f"Data_{args.difficulty}" / "extracted" / args.env / f"Data_{args.difficulty}" / args.traj
    pose_file = traj_dir / "pose_lcam_front.txt"
    rows = np.loadtxt(pose_file, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    ids = list(range(0, min(len(rows), args.num_frames * args.stride), args.stride))
    raws = [load_rgba(traj_dir / "depth_lcam_front" / f"{i:06d}_lcam_front_depth.png") for i in ids]
    fx, fy, cx, cy = TARTANAIR_INTRINSICS
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    print(f"== {args.env}/Data_{args.difficulty}/{args.traj}: frames={len(ids)} ==")
    if not print_data_stats(traj_dir, ids, raws, rows):
        print("[DATA] 모든 depth 디코딩에서 유효 픽셀 0% - png 채널 구조 재확인 필요")
        return 4

    # NaN/zero-quaternion 행이 있으면 해당 프레임 쌍만 제외(전체 중단 없이 판별)
    bad_rows = ~np.isfinite(rows[ids]).all(axis=1) | (np.linalg.norm(rows[ids][:, 3:7], axis=1) < 1e-6)
    pair_ok = ~(bad_rows[:-1] | bad_rows[1:])
    if pair_ok.sum() == 0:
        print("[DATA] pose의 유한·정규화 가능한 프레임 쌍이 없음")
        return 4

    results = []
    for name in DECODERS:
        decoded = [decode_depth(r, name) for r in raws]
        for unit in units_for(name):
            dd = [scale_depth(v, unit) for v in decoded]
            for scale, qorder, conv, invert in itertools.product(
                POSE_SCALES, QORDERS, ("cv", "gl"), (False, True)
            ):
                t44 = poses_for(rows[ids], conv, scale, invert, qorder)
                errs = [
                    depth_consistency(dd[i], dd[i + 1], t44[i], t44[i + 1], k)
                    for i in range(len(ids) - 1)
                    if pair_ok[i]
                ]
                n_fin = sum(np.isfinite(e) for e in errs)
                finite = [e for e in errs if np.isfinite(e)]
                med = float(np.median(finite)) if finite else float("inf")
                results.append((med, n_fin, {
                    "decode": name, "unit": unit, "pose_scale": scale,
                    "conv": conv, "invert": invert, "qorder": qorder,
                }))

    results.sort(key=lambda r: r[0])
    for med, n_fin, combo in results[:8]:
        print(f"  median_relerr={med:9.5f} (finite쌍={n_fin})  decode={combo['decode']}"
              f"  unit={combo['unit']:g}  pose_scale={combo['pose_scale']:g}"
              f"  conv={combo['conv']}  invert={combo['invert']}  qorder={combo['qorder']}")
    best_err, _, best = results[0]
    second_err = results[1][0]
    is_default = best == DEFAULTS
    unique = best_err * 3 < second_err or best_err < 1e-4
    print(f"best combo: {best} relerr={best_err:.5f} (2nd={second_err:.5f}) | gt.py defaults: {DEFAULTS}")
    if best_err >= args.err_thresh:
        print(f"[FAIL] 최적 조합 상대 깊이 오차 {best_err:.5f} >= {args.err_thresh} - 통계 기반 규격 재점검 필요")
        return 1
    if not unique:
        print("[AMBIGUOUS] 상위 두 조합 오차가 유사해 판별 불가 - 프레임/보폭 늘려 재실측 필요")
        return 2
    if not is_default:
        print("[MISMATCH] gt.py 기본값과 다른 조합이 최적 - gt.py 상수 갱신 필요")
        return 2
    print(f"[PASS] 기본값 조합이 최적(상대 오차 {best_err:.5f} < {args.err_thresh})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
