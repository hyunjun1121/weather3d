"""Neural-RGBD 원본 배포본 -> weather3d 로더 형식 변환.

2026-08-24 서버 실측(TUM neural_rgbd_data.zip의 whiteroom) 구조:
    <scene>/images/*.png              RGB(파일명에 프레임 번호 포함)
    <scene>/depth/depth<i>.png        16bit mm depth
    <scene>/depth_filtered/, depth_with_noise/   (다른 depth 변형, 미사용)
    <scene>/focal.txt, poses.txt, trainval_poses.txt

구형 배포(scene 루트에 im_*.png가 있는 형식)도 함께 지원한다.

변환 결과(sequences.py/gt.py가 기대하는 형식):
    <scene>/images/img<IDX:06d>.png
    <scene>/depth/depth<IDX:06d>.png
    <scene>/poses.txt                 (원본 그대로)

- IDX는 원본 프레임 번호를 보존한다. gt.py가 poses.txt의 행 인덱스로
  pose를 찾으므로 번호를 다시 매기면 안 된다.
- img 접두 파일은 원본 위치에서 이름만 정규화한다(멱등). 로더는
  `img` 접두 파일만 보므로, depth가 없어 제외된 img 접두 파일은
  로더에 걸리지 않도록 images/_excluded/ 로 옮긴다.
- 다른 디렉터리에 있는 원본은 하드링크로 연결(실패 시 복사)한다.

사용(experiments/ 기준):
    python scripts/convert_nrgbd.py --root ../data/neural_rgbd
    python scripts/convert_nrgbd.py --root ../data/neural_rgbd --scenes whiteroom
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

IMG_RE = re.compile(r"^(?:im|img|image|rgb|frame)[_-]?(\d+)\.png$", re.IGNORECASE)
BARE_IMG_RE = re.compile(r"^(\d+)\.png$")
DEPTH_NAMED_RE = re.compile(r"^depth[_-]?(\d+)\.png$", re.IGNORECASE)
DEPTH_BARE_RE = re.compile(r"^(\d+)\.png$")


def _collect(
    folder: Path, patterns: tuple[re.Pattern, ...]
) -> tuple[dict[int, Path], list[Path]]:
    """folder 내 파일을 프레임 번호로 매핑. 중복/불일치 파일은 별도 반환."""
    found: dict[int, Path] = {}
    extras: list[Path] = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        idx = None
        for pat in patterns:
            m = pat.match(p.name)
            if m:
                idx = int(m.group(1))
                break
        if idx is None:
            continue
        if idx in found:
            extras.append(p)
        else:
            found[idx] = p
    return found, extras


def convert_scene(scene: Path) -> str:
    if not scene.is_dir():
        return f"[FAIL] {scene}: not a directory"
    if not (scene / "poses.txt").is_file():
        return f"[FAIL] {scene.name}: poses.txt not found"

    img_src = scene / "images" if (scene / "images").is_dir() else scene
    imgs, img_dups = _collect(img_src, (IMG_RE, BARE_IMG_RE))
    if not imgs:
        sample = sorted(p.name for p in img_src.iterdir())[:12]
        return f"[FAIL] {scene.name}: no numbered png in {img_src.name}/; sample={sample}"

    dep_dir = scene / "depth"
    if not dep_dir.is_dir():
        return f"[FAIL] {scene.name}: depth/ not found"
    depths, _ = _collect(dep_dir, (DEPTH_NAMED_RE, DEPTH_BARE_RE))
    if not depths:
        sample = sorted(p.name for p in dep_dir.iterdir())[:8]
        return f"[FAIL] {scene.name}: no depth png matched; sample={sample}"

    keep = sorted(set(imgs) & set(depths))
    dropped = sorted(set(imgs) - set(depths))
    if not keep:
        return f"[FAIL] {scene.name}: image/depth 프레임 번호가 하나도 겹치지 않음"

    img_dst = scene / "images"
    img_dst.mkdir(exist_ok=True)
    excluded = img_dst / "_excluded"
    same_dir = img_src == img_dst

    def place(src: Path, dst: Path) -> None:
        if src == dst:
            return
        if not same_dir:
            if dst.exists():
                return
            try:
                os.link(src, dst)
                return
            except OSError:
                pass
        if dst.exists():
            excluded.mkdir(exist_ok=True)
            os.replace(src, excluded / src.name)
        else:
            os.replace(src, dst)

    moved = 0
    for idx in keep:
        i_dst = img_dst / f"img{idx:06d}.png"
        d_dst = dep_dir / f"depth{idx:06d}.png"
        before = (imgs[idx].name, depths[idx].name)
        place(imgs[idx], i_dst)
        place(depths[idx], d_dst)
        if before != (i_dst.name, d_dst.name):
            moved += 1

    # 로더 glob(img*)에 걸리지만 depth가 없는/중복인 파일은 치운다.
    # 단 keep의 최종 이름(img%06d.png)은 정규화 결과물이므로 건드리지 않는다.
    keep_names = {f"img{idx:06d}.png" for idx in keep}
    cleanup = [imgs[i] for i in dropped] + img_dups
    n_excl = 0
    for p in cleanup:
        try:
            if (
                p.name not in keep_names
                and p.is_file()
                and p.parent == img_dst
                and p.name.startswith("img")
            ):
                excluded.mkdir(exist_ok=True)
                os.replace(p, excluded / p.name)
                n_excl += 1
        except OSError:
            pass

    msg = f"[OK] {scene.name}: {len(keep)} frames (idx {keep[0]}..{keep[-1]}, renamed {moved})"
    if dropped:
        msg += f", no-depth {len(dropped)}"
    if n_excl:
        msg += f", excluded {n_excl}"
    return msg


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", required=True, help="data/neural_rgbd 경로")
    parser.add_argument("--scenes", nargs="*", default=None, help="대상 scene(기본: root 하위 전체)")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"[FAIL] root not found: {root}")
        return 1

    if args.scenes:
        targets = [root / s for s in args.scenes]
    else:
        targets = [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    if not targets:
        print(f"no scene directories under {root}")
        return 1

    failures = 0
    for scene in targets:
        msg = convert_scene(scene)
        print(msg)
        if msg.startswith("[FAIL]"):
            failures += 1

    print(f"\n{'CONVERT OK' if failures == 0 else f'{failures} SCENES FAILED'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
