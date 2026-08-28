"""resolve_autoseq(seq: auto 확정) 단위 테스트.

배경: 서버 스크립트 11의 인라인 resolver가 TestSplit.txt의 'sequence2'를
디렉터리 이름 그대로 찾아 auto 장면 5개를 전부 drop시켰다. 매핑 로직
회귀 방지용. run_all.py가 인자 없이 호출하므로 pytest fixture 대신
tempfile로 임시 디렉터리를 만든다.
"""

from __future__ import annotations

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from resolve_autoseq import resolve_config, seq_dir_name


def test_seq_dir_name_mapping():
    assert seq_dir_name("sequence2") == "seq-02"
    assert seq_dir_name("sequence10") == "seq-10"
    assert seq_dir_name("seq-01") == "seq-01"
    assert seq_dir_name(" seq-03 ") == "seq-03"
    assert seq_dir_name("other") == "other"


@contextmanager
def _tmp():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _mk_scene(root: Path, scene: str, splits: dict[str, str], seqs: list[str]):
    sd = root / scene
    sd.mkdir(parents=True, exist_ok=True)
    for name, content in splits.items():
        (sd / name).write_text(content, encoding="utf-8")
    for s in seqs:
        d = sd / s
        d.mkdir(parents=True, exist_ok=True)
        (d / "frame-000000.color.png").write_bytes(b"x")


def _cfg(tmp: Path, sequences: list[dict]) -> dict:
    return {
        "_config_path": tmp / "cfg.yaml",
        "data": {
            "seven_scenes_root": str(tmp / "7scenes"),
            "neural_rgbd_root": str(tmp / "nrgbd"),
            "tartanair2_root": str(tmp / "ta"),
            "sequences": sequences,
        },
    }


def test_resolve_auto_picks_test_split_seq():
    with _tmp() as tmp:
        _mk_scene(tmp / "7scenes", "office", {"TestSplit.txt": "sequence2\nsequence6\n", "TrainSplit.txt": "sequence1\n"}, ["seq-01", "seq-02"])
        kept, dropped, log = resolve_config(_cfg(tmp, [
            {"id": "7scenes_office", "dataset": "seven_scenes", "scene": "office", "seq": "auto", "stride": 50},
        ]))
        assert dropped == []
        assert kept[0]["seq"] == "seq-02"  # TestSplit 첫 후보(sequence2)가 우선
        assert any("[RESOLVE]" in x for x in log)


def test_resolve_auto_drops_when_test_split_has_no_frames():
    with _tmp() as tmp:
        # TestSplit 후보(seq-04)에 프레임이 없고 seq-02만 존재(Train 쪽).
        # 평가 오염 방지를 위해 glob 백업 없이 drop이 정책이다.
        _mk_scene(tmp / "7scenes", "heads", {"TestSplit.txt": "sequence4\n"}, ["seq-02"])
        kept, dropped, _ = resolve_config(_cfg(tmp, [
            {"id": "7scenes_heads", "dataset": "seven_scenes", "scene": "heads", "seq": "auto", "stride": 50},
        ]))
        assert kept == []
        assert dropped == ["7scenes_heads"]


def test_resolve_drop_when_no_frames():
    with _tmp() as tmp:
        (tmp / "7scenes" / "pumpkin" / "seq-01").mkdir(parents=True)  # 프레임 없음
        kept, dropped, _ = resolve_config(_cfg(tmp, [
            {"id": "7scenes_pumpkin", "dataset": "seven_scenes", "scene": "pumpkin", "seq": "seq-01", "stride": 50},
        ]))
        assert kept == []
        assert dropped == ["7scenes_pumpkin"]


def test_resolve_nrgbd_needs_images_dir():
    with _tmp() as tmp:
        (tmp / "nrgbd" / "whiteroom").mkdir(parents=True)  # images/ 없음
        kept, dropped, _ = resolve_config(_cfg(tmp, [
            {"id": "nrgbd_whiteroom", "dataset": "neural_rgbd", "scene": "whiteroom", "stride": 100},
        ]))
        assert kept == []
        assert dropped == ["nrgbd_whiteroom"]


def test_resolve_tartanair_checks_traj():
    with _tmp() as tmp:
        rel = tmp / "ta" / "GreatMarsh" / "Data_easy" / "extracted" / "GreatMarsh" / "Data_easy" / "P000"
        (rel / "image_lcam_front").mkdir(parents=True)
        kept, dropped, _ = resolve_config(_cfg(tmp, [
            {"id": "ta_ok", "dataset": "tartanair2", "env": "GreatMarsh", "difficulty": "easy", "traj": "P000", "stride": 100},
            {"id": "ta_bad", "dataset": "tartanair2", "env": "GreatMarsh", "difficulty": "easy", "traj": "P001", "stride": 100},
        ]))
        assert [s["id"] for s in kept] == ["ta_ok"]
        assert dropped == ["ta_bad"]
