"""단위 테스트 실행기(pytest 불필요).

사용:
    python tests/run_all.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import test_atmosphere
import test_c3_train
import test_noise
import test_pose_metrics
import test_recon_metrics
import test_resolve_autoseq
import test_tartanair_depth
import test_weather_ext
import test_stats


def main() -> int:
    failures = 0
    for mod in (test_atmosphere, test_c3_train, test_noise, test_pose_metrics, test_recon_metrics, test_resolve_autoseq, test_tartanair_depth, test_weather_ext, test_stats):
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                print(f"[PASS] {mod.__name__}.{name}")
            except Exception:
                failures += 1
                print(f"[FAIL] {mod.__name__}.{name}")
                traceback.print_exc()
    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURES'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
