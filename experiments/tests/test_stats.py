"""stats.py 단위테스트(스크립트 15). pytest 없이 run_all.py로 실행.

scipy가 설치된 환경에서만 정규 근사 교차검증을 수행한다(서버에서는
scipy 부재 시 자동 스킵 - 구현 자체는 numpy만 의존).
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from weather3d import stats as S


def test_average_ranks():
    assert np.allclose(S._average_ranks(np.array([1.0, 1.0, 2.0])), [1.5, 1.5, 3.0])
    assert np.allclose(S._average_ranks(np.array([3.0, 1.0, 2.0])), [3.0, 1.0, 2.0])
    assert np.allclose(S._average_ranks(np.array([5.0, 5.0, 5.0, 1.0])), [3.0, 3.0, 3.0, 1.0])


def test_wilcoxon_exact_constant_shift():
    x = np.arange(1, 11, dtype=float)
    d = x - (x + 1.0)  # 전부 -1
    r = S.wilcoxon_signed_rank(d)
    assert r["n"] == 10
    assert r["stat"] == 0.0
    assert r["method"] == "exact"
    assert abs(r["p_value"] - 2.0 / 1024.0) < 1e-12  # 0.001953125


def test_wilcoxon_textbook_independent_enumeration():
    x = np.array([125, 115, 130, 140, 140, 115, 105, 145], dtype=float)
    y = np.array([110, 122, 125, 120, 140, 124, 123, 140], dtype=float)
    d = x - y
    r = S.wilcoxon_signed_rank(d)
    assert r["n"] == 7  # 영차 1쌍 제외
    assert r["stat"] == 13.0

    # 독립 오라클: itertools 전열거로 P(min(W+,W-) <= stat) 직접 계산.
    # 랭크는 동순위 평균 랭크(이 fixture는 |d|=5가 두 번 등장).
    dnz = d[d != 0]
    absd = np.abs(dnz)
    ranks = np.empty(dnz.size)
    for i in range(dnz.size):
        less = float(np.sum(absd < absd[i]))
        equal = float(np.sum(absd == absd[i]))
        ranks[i] = less + (equal + 1.0) / 2.0
    total = ranks.sum()
    count = 0
    for signs in itertools.product((1.0, -1.0), repeat=dnz.size):
        wp = sum(r * s for r, s in zip(ranks, signs) if s > 0)
        if min(wp, total - wp) <= 13.0:
            count += 1
    p_oracle = count / float(2**dnz.size)
    assert abs(r["p_value"] - p_oracle) < 1e-12


def test_wilcoxon_normal_matches_scipy():
    try:
        from scipy.stats import wilcoxon as sp_wilcoxon
    except ImportError:
        return  # 서버 등 scipy 부재 환경 - 스킵
    rng = np.random.default_rng(7)
    d = rng.normal(0.3, 1.0, size=40)
    ours = S.wilcoxon_signed_rank(d, max_exact_n=0)  # 정규 근사 강제
    ref = sp_wilcoxon(d, zero_method="wilcox", method="approx", correction=False)
    assert ours["method"] == "normal"
    assert abs(ours["stat"] - ref.statistic) < 1e-9
    assert abs(ours["p_value"] - ref.pvalue) < 1e-9


def test_wilcoxon_zero_and_nan_dropped():
    r = S.wilcoxon_signed_rank([1.0, -1.0, 0.0, 2.0, float("nan"), 3.0])
    assert r["n"] == 4  # 0과 NaN 제외(1,-1,2,3)
    assert r["method"] == "exact"
    assert 0.0 <= r["p_value"] <= 1.0
    empty = S.wilcoxon_signed_rank([0.0, 0.0, float("nan")])
    assert empty["p_value"] is None and empty["method"] == "no_data"


def test_bootstrap_constant_and_reproducible():
    r1 = S.paired_bootstrap_ci([2.0] * 5, seed=0)
    assert r1["mean_diff"] == 2.0 and r1["ci_low"] == 2.0 and r1["ci_high"] == 2.0
    rng = np.random.default_rng(3)
    d = rng.normal(1.0, 1.0, size=500)
    a = S.paired_bootstrap_ci(d, seed=42)
    b = S.paired_bootstrap_ci(d, seed=42)
    c = S.paired_bootstrap_ci(d, seed=43)
    assert a == b  # 시드 고정 = 결정적
    assert a["ci_low"] < a["mean_diff"] < a["ci_high"]
    assert a != c  # 시드가 다르면 값이 다르다
    assert abs(a["mean_diff"] - 1.0) < 0.1 and (a["ci_high"] - a["ci_low"]) < 0.4


def test_holm_known_values():
    h = S.holm([0.01, 0.04, 0.03])
    assert h["adjusted"] == [0.03, 0.06, 0.06]
    assert h["reject"] == [True, False, False]
    assert h["m"] == 3
    # None은 보정 대상에서 제외(유효 m 감소).
    h2 = S.holm([0.01, None, 0.03])
    assert h2["m"] == 2
    assert h2["adjusted"][0] == 0.02 and h2["adjusted"][1] is None
    assert h2["adjusted"][2] == 0.03


def test_load_results_csv():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "r.csv"
        header = ["case", "seq_id", "video_depth.Abs Rel", "recon.nc", "video_depth.δ < 1.25"]
        rows = [
            ["c0", "s1", "0.10", "0.70", "0.90"],
            ["c1_fog_mid", "s1", "0.20", "", "0.80"],
            [],  # run_report의 구분 빈 행
            ["case", "mean", "0.1", "0.7", "0.9"],  # 집계 행
        ]
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        t = S.load_results_csv(p)
        assert set(t.keys()) == {("c0", "s1"), ("c1_fog_mid", "s1")}
        assert t[("c0", "s1")]["recon.nc"] == 0.70
        assert t[("c0", "s1")]["video_depth.δ < 1.25"] == 0.90
        assert "recon.nc" not in t[("c1_fog_mid", "s1")]  # 빈 셀 스킵


def test_oriented_improvement():
    lower = S.oriented_improvement([0.2], [0.1], "video_depth.Abs Rel")
    assert np.allclose(lower, [0.1])  # base-arm
    higher = S.oriented_improvement([0.8], [0.9], "recon.nc")
    assert np.allclose(higher, [0.1])  # arm-base
    worse = S.oriented_improvement([0.1], [0.2], "video_depth.Abs Rel")
    assert np.allclose(worse, [-0.1])


def test_analyze_arm_synthetic():
    n = 8  # 변형 2종 x 8시퀀스 = 16쌍 -> exact 경로
    better = {
        "video_depth.Abs Rel": (0.30, 0.20),  # (base, arm)
        "video_depth.δ < 1.25": (0.80, 0.90),
    }
    cases = {
        "c1_fog_mid": better,
        "c1_rain_light": better,
        "c2_fog_mid": better,
        "c0": {"video_depth.Abs Rel": (0.10, 0.10),  # 동일 -> 회귀 0%
               "video_depth.δ < 1.25": (0.95, 0.95)},
    }
    base, arm = {}, {}
    for case, metrics in cases.items():
        # arm의 c2_<variant> 대응 키는 c1_<variant>(vs C2 비교 = base c2 x arm c1).
        arm_case = f"c1_{case.split('_', 1)[1]}" if case.startswith("c2_") else case
        for i in range(n):
            sid = f"s{i:02d}"
            for m, (b, a) in metrics.items():
                base.setdefault((case, sid), {})[m] = b
                arm.setdefault((arm_case, sid), {})[m] = a

    res = S.analyze_arm(base, arm, "t")
    assert res["comparisons"]["vs_c1"]["metrics"]["video_depth.Abs Rel"]["n"] == 16
    assert res["comparisons"]["vs_c2"]["metrics"]["video_depth.Abs Rel"]["n"] == 8
    imp = res["comparisons"]["vs_c1"]["metrics"]["video_depth.Abs Rel"]["improve_mean"]
    assert abs(imp - 0.10) < 1e-12
    # 제공 지표가 2/7이라 G1/G2(>=4)는 FAIL. clean도 2/7로 G3(>=6) FAIL
    # (미제공 지표는 통과로 세지 않는 엄격 기준 - 실데이터는 7종 모두 존재).
    assert res["gates"]["degraded_vs_c1"] is False
    assert res["gates"]["degraded_vs_c2"] is False
    assert res["gates"]["clean_regression"] is False
    assert res["comparisons"]["clean_c0"]["pass_count"] == 2
    assert res["verdict"] == "FAIL"
    json.dumps(res)  # 직렬화 가능 확인
    fam = res["family_breakdown"]["fog"]["video_depth.Abs Rel"]
    assert fam["n"] == 8 and abs(fam["arm_mean"] - 0.20) < 1e-12


def test_stats_cli_end_to_end():
    n = 9
    metrics_vals = {
        "video_depth.Abs Rel": (0.30, 0.20),
        "video_depth.δ < 1.25": (0.80, 0.90),
        "pose.ate_rmse": (0.12, 0.08),
        "pose.rpe_rot_deg_mean": (2.0, 1.5),
        "recon.acc": (0.09, 0.07),
        "recon.comp": (0.11, 0.08),
        "recon.nc": (0.60, 0.70),
    }
    header = ["case", "seq_id", *S.KEY_METRICS]

    def rows(case, arm_mode):
        out = []
        for i in range(n):
            sid = f"s{i:02d}"
            vals = []
            for m in S.KEY_METRICS:
                b, a = metrics_vals[m]
                # clean은 arm==base(회귀 0%), 열화는 arm이 우위.
                v = b if (case == "c0" and arm_mode == "arm") else (a if arm_mode == "arm" else b)
                vals.append(f"{v}")
            out.append([case, sid, *vals])
        return out

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        base_csv, arm_csv = td / "base.csv", td / "arm.csv"
        with open(base_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows("c0", "base"))
            w.writerows(rows("c1_fog_mid", "base"))
            w.writerows(rows("c2_fog_mid", "base"))
        with open(arm_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows("c0", "arm"))
            w.writerows(rows("c1_fog_mid", "arm"))

        out_dir = td / "out"
        old_argv = sys.argv
        sys.argv = ["stats", "--base-csv", str(base_csv),
                    "--arm", f"r1={arm_csv}", "--out-dir", str(out_dir)]
        try:
            rc = S.main()
        finally:
            sys.argv = old_argv
        assert rc == 0
        payload = json.loads((out_dir / "stats_results.json").read_text(encoding="utf-8"))
        assert payload["arms"]["r1"]["verdict"] == "PASS"
        md = (out_dir / "stats_report.md").read_text(encoding="utf-8")
        assert "verdict: **PASS**" in md


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith("test_") and callable(v)}.items()):
        fn()
        print(f"[PASS] {name}")
    print("ALL PASS")
