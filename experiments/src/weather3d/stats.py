"""C3 학습 결과 통계 분석(스크립트 15).

사전 등록 성공 기준(실험맥락.md §8, 2026-08-23 사용자 확정)의 연산화:
- C3(파인튜닝 StreamVGGT)가 열화 입력(c1)에서 C1(base 모델 저하)·
  C2(ViWS 복원 후 base)보다 우위인지 대응표본 검정.
- clean(c0) 회귀가 기준(C0 = base 모델 clean) 대비 5% 이내인지.

외부 통계 의존성 없이 numpy만 사용한다(서버 .venv 재현성 보장).
scipy가 있는 환경에서는 단위테스트가 정규 근사치를 교차검증한다.

구성:
- wilcoxon_signed_rank: 대응표본 부호순위 검정. n<=16은 2^n 전열거
  정확분포, n>16은 동순위 보정 정규 근사. 영차(같은 값 쌍)는
  Wilcoxon 표준 zero_method="wilcox"처럼 제외한다.
- paired_bootstrap_ci: 대응 재표본 평균 개선량의 퍼센타일 95% CI
  (numpy default_rng 시드 고정 - 결정적).
- holm: Holm-Bonferroni 다중검정 보정.
- load_results_csv: run_report.py 결과 CSV(parsed by csv 모듈,
  δ 등 유니코드 컬럼명 지원) -> {(case, seq_id): {컬럼: 값}}.
- analyze_arm / analyze_ta / main: 비교·판정·CLI.

방향 규약: improve = base - arm(낮을수록 좋은 지표) 또는
arm - base(높을수록 좋은 지표). 양수면 arm이 더 좋다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# 낮을수록 좋은 지표(run_report.LOWER_IS_BETTER와 동일 집합을
# 결과 CSV의 '.' 연결 컬럼명 기준으로 옮긴 것).
LOWER_IS_BETTER = {
    "video_depth.Abs Rel",
    "video_depth.Sq Rel",
    "video_depth.RMSE",
    "video_depth.Log RMSE",
    "pose.ate_rmse",
    "pose.rpe_trans_mean",
    "pose.rpe_rot_deg_mean",
    "recon.acc",
    "recon.acc_med",
    "recon.comp",
    "recon.comp_med",
}

# 논문 표 7대 핵심 지표(run_report key_metrics와 동일).
KEY_METRICS = [
    "video_depth.Abs Rel",
    "video_depth.δ < 1.25",
    "pose.ate_rmse",
    "pose.rpe_rot_deg_mean",
    "recon.acc",
    "recon.comp",
    "recon.nc",
]

FAMILIES = ("fog", "smoke", "rain", "lowlight")

# 기계 판정 규칙(사전 등록 기준의 연산화 - 보고서에도 그대로 기재).
# G1: vs C1 개선(oriented) 평균이 양수인 핵심 지표 >= 4/7
# G2: vs C2 개선(oriented) 평균이 양수인 핵심 지표 >= 4/7
# G3: clean 회귀 5% 이내인 핵심 지표 >= 6/7
GATE_RULES = {
    "degraded_vs_c1": "oriented mean improvement > 0 on >= 4/7 key metrics",
    "degraded_vs_c2": "oriented mean improvement > 0 on >= 4/7 key metrics",
    "clean_regression": "regression within 5% on >= 6/7 key metrics",
}
CLEAN_REGRESSION_TOL = 0.05


# ---------------------------------------------------------------- 랭킹/검정

def _average_ranks(a: np.ndarray) -> np.ndarray:
    """동순위 평균 랭킨지. a>=0 (abs 차이에 적용)."""
    order = np.argsort(a, kind="stable")
    ranks = np.empty(a.size, dtype=np.float64)
    sa = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def wilcoxon_signed_rank(diffs, max_exact_n: int = 16) -> dict:
    """대응표본 Wilcoxon 부호순위 검정(양측).

    diffs: base - arm 원방향(방향은 검정에 무관 - 부호 대칭).
    반환: {"n", "stat"(=min(W+,W-)), "p_value", "method"}.
    데이터가 없거나 영차만 있으면 p_value=None.
    """
    d = np.asarray(diffs, dtype=np.float64)
    d = d[d == d]  # NaN 제거
    d = d[d != 0.0]  # 영차 제거(zero_method="wilcox")
    n = int(d.size)
    if n == 0:
        return {"n": 0, "stat": None, "p_value": None, "method": "no_data"}

    ranks = _average_ranks(np.abs(d))
    total = float(ranks.sum())
    w_plus = float(ranks[d > 0].sum())
    stat = min(w_plus, total - w_plus)

    if n <= max_exact_n:
        # 2^n 부호 배치 전열거: P(min(W+,W-) <= stat).
        bits = ((np.arange(1 << n, dtype=np.int64)[:, None] >> np.arange(n)) & 1).astype(np.float64)
        wplus_all = bits @ ranks
        min_all = np.minimum(wplus_all, total - wplus_all)
        p = float(np.mean(min_all <= stat + 1e-9))
        return {"n": n, "stat": stat, "p_value": p, "method": "exact"}

    # 정규 근사(동순위 보정, 연속 보정 없음 - scipy approx와 동일식).
    mu = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_corr = float(sum(c**3 - c for c in counts if c > 1)) / 48.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr
    if var <= 0:
        return {"n": n, "stat": stat, "p_value": None, "method": "degenerate"}
    z = (stat - mu) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))  # = 2*(1-Phi(|z|))
    return {"n": n, "stat": stat, "p_value": p, "method": "normal"}


def paired_bootstrap_ci(diffs, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> dict:
    """대응표본 평균 차이의 퍼센타일 부트스트랩 CI(결정적 시드)."""
    d = np.asarray(diffs, dtype=np.float64)
    d = d[d == d]
    n = int(d.size)
    if n == 0:
        return {"n": 0, "mean_diff": None, "ci_low": None, "ci_high": None,
                "n_boot": n_boot, "seed": seed}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "n": n,
        "mean_diff": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": n_boot,
        "seed": seed,
    }


def holm(pvals, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni 보정. pvals의 None(검정 불능)은 보정 대상에서
    제외하고 그대로 None을 유지한다(유효 m에서 제외)."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    m = len(idx)
    adj: list = [None] * len(pvals)
    reject = [False] * len(pvals)
    order = sorted(idx, key=lambda i: pvals[i])
    prev = 0.0
    for r, i in enumerate(order):
        a = min(1.0, (m - r) * pvals[i])
        a = max(a, prev)  # 단조성 유지
        adj[i] = a
        prev = a
    for i in idx:
        reject[i] = bool(adj[i] < alpha)
    return {"adjusted": adj, "reject": reject, "alpha": alpha, "m": m}


# ---------------------------------------------------------------- 데이터 적재

def load_results_csv(path: str | Path) -> dict[tuple[str, str], dict[str, float]]:
    """run_report.py 출력 CSV -> {(case, seq_id): {컬럼: float}}.

    헤더 ['case','seq_id',...]; seq_id=='mean' 집계행과 빈 행은 제외.
    빈 셀/비수치 셀은 그 컬럼을 스킵한다.
    """
    table: dict[tuple[str, str], dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 2 or row[0] == "case" or row[1] in ("mean", ""):
                continue
            case, seq_id = row[0], row[1]
            vals: dict[str, float] = {}
            for col, raw in zip(header[2:], row[2:]):
                if raw == "":
                    continue
                try:
                    vals[col] = float(raw)
                except (TypeError, ValueError):
                    continue
            if vals:
                table[(case, seq_id)] = vals
    return table


def oriented_improvement(base_vals, arm_vals, metric: str) -> np.ndarray:
    """개선량(양수 = arm 우위). 낮을수록 좋은 지표는 base-arm,
    높을수록 좋은 지표는 arm-base."""
    b = np.asarray(base_vals, dtype=np.float64)
    a = np.asarray(arm_vals, dtype=np.float64)
    return (b - a) if metric in LOWER_IS_BETTER else (a - b)


# ---------------------------------------------------------------- 비교 분석

def _pair_cases(base: dict, arm: dict, base_prefix: str, arm_prefix: str, metric: str):
    """base의 <prefix>_<variant>/<seq>와 arm의 대응 행을 모은다."""
    xs, ys = [], []
    for (b_case, seq_id), bvals in sorted(base.items()):
        if not b_case.startswith(base_prefix + "_"):
            continue
        variant = b_case[len(base_prefix) + 1 :]
        avals = arm.get((f"{arm_prefix}_{variant}", seq_id))
        if avals is None or metric not in bvals or metric not in avals:
            continue
        b, a = bvals[metric], avals[metric]
        if b != b or a != a:  # NaN 방어
            continue
        xs.append(b)
        ys.append(a)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def _pair_single(base: dict, arm: dict, case: str, metric: str):
    xs, ys = [], []
    for (c, seq_id), bvals in sorted(base.items()):
        if c != case:
            continue
        avals = arm.get((case, seq_id))
        if avals is None or metric not in bvals or metric not in avals:
            continue
        b, a = bvals[metric], avals[metric]
        if b != b or a != a:
            continue
        xs.append(b)
        ys.append(a)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def _comparison(base: dict, arm: dict, base_prefix: str, arm_prefix: str) -> dict:
    """핵심 지표별 Wilcoxon + bootstrap CI + 평균. Holm은 호출자가 적용."""
    metrics_out: dict[str, dict] = {}
    raw_ps: list = []
    for metric in KEY_METRICS:
        xs, ys = _pair_cases(base, arm, base_prefix, arm_prefix, metric)
        if xs.size == 0:
            metrics_out[metric] = {"n": 0, "base_mean": None, "arm_mean": None,
                                   "improve_mean": None, "ci95": None,
                                   "wilcoxon_p": None, "holm_p": None,
                                   "significant": False}
            raw_ps.append(None)
            continue
        imp = oriented_improvement(xs, ys, metric)
        w = wilcoxon_signed_rank(imp)
        boot = paired_bootstrap_ci(imp)
        metrics_out[metric] = {
            "n": int(xs.size),
            "base_mean": float(xs.mean()),
            "arm_mean": float(ys.mean()),
            "improve_mean": float(imp.mean()),
            "ci95": [boot["ci_low"], boot["ci_high"]],
            "wilcoxon_p": w["p_value"],
            "holm_p": None,  # 아래에서 채움
            "significant": False,
        }
        raw_ps.append(w["p_value"])
    h = holm(raw_ps)
    for metric, adj, rej in zip(KEY_METRICS, h["adjusted"], h["reject"]):
        metrics_out[metric]["holm_p"] = adj
        metrics_out[metric]["significant"] = bool(rej) if adj is not None else False
    return {"metrics": metrics_out, "holm_m": h["m"]}


def _clean_regression(base: dict, arm: dict) -> dict:
    """clean(c0) 회귀 검사: arm/base 비율이 허용 오차 이내인지."""
    metrics_out: dict[str, dict] = {}
    ok_count = 0
    for metric in KEY_METRICS:
        xs, ys = _pair_single(base, arm, "c0", metric)
        if xs.size == 0 or xs.mean() == 0:
            metrics_out[metric] = {"n": int(xs.size), "base_mean": None,
                                   "arm_mean": None, "ratio": None,
                                   "regression_ok": False}
            continue
        ratio = float(ys.mean() / xs.mean())
        # 낮을수록 좋은 지표: arm/base <= 1+tol이면 회귀 없음.
        # 높을수록 좋은 지표: arm/base >= 1-tol이면 회귀 없음.
        if metric in LOWER_IS_BETTER:
            ok = ratio <= 1.0 + CLEAN_REGRESSION_TOL
        else:
            ok = ratio >= 1.0 - CLEAN_REGRESSION_TOL
        ok_count += int(ok)
        metrics_out[metric] = {
            "n": int(xs.size),
            "base_mean": float(xs.mean()),
            "arm_mean": float(ys.mean()),
            "ratio": ratio,
            "regression_ok": bool(ok),
        }
    return {"metrics": metrics_out, "pass_count": ok_count, "of": len(KEY_METRICS)}


def _family_breakdown(base: dict, arm: dict) -> dict:
    """변형군(fog/smoke/rain/lowlight)별 c1 평균 - 논문 표용."""
    out: dict[str, dict[str, dict]] = {}
    for family in FAMILIES:
        rows: dict[str, dict] = {}
        for metric in KEY_METRICS:
            bs, as_ = [], []
            for (b_case, seq_id), bvals in sorted(base.items()):
                if not b_case.startswith(f"c1_{family}"):
                    continue
                avals = arm.get((b_case, seq_id))
                if avals is None or metric not in bvals or metric not in avals:
                    continue
                bs.append(bvals[metric])
                as_.append(avals[metric])
            if bs:
                rows[metric] = {
                    "n": len(bs),
                    "base_mean": float(np.mean(bs)),
                    "arm_mean": float(np.mean(as_)),
                }
        out[family] = rows
    return out


def analyze_arm(base: dict, arm: dict, name: str) -> dict:
    """arm 하나에 대한 전체 비교(vs C1 / vs C2 / clean) + 판정."""
    vs_c1 = _comparison(base, arm, "c1", "c1")
    vs_c2 = _comparison(base, arm, "c2", "c1")
    clean = _clean_regression(base, arm)

    g1 = sum(1 for m in KEY_METRICS
             if (vs_c1["metrics"][m]["improve_mean"] or 0) > 0) >= 4
    g2 = sum(1 for m in KEY_METRICS
             if (vs_c2["metrics"][m]["improve_mean"] or 0) > 0) >= 4
    g3 = clean["pass_count"] >= 6
    return {
        "name": name,
        "comparisons": {"vs_c1": vs_c1, "vs_c2": vs_c2, "clean_c0": clean},
        "family_breakdown": _family_breakdown(base, arm),
        "gates": {"degraded_vs_c1": g1, "degraded_vs_c2": g2, "clean_regression": g3},
        "verdict": "PASS" if (g1 and g2 and g3) else "FAIL",
    }


def analyze_ta(base_ta: dict, arm_ta: dict) -> dict:
    """TartanAir V2 native weather c0(미학습 도메인 일반화, 서술 통계)."""
    out: dict[str, dict] = {}
    for metric in KEY_METRICS:
        xs, ys = _pair_single(base_ta, arm_ta, "c0", metric)
        if xs.size == 0 or xs.mean() == 0:
            out[metric] = {"n": 0, "base_mean": None, "arm_mean": None, "ratio": None}
            continue
        out[metric] = {
            "n": int(xs.size),
            "base_mean": float(xs.mean()),
            "arm_mean": float(ys.mean()),
            "ratio": float(ys.mean() / xs.mean()),
        }
    return out


# ---------------------------------------------------------------- 보고서

def _fmt(v, digits: int = 4) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _md_arm(res: dict) -> list[str]:
    lines: list[str] = [f"## arm {res['name']} — verdict: **{res['verdict']}**", ""]
    for label, comp in (("vs C1 (base, degraded)", res["comparisons"]["vs_c1"]),
                        ("vs C2 (ViWS restore + base)", res["comparisons"]["vs_c2"])):
        lines += [f"### {label}", "",
                  "| 지표 | base | C3(arm) | 개선(oriented) | 95% CI | p(Wilcoxon) | p(Holm) | 유의 |",
                  "|---|---|---|---|---|---|---|---|"]
        for m in KEY_METRICS:
            r = comp["metrics"][m]
            ci = "-" if r["ci95"] is None else f"[{_fmt(r['ci95'][0])}, {_fmt(r['ci95'][1])}]"
            p = "-" if r["wilcoxon_p"] is None else f"{r['wilcoxon_p']:.4g}"
            ph = "-" if r["holm_p"] is None else f"{r['holm_p']:.4g}"
            lines.append(
                f"| {m} | {_fmt(r['base_mean'])} | {_fmt(r['arm_mean'])} | "
                f"{_fmt(r['improve_mean'])} | {ci} | {p} | {ph} | "
                f"{'O' if r['significant'] else '-'} |")
        lines.append("")
    clean = res["comparisons"]["clean_c0"]
    lines += ["### clean(c0) 회귀 검사", "",
              "| 지표 | C0(base) | C3(arm) | arm/base | 5% 이내 |", "|---|---|---|---|---|"]
    for m in KEY_METRICS:
        r = clean["metrics"][m]
        lines.append(
            f"| {m} | {_fmt(r['base_mean'])} | {_fmt(r['arm_mean'])} | "
            f"{_fmt(r['ratio'])} | {'O' if r['regression_ok'] else 'X'} |")
    lines += [f"\n통과 {clean['pass_count']}/{clean['of']}", ""]

    lines += ["### 변형군별 평균 (C1 base vs C3 arm)", "",
              "| 군 | 지표 | n | base | arm |", "|---|---|---|---|---|"]
    for family, rows in res["family_breakdown"].items():
        for m in KEY_METRICS:
            r = rows.get(m)
            if r is None:
                continue
            lines.append(f"| {family} | {m} | {r['n']} | "
                         f"{_fmt(r['base_mean'])} | {_fmt(r['arm_mean'])} |")
    g = res["gates"]
    lines += ["", "### 판정 게이트", "",
              f"- G1 열화 vs C1 우위(>=4/7): {'PASS' if g['degraded_vs_c1'] else 'FAIL'}",
              f"- G2 열화 vs C2 우위(>=4/7): {'PASS' if g['degraded_vs_c2'] else 'FAIL'}",
              f"- G3 clean 회귀 5% 이내(>=6/7): {'PASS' if g['clean_regression'] else 'FAIL'}",
              f"- **verdict: {res['verdict']}**", ""]
    return lines


# ---------------------------------------------------------------- CLI

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", required=True,
                        help="base 모델 결과 CSV(스크립트 12 ext_v2_results.csv)")
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=PATH",
                        help="학습 arm 결과 CSV(반복 가능: r1=... r2=...)")
    parser.add_argument("--base-ta-csv", default=None,
                        help="base TartanAir 결과 CSV(선택)")
    parser.add_argument("--arm-ta", action="append", default=[], metavar="NAME=PATH",
                        help="arm TartanAir 결과 CSV(선택, 반복 가능)")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    base = load_results_csv(args.base_csv)
    if not base:
        print(f"[FATAL] base CSV 비었거나 읽을 수 없음: {args.base_csv}")
        return 2

    arms: dict[str, dict] = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        if not name or not path:
            print(f"[FATAL] --arm 형식은 NAME=PATH: {spec}")
            return 2
        arms[name] = load_results_csv(path)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_csv": str(args.base_csv),
        "key_metrics": KEY_METRICS,
        "gate_rules": GATE_RULES,
        "clean_regression_tol": CLEAN_REGRESSION_TOL,
        "arms": {},
    }

    report: list[str] = [
        "# C3 학습 결과 통계 보고서",
        "",
        f"- 생성: {results['generated_at']}",
        f"- base: {args.base_csv} (쌍 {len(base)}행)",
        f"- 핵심 지표 {len(KEY_METRICS)}종, Holm 보정 군: vs C1 / vs C2 각각",
        f"- 판정 규칙: {json.dumps(GATE_RULES, ensure_ascii=False)}",
        "",
    ]

    for name, table in arms.items():
        if not table:
            print(f"[WARN] arm {name} CSV 비음 - 건너뜀")
            continue
        res = analyze_arm(base, table, name)
        results["arms"][name] = res
        report += _md_arm(res)
        print(f"[VERDICT] {name}: {res['verdict']} "
              f"(g1={res['gates']['degraded_vs_c1']} "
              f"g2={res['gates']['degraded_vs_c2']} "
              f"g3={res['gates']['clean_regression']})")

    if args.base_ta_csv and args.arm_ta:
        base_ta = load_results_csv(args.base_ta_csv)
        ta: dict[str, dict] = {}
        for spec in args.arm_ta:
            name, _, path = spec.partition("=")
            ta[name] = analyze_ta(base_ta, load_results_csv(path))
        results["tartanair"] = ta
        report += ["## TartanAir V2 native weather c0 (미학습 도메인, 서술 통계)", "",
                   "| arm | 지표 | n | base | arm | arm/base |", "|---|---|---|---|---|---|"]
        for name, rows in ta.items():
            for m in KEY_METRICS:
                r = rows[m]
                report.append(f"| {name} | {m} | {r['n']} | {_fmt(r['base_mean'])} | "
                              f"{_fmt(r['arm_mean'])} | {_fmt(r['ratio'])} |")
        report.append("")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "stats_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"json:   {out_dir / 'stats_results.json'}")
    print(f"report: {out_dir / 'stats_report.md'}")
    print(f"STATS OK arms={','.join(sorted(results['arms']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
