"""5단계: 평가 결과 집계 -> results.csv + report.md.

사용:
    python run_report.py --config configs/core_v1.yaml

C0(clean 기준선) 대비 저하율/회복률을 계산한다.
- 오류 지표(AbsRel, ATE, Acc, Comp 등; 낮을수록 좋음):
    저하율 = (M_case - M_c0) / M_c0
    회복률 = (M_c1 - M_case) / (M_c1 - M_c0)   [c2/c3용]
- 품질 지표(delta<1.25, NC 등; 높을수록 좋음):
    저하율 = (M_c0 - M_case) / M_c0
    회복률 = (M_case - M_c1) / (M_c0 - M_c1)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from weather3d.config import load_config, output_dir

# 낮을수록 좋은 지표(그 외 품질 지표는 높을수록 좋음으로 취급)
LOWER_IS_BETTER = {
    "Abs Rel", "Sq Rel", "RMSE", "Log RMSE",
    "ate_rmse", "rpe_trans_mean", "rpe_rot_deg_mean",
    "acc", "acc_med", "comp", "comp_med",
}


def flatten_metrics(entry: dict) -> dict[str, float]:
    """video_depth/pose/recon 하위 dict를 '.' 연결 단일 키로 편다."""
    flat: dict[str, float] = {}
    for section in ("video_depth", "pose", "recon"):
        sub = entry.get(section)
        if isinstance(sub, dict):
            for k, v in sub.items():
                if isinstance(v, (int, float)):
                    flat[f"{section}.{k}"] = float(v)
    return flat


def collect(eval_root: Path) -> dict[str, dict[str, dict]]:
    """{case_label: {seq_id: flat_metrics}}"""
    collected: dict[str, dict[str, dict]] = defaultdict(dict)
    for case_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
        for jf in sorted(case_dir.glob("*.json")):
            entry = json.loads(jf.read_text(encoding="utf-8"))
            if "error" in entry:
                continue
            flat = flatten_metrics(entry)
            if flat:
                collected[case_dir.name][entry["seq_id"]] = flat
    return dict(collected)


def mean_over_sequences(seq_metrics: dict[str, dict]) -> dict[str, float]:
    keys = set().union(*(m.keys() for m in seq_metrics.values()))
    out = {}
    for k in keys:
        vals = [m[k] for m in seq_metrics.values() if k in m and m[k] == m[k]]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def degradation(metric: str, m_case: float, m_c0: float) -> float | None:
    if m_c0 == 0 or m_c0 != m_c0 or m_case != m_case:
        return None
    if metric in LOWER_IS_BETTER:
        return (m_case - m_c0) / abs(m_c0)
    return (m_c0 - m_case) / abs(m_c0)


def recovery(metric: str, m_case: float, m_c1: float, m_c0: float) -> float | None:
    d_c1 = degradation(metric, m_c1, m_c0)
    if d_c1 is None or d_c1 <= 0:
        return None
    d_case = degradation(metric, m_case, m_c0)
    if d_case is None:
        return None
    return 1.0 - d_case / d_c1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "core_v1.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = output_dir(cfg)
    eval_root = out / "eval"
    if not eval_root.is_dir():
        print(f"no eval results under {eval_root}; run_evaluate.py 먼저 실행하세요.")
        return 2

    collected = collect(eval_root)
    means = {case: mean_over_sequences(seqs) for case, seqs in collected.items()}

    # ---- CSV: 시퀀스별 전체 지표 ----
    csv_path = out / "results.csv"
    all_keys = sorted(set().union(*(m.keys() for m in means.values()))) if means else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "seq_id", *all_keys])
        for case, seqs in sorted(collected.items()):
            for seq_id, m in sorted(seqs.items()):
                writer.writerow([case, seq_id, *[m.get(k, "") for k in all_keys]])
        writer.writerow([])
        writer.writerow(["case", "mean", *all_keys])
        for case, m in sorted(means.items()):
            writer.writerow([case, "mean", *[round(m.get(k, float("nan")), 6) for k in all_keys]])

    # ---- Markdown 보고서 ----
    report: list[str] = [
        f"# {cfg['experiment']} 결과 보고서",
        "",
        f"- 생성: {datetime.now(timezone.utc).isoformat()}",
        f"- 설정: {cfg['_config_path']}",
        f"- case 목록: {', '.join(sorted(collected)) or '(없음)'}",
        "",
    ]

    key_metrics = [
        "video_depth.Abs Rel", "video_depth.δ < 1.25",
        "pose.ate_rmse", "pose.rpe_rot_deg_mean",
        "recon.acc", "recon.comp", "recon.nc",
    ]
    c0 = means.get("c0", {})
    for case in sorted(means):
        if case == "c0":
            continue
        report += [f"## {case}", "", "| 지표 | C0(clean) | 해당 case | 저하율 | 회복률(vs C1) |", "|---|---|---|---|---|"]
        variant = case.split("_", 1)[1] if "_" in case else None
        c1_mean = means.get(f"c1_{variant}") if variant else None
        for metric in key_metrics:
            m0, mc = c0.get(metric), means[case].get(metric)
            m1 = c1_mean.get(metric) if c1_mean else None
            deg = degradation(metric, mc, m0) if mc is not None and m0 is not None else None
            rec = recovery(metric, mc, m1, m0) if m1 is not None and mc is not None and m0 is not None else None
            fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else "-"
            report.append(
                f"| {metric} | {fmt(m0)} | {fmt(mc)} | "
                f"{f'{deg:+.1%}' if deg is not None else '-'} | "
                f"{f'{rec:.1%}' if rec is not None else '-'} |"
            )
        report.append("")

    report_path = out / "report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"csv:    {csv_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
