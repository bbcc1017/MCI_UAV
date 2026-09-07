# -*- coding: utf-8 -*-
"""v21 — 정보수준 × 함수형 격자 집계 + test750 30시드 판정 재계산.

서브커맨드
  ladder    budget750 격자에서 셀별 최적 lambda 선정 (튜닝셋 전용)
  judge     test750 × seed 0-29 사다리 + paired 판정 (판정셋)
  audit     v19(30시드) 대비 v20(10시드) 정합성 감사 — 시드 수가 결론을 만들었는가

정보수준(부하 신호 출처) × 함수형(벌점 식) 두 축만 바꾸고 나머지는 고정한다.
lambda 는 셀 안에서만 비교한다 — 셀을 넘어 비교하면 축이 섞인다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from v20_threshold_report import ci, cube, paired  # noqa: E402

V20 = REPO / "results/scoreboard/v20"
V21 = REPO / "results/scoreboard/v21"
V19 = REPO / "results/scoreboard/v19"

# 셀 정의: 접두사 → (정보수준, 함수형, 통신 필요, 부하항)
CELLS = {
    "Q":  ("I3 census+이송중", "rate  (q+1-c)+/c", True,  "hingerate"),
    "H":  ("I3 census+이송중", "hinge (q+1-c)+",   True,  "hinge1"),
    "T":  ("I3 census+이송중", "lin   q",          True,  "load"),
    "OR": ("I2 census만",      "rate  (q+1-c)+/c", True,  "hingerate_occ"),
    "OH": ("I2 census만",      "hinge (q+1-c)+",   True,  "hinge1_occ"),
    "OL": ("I2 census만",      "lin   q",          True,  "occ"),
    "F":  ("I1a 이송중만",     "rate  (q+1-c)+/c", False, "hingerate_if"),
    "FH": ("I1a 이송중만",     "hinge (q+1-c)+",   False, "hinge1_if"),
    "FL": ("I1a 이송중만",     "lin   q",          False, "in_flight"),
    "P":  ("I1b 보낸누적",     "rate  (q+1-c)+/c", False, "hingerate_psent"),
    "PH": ("I1b 보낸누적",     "hinge (q+1-c)+",   False, "hinge1_psent"),
    "PL": ("I1b 보낸누적",     "lin   q",          False, "p_sent"),
    "Z":  ("I0 부하 무시",     "-",                False, "zero"),
}
ARM = re.compile(r"^([A-Z]{1,2})([0-9.]+)$")


def _load(paths):
    fs = [p for p in paths if (p.parent / (p.name + ".meta.json")).exists()]
    if not fs:
        raise SystemExit(f"완주한 CSV 없음: {[str(p) for p in paths]}")
    return pd.concat([pd.read_csv(p) for p in fs], ignore_index=True)


def _fams(df):
    """{셀 접두사: [(lam, cube), ...]}"""
    out = {}
    for pol in sorted(df.policy.unique()):
        m = ARM.match(pol)
        if not m or m.group(1) not in CELLS:
            continue
        out.setdefault(m.group(1), []).append((float(m.group(2)), cube(df, pol)))
    for k in out:
        out[k].sort(key=lambda x: x[0])
    return out


def _interp_min(lams, vals):
    """격자 최소 주변 2차 보간 정점. 경계면 격자값 그대로."""
    i = int(np.argmin(vals))
    if i in (0, len(vals) - 1):
        return float(lams[i])
    x, y = np.array(lams[i - 1:i + 2], float), np.array(vals[i - 1:i + 2], float)
    a = np.polyfit(x, y, 2)
    return float(-a[1] / (2 * a[0])) if a[0] > 0 else float(lams[i])


def cmd_ladder(args):
    df = _load([V21 / "infoladder/budget750.csv",
                V20 / "budget/lam_base.csv",
                V20 / "fieldinfo/budget750.csv"])
    fams = _fams(df)
    rows = []
    for pre, (info, form, comms, term) in CELLS.items():
        if pre not in fams:
            continue
        lams = [l for l, _ in fams[pre]]
        vals = [c.mean() for _, c in fams[pre]]
        i = int(np.argmin(vals))
        rows.append(dict(cell=pre, info=info, form=form, comms="필요" if comms else "불요",
                         load_term=term, n_arms=len(lams),
                         lam_best=lams[i], lam_interp=round(_interp_min(lams, vals), 2),
                         pdr=round(float(vals[i]), 6),
                         grid=f"{min(lams):g}~{max(lams):g}"))
    t = pd.DataFrame(rows).sort_values("pdr").reset_index(drop=True)
    print("=== budget750 (750좌표 × 10시드, 튜닝 전용) — 셀별 최적 ===")
    print(t.to_string(index=False))
    out = V21 / "infoladder/ladder_optima.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out}")

    # 통신 비용: 같은 함수형에서 I3 대비
    print("\n=== 통신 비용 (같은 함수형, I3 기준 paired) ===")
    for form_key, (hi, lows) in {"rate": ("Q", ["OR", "F", "P"]),
                                 "hinge": ("H", ["OH", "FH", "PH"]),
                                 "lin": ("T", ["OL", "FL", "PL"])}.items():
        if hi not in fams:
            continue
        bl = {r["cell"]: r for r in rows}
        a = dict(fams[hi])[bl[hi]["lam_best"]]
        for lo in lows:
            if lo not in fams:
                continue
            b = dict(fams[lo])[bl[lo]["lam_best"]]
            r = paired(a, b)
            print(f"  {form_key:5s} {hi}→{lo:2s} Δ={r['delta']:+.6f} ±{r['ci95']:.6f} "
                  f"W/T/L={r['win']}/{r['tie']}/{r['loss']}  ({CELLS[lo][0]})")


def _judge_arms(smax=29):
    v20 = _load([V20 / "theory/test750_final.csv",
                 V20 / "final30/test750_rules_s10_29.csv"])
    extra = V21 / "final/test750_info_s0_29.csv"
    if (extra.parent / (extra.name + ".meta.json")).exists():
        v20 = pd.concat([v20, pd.read_csv(extra)], ignore_index=True)
    v19p = pd.read_csv(V19 / "ppo_test750.csv")
    v19r = pd.read_csv(V19 / "rules_test750.csv")
    v19c = pd.read_csv(V19 / "cards_test750.csv")
    arms = {}
    for pol in sorted(v20.policy.unique()):
        d = v20[(v20.policy == pol) & (v20.seed <= smax)]
        if d.seed.nunique() < smax + 1:
            continue
        arms[pol] = cube(d, pol)
    for src, pol, name in [(v19p, "NATIONAL", "PPO_NATIONAL"), (v19p, "SIDO", "PPO_SIDO"),
                           (v19r, "FULL64_LB3", "FULL64_LB3"), (v19r, "LB3_AGN", "LB3_AGN"),
                           (v19c, "CARD_GRID_S", "CARD_GRID_S")]:
        arms[name] = cube(src[(src.policy == pol) & (src.seed <= smax)], pol)
    return arms


def cmd_judge(args):
    arms = _judge_arms(args.smax)
    print(f"=== test750 × seed 0-{args.smax} (정책당 {750*(args.smax+1):,}ep) ===")
    tbl = sorted(((k, v.mean()) for k, v in arms.items()), key=lambda x: x[1])
    for k, m in tbl:
        print(f"  {k:16s} {m:.6f}")
    base = args.base
    print(f"\n=== {base} 대비 paired (양수 = {base} 가 좋음) ===")
    a = arms[base]
    for k, _ in tbl:
        if k == base:
            continue
        r = paired(a, arms[k])
        sig = "유의" if abs(r["delta"]) > r["ci95"] else "무의"
        rule = "동률(판정선 0.00053 미만)" if abs(r["delta"]) < 0.00053 else ""
        print(f"  vs {k:16s} Δ={r['delta']:+.6f} ±{r['ci95']:.6f} "
              f"W/T/L={r['win']}/{r['tie']}/{r['loss']} {sig} {rule}")
    if args.out:
        rows = [dict(policy=k, pdr=float(v.mean())) for k, v in arms.items()]
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n→ {args.out}")


def cmd_audit(args):
    """시드 수가 결론을 만들었는가 — 10시드와 30시드 판정을 나란히."""
    v20 = _load([V20 / "theory/test750_final.csv", V20 / "final30/test750_rules_s10_29.csv"])
    v19p = pd.read_csv(V19 / "ppo_test750.csv")
    print("=== 좌표·난수 정합 (v20 seed0-9 vs v19 seed0-9 부분집합) ===")
    v19r = pd.read_csv(V19 / "rules_test750.csv")
    for p20, src, p19 in [("CARD_K12", v19r, "CARD"), ("START_LB3", v19r, "START_LB3")]:
        a = cube(v20[(v20.policy == p20) & (v20.seed <= 9)], p20)
        b = cube(src[(src.policy == p19) & (src.seed <= 9)], p19)
        print(f"  {p20:10s} maxΔ={np.abs(a-b).max():.3e}")
    print("\n=== 시드 수별 마진 (규칙 − 교사) ===")
    for arm in ["CARD_Q18", "CARD_S062", "CARD_P18", "CARD_K12"]:
        for tea, pol in [("PPO_NATIONAL", "NATIONAL"), ("PPO_SIDO", "SIDO")]:
            row = []
            for smax in (9, 29):
                A = cube(v20[(v20.policy == arm) & (v20.seed <= smax)], arm)
                T = cube(v19p[(v19p.policy == pol) & (v19p.seed <= smax)], pol)
                r = paired(A, T)
                row.append(f"s0-{smax}: {r['delta']:+.6f}±{r['ci95']:.6f} "
                           f"({r['win']}/{r['tie']}/{r['loss']})")
            print(f"  {arm:10s} vs {tea:13s} " + "   →   ".join(row))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ladder")
    j = sub.add_parser("judge")
    j.add_argument("--smax", type=int, default=29)
    j.add_argument("--base", default="CARD_Q18")
    j.add_argument("--out", default="")
    sub.add_parser("audit")
    a = p.parse_args()
    {"ladder": cmd_ladder, "judge": cmd_judge, "audit": cmd_audit}[a.cmd](a)


if __name__ == "__main__":
    main()
