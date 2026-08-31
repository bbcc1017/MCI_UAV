# -*- coding: utf-8 -*-
"""v18 E4 — 지역화 사다리 L0/L1/L2 파라미터표 생성과 판정 (사전등록 e4_localize_prereg.json).

핵심 설계: **train1000 위 λ×red_km 스윕 한 번**에서 모든 단계가 나온다. 지역이 정확히
한 층에만 속하므로 같은 스윕 데이터로 전국 argmin(L0) · 지형층별 argmin(L1) ·
시군구별 argmin(L2) · leave-province-out 이 전부 도출된다.

  build   스윕 CSV → 파라미터표 JSON (cardloc 이 소비하는 `{"_default":..., "<sigcd>":...}`)
  judge   평가셋 CSV → L0/L1/L2 대 기준선 paired 판정

⚠️ 누수 통제: 지형층 경계는 `scenarios/manifests/sigungu250/_index.json` 의 **학습좌표**
특징에서만 잡는다. v17 의 지형 상관은 대표점250 에서 계산된 값이라 여기 쓰지 않는다.
⚠️ L2 는 자기 학습좌표에서 argmin 을 고르므로 선택편향을 안는다 → train 성능은 보고하지
않고 **전이 성능만** 보고한다.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
INDEX = REPO / "scenarios/manifests/sigungu250/_index.json"
_SPEC = re.compile(r"^L(?P<lam>[\d.]+)_R(?P<red>[\d.]+)$")


def ci(x):
    x = np.asarray(x, float)
    return float(1.96 * x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else 0.0


def sigcd_of(key: str) -> str:
    for tok in reversed(str(key).split("_")):
        if tok.isdigit() and len(tok) == 5:
            return tok
    raise ValueError(key)


def load_sweep(path: Path) -> pd.DataFrame:
    """→ (sigcd, lam, red, pdr) 지역별 평균. 지역당 4좌표 × 30seed 를 평균한다."""
    d = pd.read_csv(path)
    m = d.policy.str.extract(_SPEC)
    if m.isna().any().any():
        bad = sorted(set(d.policy[m.isna().any(axis=1)]))[:3]
        raise ValueError(f"팔 이름 파싱 실패: {bad}")
    d["lam"] = m["lam"].astype(float)
    d["red"] = m["red"].astype(float)
    d["sigcd"] = d.region.map(sigcd_of)
    # 좌표 단위 평균 → 지역 단위 평균 (좌표 수가 같아 단순평균과 동일하나 명시)
    per_coord = d.groupby(["sigcd", "region", "lam", "red"]).pdr_woG.mean().reset_index()
    return per_coord.groupby(["sigcd", "lam", "red"]).pdr_woG.mean().reset_index()


def argmin_over(df: pd.DataFrame) -> dict:
    g = df.groupby(["lam", "red"]).pdr_woG.mean()
    (lam, red) = g.idxmin()
    return {"lam": float(lam), "red_km": float(red), "yhold": 0.0,
            "fit_pdr": float(g.min()), "n_sigcd": int(df.sigcd.nunique())}


def terrain_bins(idx: dict, n_bins: int = 4) -> tuple[dict, list]:
    """near_t3 (학습좌표 중앙값) 분위로 층 부여. 반환 (sigcd→bin, 경계)."""
    regs = idx["regions"]
    rows = [(sigcd_of(r), v["near_t3"]) for r, v in regs.items()]
    vals = np.array([v for _, v in rows], float)
    edges = list(np.nanquantile(vals, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return {s: int(np.searchsorted(edges, v, side="right")) for s, v in rows}, edges


def build(args) -> None:
    sw = load_sweep(Path(args.sweep))
    idx = json.load(open(args.index, encoding="utf-8"))
    bins, edges = terrain_bins(idx, args.n_bins)
    sw["bin"] = sw.sigcd.map(bins)
    sido = {sigcd_of(r): v["stratum"] and idx["regions"][r].get("sido") for r, v in idx["regions"].items()}
    # _index.json 에 sido 가 없으면 매니페스트 points 에서 받아온다
    if any(v is None for v in sido.values()):
        pts = json.load(open(REPO / "scenarios/manifests/sigungu_osrm_train1000_random4_points.json",
                             encoding="utf-8"))
        sido = {sigcd_of(k): v["sido"] for k, v in pts.items()}
    sw["sido"] = sw.sigcd.map(sido)

    out = {"generated": "v18_localize_report.build", "sweep": str(Path(args.sweep).resolve()),
           "n_sigcd": int(sw.sigcd.nunique()), "n_arms": int(len(sw.groupby(["lam", "red"]))),
           "terrain_edges_near_t3_km": [round(float(e), 3) for e in edges],
           "leakage_note": "층 경계·argmin 전부 train1000 학습좌표에서만 도출"}

    L0 = argmin_over(sw)
    out["L0"] = L0
    print(f"[L0] 전국 단일 λ={L0['lam']:g} red_km={L0['red_km']:g}  train PDR {L0['fit_pdr']:.6f}")

    L1 = {}
    for b, gg in sw.groupby("bin"):
        L1[int(b)] = argmin_over(gg)
    out["L1_by_bin"] = L1
    print(f"[L1] 지형층 경계(near_t3 km) {[round(e,1) for e in edges]}")
    for b in sorted(L1):
        v = L1[b]
        print(f"     bin{b}: λ={v['lam']:g} red={v['red_km']:g}  n={v['n_sigcd']}  fit {v['fit_pdr']:.6f}")

    L2 = {}
    for s, gg in sw.groupby("sigcd"):
        L2[s] = argmin_over(gg)
    out["L2_n"] = len(L2)
    lam2 = np.array([v["lam"] for v in L2.values()])
    red2 = np.array([v["red_km"] for v in L2.values()])
    print(f"[L2] 시군구 {len(L2)}곳 · λ 중위 {np.median(lam2):g} "
          f"[{np.percentile(lam2,25):g}, {np.percentile(lam2,75):g}] · "
          f"red 중위 {np.median(red2):g}")
    out["L2_lambda_quartiles"] = [float(np.percentile(lam2, q)) for q in (25, 50, 75)]

    # ---- cardloc 파라미터표 3종 ----
    P = Path(args.out_dir); P.mkdir(parents=True, exist_ok=True)
    base = {"lam": L0["lam"], "red_km": L0["red_km"], "yhold": 0.0}
    (P / "params_L0.json").write_text(json.dumps({"_default": base}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    p1 = {"_default": base}
    for s, b in bins.items():
        v = L1[int(b)]
        p1[s] = {"lam": v["lam"], "red_km": v["red_km"], "yhold": 0.0}
    (P / "params_L1.json").write_text(json.dumps(p1, ensure_ascii=False, indent=1), encoding="utf-8")
    p2 = {"_default": base}
    for s, v in L2.items():
        p2[s] = {"lam": v["lam"], "red_km": v["red_km"], "yhold": 0.0}
    (P / "params_L2.json").write_text(json.dumps(p2, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- leave-province-out: 시도 하나 빼고 L0/L1 재적합 → 그 시도용 표 ----
    lpo0, lpo1 = {"_default": base}, {"_default": base}
    for sd, gg in sw.groupby("sido"):
        rest = sw[sw.sido != sd]
        v0 = argmin_over(rest)
        for s in gg.sigcd.unique():
            lpo0[s] = {"lam": v0["lam"], "red_km": v0["red_km"], "yhold": 0.0}
            b = bins[s]
            sub = rest[rest.bin == b]
            v1 = argmin_over(sub) if len(sub) else v0
            lpo1[s] = {"lam": v1["lam"], "red_km": v1["red_km"], "yhold": 0.0}
    (P / "params_LPO_L0.json").write_text(json.dumps(lpo0, ensure_ascii=False, indent=1), encoding="utf-8")
    (P / "params_LPO_L1.json").write_text(json.dumps(lpo1, ensure_ascii=False, indent=1), encoding="utf-8")
    n_sd = sw.sido.nunique()
    print(f"[LPO] 시도 {n_sd}개 fold — 각 시도를 빼고 적합한 표 2종 기록")
    out["n_sido_folds"] = int(n_sd)

    (P / "ladder_build.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[build] → {P}/params_{{L0,L1,L2,LPO_L0,LPO_L1}}.json · ladder_build.json")


def judge(args) -> None:
    import sys
    sys.path.insert(0, str(REPO / "tools"))
    from v17_fieldcard_report import cube, wtl

    base_cube = cube(args.baseline_csv, args.baseline_policy)
    print(f"기준선 {args.baseline_policy}: {base_cube.mean(1).mean():.6f}")
    rec = {"baseline": {"policy": args.baseline_policy,
                        "pdr": round(float(base_cube.mean(1).mean()), 6)}, "arms": {}}
    for spec in args.arms.split(","):
        name, csvp = spec.split("=", 1)
        X = cube(csvp, name)
        d = X.mean(1) - base_cube.mean(1)
        w, t, l = wtl(X, base_cube)
        sig = abs(d.mean()) > ci(d)
        print(f"  {name:14s} {X.mean(1).mean():.6f}  Δ {d.mean():+.6f} ± {ci(d):.6f}"
              f"  기준선 승/무/패 {w}/{t}/{l}  {'유의' if sig else '동률'}")
        rec["arms"][name] = {"pdr": round(float(X.mean(1).mean()), 6),
                             "delta_vs_baseline": round(float(d.mean()), 6),
                             "ci95": round(float(ci(d)), 6),
                             "baseline_wtl": [int(w), int(t), int(l)], "significant": bool(sig)}
    Path(args.out).write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[judge] → {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sweep", default=str(REPO / "results/scoreboard/v18/e4_sweep_train1000.csv"))
    b.add_argument("--index", default=str(INDEX))
    b.add_argument("--n_bins", type=int, default=4)
    b.add_argument("--out_dir", default=str(REPO / "results/scoreboard/v18/ladder"))
    j = sub.add_parser("judge")
    j.add_argument("--baseline_csv", required=True)
    j.add_argument("--baseline_policy", default="CARD")
    j.add_argument("--arms", required=True, help="이름=csv 경로, 쉼표 구분")
    j.add_argument("--out", required=True)
    a = ap.parse_args()
    {"build": build, "judge": judge}[a.cmd](a)


if __name__ == "__main__":
    main()
