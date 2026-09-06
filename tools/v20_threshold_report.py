# -*- coding: utf-8 -*-
"""v20 임계값 일반화 스윕 집계 — 최적점·응답곡선·스케일링 법칙.

`tools/exp_drivers/run_v20_threshold_sweep.sh` 가 만든 설정별 CSV 를 읽어
(1) 설정별·팔가족별 최적 임계값과 U 자 곡선 전체, (2) 파라미터 대비 최적값의 스케일링,
(3) 팔 사이 paired 비교를 낸다.

**규약은 전부 기존 것을 그대로 쓴다** — `tools/v17_fieldcard_report.py` 의
`cube` / `ci` / `wtl`. 특히 승/무/패는 **지역별 에피소드 배열의 95%CI 유의성**이지
지역평균 임계값이 아니다(v5 에서 이 혼동으로 tie 수가 안 맞은 전례).

⚠️ 최적점만 보고하면 안 된다. 이 실험의 결론은 "최적값이 어디로 움직이나" 인데,
곡선이 평탄하면 최적점 이동은 잡음일 수 있다. 그래서 학습 시드 잡음바닥
(0.00114, v12 3시드 실측)보다 나쁘지 않은 격자점 구간을 **평탄대(plateau)** 로 같이 낸다.

사용:
    python tools/v20_threshold_report.py optima  [--sweep DIR] [--out JSON]
    python tools/v20_threshold_report.py scaling [--optima JSON]
    python tools/v20_threshold_report.py paired  --csv F --a POL --b POL
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

SWEEP = REPO / "results/scoreboard/v20/sweep"
OPTIMA = REPO / "results/scoreboard/v20/optima.json"

# v12 3시드 실측 학습 시드 잡음바닥. 이보다 작은 차이는 "구분 못 함" 으로 읽는다.
SEED_NOISE = 0.00114

# 설정 태그 → (축, 수치값). base 는 모든 축의 원점이다.
AXIS_BASE = {"amb": 30.0, "uav": 26.0, "n": 100.0, "vamb": 50.0, "vuav": 200.0,
             "capa": 1.0, "hamb": 5.0, "huav": 10.0}


def parse_tag(tag: str):
    """'vamb100' → ('vamb', 100.0) · 'capa075' → ('capa', 0.75) · 'base' → (None, None)."""
    if tag == "base":
        return None, None
    m = re.match(r"^([a-z]+)([0-9]+)$", tag)
    if not m:
        return None, None
    axis, num = m.group(1), m.group(2)
    if axis == "capa":                      # capa05 → 0.5, capa075 → 0.75, capa20 → 2.0
        val = float(num) / (10.0 ** (len(num) - 1))
    else:
        val = float(num)
    return axis, val


def ci(x) -> float:
    x = np.asarray(x, float)
    return float(1.96 * x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else 0.0


def cube(df: pd.DataFrame, pol: str) -> np.ndarray:
    """지역 × 시드 PDR_woG 행렬. 지역 순서는 CSV 의 region 열로 정렬한다(삽입순 가정 금지)."""
    d = df[df.policy == pol]
    if d.empty:
        raise ValueError(f"{pol} 없음")
    p = d.pivot_table(index="region", columns="seed", values="pdr_woG").sort_index()
    if p.isna().any().any():
        raise ValueError(f"{pol}: 결측 (부분 실행 CSV 를 판정에 쓰지 마라)")
    return p.to_numpy(float)


def wtl(a: np.ndarray, b: np.ndarray):
    """a=후보, b=기준. 지역별 에피소드 배열의 95%CI 로 승/무/패."""
    w = t = l = 0
    for i in range(a.shape[0]):
        dd = b[i] - a[i]
        m, c = dd.mean(), ci(dd)
        w += m > c
        l += m < -c
        t += (-c <= m <= c)
    return w, t, l


def paired(a: np.ndarray, b: np.ndarray):
    """b − a 의 지역평균 차이와 95%CI, 승무패."""
    dd = b.mean(1) - a.mean(1)
    w, t, l = wtl(a, b)
    return {"delta": float(dd.mean()), "ci95": ci(dd),
            "win": int(w), "tie": int(t), "loss": int(l)}


ARM_RE = re.compile(r"^([KTHLQS])([0-9.]+)$")


def load_setting(paths):
    """설정 CSV 들 → {가족: [(lam, cube), ...]}. 완주(meta) 한 파일만 읽는다. 없으면 None.

    한 설정의 팔이 여러 파일에 나뉘어 있을 수 있다(stage1a/b 의 K·T·H 와 나중에 추가한
    stage1L 의 L). 매니페스트·seed0 가 같으면 시나리오 난수가 같으므로 CRN 은 유지된다."""
    if isinstance(paths, Path):
        paths = [paths]
    done = [p for p in paths if (p.parent / (p.name + ".meta.json")).exists()]
    if not done:
        return None
    df = pd.concat([pd.read_csv(p) for p in done], ignore_index=True)
    fams: dict[str, list] = {}
    for pol in sorted(df.policy.unique()):
        m = ARM_RE.match(pol)
        if not m:
            continue
        fams.setdefault(m.group(1), []).append((float(m.group(2)), cube(df, pol)))
    for f in fams:
        fams[f].sort(key=lambda x: x[0])
    return fams


def cmd_optima(args):
    out = {}
    rows = []
    tags: dict[str, list] = {}
    for path in sorted(Path(args.sweep).glob("lam*_*.csv")):
        stem = path.stem
        tag = stem.split("_", 1)[1]
        tags.setdefault(tag, []).append(path)
    for tag, paths in sorted(tags.items()):
        fams = load_setting(paths)
        if fams is None:
            print(f"  [skip] {tag} — 미완주(meta 없음)")
            continue
        axis, val = parse_tag(tag)
        out[tag] = {"axis": axis, "axis_value": val}
        for fam, arms in fams.items():
            lams = np.array([a for a, _ in arms])
            means = np.array([c.mean() for _, c in arms])
            k = int(np.argmin(means))
            best = arms[k][1]
            # 평탄대 = 최적 대비 시드 잡음바닥 안쪽인 격자점
            flat = lams[means <= means[k] + SEED_NOISE]
            # 최적 팔 대비 각 격자점의 paired 열세(설계 확인용, 최적 근방만)
            out[tag][fam] = {
                "lam": float(lams[k]), "pdr": float(means[k]),
                "plateau": [float(flat.min()), float(flat.max())],
                "grid": [[float(a), float(m)] for a, m in zip(lams, means)],
                "n_regions": int(best.shape[0]), "n_seeds": int(best.shape[1]),
            }
            rows.append({"setting": tag, "axis": axis, "axis_value": val, "family": fam,
                         "lam_opt": float(lams[k]), "pdr_opt": float(means[k]),
                         "plateau_lo": float(flat.min()), "plateau_hi": float(flat.max())})
        # 가족 간 최적끼리 paired
        if len(fams) > 1:
            bestc = {f: min(a, key=lambda x: x[1].mean())[1] for f, a in fams.items()}
            ref = "K" if "K" in bestc else sorted(bestc)[0]
            out[tag]["paired_vs_" + ref] = {
                f: paired(c, bestc[ref]) for f, c in bestc.items() if f != ref}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    tab = pd.DataFrame(rows)
    if not tab.empty:
        tab = tab.sort_values(["family", "axis", "axis_value"])
        md = Path(args.out).with_suffix(".md")
        # tabulate 의존 없이 마크다운 표를 직접 만든다(공유 conda 환경을 건드리지 않는다).
        hdr = list(tab.columns)
        lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
        for _, r in tab.iterrows():
            lines.append("| " + " | ".join(
                f"{v:.5f}" if isinstance(v, float) else str(v) for v in r) + " |")
        md.write_text("# v20 설정별 최적 임계값\n\n" + "\n".join(lines)
                      + f"\n\n평탄대 = 최적 대비 {SEED_NOISE} (학습 시드 잡음바닥) 안쪽 격자 구간.\n",
                      encoding="utf-8")
        print(tab.to_string(index=False))
        print(f"\n[optima] {len(out)} 설정 → {args.out} · {md}")
    return out


def cmd_transfer(args):
    """전이 후회 — base 에서 고른 임계값을 다른 조건에 그대로 가져가면 얼마나 손해인가.

    이것이 "임계값이 일반화되는가" 의 직접 측정이다. 각 조건에서 자기 최적을 다시 고르는 것은
    배포 불가(그 조건을 미리 알아야 한다)이므로 오라클이고, 전이값은 실제 배포 가능한 구성이다.
    후회가 학습 시드 잡음바닥(0.00114) 안쪽이면 "그 조건까지 같은 상수로 간다" 고 읽는다."""
    d = json.loads(Path(args.optima).read_text(encoding="utf-8"))
    base = d.get("base")
    if base is None:
        print("base 설정 없음"); return
    rows = []
    for tag, v in sorted(d.items()):
        if tag == "base":
            continue
        for fam in ("K", "T", "H", "L", "Q", "S"):
            if fam not in v or fam not in base:
                continue
            g = {a: m for a, m in v[fam]["grid"]}
            lam_b = base[fam]["lam"]
            if lam_b not in g:                      # 격자에 없으면 가장 가까운 점
                lam_b = min(g, key=lambda a: abs(a - lam_b))
            regret = g[lam_b] - v[fam]["pdr"]
            lo, hi = v[fam]["plateau"]
            rows.append({"setting": tag, "axis": v.get("axis"), "family": fam,
                         "lam_base": base[fam]["lam"], "lam_own": v[fam]["lam"],
                         "pdr_transfer": g[lam_b], "pdr_own": v[fam]["pdr"],
                         "regret": regret,
                         "in_plateau": bool(lo <= base[fam]["lam"] <= hi)})
    t = pd.DataFrame(rows)
    if t.empty:
        print("아직 비교할 설정이 없다"); return
    t = t.sort_values(["family", "setting"])
    for fam, g in t.groupby("family"):
        ok = int(g.in_plateau.sum())
        print(f"\n== {fam} — base 최적 lam={g.lam_base.iloc[0]:g} 를 그대로 전이 ==")
        print(f"   평탄대 안에 드는 조건 {ok}/{len(g)} · 평균 후회 {g.regret.mean():+.5f} "
              f"· 최대 후회 {g.regret.max():+.5f} ({g.loc[g.regret.idxmax(),'setting']})")
        for _, r in g.iterrows():
            flag = "  " if r.in_plateau else "★"
            print(f"   {flag} {r.setting:9s} 자기최적 {r.lam_own:5g} → 전이후회 {r.regret:+.5f}"
                  f"  (자기 {r.pdr_own:.5f} / 전이 {r.pdr_transfer:.5f})")
    print(f"\n판단선: 후회 < {SEED_NOISE} (학습 시드 잡음바닥) 이면 같은 상수로 배포 가능.")
    t.to_csv(Path(args.optima).with_name("transfer.csv"), index=False)
    print(f"저장 {Path(args.optima).with_name('transfer.csv')}")


def cmd_scaling(args):
    d = json.loads(Path(args.optima).read_text(encoding="utf-8"))
    base = d.get("base")
    if base is None:
        print("base 설정이 아직 없다 — 스케일링은 base 대비 비율로 읽는다."); return
    print("== 최적 임계값의 파라미터 스케일링 ==")
    print("   K = 거리(km)축 CARD · T = 시간(분)축 CARD-T · H = 시간축 + 대기행렬 hinge\n")
    for fam in ("K", "T", "H", "L", "Q", "S"):
        if fam not in base:
            continue
        b = base[fam]["lam"]
        by_axis: dict[str, list] = {}
        for tag, v in d.items():
            if tag == "base" or fam not in v or not v.get("axis"):
                continue
            by_axis.setdefault(v["axis"], []).append((v["axis_value"], v[fam]["lam"],
                                                      v[fam]["plateau"]))
        for axis, pts in sorted(by_axis.items()):
            pts.sort()
            x = np.array([AXIS_BASE.get(axis, 1.0)] + [p[0] for p in pts], float)
            y = np.array([b] + [p[1] for p in pts], float)
            ok = (x > 0) & (y > 0)
            slope = np.nan
            if ok.sum() >= 3:
                slope = float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])
            desc = " ".join(f"{p[0]:g}:{p[1]:g}[{p[2][0]:g}-{p[2][1]:g}]" for p in pts)
            print(f" {fam} lam* vs {axis:5s} (base {AXIS_BASE.get(axis)}→{b:g}) "
                  f"log-log 기울기 {slope:+.3f} | {desc}")
    print("\n예측(이론): K 는 v_amb 에 비례(기울기 +1), T·H 는 v_amb 에 무관(0),"
          " T·H 는 capa 에 반비례(−1), 셋 다 N·자원수에는 1차로 무관.")


def cmd_paired(args):
    df = pd.read_csv(args.csv)
    a, b = cube(df, args.a), cube(df, args.b)
    r = paired(a, b)
    print(f"{args.a} {a.mean():.6f} vs {args.b} {b.mean():.6f} — "
          f"{args.b}−{args.a} = {r['delta']:+.6f} ± {r['ci95']:.6f} "
          f"({r['win']}승 {r['tie']}무 {r['loss']}패, n={a.shape[0]})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("optima"); o.add_argument("--sweep", default=str(SWEEP))
    o.add_argument("--out", default=str(OPTIMA)); o.set_defaults(fn=cmd_optima)
    s = sub.add_parser("scaling"); s.add_argument("--optima", default=str(OPTIMA))
    s.set_defaults(fn=cmd_scaling)
    tr = sub.add_parser("transfer"); tr.add_argument("--optima", default=str(OPTIMA))
    tr.set_defaults(fn=cmd_transfer)
    p = sub.add_parser("paired"); p.add_argument("--csv", required=True)
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.set_defaults(fn=cmd_paired)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
