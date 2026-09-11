# -*- coding: utf-8 -*-
"""v22 E8 — 입력 오추정 강건성. 현장 지휘관이 넣는 숫자가 틀렸을 때 공식 카드가 얼마나 남는가.

시뮬레이션을 **재실행하지 않는다**. 이미 완주된 budget750 에피소드 CSV 만 읽어 paired 통계를 낸다.
판정 커널(`ci`/`cube`/`paired`/`_interp_opt`)은 `tools/v20_threshold_report.py` 를 그대로 import 한다.
조건 태그 → 축값 환산(`_axis_value`)은 `tools/v22_field_report.py` 를 import 한다
(`ts05`=0.5 · `ts075`=0.75 · `ts40`=4.0 — 직접 `float()` 하면 조용히 10배 틀린다).

**왜 이 측정이 필요한가.** v22 정본은 `λ = 18.97 × 치료시간배수` 공식 한 줄이 봉투 밖 손실의
96.6% 를 에피소드 0개로 회수한다고 말한다. 그런데 그 공식의 입력인 "치료 회전율" 을 현장
지휘관은 정확히 모른다. 이 리포트는 그 입력을 배율 `e` 만큼 틀렸을 때
① 조건 최적 팔 대비 regret 과 ② 고정 카드(`Q18`) 대비 개선이 **어디서 부호가 뒤집히는가**를 잰다.

서브커맨드
  grid       오추정 격자. 참 조건 ts_true × 추정 배율 e 의 모든 셀에서 세 전략을
             (고정 카드 / 공식+오차 / 무튜닝 S족) 같은 표에 놓는다. 주 산출.
  replicate  같은 격자를 `budgetcurve` eval CSV(60좌표 × **서로소 평가시드 1000..1059**)로
             재현한다. 공식 계수 18.97 이 seed0..9 에서 뽑혔으므로 이 재현이 표본외 검증이다.
  nmisspec   환자 수 N 오추정. N 에는 공식이 없으므로 격자가 아니라 **전이 행렬**을 낸다
             (λ 는 N=100↔500, 등급 임계는 N=50/100/200). 정본 예측은 null 이다.

규약
  * 판정선 0.00053 = v21 실측 CRN paired 두 팔 차이 95%CI. 새로 만들지 않고 이 값을 쓰되
    셀마다 실제 paired 95%CI 를 항상 병기한다. `regret < ci95` 면 그건 곡선의 바닥이다.
  * Δ 부호는 기존 규약 그대로 `paired(cand, base)` → `delta = base − cand` (양수 = base 열세).
  * λ_hat → 격자 팔 반올림은 **로그 거리** 기본(`--round log`). 어느 팔로 반올림됐는지,
    그리고 격자 절단(λ_hat 이 격자 밖)인지를 셀마다 기록한다. 절단 셀의 regret 은
    **하한**이다(공식이 실제로 지시한 팔보다 덜 틀린 팔을 평가하므로).
  * `.meta.json` 없는 CSV = 미완주 → 즉시 중단. `cube()` 는 결측에 예외를 던지므로 완주 검사를 겸한다.
  * 한 조건을 여러 CSV 로 합칠 때 공통 팔의 **비트동일(CRN)** 을 검사한다.

사용:
  python tools/v22_misspec_report.py grid      --out results/field/misspec/treat_misspec.json
  python tools/v22_misspec_report.py replicate --out results/field/misspec/treat_misspec_oos.json
  python tools/v22_misspec_report.py nmisspec  --out results/field/misspec/n_misspec.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from v20_threshold_report import _interp_opt, ci, cube, paired  # noqa: E402

try:
    from v22_field_report import _axis_value  # noqa: E402
except Exception as _e:                                             # pragma: no cover
    raise SystemExit(
        f"[치명] tools/v22_field_report.py 의 _axis_value 를 import 못 했다 ({_e}). "
        "다른 에이전트가 그 파일을 편집 중일 수 있다 — 환산 규약을 복제하지 말고 다시 실행하라.")

# v21 실측 CRN paired 판정선. 보편 상수가 아니라 그 조건의 실측값이므로 셀 CI 를 항상 병기한다.
JUDGE_LINE = 0.00053
# v22 정본 공식: λ = LAM_COEF × 치료시간배수 (budget750 750좌표 × seed0..9, Q족 hingerate 보간 최적).
LAM_COEF = 18.97

DEF_STAGE = "results/scoreboard/v22/retune"
DEF_ANCHOR = ("1.0=results/scoreboard/v20/budget/lam_base.csv,"
              "results/scoreboard/v22/retune/base_base.csv")
DEF_ERRS = "0.5,0.7,0.85,1.0,1.2,1.5,2.0"
DEF_BC = ("1.0=results/scoreboard/v22/budgetcurve/base_eval.csv;"
          "4.0=results/scoreboard/v22/budgetcurve/ts40_eval.csv")
DEF_N_LAM = ("100=results/scoreboard/v20/budget/lam_base.csv,"
             "results/scoreboard/v22/retune/base_base.csv;"
             "500=results/scoreboard/v22/retune/extreme_n500.csv")
DEF_N_Y = ("50=results/scoreboard/v22/retune/yhold_n50.csv;"
           "100=results/scoreboard/v22/retune/yhold_base.csv;"
           "200=results/scoreboard/v22/retune/yhold_n200.csv")


# --------------------------------------------------------------------------- 출력 tee
class Out:
    """표를 화면과 텍스트 파일에 동시에 쓴다(사람이 읽을 표를 산출물로 남기기 위해)."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, s: str = "") -> None:
        print(s)
        self.lines.append(s)

    def save(self, path: str | None) -> None:
        if not path:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        print(f"[기록] {path}")


def _verdict(delta: float, ci95: float) -> str:
    """양수 delta = 기준 팔이 열세. 판정선과 셀 CI 를 둘 다 통과해야 '초과'다."""
    if abs(delta) <= ci95:
        return "동률"
    if delta > JUDGE_LINE:
        return "★초과"
    if delta < -JUDGE_LINE:
        return "☆역전"
    return "유의" if delta > 0 else "유의(-)"


# --------------------------------------------------------------------------- 로딩
def _read_one(path: str) -> tuple[pd.DataFrame, dict]:
    """완주 CSV 하나. 메타 부재·행수 불일치·(region,policy,seed) 중복은 전부 치명이다."""
    meta = path + ".meta.json"
    if not os.path.exists(meta):
        raise SystemExit(f"[치명] 미완주 CSV (.meta.json 없음): {path}")
    m = json.load(open(meta, encoding="utf-8"))
    df = pd.read_csv(path, encoding="utf-8-sig")
    dup = int(df.duplicated(["region", "policy", "seed"]).sum())
    if dup:
        raise SystemExit(f"[치명] {path}: (region,policy,seed) 중복 {dup} 행 — cube() 가 조용히 평균낸다")
    nr = m.get("n_rows")
    if nr is not None and int(nr) != len(df):
        raise SystemExit(f"[치명] {path}: 메타 n_rows={nr} ≠ 실제 {len(df)} 행 (미완주)")
    if not np.isfinite(df.pdr_woG.to_numpy(float)).all():
        raise SystemExit(f"[치명] {path}: pdr_woG 에 비유한값")
    return df, m


def _load_cond(paths: list[str]) -> dict:
    """한 물리조건 = CSV 여러 개의 팔 합집합. 공통 팔은 비트동일(CRN)을 검사한다."""
    arms: dict[str, np.ndarray] = {}
    src: dict[str, str] = {}
    mans: set[str] = set()
    knobs: dict[str, str] = {}
    regions: list[str] | None = None
    seeds: set[int] = set()
    for p in paths:
        df, m = _read_one(p)
        mans.add(os.path.basename(str(m.get("manifest", "?"))))
        knobs.update({k.replace("MCI_", ""): str(v)
                      for k, v in (m.get("scenario_knobs") or {}).items() if v})
        seeds |= set(int(s) for s in df.seed.unique())
        rg = sorted(df.region.unique())
        if regions is None:
            regions = rg
        elif regions != rg:
            raise SystemExit(f"[치명] {p}: 좌표 집합이 앞 파일과 다르다 — 합치면 paired 가 깨진다")
        for a in sorted(df.policy.unique()):
            c = cube(df, a)               # 결측이면 예외 = 완주 검사
            if a in arms:
                if arms[a].shape != c.shape or not np.array_equal(arms[a], c):
                    raise SystemExit(f"[치명] 팔 {a} 가 {src[a]} 와 {p} 에서 다르다 (CRN 깨짐)")
                continue
            arms[a], src[a] = c, p
    if len(mans) > 1:
        raise SystemExit(f"[치명] 좌표셋이 섞였다: {sorted(mans)}")
    man = mans.pop() if mans else "<메타 없음>"
    if any(bad in man for bad in ("test750", "tradeoff250", "eval250")):
        raise SystemExit(f"[치명] 판정 금지 좌표셋: {man} (budget750 만 쓴다)")
    return {"arms": arms, "src": src, "files": list(paths), "manifest": man,
            "knobs": knobs, "n_region": len(regions or []), "seeds": sorted(seeds)}


def _parse_conds(spec: str) -> dict[float, list[str]]:
    """'1.0=a.csv,b.csv;4.0=c.csv' → {1.0: [a,b], 4.0: [c]}."""
    out: dict[float, list[str]] = {}
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(f"[치명] 조건 스펙 형식은 '값=경로[,경로]' 다: {chunk}")
        k, v = chunk.split("=", 1)
        out[float(k)] = [str(REPO / p) if not os.path.isabs(p) else p
                         for p in (q.strip() for q in v.split(",")) if p]
    return out


def _discover(stage_dir: str, prefix: str, token: str, scale: str) -> dict[float, list[str]]:
    """`treat_ts40.csv` 같은 스테이지 파일을 축값으로 색인한다."""
    out: dict[float, list[str]] = {}
    for f in sorted(glob.glob(os.path.join(stage_dir, f"{prefix}_*.csv"))):
        tag = os.path.basename(f)[len(prefix) + 1: -4]
        out.setdefault(_axis_value(tag, token, scale), []).append(f)
    return out


# --------------------------------------------------------------------------- 격자 코어
def _fam_arms(arms: dict, fam: str, suffix: str) -> list[tuple[float, str]]:
    pat = re.compile(re.escape(fam) + r"([0-9.]+)" + re.escape(suffix) + r"$")
    got = [(float(m.group(1)), a) for a in arms if (m := pat.fullmatch(a))]
    return sorted(got)


def _snap(lams: np.ndarray, target: float, mode: str) -> int:
    """λ_hat 을 격자 팔로 반올림. 기본은 로그 거리(비율 격자라 로그가 옳다)."""
    if mode == "log":
        d = np.abs(np.log(lams) - math.log(target))
    elif mode == "linear":
        d = np.abs(lams - target)
    else:
        raise SystemExit(f"[치명] --round 는 log|linear: {mode}")
    return int(d.argmin())


def _band(errs: list[float], vals: list[float], thr: float) -> dict:
    """e=1 을 포함하는 연속 허용구간 [lo,hi]: vals<=thr. 경계는 log e 선형보간(=추정)."""
    i1 = int(np.argmin([abs(math.log(e)) for e in errs]))
    if vals[i1] > thr:
        return {"lo": None, "hi": None, "note": "e=1 셀이 이미 임계 초과 — 허용구간 없음"}

    def walk(step: int):
        i = i1
        while 0 <= i + step < len(errs) and vals[i + step] <= thr:
            i += step
        j = i + step
        if not (0 <= j < len(errs)):
            return errs[i], True                      # 격자 끝까지 통과 = 절단
        x0, x1 = math.log(errs[i]), math.log(errs[j])
        y0, y1 = vals[i], vals[j]
        t = 0.5 if y1 == y0 else (thr - y0) / (y1 - y0)
        return math.exp(x0 + max(0.0, min(1.0, t)) * (x1 - x0)), False

    lo, lo_edge = walk(-1)
    hi, hi_edge = walk(+1)
    return {"lo": float(lo), "hi": float(hi), "lo_grid_edge": lo_edge, "hi_grid_edge": hi_edge,
            "note": "경계는 보간 추정" + (" · 격자 절단 있음" if (lo_edge or hi_edge) else "")}


def _grid_analyze(conds: dict[float, dict], errs: list[float], fam: str, suffix: str,
                  coef: float, fixed: str, untuned: str, round_mode: str) -> dict:
    res = {"judge_line": JUDGE_LINE, "lam_coef": coef, "family": fam + "*" + suffix,
           "fixed_arm": fixed, "untuned_arm": untuned or None,
           "round": round_mode, "err_grid": errs, "conditions": []}
    for tv in sorted(conds):
        cd = conds[tv]
        arms = cd["arms"]
        fa = _fam_arms(arms, fam, suffix)
        if len(fa) < 3:
            raise SystemExit(f"[치명] ts={tv}: {fam}족 팔이 {len(fa)}개 — 격자가 없다")
        lams = np.array([v for v, _ in fa], float)
        names = [n for _, n in fa]
        means = np.array([arms[n].mean() for n in names], float)
        ib = int(means.argmin())
        best, bcube = names[ib], arms[names[ib]]
        lam_star = _interp_opt(lams.tolist(), means.tolist())

        # 조건 최적과 통계적 동률인 팔 = 응답곡선의 바닥 폭. argmin 하나만 보면 격자 잡음을 최적으로 읽는다.
        tie = []
        for lv, n in fa:
            r = paired(bcube, arms[n])
            if abs(r["delta"]) <= r["ci95"]:
                tie.append(n)

        if fixed not in arms:
            raise SystemExit(f"[치명] ts={tv}: 고정 팔 {fixed} 없음")
        fcube = arms[fixed]
        ucube = arms.get(untuned) if untuned else None

        rec = {
            "axis_value": tv, "manifest": cd["manifest"], "knobs": cd["knobs"],
            "n_region": cd["n_region"], "n_seed": len(cd["seeds"]),
            "seeds": [cd["seeds"][0], cd["seeds"][-1]] if cd["seeds"] else [],
            "files": [os.path.relpath(p, REPO) for p in cd["files"]],
            "grid": [[float(lv), n, float(arms[n].mean())] for lv, n in fa],
            "best_arm": best, "best_pdr": float(means[ib]),
            "best_grid_edge": ib in (0, len(fa) - 1),
            "lam_star_interp": float(lam_star),
            "tie_with_best": tie,
            "lam_formula_exact": float(coef * tv),
            "formula_over_lamstar": float(coef * tv / lam_star),
            "fixed": {"arm": fixed, "pdr": float(fcube.mean()),
                      **paired(bcube, fcube)},
            "cells": [],
        }
        rec["fixed"]["verdict"] = _verdict(rec["fixed"]["delta"], rec["fixed"]["ci95"])
        # 정본 ④ 는 "전 팔족 최적" 정의를 쓴다. 족을 Q 로 묶은 이 리포트의 수치와 갈리므로 함께 낸다.
        ab = min(arms, key=lambda n: arms[n].mean())
        rec["all_family_best"] = {"arm": ab, "pdr": float(arms[ab].mean()),
                                  **paired(arms[ab], fcube)}
        if ucube is not None:
            reg = paired(bcube, ucube)
            imp = paired(ucube, fcube)
            rec["untuned"] = {"arm": untuned, "pdr": float(ucube.mean()),
                              "regret": reg, "regret_verdict": _verdict(reg["delta"], reg["ci95"]),
                              "vs_fixed": imp, "vs_fixed_verdict": _verdict(imp["delta"], imp["ci95"])}

        for e in errs:
            ts_hat = tv * e
            lam_hat = coef * ts_hat
            i = _snap(lams, lam_hat, round_mode)
            arm = names[i]
            clip = "high" if lam_hat > lams[-1] else ("low" if lam_hat < lams[0] else "")
            reg = paired(bcube, arms[arm])          # delta = 선택 팔 − 조건 최적 (양수 = 손실)
            imp = paired(arms[arm], fcube)          # delta = 고정 카드 − 선택 팔 (양수 = 공식 승)
            rec["cells"].append({
                "err": e, "ts_hat": ts_hat, "lam_hat": lam_hat,
                "snapped_arm": arm, "snapped_lam": float(lams[i]),
                "snap_ratio": float(lams[i] / lam_hat),
                "clip": clip, "same_as_fixed": arm == fixed,
                "regret": reg, "regret_verdict": _verdict(reg["delta"], reg["ci95"]),
                "vs_fixed": imp, "vs_fixed_verdict": _verdict(imp["delta"], imp["ci95"]),
            })
        rec["tolerance_band_judge"] = _band(errs, [c["regret"]["delta"] for c in rec["cells"]],
                                            JUDGE_LINE)
        res["conditions"].append(rec)
    return res


def _print_grid(res: dict, o: Out, title: str) -> None:
    o("=" * 118)
    o(title)
    o(f"  공식 λ = {res['lam_coef']} × 치료시간배수 → {res['round']} 거리로 {res['family']} 격자에 반올림")
    o(f"  regret  = paired(조건최적, 선택팔)  delta = 선택 − 최적 (양수 = 공식이 손실)")
    o(f"  Q18대비 = paired(선택팔, {res['fixed_arm']})  delta = 고정 − 선택 (양수 = 공식이 이김)")
    o(f"  판정선 {JUDGE_LINE} · ★초과=판정선+CI 통과 · 유의=CI만 통과 · 동률=|delta|<=CI · ☆역전=고정 카드 승")
    o("=" * 118)
    for rec in res["conditions"]:
        o("")
        o(f"[ts_true={rec['axis_value']}]  좌표셋 {rec['manifest']} · {rec['n_region']}좌표 × "
          f"{rec['n_seed']}시드(seed {rec['seeds'][0]}..{rec['seeds'][-1]}) · 팔 {len(rec['grid'])}개 "
          f"· 노브 {rec['knobs'] or '{}'}")
        o(f"  조건최적 {rec['best_arm']} {rec['best_pdr']:.6f}"
          f"{'  ⚠️격자끝(진짜 최적은 격자 밖)' if rec['best_grid_edge'] else ''}"
          f" · 보간 λ*={rec['lam_star_interp']:.2f} · 최적과 동률인 팔 {rec['tie_with_best']}")
        fx = rec["fixed"]
        o(f"  고정 카드 {fx['arm']} {fx['pdr']:.6f} → 손실 +{fx['delta']:.5f} ±{fx['ci95']:.5f} "
          f"(W/T/L {fx['win']}/{fx['tie']}/{fx['loss']}) {fx['verdict']}")
        ab = rec["all_family_best"]
        o(f"  참고) 전 팔족 최적 {ab['arm']} {ab['pdr']:.6f} → 고정 대비 +{ab['delta']:.5f} "
          f"±{ab['ci95']:.5f} (W/T/L {ab['win']}/{ab['tie']}/{ab['loss']}) = 정본 ④ 의 정의")
        if "untuned" in rec:
            u = rec["untuned"]
            o(f"  무튜닝 {u['arm']} {u['pdr']:.6f} → regret +{u['regret']['delta']:.5f} "
              f"±{u['regret']['ci95']:.5f} {u['regret_verdict']} · 고정대비 "
              f"{u['vs_fixed']['delta']:+.5f} ±{u['vs_fixed']['ci95']:.5f} "
              f"(W/T/L {u['vs_fixed']['win']}/{u['vs_fixed']['tie']}/{u['vs_fixed']['loss']}) "
              f"{u['vs_fixed_verdict']}")
        o(f"  공식 λ(오차없음)={rec['lam_formula_exact']:.2f} = 보간 최적의 "
          f"{rec['formula_over_lamstar']:.3f}배")
        o("     e   ts_hat   λ_hat    팔    snap배율 절단 |   regret ±CI            판정 |"
          "   고정대비 ±CI            W/T/L        판정")
        for c in rec["cells"]:
            o(f"  {c['err']:5.2f} {c['ts_hat']:7.2f} {c['lam_hat']:7.1f} {c['snapped_arm']:>6s}"
              f"  {c['snap_ratio']:7.3f} {c['clip'] or '-':>4s} |"
              f" +{c['regret']['delta']:.5f} ±{c['regret']['ci95']:.5f} {c['regret_verdict']:>7s} |"
              f" {c['vs_fixed']['delta']:+.5f} ±{c['vs_fixed']['ci95']:.5f}"
              f" {c['vs_fixed']['win']:>4d}/{c['vs_fixed']['tie']:>3d}/{c['vs_fixed']['loss']:>3d}"
              f" {c['vs_fixed_verdict']:>8s}"
              + ("  (= 고정 카드와 같은 팔)" if c["same_as_fixed"] else ""))
        b = rec["tolerance_band_judge"]
        if b["lo"] is None:
            o(f"  허용 오차구간(regret <= 판정선): 없음 — {b['note']}")
        else:
            o(f"  허용 오차구간(regret <= 판정선 {JUDGE_LINE}): e ∈ [{b['lo']:.2f}, {b['hi']:.2f}]"
              f"  ({b['note']})")


def _print_matrix(res: dict, o: Out) -> None:
    errs = res["err_grid"]
    for key, lab in (("regret", "regret (선택 − 조건최적, 작을수록 좋다)"),
                     ("vs_fixed", f"고정 {res['fixed_arm']} 대비 개선 (양수 = 공식 승)")):
        o("")
        o("-" * 118)
        o(f"요약 행렬 — {lab}")
        o("-" * 118)
        o("ts_true |" + "".join(f"{e:>11.2f}" for e in errs) + "   |    무튜닝")
        for rec in res["conditions"]:
            cells = "".join(f"{c[key]['delta']:>11.5f}" for c in rec["cells"])
            u = rec.get("untuned")
            uv = (u["regret"]["delta"] if key == "regret" else u["vs_fixed"]["delta"]) if u else None
            o(f"{rec['axis_value']:7} |{cells}   | " + (f"{uv:+.5f}" if uv is not None else "-"))
    o("")
    o("-" * 118)
    o("반올림된 팔 (h = 격자 절단: λ_hat 이 격자 밖 → regret 은 하한이다)")
    o("-" * 118)
    o("ts_true |" + "".join(f"{e:>11.2f}" for e in errs))
    for rec in res["conditions"]:
        o(f"{rec['axis_value']:7} |" + "".join(
            f"{c['snapped_arm'] + ('h' if c['clip'] else ''):>11s}" for c in rec["cells"]))

    o("")
    o("-" * 118)
    o("부호 뒤집힘 — 공식이 고정 카드보다 나빠지는 e (☆역전 = 판정선 넘어 고정 카드 승)")
    o("-" * 118)
    flips = []
    for rec in res["conditions"]:
        good = [c["err"] for c in rec["cells"] if c["vs_fixed"]["delta"] > c["vs_fixed"]["ci95"]]
        bad = [c["err"] for c in rec["cells"] if c["vs_fixed"]["delta"] < -c["vs_fixed"]["ci95"]]
        hard = [c["err"] for c in rec["cells"] if c["vs_fixed_verdict"] == "☆역전"]
        flips.append({"axis_value": rec["axis_value"], "formula_wins": good,
                      "formula_loses": bad, "formula_loses_beyond_judge": hard})
        o(f"  ts_true={rec['axis_value']:<5} 공식 유의승 e={good or '-'} · 유의패 e={bad or '-'}"
          f" · 판정선 넘어 패 e={hard or '-'}")
    res["flips"] = flips

    # 예측 규칙: 공식은 "추정이 배율 1 보다 참값에 로그거리로 더 가까울 때" 고정 카드를 이긴다.
    # 셀마다 예측과 관측 부호를 대조해 규칙의 적중/빗나감을 센다(빗나감은 응답면 비대칭의 증거다).
    o("")
    o("-" * 118)
    o("예측 규칙 |log e| < |log ts_true| (추정이 배율1 보다 참값에 가까우면 공식 승) 대조")
    o("-" * 118)
    hit = miss = 0
    rule = []
    for rec in res["conditions"]:
        tv = rec["axis_value"]
        marks = []
        for c in rec["cells"]:
            pred = abs(math.log(c["err"])) < abs(math.log(tv)) - 1e-12
            d, cc = c["vs_fixed"]["delta"], c["vs_fixed"]["ci95"]
            obs = None if abs(d) <= cc else (d > 0)
            ok = (obs is None) or (obs == pred)
            hit += ok
            miss += (not ok)
            marks.append(("o" if pred else "x") + ("=" if ok else "!"))
            rule.append({"axis_value": tv, "err": c["err"], "pred_win": pred,
                         "obs_win": obs, "agree": bool(ok)})
        o(f"  ts_true={tv:<5} " + " ".join(f"e{c['err']:.2f}:{m}"
                                           for c, m in zip(rec["cells"], marks)))
    o(f"  적중 {hit} / 빗나감 {miss} (o=규칙이 공식 승 예측 · != 관측과 불일치)")
    o("  빗나감은 응답면이 로그축에서 비대칭이라는 뜻이다 — 과대추정이 과소추정보다 싸다.")
    res["flip_rule"] = {"formula": "|log e| < |log ts_true|", "hit": hit, "miss": miss,
                        "cells": rule}

    # 배포 판단은 minimax 다 — "입력이 이만큼 틀릴 수 있다" 는 가정마다 세 전략의 최악 regret.
    o("")
    o("-" * 118)
    o("전략 비교(minimax) — 오차 가정별 조건간 평균/최악 regret. 배포는 최악값으로 정한다")
    o("-" * 118)
    o(f"{'전략':<28}{'평균 regret':>14}{'최악 regret':>14}   최악 조건")
    strat = []

    def row(label: str, vals: dict):
        wv = max(vals, key=lambda k: vals[k])
        strat.append({"strategy": label, "mean": float(np.mean(list(vals.values()))),
                      "max": float(vals[wv]), "worst_cond": wv,
                      "per_condition": {str(k): float(v) for k, v in vals.items()}})
        o(f"{label:<28}{np.mean(list(vals.values())):>14.5f}{vals[wv]:>14.5f}   ts_true={wv}")

    row(f"고정 {res['fixed_arm']} (적응 없음)",
        {r["axis_value"]: r["fixed"]["delta"] for r in res["conditions"]})
    row("공식 (오차 없음, e=1)",
        {r["axis_value"]: [c for c in r["cells"] if abs(math.log(c["err"])) < 1e-9][0]["regret"]["delta"]
         for r in res["conditions"]})
    for f in (1.2, 1.5, 2.0):
        sel = [e for e in errs if 1.0 / f - 1e-9 <= e <= f + 1e-9]
        if len(sel) < 2:
            continue
        row(f"공식 (오차 ×{1 / f:.2f}~{f:.1f}, {len(sel)}셀 최악)",
            {r["axis_value"]: max(c["regret"]["delta"] for c in r["cells"] if c["err"] in sel)
             for r in res["conditions"]})
    if all("untuned" in r for r in res["conditions"]):
        row(f"무튜닝 {res['untuned_arm']} (명부 직독)",
            {r["axis_value"]: r["untuned"]["regret"]["delta"] for r in res["conditions"]})
        o("⚠️ 무튜닝 S족은 시뮬 안에서 `MCI_TREAT_SCALE` 이 스케일한 병원 명부를 직접 읽는다 —")
        o("   즉 이 행의 '오추정 위험 0' 은 시뮬의 성질이고 현실 주장이 아니다(명부는 명목값뿐).")
    res["minimax"] = strat

    o("")
    o("-" * 118)
    o("허용 오차구간 요약 (regret <= 판정선, 경계는 log e 선형보간 = 추정)")
    o("-" * 118)
    los = [r["tolerance_band_judge"]["lo"] for r in res["conditions"]
           if r["tolerance_band_judge"]["lo"]]
    his = [r["tolerance_band_judge"]["hi"] for r in res["conditions"]
           if r["tolerance_band_judge"]["hi"]]
    for r in res["conditions"]:
        b = r["tolerance_band_judge"]
        o(f"  ts_true={r['axis_value']:<5} " + ("없음" if b["lo"] is None else
                                                f"e ∈ [{b['lo']:.2f}, {b['hi']:.2f}]"))
    if los and his:
        res["tolerance_summary"] = {"lo_max": max(los), "hi_min": min(his),
                                    "lo_mean": float(np.mean(los)), "hi_mean": float(np.mean(his))}
        o(f"  → 전 조건 공통 구간(가장 좁은 쪽) e ∈ [{max(los):.2f}, {min(his):.2f}] · "
          f"평균 [{np.mean(los):.2f}, {np.mean(his):.2f}]")


def cmd_grid(args: argparse.Namespace) -> None:
    conds_paths = _discover(str(REPO / args.stage_dir), args.prefix, args.axis_token, args.axis_scale)
    for k, v in _parse_conds(args.anchor).items():
        conds_paths.setdefault(k, []).extend(v)
    if args.true_grid:
        want = [float(x) for x in args.true_grid.split(",")]
        missing = [w for w in want if w not in conds_paths]
        if missing:
            raise SystemExit(f"[치명] 요청한 참 조건이 없다: {missing} (있는 것 {sorted(conds_paths)})")
        conds_paths = {k: v for k, v in conds_paths.items() if k in want}
    conds = {k: _load_cond(v) for k, v in sorted(conds_paths.items())}
    errs = [float(x) for x in args.err_grid.split(",")]
    res = _grid_analyze(conds, errs, args.family, args.arm_suffix, args.lam_coef,
                        args.fixed_arm, args.untuned_arm, args.round)
    res["source"] = {"stage_dir": args.stage_dir, "prefix": args.prefix, "anchor": args.anchor}
    o = Out()
    _print_grid(res, o, args.title)
    _print_matrix(res, o)
    _dump(res, o, args)


def cmd_replicate(args: argparse.Namespace) -> None:
    conds = {k: _load_cond(v) for k, v in sorted(_parse_conds(args.conds).items())}
    errs = [float(x) for x in args.err_grid.split(",")]
    res = _grid_analyze(conds, errs, args.family, args.arm_suffix, args.lam_coef,
                        args.fixed_arm, args.untuned_arm, args.round)
    res["source"] = {"conds": args.conds}
    o = Out()
    _print_grid(res, o, args.title)
    _print_matrix(res, o)
    o("")
    o("※ 이 표의 평가 시드는 공식 계수 도출에 쓰인 seed0..9 와 서로소다(표본외 재현).")
    _dump(res, o, args)


# --------------------------------------------------------------------------- N 오추정
def _transfer(conds: dict[float, dict], fam: str, suffix: str, axis_name: str,
              fixed: str, o: Out) -> dict:
    """조건 A 에서 고른 팔을 조건 B 에서 평가 — 공식이 없는 축의 오추정 비용.

    격자 팔의 교집합만 쓴다(A 에만 있는 팔은 B 에서 평가할 수 없다)."""
    vals = sorted(conds)
    common = None
    for v in vals:
        s = {n for _, n in _fam_arms(conds[v]["arms"], fam, suffix)}
        common = s if common is None else (common & s)
    if not common or len(common) < 3:
        raise SystemExit(f"[치명] {axis_name}: 공통 {fam}족 팔이 {len(common or [])}개 — 전이 행렬 불가")
    fa = sorted((float(re.fullmatch(re.escape(fam) + r"([0-9.]+)" + re.escape(suffix), n).group(1)), n)
                for n in common)
    names = [n for _, n in fa]
    lams = [v for v, _ in fa]

    per = {}
    for v in vals:
        arms = conds[v]["arms"]
        means = np.array([arms[n].mean() for n in names], float)
        ib = int(means.argmin())
        # 등급 임계 격자는 0 을 포함하므로 로그축 포물선 보간이 정의되지 않는다 → argmin 을 쓴다.
        interp = float(_interp_opt(lams, means.tolist())) if min(lams) > 0 else None
        ab = min(arms, key=lambda n: arms[n].mean())
        per[v] = {"best_arm": names[ib], "best_pdr": float(means[ib]),
                  "grid_edge": ib in (0, len(names) - 1),
                  "lam_star_interp": interp,
                  "all_family_best_arm": ab, "all_family_best_pdr": float(arms[ab].mean()),
                  "all_family_best_vs_fixed": (paired(arms[ab], arms[fixed])
                                               if fixed in arms else None),
                  "grid": [[float(lams[i]), names[i], float(means[i])] for i in range(len(names))],
                  "manifest": conds[v]["manifest"], "knobs": conds[v]["knobs"],
                  "n_region": conds[v]["n_region"], "n_seed": len(conds[v]["seeds"]),
                  "files": [os.path.relpath(p, REPO) for p in conds[v]["files"]]}

    o("")
    o("=" * 112)
    o(f"{axis_name} 오추정 — {fam}족 전이 행렬 (공식이 없는 축이라 격자가 아니라 전이로 잰다)")
    o(f"  공통 팔 {names}")
    o("=" * 112)
    for v in vals:
        p = per[v]
        o(f"  {axis_name}={v:<6} 최적 {p['best_arm']:>7s} {p['best_pdr']:.6f}"
          f"{' ⚠️격자끝' if p['grid_edge'] else ''} · 보간 최적값 "
          + (f"{p['lam_star_interp']:.2f}" if p["lam_star_interp"] is not None
             else "없음(격자에 0 포함 → 로그 보간 불가)")
          + f" · {p['n_region']}좌표 × {p['n_seed']}시드 · {p['manifest']}")
        afb = p["all_family_best_vs_fixed"]
        o(f"           참고) 전 팔족 최적 {p['all_family_best_arm']} "
          f"{p['all_family_best_pdr']:.6f}"
          + (f" → 고정 {fixed} 대비 +{afb['delta']:.5f} ±{afb['ci95']:.5f} = 정본 ④ 의 정의"
             if afb else ""))
    o("")
    o(f"  행 = 참 {axis_name}(평가 조건) · 열 = 추정 {axis_name}(팔 선택 조건) · 값 = regret ±CI")
    cells = []
    for vt in vals:
        arms = conds[vt]["arms"]
        bcube = arms[per[vt]["best_arm"]]
        row = []
        for vh in vals:
            arm = per[vh]["best_arm"]
            r = paired(bcube, arms[arm])
            row.append({"true": vt, "hat": vh, "arm": arm, **r,
                        "verdict": _verdict(r["delta"], r["ci95"])})
        cells.append(row)
        o(f"  참={vt:<6} " + " | ".join(
            f"추정={c['hat']:<5} {c['arm']:>6s} +{c['delta']:.5f}±{c['ci95']:.5f} {c['verdict']:>6s}"
            for c in row))
    if fixed and all(fixed in conds[v]["arms"] for v in vals):
        o("")
        o(f"  0-에피소드 고정 카드 {fixed} (공식에 {axis_name} 항이 없으므로 모든 {axis_name} 에서 같은 팔)")
        fixrows = []
        for vt in vals:
            arms = conds[vt]["arms"]
            r = paired(arms[per[vt]["best_arm"]], arms[fixed])
            fixrows.append({"true": vt, **r, "verdict": _verdict(r["delta"], r["ci95"])})
            o(f"    참={vt:<6} 손실 +{r['delta']:.5f} ±{r['ci95']:.5f} "
              f"(W/T/L {r['win']}/{r['tie']}/{r['loss']}) {_verdict(r['delta'], r['ci95'])}")
    else:
        fixrows = []
        o(f"  (고정 팔 {fixed} 이 모든 조건에 없어 0-에피소드 기준선은 생략)")
    return {"axis": axis_name, "family": fam + "*" + suffix, "arms": names,
            "per_condition": {str(k): v for k, v in per.items()},
            "transfer": cells, "fixed_arm": fixed, "fixed_loss": fixrows}


def cmd_nmisspec(args: argparse.Namespace) -> None:
    o = Out()
    o("=" * 112)
    o("N(환자 수) 오추정 — 정본 예측은 null 이다(공식 λ = 18.97 × 치료시간배수 에 N 항이 없다)")
    o(f"  판정선 {JUDGE_LINE} · regret = paired(그 조건 최적, 전이된 팔)")
    o("=" * 112)
    out = {"judge_line": JUDGE_LINE, "blocks": []}
    if args.lam_conds:
        conds = {k: _load_cond(v) for k, v in sorted(_parse_conds(args.lam_conds).items())}
        out["blocks"].append(_transfer(conds, args.family, args.arm_suffix,
                                       "N(λ 격자)", args.fixed_arm, o))
    if args.y_conds:
        conds = {k: _load_cond(v) for k, v in sorted(_parse_conds(args.y_conds).items())}
        out["blocks"].append(_transfer(conds, args.y_family, "", "N(등급 임계)", args.y_fixed, o))
    _dump(out, o, args)


def _dump(res: dict, o: Out, args: argparse.Namespace) -> None:
    if args.out:
        Path(str(REPO / args.out)).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(str(REPO / args.out), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"[기록] {args.out}")
    o.save(str(REPO / args.out_txt) if args.out_txt else
           (str(REPO / args.out).rsplit(".", 1)[0] + ".txt" if args.out else None))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--err_grid", default=DEF_ERRS, help="추정 배율 e 격자 (ts_hat = ts_true × e)")
        p.add_argument("--lam_coef", type=float, default=LAM_COEF, help="공식 계수 λ/치료시간배수")
        p.add_argument("--family", default="Q", help="λ 격자 팔 족 (Q=hingerate)")
        p.add_argument("--arm_suffix", default="", help="팔 이름 접미(budgetcurve 는 Y0)")
        p.add_argument("--fixed_arm", default="Q18", help="0-에피소드 고정 카드")
        p.add_argument("--untuned_arm", default="S0.62", help="무튜닝 S족 (없으면 빈 문자열)")
        p.add_argument("--round", default="log", choices=["log", "linear"], help="λ_hat→팔 반올림 규약")
        p.add_argument("--out", default="")
        p.add_argument("--out_txt", default="")

    g = sub.add_parser("grid", help="오추정 격자 (ts_true × e) 3전략 비교")
    g.add_argument("--stage_dir", default=DEF_STAGE)
    g.add_argument("--prefix", default="treat")
    g.add_argument("--axis_token", default="ts")
    g.add_argument("--axis_scale", default="dot", choices=["dot", "raw"],
                   help="dot = 소수점 생략 규약(ts40=4.0) · raw = 태그 그대로")
    g.add_argument("--anchor", default=DEF_ANCHOR, help="스테이지에 없는 조건 추가 '값=경로[,경로]'")
    g.add_argument("--true_grid", default="", help="참 조건 부분집합 (쉼표, 비우면 전부)")
    g.add_argument("--title", default="E8 입력 오추정 격자 — 치료시간 축 (budget750)")
    common(g)
    g.set_defaults(func=cmd_grid)

    r = sub.add_parser("replicate", help="서로소 평가시드로 격자 재현 (budgetcurve)")
    r.add_argument("--conds", default=DEF_BC)
    r.add_argument("--title", default="E8 재현 — budgetcurve eval (60좌표 × 서로소 시드 1000..1059)")
    common(r)
    r.set_defaults(func=cmd_replicate, fixed_arm="Q18Y0", arm_suffix="Y0", untuned_arm="")

    n = sub.add_parser("nmisspec", help="환자 수 N 오추정 (전이 행렬)")
    n.add_argument("--lam_conds", default=DEF_N_LAM)
    n.add_argument("--y_conds", default=DEF_N_Y)
    n.add_argument("--y_family", default="Y")
    n.add_argument("--y_fixed", default="Y0")
    common(n)
    n.set_defaults(func=cmd_nmisspec)

    args = ap.parse_args()
    if args.cmd == "replicate":
        # set_defaults 는 common() 기본값에 덮이므로 여기서 강제한다.
        if args.fixed_arm == "Q18":
            args.fixed_arm = "Q18Y0"
        if args.arm_suffix == "":
            args.arm_suffix = "Y0"
        if args.untuned_arm == "S0.62":
            args.untuned_arm = ""
    args.func(args)


if __name__ == "__main__":
    main()
