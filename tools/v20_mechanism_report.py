# -*- coding: utf-8 -*-
"""v20 기전 집계 — PDR 을 "미진입 손실" 과 "지연 손실" 로 분해해 규칙별로 비교한다.

읽는 법
    PDR_woG = 미진입손실 + 지연손실   (항등식. 계측기가 잔차를 검증한다)
      미진입손실 = 구조됐지만 에피소드 안에 **수술실에 못 들어간** 환자의 잠재 생존확률 합 / preventable
      지연손실   = 들어갔지만 구조 시점보다 **늦게** 들어가서 잃은 생존확률 / preventable

⚠️ 해석 함정 두 개(둘 다 실측으로 걸렸다)
  1. 처치를 시작한 환자만 보면 대기가 거의 0 으로 보인다 — **줄에서 못 빠져나온 환자가 표본에 없다**(생존자 편향).
  2. 시뮬 trace 에 `ev_p_def_care` 경로의 `care_start` 가 빠져 있었다 — **기다린 환자만 정확히** 누락됐다.
     v20 에서 보완(trace 전용, 동작 불변). 그 전 trace 로 계산한 대기 통계는 전부 무효다.

사용:
    python tools/v20_mechanism_report.py            # 전 설정 표
    python tools/v20_mechanism_report.py --paired A B   # 두 정책 paired 비교(전 지표)
"""
from __future__ import annotations

import argparse
import glob
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MECH = REPO / "results/scoreboard/v20/mechanism"
LABEL = {
    "NOLOAD": "부하항 없음(최근접)", "START_LB3": "START-LB3 휴리스틱",
    "K12": "현행 CARD (거리+선형)", "H12": "시간+대기초과",
    "Q6": "유도형 lam=6(과소)", "Q18": "유도형 lam=18(채택)", "Q50": "유도형 lam=50(과대)",
    "P18": "유도형·통신불필요", "S062": "목적함수 직접(무튜닝)",
}
ORDER = ["NOLOAD", "START_LB3", "K12", "H12", "Q6", "Q18", "Q50", "P18", "S062"]


def ci(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.96 * x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else 0.0


def table(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("policy").agg(
        PDR=("pdr_woG", "mean"),
        미진입손실=("share_missing", "mean"),
        지연손실=("share_delay", "mean"),
        진입률_RY=("or_rate_RY", "mean"),
        진입률_R=("or_rate_R", "mean"),
        진입률_Y=("or_rate_Y", "mean"),
        대기후진입비=("frac_queued", "mean"),
        대기중위분=("wait_med_queued", "mean"),
        R진입시각=("t_admit_med_R", "mean"),
        Y진입시각=("t_admit_med_Y", "mean"),
        사용병원수=("n_hosp_used", "mean"),
        집중도gini=("gini_hosp", "mean"),
        전원건수=("n_diversion", "mean"),
    )
    idx = [p for p in ORDER if p in g.index] + [p for p in g.index if p not in ORDER]
    return g.reindex(idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech", default=str(MECH))
    ap.add_argument("--paired", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    pd.set_option("display.width", 240)

    files = sorted(glob.glob(os.path.join(a.mech, "*.csv")))
    files = [f for f in files if os.path.exists(f + ".meta.json")]
    if not files:
        print("완주한 CSV 가 없다"); return

    for f in files:
        tag = Path(f).stem
        d = pd.read_csv(f)
        resid = (d.share_delay + d.share_missing - d.pdr_woG).abs().max()
        t = table(d)
        print("=" * 132)
        print(f"[{tag}]  {d.region.nunique()} 좌표 x {d.seed.nunique()} 시드 · 분해 항등식 잔차 {resid:.2e}")
        print("=" * 132)
        t2 = t.copy()
        t2.index = [f"{LABEL.get(p, p)}" for p in t.index]
        print(t2.round(4).to_string())
        if a.paired and all(p in set(d.policy) for p in a.paired):
            A, B = a.paired
            key = ["region", "seed"]
            m = d[d.policy == A].merge(d[d.policy == B], on=key, suffixes=("_a", "_b"))
            print(f"\n  paired {A} - {B} (같은 좌표·같은 시드, n={len(m)})")
            for col in ("pdr_woG", "share_missing", "share_delay", "or_rate_RY",
                        "frac_queued", "t_admit_med_Y", "n_hosp_used"):
                dd = (m[col + "_a"] - m[col + "_b"]).to_numpy(float)
                dd = dd[np.isfinite(dd)]
                print(f"    {col:16s} {dd.mean():+10.5f} ± {ci(dd):.5f}")
        print()


if __name__ == "__main__":
    main()
