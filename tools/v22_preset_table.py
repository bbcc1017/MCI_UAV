# -*- coding: utf-8 -*-
"""v22 E5 — 심각도 프리셋 표(Mode B). **시뮬 재실행 0회의 완주본 재분석.**

현장에서 폐루프를 못 돌리는 경우의 대안. 지휘관이 t=0 에 아는 값만으로 인덱스된 표를 만들고
그 표를 **0-에피소드 정책**으로 평가한다.

정보수준 계약(`feedback-field-rules-information-level`): 도달시간(거리표÷속도)·수술실수·
내가 보낸 누적은 가용, **병원 입원 census 만 통신 필요**. 이 표의 입력은 전부 앞의 셋 + 환자수 +
가용 AMB 대수 + 관측 치료시간배수이므로 통신이 필요 없다(카드 본문의 부하항은 기존 CARD 와 동일).

데이터
------
`results/scoreboard/v22/retune/` 36개 스테이지 파일(= **물리조건 25개**) + 조건별 `.meta.json` 의
`scenario_knobs`. 같은 물리조건이 여러 스테이지에 중복 등장하며(예: `MCI_TREAT_SCALE=4.0` 은
extreme/lamx/treat 3벌) 매니페스트·시드가 같아 CRN 이 유지된다 — `merge_condition()` 이 중복 팔의
비트동일성을 **검사한 뒤** 병합한다(가정이 아니라 측정).

표의 두 형태
------------
* **(A) 공식형** : `λ = 18.97 × 치료시간배수`(v22 정본 ①②) + `y = f(여유·이송률)`.
  y 는 지휘관 가용 입력의 함수로만 쓴다 — 조건 이름(`capa20` 등)은 현장에 없다.
* **(B) 조건별 조회표** : 조건 → 격자 argmin `(λ*, y*)`. 조건을 완벽히 아는 오라클이므로
  **상한**이고, 같은 10 시드로 고르고 같은 10 시드로 재므로 승자의 저주가 들어 있다
  (v22 ⑨). 그래서 시드 분할(0-4 선택 / 5-9 평가) 정직판을 함께 낸다.

축 분해의 근거
--------------
축별 단독 최적을 결합해도 되는 이유는 v22 정본 ⑥ 이다 — (λ × yhold) 결합 격자 5조건 중 4곳에서
결합 최적이 축별 단독 최적과 **좌표까지 일치**하고, 어긋난 `vamb100` 도 +0.000073 ± 0.000143 =
동률이었다. 그래서 λ 는 y=0 절단면에서, y 는 λ=18 절단면에서 읽는다. 결합 격자가 있는 5조건에서는
결합 전수로 다시 확인한다(`eval` 의 D1).

판정선
------
0.00053 (v21 CRN paired 실측). 새로 만들지 않고 이 값을 쓰며 모든 비교에 실제 paired 95%CI 를
병기한다. 통계 커널(`ci`/`cube`/`paired`/`wtl`)은 `tools/v20_threshold_report.py` 를 재사용한다 —
재구현하면 W/T/L 정의가 갈린다.

사용:
    python tools/v22_preset_table.py all       # readmap→optima→table→eval→round 전부
    python tools/v22_preset_table.py readmap   # 조건별로 무엇을 읽을 수 있는지
    python tools/v22_preset_table.py optima    # 조건별 (λ*, y*) + 평탄대
    python tools/v22_preset_table.py table     # 표 A/B 생성
    python tools/v22_preset_table.py eval      # paired 평가
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from v20_threshold_report import ci, cube, paired, wtl  # noqa: E402  (통계 커널 재사용)

RETUNE = REPO / "results/scoreboard/v22/retune"
MANIFEST = REPO / "scenarios/manifests/sigungu30_budget750_manifest.json"
OUT = REPO / "results/field/preset"

# v21 CRN paired 실측 판정선. 이 파일에서 새로 만들지 않는다.
DECIDE = 0.00053
# v22 정본 ①: λ* = 18.97 × 치료시간배수 (치료시간 축 로그-로그 기울기 +1.121, 이론 +1)
LAM_COEF = 18.97

# 기준조건(정식 시나리오) 물리값. `AXIS_BASE`(v20) 과 같은 값 + 치료배수/인계.
BASE_KNOBS = {
    "MCI_INCIDENT_SIZE": 100.0, "MCI_AMB_NUM": 30.0, "MCI_UAV_NUM": 26.0,
    "MCI_CAPA_SCALE": 1.0, "MCI_TREAT_SCALE": 1.0, "MCI_AMB_VELOCITY": 50.0,
    "MCI_UAV_VELOCITY": 200.0, "MCI_AMB_HANDOVER": 5.0, "MCI_UAV_HANDOVER": 10.0,
}


# --------------------------------------------------------------------------- 팔 파싱
def parse_spec(spec: str):
    """`'Q18Y2=cardt:18,6.6,2,hingerate'` → `('Q18Y2', {...})`.

    팔 이름으로 추측하지 않고 **meta 의 policy_spec 문자열**을 파싱한다(스테이지마다 같은
    파라미터에 다른 이름을 붙였다: `Y0`·`Q18`·`G6.6`·`Q18Y0` 이 모두 같은 팔이다).
    """
    name, body = spec.split("=", 1)
    name, body = name.strip(), body.strip()
    if body.startswith("cardt:"):
        t = body[len("cardt:"):].split(",")
        term = t[3].strip() if len(t) > 3 else "load"
        fam = {"hingerate": "Q", "hinge1": "H", "hingerate_psent": "P"}.get(term, term)
        return name, {"kind": "cardt", "fam": fam, "axis": float(t[0]),
                      "red": float(t[1]), "y": float(t[2]), "term": term}
    if body.startswith("cards:"):
        t = body[len("cards:"):].split(",")
        return name, {"kind": "cards", "fam": "S", "axis": float(t[0]),
                      "red": float(t[1]), "y": float(t[2]), "term": "surv"}
    if body.startswith("card:"):
        t = body[len("card:"):].split(",")
        return name, {"kind": "card", "fam": "K", "axis": float(t[0]),
                      "red": float(t[1]), "y": float(t[2]), "term": "load"}
    return name, {"kind": "other", "fam": "X", "axis": float("nan"),
                  "red": float("nan"), "y": float("nan"), "term": body}


def is_canon_q(p: dict) -> bool:
    """카드 정본 팔 = CARD-T · hingerate · red_gain 6.6. (λ, y) 표의 좌표축이 되는 팔."""
    return p["kind"] == "cardt" and p["fam"] == "Q" and abs(p["red"] - 6.6) < 1e-9


def cond_key(knobs: dict) -> str:
    return "base" if not knobs else ",".join(f"{k}={v}" for k, v in sorted(knobs.items()))


def cond_short(knobs: dict) -> str:
    """사람이 읽을 짧은 조건 이름. **표 A 는 이 이름을 쓰지 않는다**(현장에 없는 이름)."""
    if not knobs:
        return "base"
    ab = {"MCI_INCIDENT_SIZE": "n", "MCI_AMB_NUM": "amb", "MCI_UAV_NUM": "uav",
          "MCI_CAPA_SCALE": "capa", "MCI_TREAT_SCALE": "ts", "MCI_AMB_VELOCITY": "vamb",
          "MCI_AMB_HANDOVER": "hamb", "MCI_UAV_HANDOVER": "huav"}
    return ",".join(f"{ab.get(k, k)}{v}" for k, v in sorted(knobs.items()))


# --------------------------------------------------------------------------- 조건 병합
def scan_stages() -> dict:
    """`.meta.json` 이 있는 CSV 만 읽는다(없으면 미완주). → {조건키: {...}}"""
    stages = {}
    for mp in sorted(RETUNE.glob("*.csv.meta.json")):
        meta = json.loads(mp.read_text())
        name = mp.name.replace(".csv.meta.json", "")
        stages[name] = meta
    conds: dict[str, dict] = {}
    for name, meta in stages.items():
        k = cond_key(meta["scenario_knobs"])
        c = conds.setdefault(k, {"knobs": meta["scenario_knobs"], "short": cond_short(meta["scenario_knobs"]),
                                 "files": [], "manifests": set(), "seeds": set()})
        c["files"].append(name)
        c["manifests"].add(meta["manifest_sha256"])
        c["seeds"].add((meta["seed_start"], meta["seed_end"]))
    for k, c in conds.items():
        c["files"].sort()
        assert len(c["manifests"]) == 1, f"{k}: 매니페스트 불일치 — CRN 병합 불가"
        assert len(c["seeds"]) == 1, f"{k}: 시드 집합 불일치"
        c["manifests"] = sorted(c["manifests"])
        c["seeds"] = sorted(c["seeds"])
    return conds


def merge_condition(cond: dict):
    """조건의 스테이지 파일들을 합쳐 {팔이름: cube} + 팔 파라미터 + 중복 검사 결과를 낸다.

    같은 팔이 두 파일에 있으면 **비트동일성을 검사**한다(같은 매니페스트·시드면 CRN 이 같아야
    한다는 주장을 측정으로 확인). 어긋나면 예외 — 조용히 평균내지 않는다.
    """
    cubes, params, dup = {}, {}, []
    regions = None
    for f in cond["files"]:
        meta = json.loads((RETUNE / f"{f}.csv.meta.json").read_text())
        spec = dict(parse_spec(s) for s in meta["policy_specs"])
        df = pd.read_csv(RETUNE / f"{f}.csv")
        key = df.groupby(["region", "policy", "seed"]).size()
        assert int(key.max()) == 1, f"{f}: (region,policy,seed) 중복"
        for pol in sorted(df.policy.unique()):
            c = cube(df, pol)                     # 결측 시 예외 = 완주 검사
            pv = df[df.policy == pol].pivot_table(index="region", columns="seed", values="pdr_woG").sort_index()
            if regions is None:
                regions = list(pv.index)
            elif list(pv.index) != regions:
                raise ValueError(f"{f}/{pol}: 지역 집합 불일치")
            p = spec.get(pol, {"kind": "other", "fam": "X", "axis": float("nan"),
                               "red": float("nan"), "y": float("nan"), "term": "?"})
            sig = (p["kind"], p["fam"], p["axis"], p["red"], p["y"], p["term"])
            if sig in cubes:
                prev, prev_name = cubes[sig]
                same = bool(np.array_equal(prev, c))
                dup.append({"arm": sig[:1] + sig[1:], "files": [prev_name, f],
                            "bit_identical": same,
                            "maxabs": float(np.abs(prev - c).max())})
                if not same:
                    raise ValueError(f"{f}/{pol}: 같은 팔이 다른 값 — CRN 병합 불가")
            else:
                cubes[sig] = (c, f)
                params[sig] = p
    arms = {sig: cu for sig, (cu, _) in cubes.items()}
    return arms, params, regions, dup


def q_grid(arms, params):
    """정본 Q 팔만 골라 {(λ, y): cube}."""
    return {(params[s]["axis"], params[s]["y"]): c for s, c in arms.items() if is_canon_q(params[s])}


def find_arm(arms, params, kind=None, fam=None, axis=None, y=None, red=6.6):
    for s, c in arms.items():
        p = params[s]
        if kind and p["kind"] != kind:
            continue
        if fam and p["fam"] != fam:
            continue
        if axis is not None and abs(p["axis"] - axis) > 1e-9:
            continue
        if y is not None and abs(p["y"] - y) > 1e-9:
            continue
        if red is not None and not (isinstance(p["red"], float) and math.isnan(p["red"])) \
                and abs(p["red"] - red) > 1e-9:
            continue
        return c
    return None


# --------------------------------------------------------------------------- 지역 입력
def region_inputs(force=False) -> pd.DataFrame:
    """지휘관이 t=0 에 아는 지역별 값 — 수술실수·도로거리(→도달시간). census 는 쓰지 않는다.

    * `or_list`  : 병원별 수술실수(용량 스케일을 `max(1, round(·×s))` 로 적용하려면 개별값 필요)
    * `d_near`   : 최근접 병원 도로거리(km) — AMB 왕복 사이클의 대리값
    * `d_t3`     : 최근접 tier3(상급종합=종별코드 1) 도로거리(km)
    """
    cache = OUT / "region_inputs.json"
    if cache.exists() and not force:
        return pd.DataFrame(json.loads(cache.read_text())).set_index("region")
    man = json.loads(MANIFEST.read_text())
    rows = []
    for reg, cfg in man.items():
        d = os.path.dirname(cfg)
        h = pd.read_csv(os.path.join(d, "hospital_info.csv"))
        t3 = h["종별코드"] == 1
        rows.append({"region": reg, "n_hos": int(len(h)), "n_t3": int(t3.sum()),
                     "or_list": [int(x) for x in h["수술실수"].tolist()],
                     "d_near": float(h["road_dist"].min()),
                     "d_t3": float(h.loc[t3, "road_dist"].min()) if t3.any() else float("nan")})
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, ensure_ascii=False))
    return pd.DataFrame(rows).set_index("region")


def field_inputs(knobs: dict, rf: pd.DataFrame, regions: list) -> pd.DataFrame:
    """물리조건 + 지역 → 지휘관 입력. 노브는 조건을 만드는 수단일 뿐이고, 카드가 읽는 값은
    여기서 나오는 **관측량**(수술실수·도달시간·환자수·AMB대수·치료시간배수)이다.

        S  여유    = (수술실 총수 ÷ 환자수) ÷ 치료시간배수        [수술실/명]
        R  이송률  = 60 × AMB대수 ÷ (환자수 × 왕복시간)          [1/시간]
                     왕복시간 = 2 × 최근접병원 도달시간 + 인계시간
        T3 접근    = 최근접 상급종합 도달시간                      [분]
    """
    g = dict(BASE_KNOBS)
    for k, v in knobs.items():
        g[k] = float(v)
    N, A, TS = g["MCI_INCIDENT_SIZE"], g["MCI_AMB_NUM"], g["MCI_TREAT_SCALE"]
    V, HO, s = g["MCI_AMB_VELOCITY"], g["MCI_AMB_HANDOVER"], g["MCI_CAPA_SCALE"]
    sub = rf.loc[regions]
    orl = np.array([np.maximum(1, np.rint(np.asarray(x, float) * s)).sum() for x in sub["or_list"]])
    t_near = sub["d_near"].to_numpy() * 60.0 / V + HO          # 현장→최근접병원 도달시간(분)
    t_t3 = sub["d_t3"].to_numpy() * 60.0 / V + HO
    cycle = 2.0 * t_near + HO                                   # 왕복 + 인계
    return pd.DataFrame({"S": (orl / N) / TS, "R": 60.0 * A / (N * cycle),
                         "T3": t_t3, "Tnear": t_near, "or_total": orl,
                         "N": N, "A": A, "TS": TS}, index=regions)


# --------------------------------------------------------------------------- 표 A 의 y 규칙
def y_rule(S, R, r_lo=0.25, r_hi=0.88, s_hi=1.5):
    """y = f(여유 S, 이송률 R) — 정본 ⑤ 의 패턴을 지휘관 입력으로 옮긴 것.

    정본 문장: "여유가 크면 Red 우선을 8명까지, 차량이 희소하면 2, 느리거나 빡빡하면 0".
    * 차량 희소 = 이송률 R 이 낮다 → y=2
    * 여유 크다 = 병원측 여유 S 가 크거나(수술실/환자, 치료회전) 이송률 R 이 높다 → y=8
    * 그 밖 = 0
    두 항이 **OR** 인 것은 측정 결과다 — `capa20`(병원측만 완화)과 `vamb70/100`(이송측만 완화)이
    둘 다 y*=8 이었다. 임계 셋은 13조건에 적합한 값이며 `eval` 이 LOO 정직판을 함께 낸다.
    """
    S, R = np.asarray(S, float), np.asarray(R, float)
    y = np.zeros_like(S)
    y[(S >= s_hi) | (R >= r_hi)] = 8.0
    y[R <= r_lo] = 2.0                      # 차량 희소가 우선(여유 판정을 덮는다)
    return y


# --------------------------------------------------------------------------- 최적점
def snap_log(target: float, grid) -> float:
    """격자에 로그거리로 스냅. λ 가 한 자릿수~세 자릿수를 걸치므로 곱셈거리가 맞다."""
    g = np.asarray(sorted(grid), float)
    return float(g[int(np.argmin(np.abs(np.log(g / target))))])


def optima(conds=None, verbose=True):
    conds = conds or scan_stages()
    rf = region_inputs()
    res = {}
    for k, c in sorted(conds.items()):
        arms, params, regions, dup = merge_condition(c)
        qg = q_grid(arms, params)
        lam_axis = sorted({l for (l, y) in qg if y == 0})
        y_axis = sorted({y for (l, y) in qg if l == 18.0})
        joint = sorted(qg)
        fi = field_inputs(c["knobs"], rf, regions)
        # λ* (y=0 절단면)
        lam_curve = {l: float(qg[(l, 0.0)].mean()) for l in lam_axis}
        lam_star = min(lam_curve, key=lam_curve.get) if lam_curve else None
        # y* (λ=18 절단면), 카드 공간 {0,2,4,8} 로 제한
        y_axis_card = [y for y in y_axis if y in (0.0, 2.0, 4.0, 8.0)]
        y_curve = {y: float(qg[(18.0, y)].mean()) for y in y_axis_card}
        y_star = min(y_curve, key=y_curve.get) if y_curve else None
        # 결합 최적
        j_star = min(joint, key=lambda t: qg[t].mean()) if len(joint) > 1 else None
        # λ 평탄대: λ* 대비 paired Δ 가 판정선 미만인 격자점
        plateau = []
        if lam_star is not None and len(lam_axis) > 1:
            ref = qg[(lam_star, 0.0)]
            for l in lam_axis:
                pr = paired(qg[(l, 0.0)], ref)      # +면 l 이 λ* 보다 좋다(있을 수 없음)
                if -pr["delta"] < DECIDE:
                    plateau.append(l)
        res[k] = {
            "short": c["short"], "knobs": c["knobs"], "files": c["files"],
            "n_regions": len(regions), "dup_checked": dup,
            "lam_axis": lam_axis, "lam_curve": lam_curve, "lam_star": lam_star,
            "lam_plateau": plateau,
            "y_axis": y_axis_card, "y_curve": y_curve, "y_star": y_star,
            "joint_n": len(joint), "joint_star": list(j_star) if j_star else None,
            "field": {"S": float(fi.S.median()), "R": float(fi.R.median()),
                      "T3": float(fi.T3.median()), "N": float(fi.N.iloc[0]),
                      "A": float(fi.A.iloc[0]), "TS": float(fi.TS.iloc[0])},
            "lam_formula": LAM_COEF * float(fi.TS.iloc[0]),
        }
        if verbose:
            r = res[k]
            print(f"{r['short']:12s} nλ={len(lam_axis):2d} λ*={str(lam_star):>5s} "
                  f"평탄대={plateau if plateau else '-'} | ny={len(y_axis_card)} y*={y_star} "
                  f"| 결합={r['joint_star']} | S={r['field']['S']:.2f} R={r['field']['R']:.2f} "
                  f"T3={r['field']['T3']:.0f}분 λ_공식={r['lam_formula']:.1f}")
    return res


# --------------------------------------------------------------------------- 표 생성
def build_tables(opt):
    """(A) 공식형 · (B) 조회표 를 만든다. B 의 승자의 저주는 시드분할로 따로 잰다."""
    A = {"form": {
        "lam": f"λ = {LAM_COEF} × 치료시간배수  (관측 치료시간 ÷ 정식 치료시간)",
        "y": "R ≤ 0.25/h → y=2 ;  (S ≥ 1.5 또는 R ≥ 0.88/h) → y=8 ;  그 밖 → y=0",
        "S": "여유 = (주변 수술실 총수 ÷ 환자수) ÷ 치료시간배수   [수술실/명]",
        "R": "이송률 = 60 × 가용 AMB 대수 ÷ (환자수 × 왕복시간),  왕복시간 = 2×최근접병원 도달시간 + 인계",
        "red_km": "red_gain 6.6분 고정(정본 카드값, 이 실험에서 재튜닝 대상 아님)",
    }, "rows": {}}
    B = {"note": "조건별 격자 argmin — 조건을 완벽히 아는 오라클(상한). 같은 시드로 고르고 재므로 승자의 저주 포함",
         "rows": {}}
    for k, r in opt.items():
        lam_f = r["lam_formula"]
        grid = r["lam_axis"] or [18.0]
        A["rows"][k] = {
            "short": r["short"],
            "lam_real": lam_f, "lam_print5": 5 * round(lam_f / 5),
            "lam_snapped": snap_log(lam_f, grid),
            "y_global": float(y_rule(r["field"]["S"], r["field"]["R"])),
        }
        B["rows"][k] = {"short": r["short"], "lam_star": r["lam_star"], "y_star": r["y_star"],
                        "joint_star": r["joint_star"]}
    return A, B


# --------------------------------------------------------------------------- 평가
def mix_cube(qg, regions, y_vec, lam):
    """지역별로 다른 y 를 지시하는 카드의 0-에피소드 평가용 cube.

    지역 r 의 행을 그 지역에 지시된 팔의 행에서 가져온다. 지시는 **t=0 관측값만**으로 정해지므로
    (수술실수·도달시간·환자수·AMB대수·치료배수) 이 혼합은 정당한 정책이다 —
    시드축을 골라 섞으면 부정이지만 지역축은 아니다.
    """
    need = sorted(set(np.unique(y_vec).tolist()))
    for y in need:
        if (lam, y) not in qg:
            return None, need
    out = np.empty_like(qg[(lam, need[0])])
    for y in need:
        m = (y_vec == y)
        out[m] = qg[(lam, y)][m]
    return out, need


def _fmt(pr):
    return f"{pr['delta']:+.5f}±{pr['ci95']:.5f} ({pr['win']}/{pr['tie']}/{pr['loss']})"


def evaluate(opt, conds=None):
    conds = conds or scan_stages()
    rf = region_inputs()
    rows = {}
    for k, c in sorted(conds.items()):
        arms, params, regions, _ = merge_condition(c)
        qg = q_grid(arms, params)
        fi = field_inputs(c["knobs"], rf, regions)
        r = opt[k]
        base = qg.get((18.0, 0.0))
        assert base is not None, f"{k}: Q18Y0 팔 없음"
        ent = {"short": r["short"], "files": c["files"], "n_regions": len(regions),
               "pdr": {"Q18Y0": float(base.mean())}, "vs_Q18Y0": {}, "arms_used": {}}

        cand = {}
        # 1) 무튜닝 S0.62
        s062 = find_arm(arms, params, kind="cards", axis=0.62, y=0.0, red=6.6)
        if s062 is not None:
            cand["S0.62"] = (s062, "cards:0.62,6.6,0")
        # 2) 표 B — Q 격자 argmin (결합격자가 있으면 결합, 없으면 축별)
        if r["joint_star"] and r["joint_n"] > 1:
            bl, by = r["joint_star"]
            cand["B_lookup"] = (qg[(bl, by)], f"Q{bl:g}Y{by:g}")
        elif r["lam_star"] is not None and len(r["lam_axis"]) > 1:
            cand["B_lookup"] = (qg[(r["lam_star"], 0.0)], f"Q{r['lam_star']:g}Y0")
        elif r["y_star"] is not None and len(r["y_axis"]) > 1:
            cand["B_lookup"] = (qg[(18.0, r["y_star"])], f"Q18Y{r['y_star']:g}")
        # 3) 표 A — λ 공식 스냅 + y 규칙(전역 중위 입력 / 지역별 입력)
        lam_snap = snap_log(r["lam_formula"], r["lam_axis"] or [18.0])
        y_g = float(y_rule(r["field"]["S"], r["field"]["R"]))
        if (lam_snap, y_g) in qg:
            cand["A_formula"] = (qg[(lam_snap, y_g)], f"Q{lam_snap:g}Y{y_g:g}")
        yv = y_rule(fi.S.to_numpy(), fi.R.to_numpy())
        mc, need = mix_cube(qg, regions, yv, lam_snap)
        if mc is not None:
            cand["A_region"] = (mc, f"Q{lam_snap:g}Y{{{','.join(f'{x:g}' for x in need)}}}(지역별)")
        # 4) 측정 팔 전체의 최소 = 오라클 상한(가족 무관)
        ok = min(arms, key=lambda s: arms[s].mean())
        cand["ORACLE_arm"] = (arms[ok], f"{ok[1]}{ok[2]:g}/red{ok[3]:g}/y{ok[4]:g}")
        # λ 성분만 (y=0 고정) / y 성분만 (λ=18 고정)
        if len(r["lam_axis"]) > 1:
            cand["A_lam_only"] = (qg[(lam_snap, 0.0)], f"Q{lam_snap:g}Y0")
            cand["B_lam_only"] = (qg[(r["lam_star"], 0.0)], f"Q{r['lam_star']:g}Y0")
        if len(r["y_axis"]) > 1:
            cand["A_y_only"] = (qg[(18.0, y_g)], f"Q18Y{y_g:g}")
            cand["B_y_only"] = (qg[(18.0, r["y_star"])], f"Q18Y{r['y_star']:g}")

        for nm, (cu, arm) in cand.items():
            ent["pdr"][nm] = float(cu.mean())
            ent["vs_Q18Y0"][nm] = paired(cu, base)
            ent["arms_used"][nm] = arm
        rows[k] = ent
    return rows


def seed_split_oracle(conds=None):
    """표 B 의 승자의 저주 — 시드 0-4 로 (λ,y) 고르고 5-9 로 평가. A 도 같은 5시드로 평가."""
    conds = conds or scan_stages()
    rf = region_inputs()
    out = {}
    for k, c in sorted(conds.items()):
        arms, params, regions, _ = merge_condition(c)
        qg = q_grid(arms, params)
        if len(qg) < 2:
            continue
        sel = {t: cu[:, :5] for t, cu in qg.items()}
        ev = {t: cu[:, 5:] for t, cu in qg.items()}
        b_in = min(sel, key=lambda t: sel[t].mean())
        b_out = min(ev, key=lambda t: ev[t].mean())
        fi = field_inputs(c["knobs"], rf, regions)
        r_lam = snap_log(LAM_COEF * float(fi.TS.iloc[0]), sorted({l for (l, y) in qg if y == 0}) or [18.0])
        r_y = float(y_rule(float(fi.S.median()), float(fi.R.median())))
        a_key = (r_lam, r_y) if (r_lam, r_y) in ev else None
        out[k] = {
            "short": cond_short(c["knobs"]),
            "inner_pick": list(b_in), "outer_best": list(b_out),
            "rank_agree": bool(b_in == b_out),
            "selection_regret": float(ev[b_in].mean() - ev[b_out].mean()),
            "regret_ci95": ci(ev[b_in].mean(1) - ev[b_out].mean(1)),
            "A_regret": (float(ev[a_key].mean() - ev[b_out].mean()) if a_key else None),
            "A_arm": list(a_key) if a_key else None,
            "Q18Y0_regret": (float(ev[(18.0, 0.0)].mean() - ev[b_out].mean())
                             if (18.0, 0.0) in ev else None),
        }
    return out


def rounding(opt, conds=None):
    """카드로 인쇄할 수 있는가 — λ 를 5의 배수로 반올림하면 얼마 손해인가.

    격자에 5의 배수 팔이 항상 있지는 않으므로 두 방식으로 답한다.
      (1) **직접 측정**: `round5(λ공식)` 이 격자에 있으면 λ* 대비 paired Δ 를 그대로 잰다.
      (2) **허용대**: λ* 대비 판정선 미만인 격자 구간 [lo, hi] 를 내고 `round5` 가 그 안에
          드는지 본다. 안에 들면 "측정 해상도에서 반올림 무해" 라고만 말할 수 있다(Δ 는 미측정).
    """
    conds = conds or scan_stages()
    out = {}
    for k, c in sorted(conds.items()):
        r = opt[k]
        if len(r["lam_axis"]) < 2:
            out[k] = {"short": r["short"], "status": "측정 없음(λ 격자 1점)"}
            continue
        arms, params, regions, _ = merge_condition(c)
        qg = q_grid(arms, params)
        p5 = 5 * round(r["lam_formula"] / 5)
        band = (min(r["lam_plateau"]), max(r["lam_plateau"])) if r["lam_plateau"] else None
        ent = {"short": r["short"], "lam_formula": r["lam_formula"], "lam_print5": p5,
               "lam_star": r["lam_star"], "plateau": r["lam_plateau"], "band": band,
               "print5_in_band": bool(band and band[0] <= p5 <= band[1]),
               "print5_on_grid": bool(any(abs(l - p5) < 1e-9 for l in r["lam_axis"]))}
        if ent["print5_on_grid"]:
            pr = paired(qg[(float(p5), 0.0)], qg[(r["lam_star"], 0.0)])
            ent["print5_loss"] = -pr["delta"]
            ent["print5_ci95"] = pr["ci95"]
        snap = snap_log(r["lam_formula"], r["lam_axis"])
        pr = paired(qg[(snap, 0.0)], qg[(r["lam_star"], 0.0)])
        ent["snap_arm"] = snap
        ent["snap_loss"] = -pr["delta"]
        ent["snap_ci95"] = pr["ci95"]
        out[k] = ent
    return out


# --------------------------------------------------------------------------- LOO 정직판
def loo_y_rule(conds=None):
    """y 규칙 임계 3개를 남은 12조건으로 재적합해 보류조건을 예측(누수 없는 정직판).

    적합 목표 = 그 조건의 y* 대비 손실 합. 임계는 격자 탐색이다(닫힌 형태 없음).
    """
    conds = conds or scan_stages()
    rf = region_inputs()
    data = []
    for k, c in sorted(conds.items()):
        arms, params, regions, _ = merge_condition(c)
        qg = q_grid(arms, params)
        ys = sorted({y for (l, y) in qg if l == 18.0 and y in (0.0, 2.0, 4.0, 8.0)})
        if len(ys) < 4:
            continue
        fi = field_inputs(c["knobs"], rf, regions)
        curve = {y: qg[(18.0, y)] for y in ys}
        best = min(curve, key=lambda y: curve[y].mean())
        data.append({"key": k, "short": cond_short(c["knobs"]),
                     "S": float(fi.S.median()), "R": float(fi.R.median()),
                     "curve": {y: float(curve[y].mean()) for y in ys},
                     "cubes": curve, "best": best})
    R_LO = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    R_HI = [0.75, 0.80, 0.85, 0.88, 0.95, 1.05]
    S_HI = [1.2, 1.4, 1.5, 1.6, 1.8]
    def loss(d, th):
        y = float(y_rule(d["S"], d["R"], *th))
        return d["curve"][y] - d["curve"][d["best"]]
    res = []
    for i, d in enumerate(data):
        tr = [x for j, x in enumerate(data) if j != i]
        best_th, best_l = None, float("inf")
        for th in itertools.product(R_LO, R_HI, S_HI):
            l = sum(loss(x, th) for x in tr)
            if l < best_l - 1e-15:
                best_l, best_th = l, th
        y_loo = float(y_rule(d["S"], d["R"], *best_th))
        y_ins = float(y_rule(d["S"], d["R"]))
        pr = paired(d["cubes"][y_loo], d["cubes"][d["best"]])
        res.append({"short": d["short"], "y_star": d["best"], "y_insample": y_ins,
                    "y_loo": y_loo, "th_loo": best_th,
                    "loo_loss": -pr["delta"], "loo_ci95": pr["ci95"],
                    "insample_loss": d["curve"][y_ins] - d["curve"][d["best"]]})
    return res


# --------------------------------------------------------------------------- 출력
def readmap(conds=None):
    conds = conds or scan_stages()
    rows = []
    for k, c in sorted(conds.items()):
        arms, params, regions, dup = merge_condition(c)
        qg = q_grid(arms, params)
        lam = sorted({l for (l, y) in qg if y == 0})
        ys = sorted({y for (l, y) in qg if l == 18.0})
        rows.append({
            "key": k, "short": c["short"], "files": c["files"], "n_arms": len(arms),
            "n_regions": len(regions),
            "lam_grid": lam, "y_grid": ys, "joint_cells": len(qg),
            "has_S062": find_arm(arms, params, kind="cards", axis=0.62, y=0.0) is not None,
            "dup_arms": len(dup), "dup_all_bit_identical": all(d["bit_identical"] for d in dup),
        })
    return rows


def write_text(rm, opt, A, B, ev, ss, rnd, loo, path: Path):
    L = []
    P = L.append
    P("=" * 108)
    P("v22 E5 — 심각도 프리셋 표 (Mode B, 시뮬 재실행 0회)")
    P("좌표셋 budget750(750지역) · 시드 0..9 · 원본 sim 코어 · 하네스 results/scoreboard/v22/retune")
    P(f"판정선 {DECIDE} (v21 CRN paired 실측) · 통계 커널 tools/v20_threshold_report.py")
    P("=" * 108)
    P("")
    P("[표 0] 무엇을 어디서 읽을 수 있는가 — 스테이지 36파일 = 물리조건 %d개" % len(rm))
    P(f"{'조건':14s} {'λ격자(y=0)':34s} {'y격자(λ=18)':14s} {'셀':>4s} {'S0.62':>6s} {'중복팔':>7s} 파일")
    for r in rm:
        lam = ",".join(f"{x:g}" for x in r["lam_grid"]) or "—"
        ys = ",".join(f"{x:g}" for x in r["y_grid"]) or "—"
        dup = f"{r['dup_arms']}{'✓' if r['dup_all_bit_identical'] else '✗'}"
        P(f"{r['short']:14s} {lam[:34]:34s} {ys:14s} {r['joint_cells']:>4d} "
          f"{'있음' if r['has_S062'] else '없음':>6s} {dup:>7s} {','.join(r['files'])}")
    P("")
    P("  · 중복팔 n✓ = 같은 팔이 n쌍 중복 등장했고 전부 **비트동일**(CRN 병합 정당) — 측정 결과")
    P("  · λ격자 1점·y격자 1점인 조건은 그 축에서 '측정 없음'이다. 추정으로 채우지 않았다.")
    P("")
    P("[표 1] 조건별 격자 최적 — λ*(y=0 절단면) · y*(λ=18 절단면) · 결합격자")
    P(f"{'조건':14s} {'λ*':>5s} {'λ공식':>7s} {'λ평탄대':>16s} {'y*':>4s} {'결합최적':>12s} "
      f"{'S':>6s} {'R':>6s} {'T3분':>6s}")
    for k, r in opt.items():
        pl = f"[{min(r['lam_plateau']):g},{max(r['lam_plateau']):g}]" if r["lam_plateau"] else "—"
        js = f"Q{r['joint_star'][0]:g}Y{r['joint_star'][1]:g}" if r["joint_star"] else "—"
        P(f"{r['short']:14s} {str(r['lam_star'] or '—'):>5s} {r['lam_formula']:>7.1f} {pl:>16s} "
          f"{str(r['y_star'] if r['y_star'] is not None else '—'):>4s} {js:>12s} "
          f"{r['field']['S']:>6.2f} {r['field']['R']:>6.2f} {r['field']['T3']:>6.0f}")
    P("")
    P("[표 2-A] 공식형 카드 — 지휘관 입력만으로 (λ, y) 를 정한다 (조건 이름을 쓰지 않는다)")
    for kk in ("lam", "y", "S", "R", "red_km"):
        P(f"    {kk:7s} : {A['form'][kk]}")
    P("")
    P(f"    {'조건(대조용)':14s} {'λ실수':>7s} {'λ5배수':>7s} {'λ격자스냅':>9s} {'y':>3s}")
    for k, r in A["rows"].items():
        P(f"    {r['short']:14s} {r['lam_real']:>7.1f} {r['lam_print5']:>7g} "
          f"{r['lam_snapped']:>9g} {r['y_global']:>3g}")
    P("")
    P("[표 2-B] 조건별 조회표(오라클 상한) — " + B["note"])
    P(f"    {'조건':14s} {'λ*':>5s} {'y*':>4s} {'결합최적':>12s}")
    for k, r in B["rows"].items():
        js = f"Q{r['joint_star'][0]:g}Y{r['joint_star'][1]:g}" if r["joint_star"] else "—"
        P(f"    {r['short']:14s} {str(r['lam_star'] or '—'):>5s} "
          f"{str(r['y_star'] if r['y_star'] is not None else '—'):>4s} {js:>12s}")
    P("")
    P("[표 3] 0-에피소드 평가 — Δ = PDR(Q18Y0) − PDR(후보), +면 후보가 좋다. (승/무/패)는 지역별 95%CI")
    order = ["S0.62", "A_formula", "A_region", "B_lookup", "ORACLE_arm"]
    P(f"{'조건':14s} {'PDR(Q18Y0)':>11s} " + " ".join(f"{o:>26s}" for o in order))
    for k, e in ev.items():
        cells = []
        for o in order:
            cells.append(f"{_fmt(e['vs_Q18Y0'][o]):>26s}" if o in e["vs_Q18Y0"] else f"{'측정 없음':>26s}")
        P(f"{e['short']:14s} {e['pdr']['Q18Y0']:>11.5f} " + " ".join(cells))
    P("")
    P("[표 3-축분해] λ 성분(y=0 고정) 과 y 성분(λ=18 고정) 을 따로 — 결합격자가 없는 조건도 읽을 수 있다")
    P(f"{'조건':14s} {'A_lam_only':>26s} {'B_lam_only':>26s} {'A_y_only':>26s} {'B_y_only':>26s}")
    for k, e in ev.items():
        cs = []
        for o in ("A_lam_only", "B_lam_only", "A_y_only", "B_y_only"):
            cs.append(f"{_fmt(e['vs_Q18Y0'][o]):>26s}" if o in e["vs_Q18Y0"] else f"{'측정 없음':>26s}")
        P(f"{e['short']:14s} " + " ".join(cs))
    P("")
    P("[표 4] 표 B 의 승자의 저주 — 시드 0-4 로 고르고 5-9 로 평가 (서로소 시드, 같은 지역)")
    P(f"{'조건':14s} {'내부선택':>10s} {'외부최적':>10s} {'일치':>5s} {'선택후회':>10s} {'±ci':>9s} "
      f"{'A후회':>10s} {'Q18Y0후회':>10s}")
    for k, r in ss.items():
        P(f"{r['short']:14s} Q{r['inner_pick'][0]:g}Y{r['inner_pick'][1]:g}".ljust(26)
          + f" Q{r['outer_best'][0]:g}Y{r['outer_best'][1]:g}".rjust(10)
          + f" {'○' if r['rank_agree'] else '×':>5s} {r['selection_regret']:>+10.5f} "
          f"{r['regret_ci95']:>9.5f} "
          + (f"{r['A_regret']:>+10.5f} " if r["A_regret"] is not None else f"{'—':>10s} ")
          + (f"{r['Q18Y0_regret']:>+10.5f}" if r["Q18Y0_regret"] is not None else f"{'—':>10s}"))
    P("")
    P("[표 5] 인쇄 가능성 — λ 를 5의 배수로 반올림하면 얼마 손해인가")
    P(f"{'조건':14s} {'λ공식':>7s} {'5배수':>6s} {'λ*':>5s} {'허용대':>14s} {'5배수∈허용대':>12s} "
      f"{'직접측정 손실':>14s} {'스냅팔':>7s} {'스냅 손실':>16s}")
    for k, r in rnd.items():
        if r.get("status"):
            P(f"{r['short']:14s} {r['status']}")
            continue
        band = f"[{r['band'][0]:g},{r['band'][1]:g}]" if r["band"] else "—"
        d5 = (f"{r['print5_loss']:+.5f}±{r['print5_ci95']:.5f}" if "print5_loss" in r else "격자에 없음")
        P(f"{r['short']:14s} {r['lam_formula']:>7.1f} {r['lam_print5']:>6g} {r['lam_star']:>5g} "
          f"{band:>14s} {'○' if r['print5_in_band'] else '×':>12s} {d5:>14s} "
          f"{r['snap_arm']:>7g} {r['snap_loss']:+.5f}±{r['snap_ci95']:.5f}")
    P("")
    P("[표 6] y 규칙의 LOO 정직판 — 임계 3개를 남은 12조건으로 재적합 후 보류조건 예측")
    P(f"{'조건':14s} {'y*':>4s} {'y(in-sample)':>13s} {'y(LOO)':>7s} {'LOO 임계':>22s} "
      f"{'LOO 손실':>18s} {'in-sample 손실':>14s}")
    for r in loo:
        P(f"{r['short']:14s} {r['y_star']:>4g} {r['y_insample']:>13g} {r['y_loo']:>7g} "
          f"{str(r['th_loo']):>22s} {r['loo_loss']:>+10.5f}±{r['loo_ci95']:.5f} "
          f"{r['insample_loss']:>+14.5f}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["readmap", "optima", "table", "eval", "round", "loo", "all"])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    conds = scan_stages()
    print(f"# 스테이지 파일 {sum(len(c['files']) for c in conds.values())}개 → 물리조건 {len(conds)}개")
    if a.what == "readmap":
        rm = readmap(conds)
        (outd / "readmap.json").write_text(json.dumps(rm, ensure_ascii=False, indent=1))
        for r in rm:
            print(f"{r['short']:14s} λ={r['lam_grid']} y={r['y_grid']} cells={r['joint_cells']} "
                  f"S0.62={r['has_S062']} dup={r['dup_arms']}/{r['dup_all_bit_identical']}")
        return
    if a.what in ("optima", "table", "eval", "round", "loo", "all"):
        opt = optima(conds, verbose=(a.what == "optima"))
        (outd / "optima.json").write_text(json.dumps(
            {k: {kk: vv for kk, vv in v.items() if kk != "dup_checked"} for k, v in opt.items()},
            ensure_ascii=False, indent=1, default=float))
    if a.what == "optima":
        return
    A, B = build_tables(opt)
    (outd / "preset_table.json").write_text(json.dumps({"A_formula": A, "B_lookup": B},
                                                       ensure_ascii=False, indent=1))
    if a.what == "table":
        print(json.dumps(A, ensure_ascii=False, indent=1))
        print(json.dumps(B, ensure_ascii=False, indent=1))
        return
    ev = evaluate(opt, conds)
    ss = seed_split_oracle(conds)
    rnd = rounding(opt, conds)
    loo = loo_y_rule(conds)
    (outd / "eval.json").write_text(json.dumps(
        {"decide_line": DECIDE, "vs": ev, "seed_split": ss, "rounding": rnd, "loo_y": loo},
        ensure_ascii=False, indent=1, default=float))
    txt = write_text(readmap(conds), opt, A, B, ev, ss, rnd, loo, outd / "preset_card.txt")
    print(txt)


if __name__ == "__main__":
    main()
