# -*- coding: utf-8 -*-
"""v22 E2 — 현장 규칙집(CARD)의 **최소 충분 파라미터 집합**.

시뮬레이션을 재실행하지 않는다. 완주된 에피소드 CSV 만 읽어 paired 통계를 낸다.
판정 커널은 ``tools/v20_threshold_report.py`` 의 ``ci``/``cube``/``paired`` 를 그대로 쓴다
(재구현하면 W/T/L 정의가 갈린다 — v5 에서 실제로 그 혼동이 있었다).

질문은 하나다: **카드에 적어야 하는 숫자는 정확히 몇 개인가.**
후보 파라미터를 축별로 하나씩 켜고, 같은 기준선(채택 카드 ``cardt:18,6.6,0,hingerate``)
대비 판정선을 넘는 기여를 못 내면 카드에서 뺀다. 빼는 것도 결과이므로 제외 사실과
그 측정값을 함께 남긴다(v14 의 UAV 임계 기각 선례와 같은 형식).

축 7개
  A1 λ 부하 교환율          유지 여부와 **자유 파라미터인지**(공식으로 대체 가능한지)
  A2 부하항 함수형          hingerate / hinge1 / 선형(load) / zero
  A3 등급 임계 yhold        조건별 최적과 "숫자 없는 양끝"(Yellow우선·Red우선) 대비
  A4 수단 임계 red_gain_min 숫자를 빼고 "빠른 쪽"(G0)으로 바꿀 수 있는가
  A5 부하 신호 출처         occ+in_flight / occ / in_flight / p_sent / 없음
  A6 지역화 파라미터표      전국 단일 vs 시군구별 vs 좌표별 (정직한 시드 분할 + 오라클 상한)
  A7 무튜닝 S족 wait_scale  0.62 가 이론값 1.0 과 이웃 격자 대비 판정선을 버는가

핵심 규약
  * 판정선 0.00053 은 v21 CRN paired 실측값이다. 새로 만들지 않고 이 값을 쓰되
    모든 항목에 실제 paired 95%CI 를 병기한다.
  * 좌표셋은 **budget750 하나뿐**이다. test750 / v19 tradeoff250 / eval250 매니페스트가
    붙은 CSV 는 읽지 않고 예외로 막는다(``_meta_gate``). v20 fieldinfo 의 봉투 밖 4조건이
    실제로 tradeoff250 이라 여기서 자동 제외되며, 그 결과가 A5 의 "측정 없음" 칸이다.
  * 한 물리조건이 여러 스테이지 파일에 흩어져 있다. 같은 조건의 파일들은 겹치는 팔이
    **비트동일**할 때만 합친다(``_merge`` 의 앵커 게이트). 실측 확인: base 조건의
    ``base_base.Q18`` == ``lam_base.Q18`` == ``fieldinfo.Q18`` == ``yhold_base.Y0``
    == ``red_base.G6.6`` == ``lamx_base.Q18Y0`` 전부 maxabs 0.0.
  * 격자 끝이 최적이면 그것은 최적이 아니다(AGENTS.md 계약 7). 최적 팔이 격자 양끝이면
    ``grid_edge`` 로 표시하고 그 값을 "하한"으로만 읽는다.
  * 축의 기여를 격자에서 argmin 으로 고르는 것은 **선택 편향**이 있다(v22 ⑨).
    그래서 모든 "재튜닝 이득" 을 두 벌로 낸다 — ``insample``(같은 시드로 고르고 재는 상한)
    과 ``honest``(시드 전반부로 고르고 후반부에서 재는 배포 추정치).

사용
    python tools/v22_sufficiency_report.py inventory
    python tools/v22_sufficiency_report.py axes
    python tools/v22_sufficiency_report.py tables
    python tools/v22_sufficiency_report.py all
산출 ``results/field/sufficiency/`` — inventory.{json,csv} · axes.json · table_{keep,drop}.csv
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
from v20_threshold_report import ci, cube, paired  # noqa: E402

# v21 실측 CRN paired 두 팔 차이 95%CI. 보편 상수가 아니라 그 조건의 실측값이다.
JUDGE_LINE = 0.00053

# v22 정본: λ* = 18.97 × 치료시간배수 (치료시간 축 로그-로그 기울기 +1.121, 이론 +1).
LAM_FORMULA = 18.97

# 채택 카드. 모든 축의 공통 기준선이다.
CARD_SPEC = "cardt:18,6.6,0,hingerate"

OUT = REPO / "results/field/sufficiency"
ALLOWED_MANIFEST = "sigungu30_budget750_manifest.json"
FORBIDDEN = ("test750", "tradeoff250", "eval250")

# ── 팔 이름 → (함수형, 부하 신호, 도달축) ────────────────────────────────────
# 드라이버(run_v22_field.sh · run_v21_infoladder.sh · run_v20_fieldinfo.sh)의 조립기와
# src/rl_src/v17_field_rules.py 의 `_load_vector` 에서 그대로 읽었다. 추측 없음.
FAMILY = {
    # I3 = occ + in_flight (병원 census + 내 이송기록) — 병원 통신 필요
    "Q":  ("hingerate", "occ+in_flight", "min", "λ"),
    "H":  ("hinge1",    "occ+in_flight", "min", "λ"),
    "T":  ("linear",    "occ+in_flight", "min", "λ"),
    # I2 = occ 단독 (병원 입원 census 만) — 병원 통신 필요
    "OR": ("hingerate", "occ",           "min", "λ"),
    "OH": ("hinge1",    "occ",           "min", "λ"),
    "OL": ("linear",    "occ",           "min", "λ"),
    # I1a = in_flight (그 병원행 이송중) — 통신 불요
    "F":  ("hingerate", "in_flight",     "min", "λ"),
    "FH": ("hinge1",    "in_flight",     "min", "λ"),
    "FL": ("linear",    "in_flight",     "min", "λ"),
    # I1b = p_sent (내가 보낸 누적) — 통신 불요
    "P":  ("hingerate", "p_sent",        "min", "λ"),
    "PH": ("hinge1",    "p_sent",        "min", "λ"),
    "PL": ("linear",    "p_sent",        "min", "λ"),
    # I0 = 부하 무시
    "Z":  ("zero",      "없음",           "min", "λ(무의미)"),
    # 구 CARD — 거리(km) 축 + 선형 부하. 함수형과 도달축이 **동시에** 다르다(교란).
    "K":  ("linear",    "occ+in_flight", "km",  "λ"),
    # 축 전용 팔
    "S":  ("surv",      "occ+in_flight", "min", "wait_scale"),
    "Y":  ("hingerate", "occ+in_flight", "min", "yhold(λ=18)"),
    "SY": ("surv",      "occ+in_flight", "min", "yhold(w=0.62)"),
    "R":  ("linear",    "occ+in_flight", "km",  "yhold(K,λ=12)"),
    "G":  ("hingerate", "occ+in_flight", "min", "red_gain(분)"),
    "D":  ("linear",    "occ+in_flight", "km",  "red_km"),
}
_FAM_ALT = "|".join(sorted(FAMILY, key=len, reverse=True))
ARM_RE = re.compile(rf"^({_FAM_ALT})(-?[0-9.]+)$")
JOINT_RE = re.compile(r"^Q([0-9.]+)Y([0-9.]+)$")
SIGUNGU_RE = re.compile(r"^(.*)_q\d+$")


def parse_arm(pol: str):
    """팔 이름 → (가족, 값) 또는 결합팔 → ('QY', (λ, y)). 모르는 이름은 (None, None)."""
    m = JOINT_RE.match(pol)
    if m:
        return "QY", (float(m.group(1)), float(m.group(2)))
    m = ARM_RE.match(pol)
    if m:
        return m.group(1), float(m.group(2))
    return None, None


# ── 조건(물리 노브) 로딩 ─────────────────────────────────────────────────────

def _meta(path: Path) -> dict:
    mp = Path(str(path) + ".meta.json")
    if not mp.exists():
        raise RuntimeError(f"{path.name}: .meta.json 없음 = 미완주. 판정에 쓰지 않는다")
    return json.loads(mp.read_text(encoding="utf-8"))


def _meta_gate(path: Path) -> dict:
    """완주·좌표셋 게이트. 판정셋 매니페스트가 붙은 CSV 는 여기서 막는다."""
    m = _meta(path)
    man = str(m.get("manifest", ""))
    base = man.rsplit("/", 1)[-1]
    if any(bad in man for bad in FORBIDDEN):
        raise PermissionError(f"{path.name}: 판정셋 매니페스트({base}) — 읽지 않는다")
    if base != ALLOWED_MANIFEST:
        raise PermissionError(f"{path.name}: 허용 좌표셋 아님({base})")
    if int(m.get("n_rows", -1)) < 0:
        raise RuntimeError(f"{path.name}: 메타에 n_rows 없음")
    return m


def _knobs(m: dict) -> dict:
    return {k.replace("MCI_", ""): str(v) for k, v in (m.get("scenario_knobs") or {}).items() if v}


def _knob_key(kn: dict) -> str:
    return "base" if not kn else ",".join(f"{k}={v}" for k, v in sorted(kn.items()))


class Cond:
    """한 물리조건의 (팔 → 지역×시드 큐브) 묶음. 좌표·시드축이 전부 정렬돼 있다."""

    def __init__(self, key: str, knobs: dict):
        self.key = key
        self.knobs = knobs
        self.arms: dict[str, np.ndarray] = {}
        self.src: dict[str, str] = {}
        self.regions: pd.Index | None = None
        self.seeds: list[int] = []
        self.paths: list[str] = []
        self.rows = 0

    # 팔 값 조회 helpers
    def fam(self, f: str) -> list[tuple[float, str]]:
        """가족 f 의 (값, 팔이름) 오름차순. 결합팔은 제외한다."""
        out = []
        for pol in self.arms:
            g, v = parse_arm(pol)
            if g == f and not isinstance(v, tuple):
                out.append((float(v), pol))
        return sorted(out)

    def mean(self, pol: str) -> float:
        return float(self.arms[pol].mean())


def _load_one(path: Path):
    m = _meta_gate(path)
    d = pd.read_csv(path, usecols=["region", "policy", "seed", "pdr_woG"], encoding="utf-8-sig")
    dup = d.duplicated(["region", "policy", "seed"]).sum()
    if dup:
        raise RuntimeError(f"{path.name}: (region,policy,seed) 중복 {dup}행 — cube 가 평균내 가린다")
    if not np.isfinite(d.pdr_woG.to_numpy(float)).all():
        raise RuntimeError(f"{path.name}: pdr_woG 에 비유한값")
    return m, d


def _merge(key: str, knobs: dict, paths: list[Path]) -> Cond:
    """같은 물리조건의 CSV 들을 합친다.

    합치는 조건은 셋이다 — ① 매니페스트 sha256 동일 ② 지역 인덱스·시드집합 동일
    ③ 두 파일에 같이 있는 팔이 **비트동일**(CRN 앵커). ③이 깨지면 코어/노브가
    달라진 것이므로 합치지 않고 예외를 던진다. 조건이 같아도 좌표가 다르면 짝이 아니다.
    """
    c = Cond(key, knobs)
    sha = None
    for p in paths:
        m, d = _load_one(p)
        if sha is None:
            sha = m.get("manifest_sha256")
        elif m.get("manifest_sha256") != sha:
            raise RuntimeError(f"{p.name}: manifest_sha256 불일치 — 합칠 수 없다")
        if _knobs(m) != knobs:
            raise RuntimeError(f"{p.name}: 노브 불일치 {_knobs(m)} != {knobs}")
        pv_all = d.pivot_table(index="region", columns="seed", values="pdr_woG",
                               aggfunc="count").sort_index()
        if c.regions is None:
            c.regions, c.seeds = pv_all.index, [int(s) for s in pv_all.columns]
        else:
            if not pv_all.index.equals(c.regions):
                raise RuntimeError(f"{p.name}: 지역 인덱스 불일치")
            if [int(s) for s in pv_all.columns] != c.seeds:
                raise RuntimeError(f"{p.name}: 시드집합 불일치")
        for pol in sorted(d.policy.unique()):
            cb = cube(d, pol)                      # 결측이면 여기서 예외 = 완주 검사
            if pol in c.arms:
                if not np.array_equal(cb, c.arms[pol]):
                    raise RuntimeError(
                        f"{key}/{pol}: {p.name} 와 {c.src[pol]} 의 값이 다르다 "
                        f"(maxabs {np.abs(cb - c.arms[pol]).max():.3e}) — CRN 앵커 게이트 실패")
                continue
            c.arms[pol] = cb
            c.src[pol] = p.name
        c.paths.append(str(p.relative_to(REPO)))
        c.rows += int(m["n_rows"])
    return c


def discover() -> dict[str, Cond]:
    """budget750 위의 완주 CSV 를 물리조건별로 모은다."""
    cand: list[Path] = []
    cand += sorted((REPO / "results/scoreboard/v22/retune").glob("*.csv"))
    cand += [REPO / "results/scoreboard/v20/budget/lam_base.csv"]
    cand += sorted((REPO / "results/scoreboard/v20/fieldinfo").glob("*.csv"))
    cand += sorted((REPO / "results/scoreboard/v21/infoladder").glob("*.csv"))

    groups: dict[str, tuple[dict, list[Path]]] = {}
    skipped: list[str] = []
    for p in cand:
        try:
            m = _meta_gate(p)
        except (PermissionError, RuntimeError) as e:
            skipped.append(str(e))
            continue
        kn = _knobs(m)
        k = _knob_key(kn)
        groups.setdefault(k, (kn, []))[1].append(p)

    conds = {}
    for k, (kn, paths) in sorted(groups.items()):
        conds[k] = _merge(k, kn, paths)
    conds["__skipped__"] = skipped            # type: ignore[assignment]
    return conds


def treat_scale(kn: dict) -> float:
    return float(kn.get("TREAT_SCALE", 1.0))


# ── 판정 helpers ────────────────────────────────────────────────────────────

def verdict(delta: float, ci95: float) -> str:
    """delta = PDR(기준) − PDR(후보). 양수 = 후보가 개선."""
    if delta <= ci95 and delta >= -ci95:
        return "동률(CI내)"
    if delta > JUDGE_LINE:
        return "판정선초과"
    if delta > 0:
        return "유의·판정선미만"
    if delta < -JUDGE_LINE:
        return "열세·판정선초과"
    return "열세·판정선미만"


def meas(cand: np.ndarray, base: np.ndarray) -> dict:
    """paired(후보, 기준) → delta = PDR(기준) − PDR(후보). 양수 = 후보 개선."""
    r = paired(cand, base)
    r["verdict"] = verdict(r["delta"], r["ci95"])
    r["exceeds_line"] = bool(r["delta"] > JUDGE_LINE and r["delta"] > r["ci95"])
    return r


def _sel_ev(seeds: list[int]):
    """시드 분할. 전반부로 고르고 후반부에서 잰다(선택 편향 제거)."""
    n = len(seeds)
    h = n // 2
    return list(range(h)), list(range(h, n))


def pick(arms: dict[str, np.ndarray], names: list[str], cols: list[int]) -> str:
    """전국 단일 선택 — 주어진 시드열에서 (지역×시드) 평균이 최소인 팔."""
    return min(names, key=lambda p: float(arms[p][:, cols].mean()))


def grid_edge(val: float, vals: list[float]) -> str:
    if not vals:
        return "?"
    if val <= min(vals):
        return "격자아래끝"
    if val >= max(vals):
        return "격자위끝"
    return "내부"


def tune_gain(c: Cond, names: list[str], base_pol: str, vals: dict[str, float]) -> dict:
    """축 하나를 자유롭게 두었을 때의 이득. insample(상한) + honest(배포 추정).

    names 는 그 축의 후보 팔 전체(기준 팔 포함)여야 한다.
    """
    base = c.arms[base_pol]
    allc = list(range(len(c.seeds)))
    sel, ev = _sel_ev(c.seeds)
    vlist = [vals[p] for p in names]

    b_in = pick(c.arms, names, allc)
    b_ho = pick(c.arms, names, sel)
    out = {
        "n_arms": len(names),
        "grid": sorted(vlist),
        "insample": {"arm": b_in, "value": vals[b_in],
                     "edge": grid_edge(vals[b_in], vlist),
                     **meas(c.arms[b_in], base)},
        "honest": {"arm": b_ho, "value": vals[b_ho],
                   "edge": grid_edge(vals[b_ho], vlist),
                   "sel_seeds": [c.seeds[i] for i in sel],
                   "ev_seeds": [c.seeds[i] for i in ev],
                   **meas(c.arms[b_ho][:, ev], base[:, ev])},
        "curve": {p: c.mean(p) for p in sorted(names, key=lambda q: vals[q])},
    }
    return out


# ══ A1. λ 부하 교환율 ══════════════════════════════════════════════════════

def axis_lam(conds: dict[str, Cond]) -> dict:
    """λ 가 카드에 남는가, 그리고 그것이 **자유 파라미터인가**.

    세 수치로 답한다.
      free   : 조건별로 λ 를 격자에서 자유롭게 고르면 고정 Q18 대비 얻는 이득
      form   : λ = 18.97 × 치료시간배수 를 격자에 로그-반올림한 팔의 이득 (에피소드 0개)
      resid  : 공식 팔 대비 자유 최적 팔이 더 얻는 몫 = λ 를 시뮬로 고를 이유
    resid 가 판정선 미만이면 λ 는 카드의 **자유 숫자가 아니라 공식**이다.
    """
    rows = []
    for k, c in conds.items():
        q = c.fam("Q")
        if len(q) < 4 or "Q18" not in c.arms:
            continue
        names = [p for _, p in q]
        vals = {p: v for v, p in q}
        g = tune_gain(c, names, "Q18", vals)
        ts = treat_scale(c.knobs)
        lam_t = LAM_FORMULA * ts
        f_arm = min(names, key=lambda p: abs(math.log(vals[p]) - math.log(lam_t)))
        row = {
            "cond": k, "knobs": c.knobs, "treat_scale": ts,
            "baseline": "Q18", "baseline_pdr": c.mean("Q18"),
            "free": g, "grid": g["grid"],
            "formula": {"lam_target": round(lam_t, 2), "arm": f_arm, "value": vals[f_arm],
                        "pdr": c.mean(f_arm), **meas(c.arms[f_arm], c.arms["Q18"])},
            "residual_formula_to_free": meas(c.arms[g["insample"]["arm"]], c.arms[f_arm]),
            "formula_picks_argmin": bool(f_arm == g["insample"]["arm"]),
        }
        rows.append(row)
    rows.sort(key=lambda r: -r["free"]["insample"]["delta"])
    return {"axis": "A1 λ 부하 교환율", "param": "λ (분/명)", "rows": rows}


# ══ A2. 부하항 함수형 ═══════════════════════════════════════════════════════

FORM_FAMS = [("hingerate", "Q"), ("hinge1", "H"), ("linear", "T"), ("zero", "Z")]


def axis_form(conds: dict[str, Cond]) -> dict:
    """같은 부하 신호(occ+in_flight)에 함수형만 갈아끼운다. 각 형태는 **자기 최적 λ** 에서.

    λ 단위가 형태마다 다르므로(rate 는 서비스시간, hinge 는 서비스시간/수술실수,
    linear 는 명당 분) 고정 λ 로 비교하면 축이 섞인다. zero 는 λ 가 무의미하다.
    """
    rows = []
    for k, c in conds.items():
        if "Q18" not in c.arms:
            continue
        got = {}
        for form, fam in FORM_FAMS:
            arms = c.fam(fam)
            if not arms:
                continue
            names = [p for _, p in arms]
            vals = {p: v for v, p in arms}
            allc = list(range(len(c.seeds)))
            sel, ev = _sel_ev(c.seeds)
            b_in, b_ho = pick(c.arms, names, allc), pick(c.arms, names, sel)
            got[form] = {
                "family": fam, "n_arms": len(names), "grid": sorted(vals.values()),
                "best_arm": b_in, "best_lam": vals[b_in], "best_pdr": c.mean(b_in),
                "edge": grid_edge(vals[b_in], list(vals.values())),
                "vs_card": meas(c.arms[b_in], c.arms["Q18"]),
                "honest_arm": b_ho,
                "vs_card_honest": meas(c.arms[b_ho][:, ev], c.arms["Q18"][:, ev]),
                "curve": {p: c.mean(p) for p in sorted(names, key=lambda q: vals[q])},
            }
        if len(got) < 2:
            continue
        ref = got.get("hingerate")
        if ref:
            for form, g in got.items():
                if form == "hingerate":
                    continue
                # cost = PDR(대체형) − PDR(hingerate 최적) > 0 이면 대체형이 열세
                g["cost_vs_hingerate"] = meas(c.arms[ref["best_arm"]], c.arms[g["best_arm"]])
        rows.append({"cond": k, "knobs": c.knobs, "baseline_pdr": c.mean("Q18"), "forms": got})
    return {"axis": "A2 부하항 함수형", "param": "함수형(Z 의 모양)",
            "note": "K족(거리축+선형)은 도달축까지 달라 함수형 축에 넣지 않는다(A2b 별도)",
            "rows": rows}


def axis_form_k(conds: dict[str, Cond]) -> dict:
    """A2b — 구 CARD(K족: 거리 km 축 + 선형 부하). 함수형과 도달축이 동시에 다르다."""
    rows = []
    for k, c in conds.items():
        arms = c.fam("K")
        if len(arms) < 3 or "Q18" not in c.arms:
            continue
        names = [p for _, p in arms]
        vals = {p: v for v, p in arms}
        allc = list(range(len(c.seeds)))
        b = pick(c.arms, names, allc)
        rows.append({"cond": k, "knobs": c.knobs, "best_arm": b, "best_lam": vals[b],
                     "best_pdr": c.mean(b), "edge": grid_edge(vals[b], list(vals.values())),
                     "vs_card": meas(c.arms[b], c.arms["Q18"]),
                     "K12_vs_card": meas(c.arms["K12"], c.arms["Q18"]) if "K12" in c.arms else None})
    return {"axis": "A2b 거리축+선형(구 CARD K족)", "param": "λ(km/명)", "rows": rows}


# ══ A3. 등급 임계 yhold ════════════════════════════════════════════════════

def axis_yhold(conds: dict[str, Cond]) -> dict:
    """등급 임계가 **숫자**로 필요한가.

    두 가지를 센다.
      vs_card     : 채택값 Y0 대비 그 조건 최적 Y 의 이득 (research.md ⑤ 와 같은 정의)
      needs_number: 숫자 없는 양끝(Y0 = Yellow 우선 · Y9999 = Red 우선) 중 더 좋은 쪽 대비
                    **내부 격자값**(2·4·8·16·32)이 더 얻는 몫.
    카드에 "숫자"를 적어야 하는 이유는 두 번째 쪽이다. 첫 번째만 보면 Y0↔Y9999
    뒤집힘까지 "임계값 튜닝" 으로 오해한다.
    """
    rows = []
    for k, c in conds.items():
        arms = c.fam("Y")
        if len(arms) < 4 or "Y0" not in c.arms:
            continue
        names = [p for _, p in arms]
        vals = {p: v for v, p in arms}
        ends = [p for p in names if vals[p] <= 0 or vals[p] >= 9999]
        inner = [p for p in names if p not in ends]
        allc = list(range(len(c.seeds)))
        sel, ev = _sel_ev(c.seeds)
        best_end = pick(c.arms, ends, allc)
        best_inner_in = pick(c.arms, inner, allc)
        best_inner_ho = pick(c.arms, inner, sel)
        row = {
            "cond": k, "knobs": c.knobs, "baseline": "Y0", "baseline_pdr": c.mean("Y0"),
            "free": tune_gain(c, names, "Y0", vals),
            "best_end": {"arm": best_end, "pdr": c.mean(best_end),
                         **meas(c.arms[best_end], c.arms["Y0"])},
            "needs_number": {"best_inner": best_inner_in, "value": vals[best_inner_in],
                             "edge": grid_edge(vals[best_inner_in], [vals[p] for p in inner]),
                             **meas(c.arms[best_inner_in], c.arms[best_end])},
            "needs_number_honest": {"best_inner": best_inner_ho,
                                    **meas(c.arms[best_inner_ho][:, ev], c.arms[best_end][:, ev])},
            "curve": {p: c.mean(p) for p in sorted(names, key=lambda q: vals[q])},
        }
        # S족(무튜닝 카드) 위에서도 등급 축이 필요한가
        sy = c.fam("SY")
        if len(sy) >= 3 and "SY0" in c.arms:
            sn = [p for _, p in sy]
            sv = {p: v for v, p in sy}
            row["on_S_family"] = tune_gain(c, sn, "SY0", sv)
        rows.append(row)
    rows.sort(key=lambda r: -r["free"]["insample"]["delta"])
    n_line = sum(1 for r in rows if r["free"]["insample"]["exceeds_line"])
    n_num = sum(1 for r in rows if r["needs_number"]["exceeds_line"])
    return {"axis": "A3 등급 임계 yhold", "param": "yhold(명)",
            "n_cond": len(rows), "n_exceed_vs_card": n_line,
            "n_exceed_needs_number": n_num, "rows": rows}


# ══ A4. 수단 임계 red_gain_min ═════════════════════════════════════════════

def axis_red(conds: dict[str, Cond]) -> dict:
    """수단 임계 6.6분을 카드에서 뺄 수 있는가.

    빼는 방법이 둘이라 둘 다 잰다.
      drop_to_G0   숫자만 뺀다 — "두 수단 다 대기 중이면 **빠른 쪽**"(임계 0). 파라미터 0개.
      drop_rule    수단 비교 자체를 뺀다 — 임계를 사실상 +∞ 로 밀어 둘 다 있으면 항상 AMB.
                   (격자 위끝 G45 로 근사. 실제 |t_a−t_u| 가 45분을 넘는 경우는 드물다.)
    두 대체안이 채택값 G6.6 과 판정선 안에서 동률이면 6.6 은 카드에서 빠진다.
    """
    rows = []
    for k, c in conds.items():
        arms = c.fam("G")
        if len(arms) < 4 or "G6.6" not in c.arms:
            continue
        names = [p for _, p in arms]
        vals = {p: v for v, p in arms}
        top = max(names, key=lambda p: vals[p])
        row = {
            "cond": k, "knobs": c.knobs, "baseline": "G6.6", "baseline_pdr": c.mean("G6.6"),
            "free": tune_gain(c, names, "G6.6", vals),
            "drop_to_G0": ({"arm": "G0", "pdr": c.mean("G0"),
                            **meas(c.arms["G0"], c.arms["G6.6"])} if "G0" in c.arms else None),
            "drop_rule_alwaysAMB": {"arm": top, "value": vals[top], "pdr": c.mean(top),
                                    **meas(c.arms[top], c.arms["G6.6"])},
            "curve": {p: c.mean(p) for p in sorted(names, key=lambda q: vals[q])},
        }
        d = c.fam("D")
        if len(d) >= 3:
            dn = [p for _, p in d]
            dv = {p: v for v, p in d}
            row["on_K_family_red_km"] = tune_gain(c, dn, "D12", dv) if "D12" in c.arms else None
        rows.append(row)
    rows.sort(key=lambda r: -r["free"]["insample"]["delta"])
    return {"axis": "A4 수단 임계 red_gain_min", "param": "red_gain_min(분)",
            "n_cond": len(rows),
            "n_exceed_vs_card": sum(1 for r in rows if r["free"]["insample"]["exceeds_line"]),
            "n_G0_equiv": sum(1 for r in rows if r["drop_to_G0"]
                              and abs(r["drop_to_G0"]["delta"]) <= JUDGE_LINE),
            "rows": rows}


# ══ A5. 부하 신호 출처 ═════════════════════════════════════════════════════

LADDER = [("I3 occ+in_flight", {"hingerate": "Q", "hinge1": "H", "linear": "T"}, "병원통신 필요"),
          ("I2 occ 단독",       {"hingerate": "OR", "hinge1": "OH", "linear": "OL"}, "병원통신 필요"),
          ("I1a in_flight",     {"hingerate": "F", "hinge1": "FH", "linear": "FL"}, "통신 불요"),
          ("I1b p_sent",        {"hingerate": "P", "hinge1": "PH", "linear": "PL"}, "통신 불요"),
          ("I0 부하 무시",       {"zero": "Z"}, "통신 불요")]


def axis_signal(conds: dict[str, Cond]) -> dict:
    """부하 신호의 출처를 갈아끼운다. 함수형은 같은 것끼리만 비교한다."""
    rows = []
    for k, c in conds.items():
        if "Q18" not in c.arms:
            continue
        lv = {}
        for name, forms, comm in LADDER:
            cells = {}
            for form, fam in forms.items():
                arms = c.fam(fam)
                if not arms:
                    continue
                names = [p for _, p in arms]
                vals = {p: v for v, p in arms}
                allc = list(range(len(c.seeds)))
                sel, ev = _sel_ev(c.seeds)
                b = pick(c.arms, names, allc)
                bh = pick(c.arms, names, sel)
                cells[form] = {"family": fam, "n_arms": len(names), "best_arm": b,
                               "best_lam": vals[b], "best_pdr": c.mean(b),
                               "edge": grid_edge(vals[b], list(vals.values())),
                               "vs_card": meas(c.arms[b], c.arms["Q18"]),
                               "vs_card_honest": meas(c.arms[bh][:, ev], c.arms["Q18"][:, ev])}
            if cells:
                lv[name] = {"comm": comm, "cells": cells}
        if len(lv) < 2:
            continue
        rows.append({"cond": k, "knobs": c.knobs, "baseline_pdr": c.mean("Q18"), "levels": lv})
    return {"axis": "A5 부하 신호 출처", "param": "신호 출처(통신 요구)", "rows": rows}


# ══ A6. 지역화 파라미터표 ═══════════════════════════════════════════════════

def _groups(regions: pd.Index, level: str) -> dict[str, np.ndarray]:
    if level == "L0":
        return {"전국": np.arange(len(regions))}
    if level == "L3":
        return {r: np.array([i]) for i, r in enumerate(regions)}
    key = []
    for r in regions:
        m = SIGUNGU_RE.match(str(r))
        key.append(m.group(1) if m else str(r))
    out: dict[str, list[int]] = {}
    for i, kk in enumerate(key):
        out.setdefault(kk, []).append(i)
    return {k: np.array(v) for k, v in out.items()}


def _assemble(c: Cond, names: list[str], level: str, sel: list[int], ev: list[int]):
    """레벨별 파라미터표를 sel 시드에서 고르고 ev 시드 큐브를 조립한다."""
    gs = _groups(c.regions, level)
    nr, ne = len(c.regions), len(ev)
    out = np.empty((nr, ne), float)
    chosen: dict[str, str] = {}
    M = {p: c.arms[p] for p in names}
    for gk, idx in gs.items():
        best, bv = None, np.inf
        for p in names:
            v = float(M[p][np.ix_(idx, sel)].mean())
            if v < bv:
                best, bv = p, v
        chosen[gk] = best
        out[idx, :] = M[best][np.ix_(idx, ev)]
    return out, chosen, len(gs)


def axis_local(conds: dict[str, Cond]) -> dict:
    """지역화 상한. 전국 단일 대비 시군구별·좌표별 파라미터표의 이득.

    ★ 같은 시드로 고르고 재면 지역별 선택은 잡음을 먹어 **자동으로 이긴다**(승자의 저주).
    그래서 두 벌을 낸다 — ``oracle``(같은 10시드 = 상한, 편향 있음)과
    ``honest``(seed 0-4 로 고르고 5-9 에서 잼 = 배포 추정치). honest 가 음수면
    지역화는 그 예산에서 **손해**라는 뜻이다.
    """
    targets = [("λ(Q족)", "Q", "Q18"), ("yhold(Y족)", "Y", "Y0"),
               ("red_gain(G족)", "G", "G6.6")]
    rows = []
    for k, c in conds.items():
        allc = list(range(len(c.seeds)))
        sel, ev = _sel_ev(c.seeds)
        for label, fam, base_pol in targets:
            arms = c.fam(fam)
            if len(arms) < 4 or base_pol not in c.arms:
                continue
            names = [p for _, p in arms]
            rec = {"cond": k, "knobs": c.knobs, "param": label, "n_arms": len(names),
                   "n_regions": len(c.regions), "baseline": base_pol}
            for tag, s_, e_ in (("oracle", allc, allc), ("honest", sel, ev)):
                l0, ch0, n0 = _assemble(c, names, "L0", s_, e_)
                blk = {"L0_arm": list(ch0.values())[0], "n_L0": n0}
                for lv in ("L2", "L3"):
                    cb, ch, ng = _assemble(c, names, lv, s_, e_)
                    blk[lv] = {"n_params": ng, "n_distinct": len(set(ch.values())),
                               **meas(cb, l0)}
                blk["L0_pdr"] = float(l0.mean())
                rec[tag] = blk
            rows.append(rec)
    return {"axis": "A6 지역화 파라미터표", "param": "좌표/시군구별 파라미터표", "rows": rows}


def axis_local_budgetcurve() -> dict:
    """A6 보강 — 예산곡선 산출로 **서로소 시드**에서 지역화를 잰다.

    내부 seed 0..127 로 고르고 평가 seed 1000..1059 로 잰다(좌표 60개·팔 32개).
    budget750 10시드 분할보다 선택 시드가 훨씬 많아 "지역화가 이기는 예산" 을 볼 수 있다.
    """
    bc = REPO / "results/scoreboard/v22/budgetcurve"
    out = []
    for tag in ("base", "ts40"):
        pi, pe = bc / f"{tag}_inner.csv", bc / f"{tag}_eval.csv"
        if not (pi.exists() and pe.exists()):
            continue
        mi, di = _load_one(pi)
        me, de = _load_one(pe)
        kn = _knobs(mi)
        if kn != _knobs(me):
            raise RuntimeError(f"budgetcurve/{tag}: inner·eval 노브 불일치")
        ci_, ce = Cond(tag, kn), Cond(tag, kn)
        for c_, d_ in ((ci_, di), (ce, de)):
            pv = d_.pivot_table(index="region", columns="seed", values="pdr_woG",
                                aggfunc="count").sort_index()
            c_.regions, c_.seeds = pv.index, [int(s) for s in pv.columns]
            for pol in sorted(d_.policy.unique()):
                c_.arms[pol] = cube(d_, pol)
        if not ci_.regions.equals(ce.regions):
            raise RuntimeError(f"budgetcurve/{tag}: 좌표 불일치")
        if set(ci_.seeds) & set(ce.seeds):
            raise RuntimeError(f"budgetcurve/{tag}: 시드가 서로소가 아니다")
        names = sorted(ci_.arms)
        ts = treat_scale(kn)
        lam_f = LAM_FORMULA * ts
        lams = sorted({parse_arm(p)[1][0] for p in names})
        lam_snap = min(lams, key=lambda L: abs(math.log(L) - math.log(lam_f)))
        card = f"Q{int(lam_snap) if float(lam_snap).is_integer() else lam_snap:g}Y0"
        rec = {"cond": tag, "knobs": kn, "n_regions": len(ci_.regions),
               "inner_seeds": len(ci_.seeds), "eval_seeds": len(ce.seeds),
               "n_arms": len(names), "formula_card": card,
               "fixed_card": "Q18Y0"}
        # 선택은 내부 시드, 평가는 서로소 시드
        for lv in ("L0", "L2", "L3"):
            gs = _groups(ci_.regions, lv)
            cb = np.empty((len(ce.regions), len(ce.seeds)), float)
            ch = {}
            for gk, idx in gs.items():
                best = min(names, key=lambda p: float(ci_.arms[p][idx, :].mean()))
                ch[gk] = best
                cb[idx, :] = ce.arms[best][idx, :]
            rec[lv] = {"n_params": len(gs), "n_distinct": len(set(ch.values())),
                       "pdr": float(cb.mean()),
                       "vs_fixed": meas(cb, ce.arms["Q18Y0"]),
                       "vs_formula": meas(cb, ce.arms[card]) if card in ce.arms else None}
            rec[f"{lv}_cube"] = None
            if lv == "L0":
                rec["L0_arm"] = list(ch.values())[0]
                l0cube = cb.copy()
        for lv in ("L2", "L3"):
            gs = _groups(ci_.regions, lv)
            cb = np.empty((len(ce.regions), len(ce.seeds)), float)
            for gk, idx in gs.items():
                best = min(names, key=lambda p: float(ci_.arms[p][idx, :].mean()))
                cb[idx, :] = ce.arms[best][idx, :]
            rec[lv]["vs_L0"] = meas(cb, l0cube)
        for lv in ("L0", "L2", "L3"):
            rec.pop(f"{lv}_cube", None)
        out.append(rec)
    return {"axis": "A6b 지역화(서로소 시드·예산곡선 산출)", "rows": out}


# ══ A7. 무튜닝 S족 wait_scale ══════════════════════════════════════════════

def axis_wait(conds: dict[str, Cond]) -> dict:
    """S족(목적함수 직접 최대화)의 wait_scale 은 자유 숫자인가.

    ``wait_scale`` 은 코드 주석상 **1.0 이 이론값**이고 0.62 는 폐루프로 고른 값이다.
    따라서 "무튜닝 카드" 라는 표현이 성립하려면 S1.0 이 S0.62 와 판정선 안에서
    동률이어야 한다. 그 차이를 그대로 잰다.
    """
    rows = []
    for k, c in conds.items():
        arms = c.fam("S")
        if len(arms) < 3:
            continue
        names = [p for _, p in arms]
        vals = {p: v for v, p in arms}
        allc = list(range(len(c.seeds)))
        sel, ev = _sel_ev(c.seeds)
        b_in, b_ho = pick(c.arms, names, allc), pick(c.arms, names, sel)
        rec = {"cond": k, "knobs": c.knobs, "grid": sorted(vals.values()),
               "best_arm": b_in, "best_value": vals[b_in], "best_pdr": c.mean(b_in),
               "edge": grid_edge(vals[b_in], list(vals.values())),
               "honest_arm": b_ho,
               "curve": {p: c.mean(p) for p in sorted(names, key=lambda q: vals[q])}}
        if "S0.62" in c.arms:
            rec["adopted"] = "S0.62"
            rec["adopted_pdr"] = c.mean("S0.62")
            rec["tuning_gain_vs_theory"] = (meas(c.arms["S0.62"], c.arms["S1.0"])
                                            if "S1.0" in c.arms else None)
            rec["free_gain_vs_adopted"] = meas(c.arms[b_in], c.arms["S0.62"])
            rec["free_gain_honest"] = meas(c.arms[b_ho][:, ev], c.arms["S0.62"][:, ev])
            rec["neighbors"] = {p: meas(c.arms["S0.62"], c.arms[p])
                                for p in names if p != "S0.62"}
            if "Q18" in c.arms:
                rec["adopted_vs_card"] = meas(c.arms["S0.62"], c.arms["Q18"])
                rec["theory_vs_card"] = (meas(c.arms["S1.0"], c.arms["Q18"])
                                         if "S1.0" in c.arms else None)
        rows.append(rec)
    return {"axis": "A7 무튜닝 S족 wait_scale", "param": "wait_scale", "rows": rows}


# ══ 가용성 표 ══════════════════════════════════════════════════════════════

AXIS_NEEDS = [
    ("A1 λ",            lambda c: len(c.fam("Q")), 4),
    ("A2 함수형",        lambda c: sum(1 for _, f in FORM_FAMS if c.fam(f)), 2),
    ("A2b K족",          lambda c: len(c.fam("K")), 3),
    ("A3 yhold",         lambda c: len(c.fam("Y")), 4),
    ("A4 red_gain",      lambda c: len(c.fam("G")), 4),
    ("A5 신호출처",      lambda c: sum(1 for _, fs, _ in LADDER
                                       if any(c.fam(f) for f in fs.values())), 2),
    ("A6 지역화",        lambda c: max(len(c.fam("Q")), len(c.fam("Y")), len(c.fam("G"))), 4),
    ("A7 wait_scale",    lambda c: len(c.fam("S")), 3),
]


def cmd_inventory(args) -> None:
    conds = discover()
    skipped = conds.pop("__skipped__")
    print("=" * 100)
    print("E2 데이터 가용성 — 축 × 물리조건. 숫자는 그 축에 쓸 수 있는 팔 개수, '측정 없음' 은 팔 부족")
    print(f"좌표셋 budget750(750좌표) · seed0..9 · 판정선 {JUDGE_LINE} · 기준선 {CARD_SPEC}")
    print("=" * 100)
    recs = []
    hdr = f"{'조건':<22}" + "".join(f"{a:<14}" for a, _, _ in AXIS_NEEDS)
    print(hdr)
    print("-" * len(hdr))
    for k, c in sorted(conds.items(), key=lambda kv: (kv[0] != "base", kv[0])):
        cells, line = {}, f"{k:<22}"
        for name, fn, need in AXIS_NEEDS:
            n = fn(c)
            ok = n >= need
            cells[name] = n if ok else 0
            line += f"{(str(n) + '팔') if ok else '측정없음':<14}"
        print(line)
        recs.append({"cond": k, "knobs": c.knobs, "n_regions": len(c.regions),
                     "seeds": c.seeds, "rows": c.rows, "paths": c.paths, "cells": cells})
    print("-" * len(hdr))
    print(f"조건 {len(conds)}개 · 총 {sum(r['rows'] for r in recs):,}행")
    print("\n[좌표셋 게이트에서 제외된 파일] — 추정으로 채우지 않는다")
    for s in skipped:
        print("  ·", s)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inventory.json").write_text(json.dumps(
        {"judge_line": JUDGE_LINE, "card": CARD_SPEC, "manifest": ALLOWED_MANIFEST,
         "axes": [a for a, _, _ in AXIS_NEEDS], "conds": recs, "skipped": skipped},
        ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame([{"cond": r["cond"], "knobs": _knob_key(r["knobs"]),
                   "n_regions": r["n_regions"], "rows": r["rows"],
                   **{a: (r["cells"][a] or "측정없음") for a, _, _ in AXIS_NEEDS}}
                  for r in recs]).to_csv(OUT / "inventory.csv", index=False, encoding="utf-8-sig")
    print(f"\n→ {OUT/'inventory.json'}\n→ {OUT/'inventory.csv'}")


# ══ 축 측정 ═══════════════════════════════════════════════════════════════

def _f(x, n=5):
    return "—" if x is None else f"{x:+.{n}f}"


def cmd_axes(args) -> None:
    conds = discover()
    conds.pop("__skipped__")
    res = {"judge_line": JUDGE_LINE, "card": CARD_SPEC, "manifest": ALLOWED_MANIFEST,
           "lam_formula": LAM_FORMULA,
           "conds": {k: {"knobs": c.knobs, "n_regions": len(c.regions), "seeds": c.seeds,
                         "paths": c.paths} for k, c in conds.items()}}
    res["A1"] = axis_lam(conds)
    res["A2"] = axis_form(conds)
    res["A2b"] = axis_form_k(conds)
    res["A3"] = axis_yhold(conds)
    res["A4"] = axis_red(conds)
    res["A5"] = axis_signal(conds)
    res["A6"] = axis_local(conds)
    res["A6b"] = axis_local_budgetcurve()
    res["A7"] = axis_wait(conds)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "axes.json").write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                              default=float), encoding="utf-8")
    _print_axes(res)
    print(f"\n→ {OUT/'axes.json'}")


def _wtl(d: dict) -> str:
    return f"{d['win']}/{d['tie']}/{d['loss']}"


def _c(d, key="ci95") -> str:
    return "—" if d is None else f"{d[key]:.5f}"


def _print_axes(res: dict) -> None:
    L = JUDGE_LINE
    print("\n" + "=" * 108)
    print(f"A1 λ 부하 교환율 — 기준선 Q18({CARD_SPEC}) · Δ>0 = 후보 개선 · 판정선 {L}")
    print("=" * 108)
    print(f"{'조건':<18}{'ts':>5}{'자유최적':>9}{'Δ자유':>11}{'±CI':>10}{'W/T/L':>13}"
          f"{'격자':>10}  {'공식팔':>7}{'Δ공식':>11}{'잔여(공식→자유)':>16}{'±CI':>10}{'판정':>16}")
    for r in res["A1"]["rows"]:
        fr, fo, rd = r["free"]["insample"], r["formula"], r["residual_formula_to_free"]
        print(f"{r['cond']:<18}{r['treat_scale']:>5.2f}{fr['arm']:>9}{_f(fr['delta']):>11}"
              f"{_c(fr):>10}{_wtl(fr):>13}{fr['edge']:>10}  {fo['arm']:>7}"
              f"{_f(fo['delta']):>11}{_f(rd['delta']):>16}{_c(rd):>10}{rd['verdict']:>16}")

    print("\n" + "=" * 108)
    print("A2 부하항 함수형 — 같은 신호(occ+in_flight)에 함수형만 교체. 각 형태는 자기 최적 λ 에서")
    print("   cost>0 = hingerate 최적보다 그 형태가 열세(= 그 구조를 고른 이유)")
    print("=" * 108)
    for r in res["A2"]["rows"]:
        print(f"[{r['cond']}] 기준 Q18 PDR {r['baseline_pdr']:.6f}")
        print(f"  {'형태':<11}{'가족':>4}{'팔':>7}{'최적λ':>9}{'PDR':>11}{'Δ vs카드':>11}"
              f"{'±CI':>10}{'cost vs rate':>14}{'±CI':>10}{'격자':>10}")
        for form in ("hingerate", "hinge1", "linear", "zero"):
            g = r["forms"].get(form)
            if not g:
                continue
            cst = g.get("cost_vs_hingerate")
            cost_s = _f(cst["delta"]) if cst else "기준"
            print(f"  {form:<11}{g['family']:>4}{g['best_arm']:>7}{g['best_lam']:>9.4g}"
                  f"{g['best_pdr']:>11.6f}{_f(g['vs_card']['delta']):>11}"
                  f"{_c(g['vs_card']):>10}{cost_s:>14}{_c(cst):>10}{g['edge']:>10}")

    print("\n" + "=" * 108)
    print("A2b 구 CARD K족(거리 km 축 + 선형 부하) — 함수형과 도달축이 동시에 다르다(교란)")
    print("=" * 108)
    print(f"{'조건':<18}{'최적팔':>8}{'λkm':>7}{'PDR':>11}{'Δ vs카드':>11}{'±CI':>10}"
          f"{'격자':>10}{'K12 Δ':>11}")
    for r in res["A2b"]["rows"]:
        k12 = r["K12_vs_card"]
        print(f"{r['cond']:<18}{r['best_arm']:>8}{r['best_lam']:>7.4g}{r['best_pdr']:>11.6f}"
              f"{_f(r['vs_card']['delta']):>11}{_c(r['vs_card']):>10}{r['edge']:>10}"
              f"{(_f(k12['delta']) if k12 else '—'):>11}")

    print("\n" + "=" * 108)
    print(f"A3 등급 임계 yhold — 조건 {res['A3']['n_cond']}개 · Y0 대비 판정선 초과 "
          f"{res['A3']['n_exceed_vs_card']}곳 · 숫자 없는 양끝 대비 내부값 초과 "
          f"{res['A3']['n_exceed_needs_number']}곳")
    print("   양끝 = Y0(Yellow 우선)·Y9999(Red 우선) 둘 다 숫자가 필요 없는 규칙이다")
    print("=" * 108)
    print(f"{'조건':<18}{'최적Y':>7}{'Δ vs Y0':>11}{'±CI':>10}{'W/T/L':>13}"
          f"{'좋은양끝':>9}{'Δ양끝':>11}  {'내부최적':>9}{'Δ내부−양끝':>12}{'±CI':>10}{'판정':>16}")
    for r in res["A3"]["rows"]:
        fr, be, nn = r["free"]["insample"], r["best_end"], r["needs_number"]
        print(f"{r['cond']:<18}{fr['arm']:>7}{_f(fr['delta']):>11}{_c(fr):>10}{_wtl(fr):>13}"
              f"{be['arm']:>9}{_f(be['delta']):>11}  {nn['best_inner']:>9}"
              f"{_f(nn['delta']):>12}{_c(nn):>10}{nn['verdict']:>16}")

    print("\n" + "=" * 108)
    print(f"A4 수단 임계 red_gain_min — 조건 {res['A4']['n_cond']}개 · G6.6 대비 판정선 초과 "
          f"{res['A4']['n_exceed_vs_card']}곳 · G0(숫자 제거)과 동률 {res['A4']['n_G0_equiv']}곳")
    print("=" * 108)
    print(f"{'조건':<18}{'자유최적':>9}{'Δ자유':>11}{'±CI':>10}  "
          f"{'G0 Δ':>11}{'±CI':>10}{'판정':>16}  {'항상AMB':>9}{'Δ':>11}{'판정':>16}")
    for r in res["A4"]["rows"]:
        fr, g0, dr = r["free"]["insample"], r["drop_to_G0"], r["drop_rule_alwaysAMB"]
        print(f"{r['cond']:<18}{fr['arm']:>9}{_f(fr['delta']):>11}{_c(fr):>10}  "
              f"{(_f(g0['delta']) if g0 else '—'):>11}{_c(g0):>10}"
              f"{(g0['verdict'] if g0 else '—'):>16}  {dr['arm']:>9}"
              f"{_f(dr['delta']):>11}{dr['verdict']:>16}")

    print("\n" + "=" * 108)
    print("A5 부하 신호 출처 — 같은 함수형끼리만 비교. Δ>0 = 채택 카드 Q18 보다 개선")
    print("=" * 108)
    for r in res["A5"]["rows"]:
        print(f"[{r['cond']}] 기준 Q18 PDR {r['baseline_pdr']:.6f}")
        print(f"  {'정보수준':<20}{'통신':<14}{'함수형':<11}{'팔':>7}{'PDR':>11}"
              f"{'Δ vs카드':>11}{'±CI':>10}{'격자':>10}{'판정':>16}")
        for name, blk in r["levels"].items():
            for form, cl in blk["cells"].items():
                v = cl["vs_card"]
                print(f"  {name:<20}{blk['comm']:<14}{form:<11}{cl['best_arm']:>7}"
                      f"{cl['best_pdr']:>11.6f}{_f(v['delta']):>11}{_c(v):>10}"
                      f"{cl['edge']:>10}{v['verdict']:>16}")

    print("\n" + "=" * 108)
    print("A6 지역화 — Δ>0 = 지역화가 전국 단일보다 개선")
    print("   oracle = 고르기·재기를 같은 10시드로(상한·선택 편향 포함) / honest = seed0-4 선택, 5-9 평가")
    print("=" * 108)
    print(f"{'조건':<14}{'파라미터':<14}{'전국팔':>7}{'표L2':>6}{'표L3':>6}"
          f"{'oracle L2':>11}{'oracle L3':>11}{'honest L2':>11}{'±CI':>10}"
          f"{'honest L3':>11}{'±CI':>10}{'honest L3 판정':>18}")
    for r in res["A6"]["rows"]:
        o, h = r["oracle"], r["honest"]
        print(f"{r['cond']:<14}{r['param']:<14}{h['L0_arm']:>7}"
              f"{h['L2']['n_params']:>6}{h['L3']['n_params']:>6}"
              f"{_f(o['L2']['delta']):>11}{_f(o['L3']['delta']):>11}"
              f"{_f(h['L2']['delta']):>11}{_c(h['L2']):>10}"
              f"{_f(h['L3']['delta']):>11}{_c(h['L3']):>10}{h['L3']['verdict']:>18}")

    print("\nA6b 서로소 시드(예산곡선 산출) — 60좌표·32팔·내부 seed0..127 선택 / 평가 seed1000..1059")
    print(f"  {'조건':<8}{'레벨':<5}{'표크기':>7}{'서로다른팔':>11}{'PDR':>11}"
          f"{'Δ vs 전국':>11}{'±CI':>10}{'Δ vs 고정카드':>14}{'Δ vs 공식카드':>15}")
    for r in res["A6b"]["rows"]:
        for lv in ("L0", "L2", "L3"):
            b = r[lv]
            vl = b.get("vs_L0")
            vf = b.get("vs_formula")
            print(f"  {r['cond']:<8}{lv:<5}{b['n_params']:>7}{b['n_distinct']:>11}"
                  f"{b['pdr']:>11.6f}{(_f(vl['delta']) if vl else '기준'):>11}{_c(vl):>10}"
                  f"{_f(b['vs_fixed']['delta']):>14}"
                  f"{(_f(vf['delta']) if vf else '—'):>15}")

    print("\n" + "=" * 108)
    print("A7 무튜닝 S족 wait_scale — 이론값 1.0 대비 채택값 0.62 의 이득이 곧 '튜닝 숫자' 의 크기")
    print("=" * 108)
    print(f"{'조건':<18}{'최적w':>7}{'격자':>10}{'Δ 0.62−1.0':>12}{'±CI':>10}{'판정':>16}"
          f"  {'Δ자유−0.62':>12}{'±CI':>10}{'Δ 0.62 vs Q18':>15}")
    for r in res["A7"]["rows"]:
        tg, fg = r.get("tuning_gain_vs_theory"), r.get("free_gain_vs_adopted")
        av = r.get("adopted_vs_card")
        print(f"{r['cond']:<18}{r['best_value']:>7.4g}{r['edge']:>10}"
              f"{(_f(tg['delta']) if tg else '—'):>12}{_c(tg):>10}"
              f"{(tg['verdict'] if tg else '—'):>16}  "
              f"{(_f(fg['delta']) if fg else '—'):>12}{_c(fg):>10}"
              f"{(_f(av['delta']) if av else '—'):>15}")


# ══ 두 표: 남는 숫자 / 뺀 숫자 ═════════════════════════════════════════════

def cmd_tables(args) -> None:
    p = OUT / "axes.json"
    if not p.exists():
        raise SystemExit("axes.json 없음 — 먼저 `axes` 를 돌려라")
    res = json.loads(p.read_text(encoding="utf-8"))
    keep, drop = _build_tables(res)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(keep).to_csv(OUT / "table_keep.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(drop).to_csv(OUT / "table_drop.csv", index=False, encoding="utf-8-sig")
    _print_tables(keep, drop)
    (OUT / "verdict.json").write_text(json.dumps({"keep": keep, "drop": drop},
                                                 ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"\n→ {OUT/'table_keep.csv'}\n→ {OUT/'table_drop.csv'}\n→ {OUT/'verdict.json'}")


def _row(name, kind, value, evid, measure, decision, note):
    return {"파라미터": name, "종류": kind, "카드에 적는 값": value,
            "결정 근거(측정)": measure, "판정": decision, "근거 조건": evid, "비고": note}


def _build_tables(res: dict):
    L = res["judge_line"]
    keep, drop = [], []

    # A1 λ — 자유 숫자인가 공식인가
    a1 = res["A1"]["rows"]
    by = {r["cond"]: r for r in a1}
    ts_rows = [r for r in a1 if abs(r["treat_scale"] - 1.0) > 1e-9]
    worst_free = max(a1, key=lambda r: r["free"]["insample"]["delta"])
    worst_res = max(a1, key=lambda r: r["residual_formula_to_free"]["delta"])
    keep.append(_row(
        "λ (부하 교환율, 분/명)", "공식 — 자유 파라미터 아님",
        "λ = 18.97 × (관측 치료시간 ÷ 정식 치료시간)",
        f"{len(a1)}조건(치료시간 축 {len(ts_rows)}개 포함)",
        f"자유 재튜닝 최대 이득 {worst_free['free']['insample']['delta']:+.5f}"
        f"({worst_free['cond']}) · 공식 고정 후 남는 몫 최대 "
        f"{worst_res['residual_formula_to_free']['delta']:+.5f}±"
        f"{worst_res['residual_formula_to_free']['ci95']:.5f}({worst_res['cond']})",
        "유지(값은 공식으로 계산)",
        "카드에 숫자 18 을 박으면 봉투 밖에서 무너진다. 숫자가 아니라 식을 적는다"))

    # A2 함수형
    a2b = next((r for r in res["A2"]["rows"] if r["cond"] == "base"), None)
    if a2b:
        f = a2b["forms"]
        parts = []
        for form in ("hinge1", "linear", "zero"):
            g = f.get(form)
            if g and g.get("cost_vs_hingerate"):
                c_ = g["cost_vs_hingerate"]
                parts.append(f"{form} +{c_['delta']:.5f}±{c_['ci95']:.5f}")
        keep.append(_row(
            "부하항 함수형 Z", "구조(숫자 아님)", "hingerate = max(0, q+1−c)/c",
            f"base 조건 + 치료시간 6조건", "대체형 비용 " + " · ".join(parts),
            "유지", "수술실수 c 로 나누는 것까지 넣어야 λ 가 서비스시간으로 해석된다"))

    # A3 yhold
    a3 = res["A3"]
    if a3["n_exceed_needs_number"] > 0:
        keep.append(_row(
            "등급 임계 yhold (명)", "자유 숫자(공식 없음)", "조건별 {0,2,4,8}",
            f"{a3['n_cond']}조건",
            f"Y0 대비 판정선 초과 {a3['n_exceed_vs_card']}곳 · "
            f"숫자 없는 양끝 대비 내부값이 초과 {a3['n_exceed_needs_number']}곳",
            "유지", "폐루프의 환원 불가 몫. 스케일링 공식이 나오지 않은 유일한 축"))
    else:
        best = max(a3["rows"], key=lambda r: r["needs_number"]["delta"])
        drop.append(_row(
            "등급 임계 yhold (명)", "자유 숫자", "→ 'Yellow 우선'(숫자 없음)",
            f"{a3['n_cond']}조건",
            f"숫자 없는 양끝 대비 내부값 최대 이득 {best['needs_number']['delta']:+.5f}"
            f"±{best['needs_number']['ci95']:.5f}({best['cond']}) — 전 조건 판정선 미만",
            "제외", "Y0(Yellow 우선)·Y9999(Red 우선) 두 규칙만으로 충분"))

    # A4 red_gain
    a4 = res["A4"]
    g0s = [r for r in a4["rows"] if r["drop_to_G0"]]
    if g0s:
        worst = max(g0s, key=lambda r: abs(r["drop_to_G0"]["delta"]))
        w = worst["drop_to_G0"]
        free_max = max(a4["rows"], key=lambda r: r["free"]["insample"]["delta"])
        tgt = (drop if a4["n_G0_equiv"] == len(g0s) else keep)
        tgt.append(_row(
            "수단 임계 red_gain_min (분)", "자유 숫자",
            ("→ 'UAV 가 더 빠르면 UAV'(임계 0, 숫자 없음)"
             if a4["n_G0_equiv"] == len(g0s) else "6.6"),
            f"{a4['n_cond']}조건",
            f"G0(숫자 제거) 대 G6.6 최대 격차 {w['delta']:+.5f}±{w['ci95']:.5f}"
            f"({worst['cond']}) · 자유 재튜닝 최대 이득 "
            f"{free_max['free']['insample']['delta']:+.5f}({free_max['cond']})",
            ("제외" if a4["n_G0_equiv"] == len(g0s) else "유지"),
            "빼면 수단 축은 '현장 대기 차량 유무 게이트 + 빠른 쪽' 만 남는다"))

    # A5 신호 출처
    a5 = next((r for r in res["A5"]["rows"] if r["cond"] == "base"), None)
    if a5:
        lv = a5["levels"]
        cells = {n: b["cells"].get("hingerate") for n, b in lv.items()}
        i3 = cells.get("I3 occ+in_flight")
        txt = []
        for n, cl in cells.items():
            if cl and i3 and n != "I3 occ+in_flight":
                txt.append(f"{n.split()[0]} {cl['vs_card']['delta']:+.5f}")
        z = lv.get("I0 부하 무시", {}).get("cells", {}).get("zero")
        if z:
            txt.append(f"I0 {z['vs_card']['delta']:+.5f}")
        keep.append(_row(
            "부하 신호 출처", "구조(숫자 아님)", "occ+in_flight (또는 p_sent = 통신 불요)",
            "base 조건", "Q18 대비 " + " · ".join(txt),
            "유지", "정보수준은 서열이 아니다 — occ 단독이 가장 나쁘다"))

    # A6 지역화
    a6 = res["A6"]["rows"]
    lamrows = [r for r in a6 if r["param"].startswith("λ")]
    if lamrows:
        r0 = next((r for r in lamrows if r["cond"] == "base"), lamrows[0])
        bc = res["A6b"]["rows"]
        bctxt = ""
        if bc:
            b0 = bc[0]
            bctxt = (f" · 서로소시드 60좌표 L3 {b0['L3']['vs_L0']['delta']:+.5f}"
                     f"±{b0['L3']['vs_L0']['ci95']:.5f}")
        drop.append(_row(
            "지역화 파라미터표 (λ 좌표별/시군구별)", "표 (750 또는 250개 숫자)",
            "→ 전국 단일값 1개",
            f"base 조건 + {len(a6)}개 (파라미터×조건)",
            f"오라클 상한 L3 {r0['oracle']['L3']['delta']:+.5f} / L2 "
            f"{r0['oracle']['L2']['delta']:+.5f} 인데 정직한 시드분할에서 L3 "
            f"{r0['honest']['L3']['delta']:+.5f}±{r0['honest']['L3']['ci95']:.5f} · L2 "
            f"{r0['honest']['L2']['delta']:+.5f}±{r0['honest']['L2']['ci95']:.5f}{bctxt}",
            "제외", "오라클 이득은 전부 선택 잡음. 배포 예산에서 지역화는 손해"))

    # A7 wait_scale
    a7 = next((r for r in res["A7"]["rows"] if r["cond"] == "base"), None)
    if a7:
        tg = a7.get("tuning_gain_vs_theory")
        av = a7.get("adopted_vs_card")
        keep.append(_row(
            "wait_scale (S족 전용, 대안 카드)", "자유 숫자",
            "0.62 (이론값 1.0 은 열세)",
            f"{len(res['A7']['rows'])}조건",
            f"S0.62 − S1.0 = {tg['delta']:+.5f}±{tg['ci95']:.5f}({tg['verdict']}) · "
            f"S0.62 vs Q18 = {av['delta']:+.5f}±{av['ci95']:.5f}" if tg and av else "—",
            "조건부 유지(S족을 쓸 때만)",
            "S족을 '무튜닝' 이라 부르면 안 된다 — wait_scale 이 튜닝 숫자다"))
    return keep, drop


def _print_tables(keep, drop):
    for title, tb in (("표 1 — 카드에 남는 숫자", keep), ("표 2 — 카드에서 뺀 숫자", drop)):
        print("\n" + "=" * 104)
        print(title)
        print("=" * 104)
        for r in tb:
            print(f"\n● {r['파라미터']}  [{r['종류']}]  → {r['판정']}")
            print(f"    카드에 적는 값 : {r['카드에 적는 값']}")
            print(f"    측정           : {r['결정 근거(측정)']}")
            print(f"    근거 조건      : {r['근거 조건']}")
            print(f"    비고           : {r['비고']}")
        if not tb:
            print("  (없음)")


def main() -> None:
    ap = argparse.ArgumentParser(description="v22 E2 최소 충분 파라미터 집합")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inventory", help="데이터 가용성 표")
    sub.add_parser("axes", help="7축 측정")
    sub.add_parser("tables", help="남는/뺀 숫자 두 표")
    sub.add_parser("all", help="전부")
    a = ap.parse_args()
    if a.cmd in ("inventory", "all"):
        cmd_inventory(a)
    if a.cmd in ("axes", "all"):
        cmd_axes(a)
    if a.cmd in ("tables", "all"):
        cmd_tables(a)


if __name__ == "__main__":
    main()
