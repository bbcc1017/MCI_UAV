# -*- coding: utf-8 -*-
"""v22 현장 어시스트 — 헤드룸 게이트와 손실 분해 리포트.

시뮬레이션을 재실행하지 않는다. 이미 완주된 에피소드 CSV 만 읽어 paired 통계를 낸다.
판정 커널은 ``tools/v20_threshold_report.py`` 의 ``cube``/``paired``/``wtl`` 를 그대로 쓴다
(재구현하면 W/T/L 정의가 갈린다 — v5 에서 실제로 그 혼동이 있었다).

서브커맨드
  headroom   G1 게이트. ① 봉투 밖 조건에서 고정 카드가 그 조건 최적 팔에 얼마나 지는가
             ② 등급 축(yhold)의 조건별 최적이 채택값에서 얼마나 떨어져 있는가
  decompose  봉투 밖 한 조건의 손실을 (격자 내 재튜닝 회수분 / 잔여분) 으로 쪼갠다.
             "λ 를 수식화하는 대신 시뮬레이션 최적값으로 대체" 의 정량 답이 이 표다.
  lamscale   물리 축 한 개에 대한 최적 파라미터의 로그-로그 기울기.
  envelope   봉투 거리 대 "그 조건에서 재튜닝하면 얻는 이득" 곡선.
  interact   (λ × yhold) 결합 격자 — 두 축이 파라미터 선택에서 독립인가.
  budgetcurve  온라인 선택의 예산 곡선. 내부 시드 K 를 늘릴 때 selection_regret 이
             0-에피소드 카드(고정 카드 / 공식 카드) 아래로 내려가는 지점을 찾는다.
             CRN 접두사 재사용이라 시뮬 재실행 0 회다.

⚠️ 좌표셋 주의. ``--stage_dir results/scoreboard/v20/theory`` 는 v20 산출물이고 그 스테이지는
   ``v19/tradeoff250_manifest.json`` = **test750 의 정확한 1/3 부분집합**을 썼다. 따라서 그 수치는
   동기 부여용 잠정값이다. 파라미터·템플릿 **선택 근거**로 쓸 값은 budget750 재도출본
   (``results/scoreboard/v22/retune``) 에서 다시 뽑아야 한다. 리포트는 읽은 매니페스트를
   메타에서 확인해 출력에 함께 적는다.

판정선: v21 실측 CRN paired 두 팔 차이 95%CI = 0.00053. 이건 그 조건의 실측값이므로
   새 설정에서는 ``v20_threshold_report.py cmd_refine`` 으로 다시 재는 것이 원칙이다.
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

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from v20_threshold_report import ci, cube, paired  # noqa: E402

# v21 실측 판정선(CRN paired 두 팔 차이 95%CI). 보편 상수가 아니라 그 조건의 실측값이다.
JUDGE_LINE = 0.00053


def _meta_manifest(csv_path: str) -> str:
    """그 CSV 가 어느 좌표셋에서 나왔는지. 누수 판정에 필요하므로 항상 함께 보고한다."""
    m = csv_path + ".meta.json"
    if not os.path.exists(m):
        return "<메타 없음>"
    try:
        return os.path.basename(str(json.load(open(m, encoding="utf-8")).get("manifest", "?")))
    except Exception:
        return "<메타 파싱 실패>"


def _knobs(csv_path: str) -> dict:
    m = csv_path + ".meta.json"
    if not os.path.exists(m):
        return {}
    try:
        kn = json.load(open(m, encoding="utf-8")).get("scenario_knobs") or {}
        return {k.replace("MCI_", ""): v for k, v in kn.items() if v}
    except Exception:
        return {}


def _load(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding="utf-8-sig")


def _verdict(delta: float, ci95: float, mult: int = 1) -> str:
    """양수 delta = 기준 팔(고정 카드)이 열세."""
    if delta > JUDGE_LINE * mult and delta > ci95:
        return f"★판정선{'' if mult == 1 else f'{mult}배'}초과"
    if delta > ci95:
        return "유의"
    return "동률"


def cmd_headroom(args) -> None:
    out = {"judge_line": JUDGE_LINE, "stage_dir": args.stage_dir, "extreme": [], "yhold": []}

    print("=" * 100)
    print(f"G1-a  봉투 밖 — 고정 카드 {args.fixed_arm} 대비 그 조건 최적 팔")
    print(f"      delta = {args.fixed_arm} − best (양수 = 고정 카드 열세) · 판정선 {JUDGE_LINE}")
    print("=" * 100)
    for f in sorted(glob.glob(os.path.join(args.stage_dir, f"{args.extreme_prefix}_*.csv"))):
        if f.endswith(".meta.json"):
            continue
        tag = os.path.basename(f)[len(args.extreme_prefix) + 1: -4]
        df = _load(f)
        if args.fixed_arm not in set(df.policy.unique()):
            print(f"{tag:10s} — 고정 팔 {args.fixed_arm} 없음, 건너뜀")
            continue
        cubes = {p: cube(df, p) for p in df.policy.unique()}
        means = {p: v.mean() for p, v in cubes.items()}
        best = min(means, key=means.get)
        r = paired(cubes[best], cubes[args.fixed_arm])
        rec = {"condition": tag, "knobs": _knobs(f), "manifest": _meta_manifest(f),
               "best_arm": best, "best_pdr": means[best],
               "fixed_pdr": means[args.fixed_arm], **r,
               "verdict_x5": _verdict(r["delta"], r["ci95"], 5)}
        out["extreme"].append(rec)
        kn = ",".join(f"{k}={v}" for k, v in rec["knobs"].items())
        print(f"{tag:10s} {kn:24s} best={best:7s} {means[best]:.5f}  "
              f"고정={means[args.fixed_arm]:.5f}  손실={r['delta']:+.5f} ±{r['ci95']:.5f}  "
              f"W/T/L={r['win']}/{r['tie']}/{r['loss']}  {rec['verdict_x5']}")

    print()
    print("=" * 100)
    print(f"G1-b  등급 축 — 채택값 {args.yhold_base_arm} 대비 그 조건 최적 yhold")
    print("=" * 100)
    for f in sorted(glob.glob(os.path.join(args.stage_dir, f"{args.yhold_prefix}_*.csv"))):
        if f.endswith(".meta.json"):
            continue
        tag = os.path.basename(f)[len(args.yhold_prefix) + 1: -4]
        df = _load(f)
        ys = {p: cube(df, p) for p in df.policy.unique() if p.startswith(args.yhold_arm_prefix)}
        if args.yhold_base_arm not in ys:
            print(f"{tag:10s} — 기준 팔 {args.yhold_base_arm} 없음, 건너뜀")
            continue
        means = {p: v.mean() for p, v in ys.items()}
        best = min(means, key=means.get)
        r = paired(ys[best], ys[args.yhold_base_arm])
        rec = {"condition": tag, "knobs": _knobs(f), "manifest": _meta_manifest(f),
               "best_arm": best, "best_pdr": means[best],
               "base_pdr": means[args.yhold_base_arm], **r,
               "verdict": _verdict(r["delta"], r["ci95"], 1)}
        out["yhold"].append(rec)
        print(f"{tag:10s} best={best:6s} {means[best]:.5f}  기준={means[args.yhold_base_arm]:.5f}  "
              f"손실={r['delta']:+.5f} ±{r['ci95']:.5f}  "
              f"W/T/L={r['win']}/{r['tie']}/{r['loss']}  {rec['verdict']}")

    n_x5 = sum(1 for r in out["extreme"] if r["verdict_x5"].startswith("★"))
    n_y = sum(1 for r in out["yhold"] if r["verdict"].startswith("★"))
    out["gate"] = {"extreme_over_5x": n_x5, "yhold_over_line": n_y,
                   "pass": bool(n_x5 >= 1 and n_y >= 2)}
    print()
    print(f"[G1 게이트] 봉투 밖 판정선5배 초과 조건 {n_x5}개(요구 ≥1) · "
          f"등급 축 판정선 초과 조건 {n_y}개(요구 ≥2) → {'PASS' if out['gate']['pass'] else 'FAIL'}")
    mans = sorted({r["manifest"] for r in out["extreme"] + out["yhold"]})
    print(f"[좌표셋] {', '.join(mans)}")
    if any("tradeoff250" in m for m in mans):
        print("  ⚠️ tradeoff250 은 test750 의 부분집합이다. 이 수치는 잠정값이며 "
              "선택 근거로 쓰려면 budget750 재도출본이 필요하다.")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[기록] {args.out}")


def cmd_decompose(args) -> None:
    """봉투 밖 손실을 '격자 내 재튜닝으로 회수되는 몫'과 '잔여'로 쪼갠다."""
    f = args.csv
    df = _load(f)
    cubes = {p: cube(df, p) for p in df.policy.unique()}
    means = {p: v.mean() for p, v in cubes.items()}

    fam = [p for p in means if p.startswith(args.family) and p[len(args.family):].replace(".", "", 1).isdigit()]
    fam.sort(key=lambda s: float(s[len(args.family):]))
    retuned = min(fam, key=lambda p: means[p]) if fam else None
    best = min(means, key=means.get)

    print("=" * 100)
    print(f"손실 분해 — {os.path.basename(f)}  (좌표셋 {_meta_manifest(f)})")
    kn = ",".join(f"{k}={v}" for k, v in _knobs(f).items())
    print(f"  조건 노브: {kn or '(없음)'}")
    print("=" * 100)
    if fam:
        print(f"  {args.family}족 λ 응답: " + " ".join(f"{p[len(args.family):]}={means[p]:.5f}" for p in fam))
        edge = fam[-1]
        if retuned == edge:
            print(f"  ⚠️ 최적이 격자 끝({edge})이다 — 진짜 최적점은 격자 밖. 격자 확장 필요.")
    rows = []
    if retuned and retuned != args.fixed_arm:
        rows.append((retuned, args.fixed_arm, f"격자 내 λ 재튜닝 {args.fixed_arm}→{retuned}"))
    if best != retuned:
        rows.append((best, retuned, f"재튜닝 후 잔여 ({best} 가 잡는 몫)"))
    rows.append((best, args.fixed_arm, "총 손실"))

    res = {}
    for a, b, lab in rows:
        r = paired(cubes[a], cubes[b])
        res[lab] = r
        print(f"  {lab:46s} {r['delta']:+.5f} ±{r['ci95']:.5f}  W/T/L={r['win']}/{r['tie']}/{r['loss']}")
    tot = res.get("총 손실", {}).get("delta")
    ret = next((v["delta"] for k, v in res.items() if k.startswith("격자 내")), None)
    if tot and ret and tot != 0:
        print(f"  → 재튜닝 회수율 {100 * ret / tot:.1f}%  ·  잔여 {100 * (1 - ret / tot):.1f}%")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"csv": f, "manifest": _meta_manifest(f), "knobs": _knobs(f),
                   "means": means, "decomposition": res},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[기록] {args.out}")




def _axis_value(tag: str, token: str, scale: str) -> float:
    """조건 태그에서 축 값을 뽑는다.

    v20/v22 드라이버의 태그 규약은 **소수점을 뺀다** — `capa05`=0.5 · `ts075`=0.75 ·
    `ts15`=1.5 · `ts40`=4.0. `float("05")`=5.0 으로 읽으면 조용히 10배 틀린다.
    `v20_threshold_report` 의 `_axis_of` 와 같은 환산을 쓴다:
    ``float(num) / 10**(len(num)-1)``.
    구 v20 theory 파일은 `ts0.5` 처럼 점이 들어 있어 그대로 파싱해야 하므로,
    점이 있으면 환산하지 않는다.
    """
    num = tag.replace(token, "", 1)
    if not num:
        raise ValueError(f"축 값을 못 읽었다: tag={tag} token={token}")
    if scale == "raw" or "." in num:
        return float(num)
    return float(num) / (10.0 ** (len(num) - 1))


def _lam_star(df: pd.DataFrame, fam: str):
    """그 CSV 에서 팔 족 ``fam`` 의 보간 최적 파라미터.

    팔 이름을 재구성하지 않고 CSV 의 원본 이름을 쓴다 — ``S1.0`` 을 ``S1`` 로 만들면
    조용히 KeyError 가 난다. 최적이 격자 끝이면 보간하지 않고 edge 플래그를 세운다
    (진짜 최적점이 격자 밖이라는 뜻이므로 기울기 추정에서 빼야 한다).
    """
    pol = [q for q in df.policy.unique() if re.fullmatch(fam + r"[0-9.]+", q)]
    if len(pol) < 3:
        return None
    vals = sorted((float(q[len(fam):]), q) for q in pol)
    lams = np.array([v for v, _ in vals])
    pdrs = np.array([cube(df, q).mean() for _, q in vals])
    i = int(pdrs.argmin())
    edge = i in (0, len(lams) - 1)
    if not edge:
        c = np.polyfit(np.log(lams[i - 1:i + 2]), pdrs[i - 1:i + 2], 2)
        return float(np.exp(-c[1] / (2 * c[0]))), edge, float(lams[i]), float(pdrs[i])
    return float(lams[i]), edge, float(lams[i]), float(pdrs[i])


def cmd_lamscale(args) -> None:
    """물리 축 한 개에 대한 최적 파라미터의 로그-로그 기울기.

    Q 족의 λ 자리는 이론상 **평균 서비스시간**이므로 치료시간 축 기울기는 +1 이어야 한다.
    S 족은 서비스시간을 병원 명부에서 직접 읽으므로 그 배율 w 는 평평해야 한다(이론 0).
    ⚠️ 격자 끝 점을 포함하면 기울기가 **낮게** 편향된다. 두 값을 나란히 보고한다.
    """
    pat = os.path.join(args.stage_dir, f"{args.prefix}_*.csv")
    data = {}
    if args.anchor_csv:
        # 같은 좌표셋·시드의 축 기준점을 다른 경로에서 끌어온다. budget750 의 ts=1.0 은
        # v22 treat 스테이지에 없고 results/scoreboard/v20/budget/lam_base.csv 에 전 Q격자가 있다.
        if not os.path.exists(args.anchor_csv):
            raise SystemExit(f"[치명] anchor_csv 없음: {args.anchor_csv}")
        _am = _meta_manifest(args.anchor_csv)
        _adf = _load(args.anchor_csv)
        data[args.anchor_value] = {"file": args.anchor_csv, "manifest": _am, "is_anchor": True,
                                   **{fam: _lam_star(_adf, fam) for fam in args.families.split(",")}}
        print(f"[앵커] {args.axis_name}={args.anchor_value} <- {args.anchor_csv} (좌표셋 {_am})")
    for f in sorted(glob.glob(pat)):
        if f.endswith(".meta.json"):
            continue
        tag = os.path.basename(f)[len(args.prefix) + 1: -4]
        val = args.base_value if tag == args.base_tag else _axis_value(tag, args.axis_token, args.axis_scale)
        df = _load(f)
        data[val] = {"file": f, "manifest": _meta_manifest(f),
                     **{fam: _lam_star(df, fam) for fam in args.families.split(",")}}
    if not data:
        print(f"[없음] {pat}")
        return

    fams = args.families.split(",")
    print("=" * 100)
    print(f"{args.axis_name} 축 — 팔 족별 보간 최적 (e = 격자끝, 최적점이 격자 밖)")
    print("=" * 100)
    print(f"{args.axis_name:>8} | " + " | ".join(f"{fam+' 최적':>10}" for fam in fams))
    for v in sorted(data):
        cells = []
        for fam in fams:
            x = data[v][fam]
            cells.append(f"{x[0]:8.2f}{'e' if x[1] else ' '} " if x else f"{'-':>10}")
        print(f"{v:8} | " + " | ".join(cells))

    out = {"axis": args.axis_name, "stage_dir": args.stage_dir, "points": {}, "slopes": {}}
    for v in sorted(data):
        out["points"][str(v)] = {fam: (data[v][fam][:3] if data[v][fam] else None) for fam in fams}
    print()
    for fam in fams:
        pts = [(v, data[v][fam][0], data[v][fam][1]) for v in sorted(data) if data[v][fam]]
        for label, sel in (("격자끝 포함", pts), ("격자끝 제외", [q for q in pts if not q[2]])):
            if len(sel) < 3:
                continue
            slope, inter = np.polyfit(np.log([q[0] for q in sel]), np.log([q[1] for q in sel]), 1)
            pred = float(np.exp(inter)) * args.predict_at ** slope
            out["slopes"].setdefault(fam, {})[label] = {
                "n": len(sel), "slope": float(slope),
                f"predict_at_{args.predict_at}": pred}
            print(f"  {fam}족 {label:9s} n={len(sel)}  기울기={slope:+.3f}  "
                  f"→ {args.axis_name}={args.predict_at} 예측 {pred:.2f}")
    mans = sorted({data[v]["manifest"] for v in data})
    print("")
    print(f"[좌표셋] {', '.join(mans)}")
    if len(mans) > 1:
        print("  주의: 좌표셋이 섞였다 — 앵커와 스테이지의 매니페스트가 다르면 기울기를 믿지 마라.")
    if any("tradeoff250" in m for m in mans):
        print("  ⚠️ tradeoff250 = test750 부분집합. 잠정값이다.")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[기록] {args.out}")



# 봉투(calibration envelope) 중심 = 정식 물리조건. 각 노브의 기준값.
ENVELOPE_CENTER = {
    "MCI_INCIDENT_SIZE": 100.0, "MCI_AMB_NUM": 30.0, "MCI_UAV_NUM": 26.0,
    "MCI_CAPA_SCALE": 1.0, "MCI_TREAT_SCALE": 1.0,
    "MCI_AMB_VELOCITY": 50.0, "MCI_UAV_VELOCITY": 200.0,
    "MCI_AMB_HANDOVER": 5.0, "MCI_UAV_HANDOVER": 10.0,
}
# 자원을 0 으로 없앤 조건은 배율이 아니라 질적 변화라 log 가 정의되지 않는다.
# 무한 대신 관례값을 쓰고, 결과에 "퇴화 노브" 로 표시해 곡선 회귀에서 제외한다.
DEGENERATE_DISTANCE = 4.0


def envelope_distance(knobs: dict):
    """정식 조건에서 얼마나 먼 조건인가 — 로그2 배율 편차 벡터의 L2 노름.

    예) ``MCI_TREAT_SCALE=4.0`` 단독이면 ``|log2 4| = 2.0``.
    반환 ``(거리, 퇴화노브 목록)``.
    """
    d2, degenerate = 0.0, []
    for k, v in (knobs or {}).items():
        key = k if k.startswith("MCI_") else "MCI_" + k
        base = ENVELOPE_CENTER.get(key)
        if base is None:
            continue
        try:
            val = float(v)
        except (TypeError, ValueError):
            continue
        if val <= 0:
            degenerate.append(key)
            d2 += DEGENERATE_DISTANCE ** 2
            continue
        d2 += math.log(val / base, 2.0) ** 2
    return math.sqrt(d2), degenerate


def cmd_envelope(args) -> None:
    """봉투 거리 대 "그 조건에서 재튜닝하면 얻는 이득" 곡선.

    각 조건 CSV 에서 배포 상수(고정 카드) 대비 **그 조건 최적 팔**의 paired 이득을 재고
    조건의 봉투 거리와 나란히 놓는다. 이득이 거리의 단조 증가 함수면
    "언제 현장에서 폐루프를 돌려야 하는가" 가 거리 하나로 환원된다.

    ``--family Q`` 면 그 족 안에서만 재튜닝한다(= λ 만 다시 고르는 현실적 시나리오).
    빈 문자열이면 전 팔 중 최선(= 함수형까지 바꾸는 상한).
    """
    rows = []
    for pat in args.globs.split(","):
        for f in sorted(glob.glob(os.path.join(args.stage_dir, pat.strip()))):
            if f.endswith(".meta.json"):
                continue
            df = _load(f)
            pols = set(df.policy.unique())
            if args.fixed_arm not in pols:
                continue
            if args.family:
                cand = [q for q in pols if re.fullmatch(args.family + r"[0-9.]+", q)]
            else:
                cand = list(pols)
            if not cand:
                continue
            cubes = {q: cube(df, q) for q in set(cand) | {args.fixed_arm}}
            means = {q: v.mean() for q, v in cubes.items()}
            best = min(cand, key=lambda q: means[q])
            r = paired(cubes[best], cubes[args.fixed_arm])
            kn = _knobs(f)
            dist, degen = envelope_distance(kn)
            edge = ""
            if args.family:
                vals = sorted(float(q[len(args.family):]) for q in cand)
                if float(best[len(args.family):]) in (vals[0], vals[-1]):
                    edge = "격자끝"
            rows.append({"tag": os.path.basename(f)[:-4], "knobs": kn,
                         "manifest": _meta_manifest(f), "envelope_distance": dist,
                         "degenerate_knobs": degen, "best_arm": best,
                         "best_pdr": means[best], "fixed_pdr": means[args.fixed_arm],
                         "grid_edge": edge, **r})
    if not rows:
        print("[없음] 조건 CSV 를 찾지 못했다")
        return
    rows.sort(key=lambda x: x["envelope_distance"])

    scope = f"{args.family}족 내부" if args.family else "전 팔(함수형 포함)"
    print("=" * 108)
    print(f"봉투 거리 대 재튜닝 이득 — 배포 상수 {args.fixed_arm} 기준 · 재튜닝 범위 {scope}")
    print("  거리 = 정식조건 대비 log2 배율 편차의 L2 노름 (자원 0 조건은 관례값 4.0, 회귀에서 제외)")
    print(f"  이득 = {args.fixed_arm} PDR - 재튜닝 PDR (양수 = 재튜닝 이득) · 판정선 {JUDGE_LINE}")
    print("=" * 108)
    print(f"{'조건':16s} {'거리':>6} {'최적팔':>8} {'재튜닝':>9} {'고정':>9} "
          f"{'이득':>10} {'±CI':>9} {'W/T/L':>14}  판정")
    for x in rows:
        note = (" [" + x["grid_edge"] + "]") if x["grid_edge"] else ""
        note += " [퇴화노브]" if x["degenerate_knobs"] else ""
        print(f"{x['tag']:16s} {x['envelope_distance']:6.2f} {x['best_arm']:>8} "
              f"{x['best_pdr']:9.5f} {x['fixed_pdr']:9.5f} {x['delta']:+10.5f} "
              f"{x['ci95']:9.5f} {x['win']:>4}/{x['tie']:>4}/{x['loss']:>3}  "
              f"{_verdict(x['delta'], x['ci95'], 1)}{note}")

    ok = [x for x in rows if not x["degenerate_knobs"]]
    summary = {}
    if len(ok) >= 3:
        d = np.array([x["envelope_distance"] for x in ok])
        g = np.array([x["delta"] for x in ok])
        rho = float(np.corrcoef(d, g)[0, 1])
        summary["pearson_r"] = rho
        summary["n"] = len(ok)
        line = f"거리-이득 상관 Pearson r={rho:+.3f}, n={len(ok)}"
        try:
            from scipy.stats import spearmanr
            sp = spearmanr(d, g)
            summary["spearman_rho"] = float(sp.statistic)
            summary["spearman_p"] = float(sp.pvalue)
            line += f" · Spearman rho={sp.statistic:+.3f} (p={sp.pvalue:.4f})"
        except Exception:
            line += " · Spearman 미계산(scipy 없음)"
        print("")
        print("  " + line)
        cross = [x for x in ok if x["delta"] > JUDGE_LINE and x["delta"] > x["ci95"]]
        if cross:
            c = min(cross, key=lambda z: z["envelope_distance"])
            summary["min_crossing_distance"] = c["envelope_distance"]
            summary["min_crossing_tag"] = c["tag"]
            print(f"  판정선 초과 최소 거리 = {c['envelope_distance']:.2f} ({c['tag']}) "
                  f"-> 이 거리 이상이면 현장 재튜닝 권고")
        else:
            summary["min_crossing_distance"] = None
            print("  판정선을 넘는 조건이 없다 — 검증 범위 안에서는 배포 상수로 충분하다")
    mans = sorted({x["manifest"] for x in rows})
    print("")
    print(f"[좌표셋] {', '.join(mans)}")
    if any("tradeoff250" in m for m in mans):
        print("  주의: tradeoff250 은 test750 부분집합이다. 이 수치는 잠정값이다.")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"fixed_arm": args.fixed_arm, "family": args.family,
                   "judge_line": JUDGE_LINE, "envelope_center": ENVELOPE_CENTER,
                   "degenerate_distance": DEGENERATE_DISTANCE,
                   "summary": summary, "rows": rows},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[기록] {args.out}")



def cmd_interact(args) -> None:
    """(λ × yhold) 결합 격자 — 두 축이 정말 독립인가.

    v20 은 λ 를 `yhold=0` 에서만, yhold 를 `λ=18` 에서만 쓸었다. 두 축이 독립이면
    결합 격자의 최소점이 **(축별 단독 최적) 그 자리**여야 한다. 어긋나면 축이 상호작용한다.

    상호작용을 크기로 재는 방법 두 가지를 함께 낸다.
      ① 좌표 이동 — 결합 최적 (λ*, y*) 가 단독 최적 (λ0, y0) 와 다른가.
      ② 성능 차 — `PDR(λ0, y0) − PDR(λ*, y*)` 의 CRN paired 이득. 이게 결합을 고려해서
         실제로 얻는 값이고, 판정선 미만이면 "축을 따로 골라도 된다" 가 맞다.
    """
    for f in sorted(glob.glob(os.path.join(args.stage_dir, f"{args.prefix}_*.csv"))):
        if f.endswith(".meta.json"):
            continue
        tag = os.path.basename(f)[len(args.prefix) + 1: -4]
        df = _load(f)
        cells = {}
        for q in df.policy.unique():
            m = re.fullmatch(r"Q([0-9.]+)Y([0-9.]+)", q)
            if m:
                cells[(float(m.group(1)), float(m.group(2)))] = q
        if not cells:
            print(f"[{tag}] Q<λ>Y<y> 꼴 팔이 없다 — 건너뜀")
            continue
        lams = sorted({k[0] for k in cells})
        ys = sorted({k[1] for k in cells})
        cubes = {k: cube(df, v) for k, v in cells.items()}
        means = {k: v.mean() for k, v in cubes.items()}

        kn = _knobs(f)
        print("=" * 96)
        print(f"[{tag}] 결합 격자 {len(lams)}x{len(ys)}  노브={kn or '(기준조건)'}  좌표셋 {_meta_manifest(f)}")
        print("=" * 96)
        head = "  λ \\ y  " + "".join(f"{y:>10.0f}" for y in ys)
        print(head)
        for l in lams:
            row = f"  {l:6.0f} "
            for y in ys:
                v = means.get((l, y))
                row += f"{v:10.5f}" if v is not None else f"{'-':>10}"
            print(row)

        joint = min(means, key=means.get)
        # 축별 단독 최적: y=y_ref 행에서 최선 λ, λ=λ_ref 열에서 최선 y
        row_at_yref = {l: means[(l, args.y_ref)] for l in lams if (l, args.y_ref) in means}
        col_at_lref = {y: means[(args.lam_ref, y)] for y in ys if (args.lam_ref, y) in means}
        if not row_at_yref or not col_at_lref:
            print(f"  참조선(λ={args.lam_ref}, y={args.y_ref})이 격자에 없다 — 좌표 이동 판정 생략")
            continue
        l0 = min(row_at_yref, key=row_at_yref.get)
        y0 = min(col_at_lref, key=col_at_lref.get)
        sep = (l0, y0)
        print("")
        print(f"  결합 최적      (λ={joint[0]:.0f}, y={joint[1]:.0f})  PDR={means[joint]:.6f}")
        print(f"  축별 단독 최적 (λ={l0:.0f}, y={y0:.0f})  PDR="
              + (f"{means[sep]:.6f}" if sep in means else "격자 밖"))
        print(f"    (단독은 y={args.y_ref:.0f} 행에서 λ, λ={args.lam_ref:.0f} 열에서 y 를 각각 고른 값)")
        if sep == joint:
            print("  → 좌표 일치: 이 조건에서는 축을 따로 골라도 같은 답에 도달한다")
        elif sep in means:
            r = paired(cubes[joint], cubes[sep])
            print(f"  → 좌표 불일치. 결합이 단독보다 {r['delta']:+.6f} ±{r['ci95']:.6f} "
                  f"(W/T/L {r['win']}/{r['tie']}/{r['loss']}) {_verdict(r['delta'], r['ci95'], 1)}")
            if r["delta"] <= JUDGE_LINE:
                print("     판정선 미만 — 좌표는 움직였지만 성능 차이는 없다. "
                      "\"축을 따로 골라도 된다\" 가 성능 기준으로는 유지된다")
        # 채택 카드 대비
        if (args.lam_ref, args.y_ref) in means:
            base = (args.lam_ref, args.y_ref)
            r2 = paired(cubes[joint], cubes[base])
            print(f"  채택 카드 (λ={args.lam_ref:.0f}, y={args.y_ref:.0f}) 대비 결합 최적 이득 "
                  f"{r2['delta']:+.6f} ±{r2['ci95']:.6f} (W/T/L {r2['win']}/{r2['tie']}/{r2['loss']}) "
                  f"{_verdict(r2['delta'], r2['ci95'], 1)}")
        print("")


# ---------------------------------------------------------------- budgetcurve (온라인 선택 예산 곡선)
# 질문: **내부 시드를 몇 개(= 계산 예산을 얼마) 써야 온라인 선택이 0-에피소드 카드보다 나아지는가.**
#
# v22 ⑨ 에서 내부 4시드·39후보의 `selection_regret` +0.0114 가 그 카드가 주장하는 이득 +0.0089
# 보다 커서 "시드가 적으면 튜닝이 아무것도 안 하는 것보다 나쁘다" 가 나왔다. 그 곡선을 다 그린다.
#
# 설계의 핵심은 **CRN 접두사 재사용**이다. 각 팔이 같은 내부 시드 0..N-1 을 공유하므로
# "내부 K 시드로 고른 argmin" 은 시드 0..K-1 **접두사 평균의 argmin** 으로 나온다.
# 따라서 곡선 전체가 시뮬 재실행 0 회로 계산된다(이 파일의 다른 서브커맨드와 같다).
#
# 시드 블록은 서로소다 — 내부(선택) 0..127, 평가 1000..1059. regret 은 평가 블록에서 잰다.
#     regret(K) = E_eval[K시드로 고른 팔] − min_c E_eval[c]
# 뒤 항이 잡음 있는 추정의 min 이라 **하향** 편향이므로 regret 은 **상향** 편향이다.
# 그 편향의 크기는 K=Kmax 의 regret(선택이 거의 최적인 지점)이 바닥으로 읽어 준다(자가 교정).
# 곡선에서 의미 있는 것은 바닥 위쪽 = `regret_excess = regret(K) − regret(Kmax)` 다.
#
# ⚠️ `outer − inner` 꼴 혼합 지표를 쓰지 않는다. v22 ⑨ 에서 그 지표가 시드집합 난이도 이동
#    (−0.0117)과 상쇄돼 −0.0002 를 보고하며 선택 편향을 가렸다. 난이도 이동은
#    `level_shift` 로 **따로** 내고, 이름으로 "선택 편향이 아니다" 를 못 박는다.
#
# 0-에피소드 기준선이 **둘**이다. 이 둘을 구분하지 않으면 곡선이 폐루프의 가치를 과대평가한다.
#   ① 고정 카드  — 현행 채택 `Q18Y0`. 조건을 모른다.
#   ② 공식 카드  — v22 ① ② 로 확정된 `λ = 18.97 × 치료시간배수` 를 격자에 반올림한 팔 + y=0.
#                  현장에서 병원에 치료 회전율을 한 번 물어보면 에피소드 0 개로 얻는다.
#                  base(ts=1.0)에서는 ①과 같은 팔이어야 하고, 그 일치가 배선 검사다.
# 판정 문장도 둘이다. ②가 더 엄격하고 더 정직하다 — 어떤 K 에서도 공식 카드를 못 이기면
# 그것이 결론이고 음성이 아니라 배포 권고다("λ 에는 폐루프가 필요 없다").
#
# 탐색 세 가지를 같은 표에 놓는다(팔 수가 곧 통계 부담이다).
#   joint    32팔 = λ8 × y4 전수.                     에피소드 32·K
#   additive 12팔 = y=y_ref 행에서 λ̂ → λ=λ̂ 열에서 ŷ.  에피소드 (8+4)·K
#            근거는 v22 ⑥ (두 축이 파라미터 선택에서 독립, 5조건 중 4곳 좌표 일치).
#   yonly     4팔 = λ 는 공식으로 고정하고 y 만 고른다.  에피소드 4·K
#            v22 정본이 가리키는 실제 현장 알고리즘이다(공식 없는 축은 등급 축 하나뿐).
#
# 타이브레이크 규약(결정론): 접두사 평균이 같으면 **λ 작은 쪽 → 그 다음 y 작은 쪽**.
# 구현은 팔 목록을 (λ, y) 오름차순으로 정렬해 두고 `np.argmin`(최초 최소 반환)을 쓰는 것이다.
# `df.policy.unique()` 등장 순서나 dict 순서에 의존하면 regret 이 실행마다 달라진다.

BC_KS_DEFAULT = "1,2,3,4,6,8,12,16,24,32,48,64,96,128"
BC_MODES = ("joint", "additive", "yonly")
BC_MODE_LABEL = {"joint": "joint(전수)", "additive": "additive(축별)", "yonly": "yonly(등급축만)"}
ARM_QY_RE = re.compile(r"^Q([0-9.]+)Y([0-9.]+)$")


def _spearman(x, y) -> float:
    """순위 상관 — 동순위 평균처리 순위의 Pearson (= Spearman ρ). scipy 없이 계산한다."""
    rx = pd.Series(np.asarray(x, float)).rank().to_numpy()
    ry = pd.Series(np.asarray(y, float)).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _bc_arms(df: pd.DataFrame):
    """CSV 의 ``Q<λ>Y<y>`` 팔 → ``[(λ, y, 팔이름)]``, **(λ, y) 오름차순**.

    이 정렬이 타이브레이크 규약의 구현체다. 뒤에서 쓰는 ``np.argmin`` 은 최초 최소를
    돌려주므로 접두사 평균이 같으면 λ 작은 쪽 → 그 다음 y 작은 쪽이 뽑힌다.
    """
    arms = []
    for p in df.policy.unique():
        m = ARM_QY_RE.match(str(p))
        if m:
            arms.append((float(m.group(1)), float(m.group(2)), str(p)))
    arms.sort(key=lambda t: (t[0], t[1]))
    return arms


def _bc_block(path: str, label: str):
    """시드 블록 CSV 하나 → ``(정보 dict, 오류문구)``. 완주본이 아니면 ``(None, 이유)``.

    완주 검사를 네 겹으로 한다(research.md 판정 계약 3·4번).
      ① ``.meta.json`` 존재 — 없으면 진행 중 CSV 이므로 판정에 쓰지 않는다
      ② ``(region, policy, seed)`` 유일성 — 중복이 있으면 ``cube()`` 가 조용히 평균낸다
      ③ ``cube()`` 의 결측 예외 — 팔·지역·시드 격자에 구멍이 있으면 여기서 터진다
      ④ 팔마다 (지역, 시드) 격자 모양이 같은가 + 전부 유한값인가
    행 순서는 ``cube()`` 가 ``sort_index()`` 로 지역명 정렬하므로 팔 사이에 일치한다.
    """
    if not os.path.exists(path):
        return None, f"{label}: CSV 없음 ({path})"
    if not os.path.exists(path + ".meta.json"):
        return None, f"{label}: 미완주(.meta.json 없음) — 진행 중 CSV 로 판정하지 않는다"
    df = _load(path)
    arms = _bc_arms(df)
    if len(arms) < 2:
        return None, f"{label}: Q<λ>Y<y> 꼴 팔이 {len(arms)}개뿐"
    if df.duplicated(["region", "policy", "seed"]).any():
        return None, f"{label}: (region,policy,seed) 중복 행이 있다"
    regions = sorted(str(r) for r in df.region.unique())
    seeds = sorted(int(s) for s in df.seed.unique())
    try:
        cubes = {nm: cube(df, nm) for _, _, nm in arms}
    except ValueError as exc:
        return None, f"{label}: {exc}"
    want = (len(regions), len(seeds))
    bad = [nm for nm, c in cubes.items() if c.shape != want]
    if bad:
        return None, f"{label}: 팔 {bad[:3]} 의 (지역,시드) 격자가 {want} 와 다르다"
    stack = np.stack([cubes[nm] for _, _, nm in arms])
    if not np.isfinite(stack).all():
        return None, f"{label}: 비유한값이 있다"
    return {"path": path, "arms": arms, "regions": regions, "seeds": seeds, "cubes": cubes,
            "manifest": _meta_manifest(path), "knobs": _knobs(path),
            "cpu_ms_median": float(df.cpu_ms.median()),
            "wall_ms_median": float(df.wall_ms.median()),
            "n_rows": int(len(df))}, ""


def _bc_treat_scale(knobs: dict) -> float:
    """그 조건의 치료시간배수. 메타에 노브가 없으면 정식조건 1.0 이다."""
    for k in ("TREAT_SCALE", "MCI_TREAT_SCALE"):
        if k in (knobs or {}):
            try:
                return float(knobs[k])
            except (TypeError, ValueError):
                pass
    return 1.0


def _bc_snap_lam(lam_vals, lam_target: float) -> float:
    """공식 λ 를 격자에 **로그 거리**로 반올림. λ 는 배율 축이라 선형 거리로 반올림하면
    큰 쪽으로 쏠린다. 동률이면 작은 λ (``lam_vals`` 오름차순 + ``min`` 최초 반환)."""
    return min(lam_vals, key=lambda L: abs(math.log(L) - math.log(lam_target)))


def _bc_pick_joint(inner_mean: np.ndarray) -> np.ndarray:
    """(팔, 지역) 접두사 평균 → 좌표별 선택 팔 인덱스. 팔 축이 (λ,y) 정렬이므로 타이브레이크 성립."""
    return np.argmin(inner_mean, axis=0).astype(int)


def _bc_pick_additive(inner_mean: np.ndarray, lam_of, y_of, y_ref: float):
    """축별 순차 선택 — ① y=y_ref 행에서 λ̂ ② λ=λ̂ 열에서 ŷ."""
    step1 = np.flatnonzero(y_of == y_ref)          # 이미 λ 오름차순
    if step1.size == 0:
        raise ValueError(f"y={y_ref:g} 행이 격자에 없다 — additive 탐색 불가")
    nr = inner_mean.shape[1]
    chosen = np.empty(nr, dtype=int)
    for r in range(nr):
        lam_hat = lam_of[step1[int(np.argmin(inner_mean[step1, r]))]]
        step2 = np.flatnonzero(lam_of == lam_hat)  # 이미 y 오름차순
        chosen[r] = step2[int(np.argmin(inner_mean[step2, r]))]
    return chosen


def _bc_pick_yonly(inner_mean: np.ndarray, lam_of, lam_fix: float):
    """λ 를 공식으로 고정하고 등급 축만 고른다. 후보가 4개뿐이라 통계 부담이 가장 작다."""
    col = np.flatnonzero(lam_of == lam_fix)        # 이미 y 오름차순
    if col.size == 0:
        raise ValueError(f"λ={lam_fix:g} 열이 격자에 없다 — yonly 탐색 불가")
    nr = inner_mean.shape[1]
    chosen = np.empty(nr, dtype=int)
    for r in range(nr):
        chosen[r] = col[int(np.argmin(inner_mean[col, r]))]
    return chosen


def _bc_stats(chosen, Ec, eval_mean, oracle_mean, inner_mean, baselines) -> dict:
    """선택 결과의 regret·paired·순위일치 지표.

    ``chosen`` = 좌표별 팔 인덱스. 선택은 **좌표별**이다(현장 지휘관은 자기 좌표 하나를 튜닝한다).
    ``baselines`` = {이름: 기준 팔의 (지역 × 평가시드) 큐브}. 오라클 항이 상쇄되므로
    기준선 대비 paired 는 regret 차이와 같고, W/T/L 은 ``paired()`` 정의를 그대로 쓴다.
    """
    nr = int(oracle_mean.size)
    sel = np.stack([Ec[int(chosen[r]), r, :] for r in range(nr)])
    reg = sel.mean(1) - oracle_mean
    rho = [_spearman(inner_mean[:, r], eval_mean[:, r]) for r in range(nr)]
    rk = [float(pd.Series(eval_mean[:, r]).rank(method="min").to_numpy()[int(chosen[r])])
          for r in range(nr)]
    rec = {"regret_mean": float(reg.mean()), "regret_ci95": ci(reg),
           "regret_median": float(np.median(reg)), "regret_p90": float(np.percentile(reg, 90)),
           "regret_max": float(reg.max()),
           "frac_over_judge": float(np.mean(reg > JUDGE_LINE)),
           "spearman_mean": float(np.nanmean(rho)),
           "top1_eval_rank_median": float(np.median(rk)),
           "n_distinct_arms": int(len({int(c) for c in chosen})),
           "vs": {}}
    for name, bcube in (baselines or {}).items():
        rec["vs"][name] = paired(sel, bcube)       # delta = 기준 − 선택 (양수 = 선택이 낫다)
    return rec


def _bc_nearest_k(ks, target: float):
    """격자에 없는 K 는 보간하지 않고 가장 가까운 계산된 K 로 표기한다(로그 거리, 동률이면 작은 쪽)."""
    if target < min(ks) or target > max(ks):
        return None
    return min(ks, key=lambda k: (abs(math.log(k) - math.log(max(target, 1e-9))), k))


def _bc_mark(d: dict) -> str:
    """paired 결과 한 칸 — 양수면 선택이 기준선보다 낫다. ★=판정선 초과·*=CI 초과."""
    s = "★" if (d["delta"] > JUDGE_LINE and d["delta"] > d["ci95"]) else \
        ("*" if d["delta"] > d["ci95"] else " ")
    return f"{d['delta']:+9.5f}{s}"


def cmd_budgetcurve(args) -> None:
    ks_req = sorted({int(k) for k in args.ks.split(",") if k.strip()})
    cores = max(1, int(args.cores))
    out = {"judge_line": JUDGE_LINE, "stage_dir": args.stage_dir,
           "fixed_arm": args.fixed_arm, "y_ref": float(args.y_ref),
           "formula_lam0": float(args.formula_lam0), "ks_requested": ks_req, "cores": cores,
           "tiebreak": "접두사 평균 동률이면 λ 작은 쪽 → 그 다음 y 작은 쪽 "
                       "(팔을 (λ,y) 오름차순 정렬 후 np.argmin = 최초 최소)",
           "regret_definition": "좌표별 E_eval[선택 팔] − min_c E_eval[c] "
                                "(E_eval = 평가 시드 블록 평균)",
           "floor_note": "min_c E_eval[c] 는 잡음 있는 추정의 min 이라 하향 편향이고 "
                         "regret 은 상향 편향이다. K=Kmax 의 regret 을 바닥 추정치로 읽고 "
                         "regret_excess = regret(K) − regret(Kmax) 를 본다.",
           "level_shift_note": "level_shift = 전 팔 mean(E_eval − E_inner(K)). "
                               "시드 블록 난이도 차이이며 선택 편향이 아니다.",
           "wall_formula": "초 = 에피소드수 × cpu_ms_중위 / 1000 / 코어수 "
                           "(cpu_ms 중위는 그 조건 내부 CSV 실측)",
           "baselines_note": "0-에피소드 기준선 둘 — 고정 카드(조건 무지) / "
                             "공식 카드(λ = formula_lam0 × 치료시간배수, 격자 로그 반올림, y=y_ref)",
           "conditions": {}, "skipped": []}

    for cond in [c.strip() for c in args.conditions.split(",") if c.strip()]:
        ipath = os.path.join(args.stage_dir, f"{cond}_{args.inner_suffix}.csv")
        epath = os.path.join(args.stage_dir, f"{cond}_{args.eval_suffix}.csv")
        inner, e1 = _bc_block(ipath, f"{cond}/{args.inner_suffix}")
        ev, e2 = _bc_block(epath, f"{cond}/{args.eval_suffix}")
        if inner is None or ev is None:
            for msg in (e1, e2):
                if msg:
                    print(f"[건너뜀] {msg}")
                    out["skipped"].append(msg)
            continue

        names = [a[2] for a in inner["arms"]]
        if names != [a[2] for a in ev["arms"]]:
            msg = f"{cond}: 내부/평가 팔 집합이 다르다 — CRN 비교 불가"
            print(f"[건너뜀] {msg}")
            out["skipped"].append(msg)
            continue
        if inner["regions"] != ev["regions"]:
            msg = f"{cond}: 내부/평가 좌표 집합이 다르다 (내부 {len(inner['regions'])} · "
            msg += f"평가 {len(ev['regions'])})"
            print(f"[건너뜀] {msg}")
            out["skipped"].append(msg)
            continue
        if args.fixed_arm not in names:
            msg = f"{cond}: 고정 카드 팔 {args.fixed_arm} 이 격자에 없다"
            print(f"[건너뜀] {msg}")
            out["skipped"].append(msg)
            continue

        lam_of = np.array([a[0] for a in inner["arms"]], float)
        y_of = np.array([a[1] for a in inner["arms"]], float)
        lam_vals = sorted(set(lam_of.tolist()))
        y_vals = sorted(set(y_of.tolist()))
        nr = len(inner["regions"])
        n_inner, n_eval = len(inner["seeds"]), len(ev["seeds"])
        Ic = np.stack([inner["cubes"][n] for n in names])      # (팔, 지역, 내부시드)
        Ec = np.stack([ev["cubes"][n] for n in names])         # (팔, 지역, 평가시드)
        eval_mean = Ec.mean(2)                                 # (팔, 지역)
        oracle_idx = np.argmin(eval_mean, axis=0)
        oracle_mean = eval_mean[oracle_idx, np.arange(nr)]

        # 공식 카드 — λ = lam0 × 치료시간배수 를 격자에 로그 반올림하고 y=y_ref 와 조합한다.
        ts = _bc_treat_scale(inner["knobs"])
        lam_formula = float(args.formula_lam0) * ts
        lam_snap = _bc_snap_lam(lam_vals, lam_formula)
        fa = [i for i in range(len(names)) if lam_of[i] == lam_snap and y_of[i] == float(args.y_ref)]
        formula_arm = names[fa[0]] if fa else None

        fixed_i = names.index(args.fixed_arm)
        baselines = {"고정카드": Ec[fixed_i]}
        if formula_arm is not None:
            baselines["공식카드"] = Ec[names.index(formula_arm)]

        ks_use = [k for k in ks_req if 1 <= k <= n_inner]
        if not ks_use:
            msg = f"{cond}: 요청 K 가 내부 시드 수 {n_inner} 를 전부 넘는다"
            print(f"[건너뜀] {msg}")
            out["skipped"].append(msg)
            continue
        kmax = ks_use[-1]

        n_arms_mode = {"joint": len(names), "additive": len(lam_vals) + len(y_vals),
                       "yonly": len(y_vals)}
        cpu_ms = inner["cpu_ms_median"]

        def _sec(n_ep: int) -> float:
            return n_ep * cpu_ms / 1000.0 / cores

        kn = ",".join(f"{k}={v}" for k, v in (inner["knobs"] or {}).items())
        print("=" * 122)
        print(f"[{cond}] 예산 곡선 — 온라인 선택 대 0-에피소드 카드")
        print(f"  좌표셋 {inner['manifest']} · 좌표 {nr}개 · 팔 {len(names)}개 "
              f"(λ {len(lam_vals)} × y {len(y_vals)}) · 노브 {kn or '(기준조건)'}")
        print(f"  내부(선택) 시드 {inner['seeds'][0]}..{inner['seeds'][-1]} ({n_inner}개) · "
              f"평가 시드 {ev['seeds'][0]}..{ev['seeds'][-1]} (M={n_eval})")
        ovl = sorted(set(inner["seeds"]) & set(ev["seeds"]))
        print(f"  시드 블록 교집합 {len(ovl)}개" + (f" ⚠️ {ovl[:5]} — regret 이 낙관 편향된다"
                                                  if ovl else " (서로소 = 설계대로)"))
        print(f"  타이브레이크: 평균 동률이면 λ 작은 쪽 → 그 다음 y 작은 쪽 (결정론)")
        print(f"  실행시간: cpu_ms/ep 중위 {cpu_ms:.1f} (내부 CSV 실측) · {cores}코어 가정 · "
              f"초 = 에피소드 × {cpu_ms:.1f} / 1000 / {cores}")
        print(f"  치료시간배수 {ts:g} → 공식 λ = {args.formula_lam0:g} × {ts:g} = {lam_formula:.2f} "
              f"→ 격자 로그 반올림 λ={lam_snap:g} → 공식 카드 팔 {formula_arm or '(격자 밖)'}")
        if formula_arm == args.fixed_arm:
            print(f"    (배선 검사: 이 조건에서는 공식 카드 = 고정 카드 = {args.fixed_arm} 로 일치한다)")
        print("=" * 122)

        # 0-에피소드 기준선 — 곡선의 판단 기준. K=0, 에피소드 0, 0.0초.
        base_rows = {}
        inner_kmax = Ic[:, :, :kmax].mean(2)
        for bname, bidx in (("고정카드", fixed_i),
                            ("공식카드", names.index(formula_arm) if formula_arm else None)):
            if bidx is None:
                continue
            st = _bc_stats(np.full(nr, bidx, int), Ec, eval_mean, oracle_mean, inner_kmax, None)
            st.update({"arm": names[bidx], "K": 0, "episodes": 0, "wall_sec": 0.0})
            base_rows[bname] = st
            print(f"  [기준선 {bname} {names[bidx]:7s}] K=0 · 에피소드 0 · 0.0초 · "
                  f"regret {st['regret_mean']:+.5f} ±{st['regret_ci95']:.5f} · "
                  f"중위 {st['regret_median']:+.5f} · p90 {st['regret_p90']:+.5f} · "
                  f">판정선 {100*st['frac_over_judge']:.0f}% · eval순위 중위 {st['top1_eval_rank_median']:.0f}")
        print("")

        # 시드 난이도 이동 — 선택과 무관한 팔 수준 양이므로 탐색 모드와 독립이다.
        shifts = {}
        for k in ks_use:
            shifts[k] = float(np.mean(eval_mean - Ic[:, :, :k].mean(2)))
        print("  [level_shift = 전 팔 mean(E_eval − E_inner(K))] — 시드 블록 난이도 차이이며 "
              "**선택 편향이 아니다**")
        print("    " + "  ".join(f"K{k}={shifts[k]:+.5f}" for k in ks_use))
        print("")

        rows = {m: [] for m in BC_MODES}
        for mode in BC_MODES:
            na = n_arms_mode[mode]
            print(f"  ── {BC_MODE_LABEL[mode]:16s} 후보 {na:2d}팔 · 에피소드 = {na}·K ──")
            print(f"  {'K':>4} {'ep':>6} {'sec':>8} {'regret':>10} {'±CI':>8} {'excess':>9} "
                  f"{'median':>9} {'p90':>9} {'>line':>6} {'rho':>6} {'rk':>4} "
                  f"{'고정−선택':>11} {'공식−선택':>11} {'팔수':>4}")
            for k in ks_use:
                inner_mean = Ic[:, :, :k].mean(2)
                try:
                    if mode == "joint":
                        chosen = _bc_pick_joint(inner_mean)
                    elif mode == "additive":
                        chosen = _bc_pick_additive(inner_mean, lam_of, y_of, float(args.y_ref))
                    else:
                        chosen = _bc_pick_yonly(inner_mean, lam_of, lam_snap)
                except ValueError as exc:
                    print(f"    [{mode}] {exc}")
                    break
                st = _bc_stats(chosen, Ec, eval_mean, oracle_mean, inner_mean, baselines)
                st.update({"K": k, "n_arms": na, "episodes": na * k,
                           "wall_sec": _sec(na * k), "level_shift": shifts[k],
                           "chosen_arm_counts": {names[i]: int((chosen == i).sum())
                                                 for i in sorted(set(int(c) for c in chosen))}})
                rows[mode].append(st)
            if not rows[mode]:
                continue
            floor = rows[mode][-1]["regret_mean"]
            for st in rows[mode]:
                st["regret_excess"] = st["regret_mean"] - floor
                vf = st["vs"].get("고정카드")
                vq = st["vs"].get("공식카드")
                print(f"  {st['K']:>4} {st['episodes']:>6} {st['wall_sec']:>8.1f} "
                      f"{st['regret_mean']:>10.5f} {st['regret_ci95']:>8.5f} "
                      f"{st['regret_excess']:>+9.5f} {st['regret_median']:>9.5f} "
                      f"{st['regret_p90']:>9.5f} {100*st['frac_over_judge']:>5.0f}% "
                      f"{st['spearman_mean']:>+6.3f} {st['top1_eval_rank_median']:>4.0f} "
                      f"{_bc_mark(vf) if vf else ' ':>11} {_bc_mark(vq) if vq else '        —':>11} "
                      f"{st['n_distinct_arms']:>4}")
            print(f"    바닥(상향 편향 추정치) = K={kmax} 의 regret {floor:+.5f} · "
                  f"excess 는 그 바닥 위쪽만 센다")
            print("")

        # 교차점 — 이 실험의 답. 두 기준선에 대해 각각 낸다.
        cross = {}
        print("  [교차점] regret(K) < regret(기준선) 이 되는 최소 K")
        for mode in BC_MODES:
            if not rows[mode]:
                continue
            for bname, bst in base_rows.items():
                hit = next((s for s in rows[mode] if s["regret_mean"] < bst["regret_mean"]), None)
                sig = next((s for s in rows[mode]
                            if s["vs"].get(bname) and
                            s["vs"][bname]["delta"] > s["vs"][bname]["ci95"]), None)
                strict = next((s for s in rows[mode]
                               if s["vs"].get(bname) and
                               s["vs"][bname]["delta"] > max(s["vs"][bname]["ci95"], JUDGE_LINE)),
                              None)
                cross[f"{mode}/{bname}"] = {
                    "mean": None if hit is None else
                            {"K": hit["K"], "episodes": hit["episodes"], "wall_sec": hit["wall_sec"]},
                    "paired_sig": None if sig is None else
                                  {"K": sig["K"], "episodes": sig["episodes"],
                                   "wall_sec": sig["wall_sec"], **sig["vs"][bname]},
                    "paired_over_judge": None if strict is None else
                                         {"K": strict["K"], "episodes": strict["episodes"],
                                          "wall_sec": strict["wall_sec"], **strict["vs"][bname]}}
                c = cross[f"{mode}/{bname}"]
                m = c["mean"]
                s2 = c["paired_sig"]
                print(f"    {BC_MODE_LABEL[mode]:16s} vs {bname} : "
                      + (f"K={m['K']:<4d} ep={m['episodes']:<6d} {m['wall_sec']:6.1f}초"
                         if m else "격자 안에서 교차 없음 — 기준선을 못 이긴다")
                      + (f"  · paired 유의는 K={s2['K']} ({s2['delta']:+.5f} ±{s2['ci95']:.5f}, "
                         f"W/T/L {s2['win']}/{s2['tie']}/{s2['loss']})" if s2 else "  · paired 유의 없음"))
        print("")

        # 동일 예산 비교 — 축 독립성·공식이 실제로 예산을 벌어 주는지 판정하는 자리.
        print("  [동일 예산] 같은 에피소드 예산 B 에서 세 탐색을 나란히 "
              f"(joint K=B/{n_arms_mode['joint']} · additive K=B/{n_arms_mode['additive']} · "
              f"yonly K=B/{n_arms_mode['yonly']})")
        print(f"  {'B(ep)':>7} {'sec':>7} | " + " | ".join(
            f"{BC_MODE_LABEL[m]:>16}" for m in BC_MODES))
        by_k = {m: {s["K"]: s for s in rows[m]} for m in BC_MODES}
        budgets = sorted({n_arms_mode["joint"] * k for k in ks_use})
        eqrows = []
        for B in budgets:
            cells, rec = [], {"budget_episodes": B, "wall_sec": _sec(B), "modes": {}}
            for m in BC_MODES:
                tgt = B / n_arms_mode[m]
                kk = _bc_nearest_k(ks_use, tgt)
                if kk is None or kk not in by_k[m]:
                    cells.append(f"{'범위밖':>16}")
                    rec["modes"][m] = None
                    continue
                st = by_k[m][kk]
                snap = "" if abs(kk - tgt) < 1e-9 else "~"
                cells.append(f"K{kk}{snap} {st['regret_mean']:+.5f}".rjust(16))
                rec["modes"][m] = {"K_target": tgt, "K_used": kk, "snapped": bool(snap),
                                   "regret_mean": st["regret_mean"],
                                   "regret_ci95": st["regret_ci95"],
                                   "regret_excess": st["regret_excess"]}
            eqrows.append(rec)
            print(f"  {B:>7} {_sec(B):>7.1f} | " + " | ".join(cells))
        print("    ~ = 그 예산의 K 가 격자에 없어 가장 가까운 계산된 K 로 대체(보간하지 않았다)")
        print("")

        out["conditions"][cond] = {
            "inner_csv": ipath, "eval_csv": epath, "manifest": inner["manifest"],
            "knobs": inner["knobs"], "treat_scale": ts,
            "n_regions": nr, "regions": inner["regions"],
            "arms": names, "lam_grid": lam_vals, "y_grid": y_vals,
            "n_arms_by_mode": n_arms_mode,
            "inner_seeds": [inner["seeds"][0], inner["seeds"][-1], n_inner],
            "eval_seeds": [ev["seeds"][0], ev["seeds"][-1], n_eval],
            "seed_overlap": ovl,
            "cpu_ms_median": cpu_ms, "wall_ms_median": inner["wall_ms_median"],
            "formula": {"lam0": float(args.formula_lam0), "treat_scale": ts,
                        "lam_formula": lam_formula, "lam_snapped": lam_snap,
                        "arm": formula_arm,
                        "same_as_fixed": bool(formula_arm == args.fixed_arm)},
            "baselines": base_rows, "level_shift": {str(k): v for k, v in shifts.items()},
            "ks_used": ks_use, "kmax": kmax,
            "floor_regret": {m: (rows[m][-1]["regret_mean"] if rows[m] else None) for m in BC_MODES},
            "curve": {m: rows[m] for m in BC_MODES},
            "crossing": cross, "equal_budget": eqrows,
        }
        if "tradeoff250" in (inner["manifest"] or ""):
            print("  ⚠️ tradeoff250 은 test750 부분집합이다 — 이 수치는 선택 근거로 쓸 수 없다")

    if not out["conditions"]:
        print("[없음] 완주한 조건이 하나도 없다 — .meta.json 이 붙은 CSV 만 판정에 쓴다")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2,
                  default=float)
        print(f"[기록] {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("headroom", help="G1 헤드룸 게이트")
    h.add_argument("--stage_dir", default="results/scoreboard/v20/theory")
    h.add_argument("--extreme_prefix", default="extreme")
    h.add_argument("--yhold_prefix", default="yhold")
    h.add_argument("--fixed_arm", default="Q18", help="현행 채택 카드의 팔 이름")
    h.add_argument("--yhold_arm_prefix", default="Y")
    h.add_argument("--yhold_base_arm", default="Y0", help="채택된 yhold 팔")
    h.add_argument("--out", default="")
    h.set_defaults(func=cmd_headroom)

    d = sub.add_parser("decompose", help="봉투 밖 손실 분해")
    d.add_argument("--csv", required=True)
    d.add_argument("--family", default="Q", help="λ 를 스윕한 팔 족 접두사")
    d.add_argument("--fixed_arm", default="Q18")
    d.add_argument("--out", default="")
    d.set_defaults(func=cmd_decompose)

    s = sub.add_parser("lamscale", help="물리 축에 대한 최적 파라미터 스케일링 기울기")
    s.add_argument("--stage_dir", default="results/scoreboard/v20/theory")
    s.add_argument("--prefix", default="treat", help="CSV 파일명 접두사")
    s.add_argument("--axis_token", default="ts", help="파일명에서 축 값 앞에 붙는 토큰")
    s.add_argument("--base_tag", default="base")
    s.add_argument("--base_value", type=float, default=1.0)
    s.add_argument("--axis_name", default="치료시간배수")
    s.add_argument("--families", default="Q,H,S")
    s.add_argument("--predict_at", type=float, default=4.0)
    s.add_argument("--axis_scale", choices=["v20", "raw"], default="v20",
                   help="v20: 태그 ts075 를 0.75 로 환산(소수점 생략 규약) · raw: 그대로 float")
    s.add_argument("--anchor_csv", default="", help="축 기준점을 담은 다른 CSV(같은 좌표셋이어야 한다)")
    s.add_argument("--anchor_value", type=float, default=1.0)
    s.add_argument("--out", default="")
    s.set_defaults(func=cmd_lamscale)

    e = sub.add_parser("envelope", help="봉투 거리 대 재튜닝 이득 곡선")
    e.add_argument("--stage_dir", default="results/scoreboard/v22/retune")
    e.add_argument("--globs", default="extreme_*.csv,treat_*.csv,yhold_*.csv")
    e.add_argument("--fixed_arm", default="Q18")
    e.add_argument("--family", default="Q", help="재튜닝을 이 족으로 제한(빈 문자열이면 전 팔)")
    e.add_argument("--out", default="")
    e.set_defaults(func=cmd_envelope)

    i = sub.add_parser("interact", help="(λ × yhold) 결합 격자 — 축 독립성 검정")
    i.add_argument("--stage_dir", default="results/scoreboard/v22/retune")
    i.add_argument("--prefix", default="lamx")
    i.add_argument("--lam_ref", type=float, default=18.0, help="채택 λ (yhold 단독 스윕이 쓴 값)")
    i.add_argument("--y_ref", type=float, default=0.0, help="채택 yhold (λ 단독 스윕이 쓴 값)")
    i.set_defaults(func=cmd_interact)

    b = sub.add_parser("budgetcurve", help="온라인 선택의 예산 곡선(내부 시드 수 대비 선택 품질)")
    b.add_argument("--stage_dir", default="results/scoreboard/v22/budgetcurve")
    b.add_argument("--conditions", default="base,ts40",
                   help="조건 태그 쉼표 목록. <조건>_<블록>.csv 를 읽는다")
    b.add_argument("--inner_suffix", default="inner", help="선택용 시드 블록 파일 접미")
    b.add_argument("--eval_suffix", default="eval", help="평가용 시드 블록 파일 접미")
    b.add_argument("--fixed_arm", default="Q18Y0", help="현행 채택 카드(조건을 모르는 0-에피소드 기준선)")
    b.add_argument("--formula_lam0", type=float, default=18.97,
                   help="정식조건(치료시간배수 1.0)의 λ*. 공식 카드는 lam0 × 치료시간배수 를 "
                        "격자에 로그 반올림한 팔 + y=y_ref 다")
    b.add_argument("--y_ref", type=float, default=0.0,
                   help="채택 yhold. additive 1단계 행이자 공식 카드의 y 다(등급 축엔 공식이 없다)")
    b.add_argument("--ks", default=BC_KS_DEFAULT, help="내부 시드 수 격자")
    b.add_argument("--cores", type=int, default=24, help="벽시계 환산에 쓸 코어 수")
    b.add_argument("--out", default="")
    b.set_defaults(func=cmd_budgetcurve)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
