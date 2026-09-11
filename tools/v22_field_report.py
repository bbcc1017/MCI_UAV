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
from v20_threshold_report import cube, paired  # noqa: E402

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

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
