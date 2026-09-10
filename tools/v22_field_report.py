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
    for f in sorted(glob.glob(pat)):
        if f.endswith(".meta.json"):
            continue
        tag = os.path.basename(f)[len(args.prefix) + 1: -4]
        val = args.base_value if tag == args.base_tag else float(tag.replace(args.axis_token, ""))
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
    print(f"\n[좌표셋] {', '.join(mans)}")
    if any("tradeoff250" in m for m in mans):
        print("  ⚠️ tradeoff250 = test750 부분집합. 잠정값이다.")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
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
    s.add_argument("--out", default="")
    s.set_defaults(func=cmd_lamscale)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
