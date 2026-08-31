# -*- coding: utf-8 -*-
"""train1000 random4 매니페스트를 시군구별 4엔트리 매니페스트 250개로 분할한다.

v18 의 지역특화 PPO 교사(시군구당 1개)를 학습하기 위한 전처리다. 산출 매니페스트는
``scenarios/manifests/eval_holdout_sido/세종.json`` 과 같은 스키마(``{키: yaml 절대경로}``)라
``train_ppo_feature.py`` 가 **무수정으로** 소비한다 — 그 스크립트의 엄격 검증(`:111`)은
파일명이 ``sigungu_osrm_train1000_random4_manifest.json`` 일 때만 걸린다.

같이 만드는 것
--------------
``_index.json``  지역별 지형 특징과 층화 정보. 특징은 **학습좌표(p0~p3)** 의
``static_train1000.npz`` 에서만 계산한다 — 대표점(평가좌표)을 쓰면 누수다.

  near_t3    현장에서 가장 가까운 Tier3 병원까지 도로거리(km), 4좌표 중앙값
  n_reach30  인계시간 포함 30분 내 도달 가능한 (병원, 수단) 조합 수, 4좌표 중앙값
             AMB = d_road/50*60 + 5 분, UAV = d_euc/200*60 + 10 분 (v17_funnel_report 규약)
  stratum    두 축의 사분위 교차 (0~15). wave1 은 각 층에서 고르게 뽑는다

``--holdout p3`` 를 주면 지역마다 ``<지역>.train3.json``(p0~p2)도 함께 쓴다. Wave 1 의
스텝 예산 결정용 내부검증에 쓴다 — **평가좌표(대표점)로 예산을 고르면 누수**이므로 p3 를 쓴다.

사용
----
    python src/sce_src/split_sigungu_manifests.py --holdout p3 --wave1 16
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json"
STATIC = REPO / "results/scoreboard/v17/fieldrules/static_train1000.npz"
OUT = REPO / "scenarios/manifests/sigungu250"

V_AMB, V_UAV, H_AMB, H_UAV, CUT = 50.0, 200.0, 5.0, 10.0, 30.0
_P = re.compile(r"^(.*)_p([0-3])$")


def region_of(key: str) -> str:
    m = _P.match(key)
    if not m:
        raise ValueError(f"p 접미 없는 키: {key}")
    return m.group(1)


def terrain_features(static_path: Path) -> dict[str, dict]:
    """학습좌표 정적표에서 좌표별 지형 특징 → 지역별 중앙값."""
    z = np.load(static_path, allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    d_road, d_euc, tier, heli = z["d_road"], z["d_euc"], z["tier"], z["heli"]
    per: dict[str, list[tuple[float, float]]] = {}
    for i, k in enumerate(keys):
        t3 = tier[i] >= 3
        near_t3 = float(d_road[i][t3].min()) if t3.any() else float("nan")
        t_amb = d_road[i] / V_AMB * 60.0 + H_AMB
        t_uav = d_euc[i] / V_UAV * 60.0 + H_UAV
        n_reach = int((t_amb <= CUT).sum() + ((heli[i] > 0) & (t_uav <= CUT)).sum())
        per.setdefault(region_of(k), []).append((near_t3, float(n_reach)))
    return {r: {"near_t3": float(np.median([v[0] for v in vs])),
                "n_reach30": float(np.median([v[1] for v in vs])),
                "n_coord": len(vs)}
            for r, vs in per.items()}


def assign_strata(feat: dict[str, dict]) -> None:
    """두 축 사분위 교차로 0~15 층 부여. 각 축은 250 지역 분포 기준."""
    regs = sorted(feat)
    for axis in ("near_t3", "n_reach30"):
        v = np.array([feat[r][axis] for r in regs], float)
        q = np.nanquantile(v, [0.25, 0.5, 0.75])
        for r in regs:
            feat[r][f"q_{axis}"] = int(np.searchsorted(q, feat[r][axis], side="right"))
    for r in regs:
        feat[r]["stratum"] = feat[r]["q_near_t3"] * 4 + feat[r]["q_n_reach30"]


def pick_wave1(feat: dict[str, dict], n: int, seed: int = 20260901) -> list[str]:
    """층마다 라운드로빈으로 n 개를 뽑는다(재현 가능한 고정 seed)."""
    rng = np.random.default_rng(seed)
    buckets: dict[int, list[str]] = {}
    for r in sorted(feat):
        buckets.setdefault(feat[r]["stratum"], []).append(r)
    for s in buckets:
        buckets[s] = list(rng.permutation(buckets[s]))
    picked, i = [], 0
    while len(picked) < n:
        progressed = False
        for s in sorted(buckets):
            if i < len(buckets[s]) and len(picked) < n:
                picked.append(buckets[s][i]); progressed = True
        if not progressed:
            break
        i += 1
    return sorted(picked)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--static", default=str(STATIC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--holdout", default="", choices=["", "p0", "p1", "p2", "p3"],
                    help="지정하면 그 좌표를 뺀 <지역>.train3.json 도 함께 쓴다")
    ap.add_argument("--wave1", type=int, default=16)
    args = ap.parse_args()

    man = json.load(open(args.src, encoding="utf-8"))
    if len(man) != 1000:
        raise ValueError(f"train1000 이 아니다: {len(man)}")
    groups: dict[str, dict[str, str]] = {}
    for k, v in man.items():
        groups.setdefault(region_of(k), {})[k] = v
    bad = {r: sorted(g) for r, g in groups.items() if len(g) != 4}
    if len(groups) != 250 or bad:
        raise ValueError(f"250지역×4좌표 구조 아님: 지역 {len(groups)}, 이상 {list(bad)[:3]}")

    feat = terrain_features(Path(args.static))
    missing = set(groups) - set(feat)
    if missing:
        raise ValueError(f"정적표에 없는 지역: {sorted(missing)[:3]}")
    assign_strata(feat)
    wave1 = pick_wave1(feat, args.wave1)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for r, g in groups.items():
        (out / f"{r}.json").write_text(
            json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")
        if args.holdout:
            g3 = {k: v for k, v in g.items() if not k.endswith("_" + args.holdout)}
            if len(g3) != 3:
                raise ValueError(f"{r}: holdout 분리 실패")
            (out / f"{r}.train3.json").write_text(
                json.dumps(g3, ensure_ascii=False, indent=1), encoding="utf-8")

    index = {
        "source_manifest": str(Path(args.src).resolve()),
        "static_table": str(Path(args.static).resolve()),
        "holdout_coord": args.holdout or None,
        "wave1_seed": 20260901,
        "wave1": wave1,
        "note": ("지형 특징은 학습좌표(p0~p3)에서만 계산했다. 대표점250·외부250 은 쓰지 않았다. "
                 "층화는 wave 배정(스케줄링)용이며 최종 판정에 쓰지 않는다."),
        "regions": {r: {**feat[r], "wave": 1 if r in wave1 else 2,
                        "manifest": str((out / f"{r}.json").resolve()),
                        "manifest_train3": (str((out / f"{r}.train3.json").resolve())
                                            if args.holdout else None)}
                    for r in sorted(groups)},
    }
    (out / "_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")

    nt = np.array([feat[r]["near_t3"] for r in sorted(groups)])
    nr = np.array([feat[r]["n_reach30"] for r in sorted(groups)])
    print(f"[split] 지역 {len(groups)} · 매니페스트 {len(list(out.glob('*.json')))} 개 → {out}")
    print(f"        near_t3   중위 {np.median(nt):6.2f} km  IQR [{np.quantile(nt,.25):.2f}, {np.quantile(nt,.75):.2f}]  최대 {nt.max():.1f}")
    print(f"        n_reach30 중위 {np.median(nr):6.1f} 개  IQR [{np.quantile(nr,.25):.0f}, {np.quantile(nr,.75):.0f}]")
    print(f"[wave1] {len(wave1)}개 층화 표집: {', '.join(wave1)}")


if __name__ == "__main__":
    main()
