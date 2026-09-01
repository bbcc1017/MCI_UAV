# -*- coding: utf-8 -*-
"""sigungu30 풀(시군구 250 × 30점)을 지역별 학습/예산/평가 매니페스트로 분할 (v18 E5).

## 분할 규약 — 24 / 3 / 3

    train24   모델 학습
    budget3   **스텝 예산 결정 전용** 내부검증. 학습에 안 쓴 같은 시군구 좌표.
              평가좌표(대표점250)로 예산을 고르면 누수이므로 반드시 이쪽을 쓴다.
    test3     **지역 내부 평가**. 예산 선택에도 안 쓴다.
              250지역 × 3점 = 750점이라 대표점250(250점)보다 통계력이 3배다.

정본 비교축은 여전히 **대표점250 · 외부250** 이다 — 이 프로젝트의 모든 기준선
(CARD 0.141493, PPO 6시드, START-LB3, L0~L3 사다리)이 그 좌표에서 측정됐고, 신규 30점 풀은
생성 시 두 셋과 1.0km 이격을 강제해 실측 최소 이격 1.001km 다.

## 중첩 부분집합

학습곡선(모델이 몇 좌표부터 포화하는가)을 재려면 부분집합이 **중첩**이어야 한다
(train4 ⊂ train8 ⊂ train16 ⊂ train24). 서로 다른 좌표를 뽑으면 좌표 운과 데이터량이
교란된다. 여기서는 지역별로 고정 seed 셔플 후 앞에서부터 잘라 중첩을 보장한다.

⚠️ 시나리오 다양성은 이미 k≈8 에서 포화한다(v18 실측: 도심 30/4 = 1.13배, 농촌 1.17배).
따라서 학습곡선의 관심 구간은 데이터량보다 **스텝 예산**이며, 부분집합은 데이터 한계와
스텝 한계를 분리하는 용도다.

사용
----
    python src/sce_src/split_sigungu30.py --wave1 16
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC_MAN = REPO / "scenarios/manifests/sigungu30_manifest.json"
SRC_PTS = REPO / "scenarios/manifests/sigungu30_points.json"
STATIC = REPO / "results/scoreboard/v18/static_sigungu30.npz"
OUT = REPO / "scenarios/manifests/sigungu30"

V_AMB, V_UAV, H_AMB, H_UAV, CUT = 50.0, 200.0, 5.0, 10.0, 30.0
NESTED = (4, 8, 16, 24)
_Q = re.compile(r"^(?P<reg>.+)_q(?P<idx>\d{2})$")


def region_of(key: str) -> str:
    m = _Q.match(key)
    if not m:
        raise ValueError(f"q 접미 없는 키: {key}")
    return m.group("reg")


def terrain_from_static(path: Path, train_keys: dict[str, list[str]]) -> dict:
    """지형 특징은 **학습좌표에서만** 계산한다(평가좌표 사용 시 누수).

    near_t3   현장 최근접 Tier3 도로거리(km) — 학습 24좌표 중앙값
    n_reach30 인계 포함 30분 내 도달 가능한 (병원, 수단) 조합 수
    diversity 병원 거리벡터의 좌표 간 평균 상대편차(%) — 그 시군구가 만들 수 있는
              시나리오 다양성. 도심 ~1.5%, 농촌 5~36% (v18 실측)
    """
    z = np.load(path, allow_pickle=True)
    idx = {str(k): i for i, k in enumerate(z["keys"])}
    d_road, d_euc, tier, heli = z["d_road"], z["d_euc"], z["tier"], z["heli"]
    out = {}
    for reg, keys in train_keys.items():
        rows = [idx[k] for k in keys if k in idx]
        if not rows:
            continue
        D = d_road[rows]
        nt, nr = [], []
        for i in rows:
            t3 = tier[i] >= 3
            nt.append(float(d_road[i][t3].min()) if t3.any() else float("nan"))
            t_amb = d_road[i] / V_AMB * 60.0 + H_AMB
            t_uav = d_euc[i] / V_UAV * 60.0 + H_UAV
            nr.append(float((t_amb <= CUT).sum() + ((heli[i] > 0) & (t_uav <= CUT)).sum()))
        mu = D.mean(0)
        out[reg] = {"near_t3": float(np.nanmedian(nt)), "n_reach30": float(np.median(nr)),
                    "diversity_pct": float(100 * (np.abs(D - mu).mean(1) / max(mu.mean(), 1e-9)).mean()),
                    "n_train_coord": len(rows)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(SRC_MAN))
    ap.add_argument("--points", default=str(SRC_PTS))
    ap.add_argument("--static", default=str(STATIC), help="없으면 지형 특징 생략")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--n_train", type=int, default=24)
    ap.add_argument("--n_budget", type=int, default=3)
    ap.add_argument("--wave1", type=int, default=16, help="층화 표집할 파일럿 지역 수")
    a = ap.parse_args()

    man = json.load(open(a.manifest, encoding="utf-8"))
    pts = json.load(open(a.points, encoding="utf-8"))
    groups: dict[str, list[str]] = {}
    for k in man:
        groups.setdefault(region_of(k), []).append(k)
    bad = {r: len(v) for r, v in groups.items() if len(v) != 30}
    if bad:
        raise SystemExit(f"30점이 아닌 지역 {len(bad)}곳: {list(bad.items())[:5]} — 생성 미완료?")
    print(f"[split30] 지역 {len(groups)} × 30점 = {len(man)}")

    rng = np.random.default_rng(a.seed)
    outp = Path(a.out); outp.mkdir(parents=True, exist_ok=True)
    n_test = 30 - a.n_train - a.n_budget
    train_keys, index = {}, {}
    for reg in sorted(groups):
        ks = sorted(groups[reg])
        order = list(rng.permutation(ks))
        tr, bu, te = (order[:a.n_train],
                      order[a.n_train:a.n_train + a.n_budget],
                      order[a.n_train + a.n_budget:])
        train_keys[reg] = tr
        write = {f"train{a.n_train}": tr, "budget": bu, "test": te}
        for n in NESTED:                       # 중첩: 앞에서부터 자른다
            if n < a.n_train:
                write[f"train{n}"] = tr[:n]
        for tag, keys in write.items():
            (outp / f"{reg}.{tag}.json").write_text(
                json.dumps({k: man[k] for k in keys}, ensure_ascii=False, indent=1),
                encoding="utf-8")
        index[reg] = {"n_train": len(tr), "n_budget": len(bu), "n_test": len(te),
                      "snap_m_median": float(np.median([pts[k]["snap_m"] for k in ks])),
                      "snap_m_max": float(max(pts[k]["snap_m"] for k in ks)),
                      "sigcd": pts[ks[0]]["sigcd"], "sido": pts[ks[0]]["sido"],
                      "manifests": {t: str((outp / f"{reg}.{t}.json").resolve())
                                    for t in write}}

    sp = Path(a.static)
    if sp.exists():
        feat = terrain_from_static(sp, train_keys)
        for r, v in feat.items():
            index.get(r, {}).update(v)
        vals = {k: np.array([index[r][k] for r in index if k in index[r]])
                for k in ("near_t3", "n_reach30", "diversity_pct")}
        for k in ("near_t3", "n_reach30"):
            q = np.nanquantile(vals[k], [0.25, 0.5, 0.75])
            for r in index:
                if k in index[r]:
                    index[r][f"q_{k}"] = int(np.searchsorted(q, index[r][k], side="right"))
        for r in index:
            if "q_near_t3" in index[r]:
                index[r]["stratum"] = index[r]["q_near_t3"] * 4 + index[r]["q_n_reach30"]
        print(f"  지형: near_t3 중위 {np.nanmedian(vals['near_t3']):.1f}km · "
              f"n_reach30 중위 {np.median(vals['n_reach30']):.0f} · "
              f"다양성 중위 {np.median(vals['diversity_pct']):.2f}%")
    else:
        print(f"  ⚠ 정적표 없음({sp}) — 지형 특징·층화 생략. "
              f"`v17_field_rules.py static --manifest {a.manifest}` 로 만들 것")

    # Wave1 층화 표집 (층이 있으면 층별 라운드로빈, 없으면 무작위)
    wave1 = []
    if any("stratum" in v for v in index.values()):
        buckets: dict[int, list[str]] = {}
        for r in sorted(index):
            buckets.setdefault(index[r].get("stratum", -1), []).append(r)
        rr = np.random.default_rng(a.seed)
        for s in buckets:
            buckets[s] = list(rr.permutation(buckets[s]))
        i = 0
        while len(wave1) < a.wave1:
            moved = False
            for s in sorted(buckets):
                if i < len(buckets[s]) and len(wave1) < a.wave1:
                    wave1.append(buckets[s][i]); moved = True
            if not moved:
                break
            i += 1
        wave1.sort()
    for r in index:
        index[r]["wave"] = 1 if r in wave1 else 2

    meta = {"generated": "split_sigungu30.py", "source_manifest": str(Path(a.manifest).resolve()),
            "split": {"train": a.n_train, "budget": a.n_budget, "test": n_test},
            "nested_train_subsets": [n for n in NESTED if n <= a.n_train],
            "seed": a.seed, "wave1": wave1,
            "roles": {
                "train": "모델 학습",
                "budget": "스텝 예산 결정 전용 내부검증 — 평가좌표로 예산을 고르면 누수",
                "test": "지역 내부 평가(예산 선택에 미사용). 250×3=750점",
                "canonical_eval": "대표점250 · 외부250 — 기존 모든 기준선이 측정된 정본 축. "
                                  "신규 30점 풀과 실측 최소 이격 1.001km"},
            "regions": index}
    (outp / "_index.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[split30] 매니페스트 {len(list(outp.glob('*.json')))}개 → {outp}")
    print(f"[wave1] {len(wave1)}곳: {', '.join(wave1[:8])}{' …' if len(wave1)>8 else ''}")


if __name__ == "__main__":
    main()
