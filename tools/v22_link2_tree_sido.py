# -*- coding: utf-8 -*-
"""링크 2 보강 — 전국 교사와 광역시도 17 교사의 **트리 gain 중요도**를 같은 특징공간에서 비교.

왜 따로 만드는가
  `tools/v22_link2_importance.py` 의 `sido` 하위명령은 **조건부 로짓 표준화 계수**만 낸다.
  전국 쪽은 `stage_tree` 로 GBDT gain 을 내지만 그 적합은 **파생 포함 56 특징** 위에서
  이뤄져, 시도 로그(원본 43 특징만 보유)와 gain 배분을 직접 비교할 수 없다.
  발표용 그림은 "같은 자를 대고 잰 값"이어야 하므로, **원본 43 특징 공간에서 전국과 17 시도를
  전부 다시 적합**한다.

계약
  * 모델·하이퍼는 `stage_tree` 와 동일(LGBMRegressor G31, v17_feature_distill 설정).
  * 타깃은 교사 softmax 확률, 홀드아웃은 **시군구 단위 5분할 중 1분할**(누수 방지).
  * ⚠️ 원본 43 에는 카드의 부하항(`hingerate_capa`)·수술실수(`max_capa`) 같은 **파생 특징이
    없다.** 이 그림은 "카드의 두 항이 뽑혔다"의 근거가 아니라 **"교사가 보는 물리량 계열이
    지역을 넘어 같은가"** 의 근거다. 두 주장을 섞지 않는다.
  * ⚠️ 캐시가 원본 row weight 를 보관하지 않아 균등 가중으로 적합한다(gain 배분에만 영향).
  * 시뮬 재실행 0회.

사용:
    python tools/v22_link2_tree_sido.py --m 160
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src" / "rl_src"))

import v22_link2_importance as L2          # noqa: E402  스트리밍·표본 규약 재사용
from v17_field_rules import decode         # noqa: E402

OUT = REPO / "results/field/link2"
GAIN_KEYS = ("gain",)


def _new_model(jobs):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(objective="regression_l2", num_leaves=31, learning_rate=0.04,
                         n_estimators=600, min_child_samples=40, subsample=0.85,
                         subsample_freq=1, colsample_bytree=0.90, reg_lambda=1.0,
                         random_state=0, n_jobs=jobs, verbosity=-1,
                         deterministic=True, force_col_wise=True)


def _fit_tree(X, y, dist_code, gsz, chosen, jobs=8):
    """gain 은 전량 적합에서, 재현율은 시군구 5개 이상일 때만 홀드아웃에서 낸다."""
    m = _new_model(jobs)
    m.fit(X, y)
    gain = np.asarray(m.booster_.feature_importance("gain"), float)
    gain = gain / max(gain.sum(), 1e-12)

    nd = int(dist_code.max()) + 1
    if nd < 5:
        return gain, None, 0

    rng = np.random.default_rng(L2.SEED)
    test_d = set(rng.permutation(nd)[: max(1, nd // 5)].tolist())
    is_test = np.asarray([c in test_d for c in dist_code])
    rmask = np.repeat(is_test, gsz)
    mh = _new_model(jobs)
    mh.fit(X[~rmask], y[~rmask])
    pred = mh.predict(X[rmask])
    gte, cte = gsz[is_test], chosen[rmask]
    offs = np.concatenate([[0], np.cumsum(gte)])
    hit = sum(int(cte[int(offs[k]) + int(np.argmax(pred[int(offs[k]):int(offs[k + 1])]))])
              for k in range(len(gte)))
    return gain, hit / max(len(gte), 1), int(len(gte))


def _sido_arrays(dec: Path, m: int, seed: int):
    """_sido_sample 과 같은 표본 규약 + target·시군구 코드까지 함께 돌려준다."""
    sk, off, teach = L2._lazy(dec, "state_key", "offsets", "teacher_action")
    sk = np.asarray([str(x) for x in sk])
    sig = np.asarray([k.rsplit("_", 2)[-2] for k in sk])
    dmap = {int(a): decode(int(a))[1] for a in np.unique(teach)}
    td = np.asarray([dmap[int(a)] for a in teach])
    rng = np.random.default_rng(seed)
    picked = []
    for c in np.unique(sig):
        idx = np.flatnonzero(sig == c)
        rng.shuffle(idx)
        picked.append(np.asarray([int(s) for s in idx if td[s] != 0][:m], int))
    states = np.sort(np.concatenate([p for p in picked if p.size]))
    keep = np.zeros(int(off[-1]), bool)
    for s in states:
        keep[int(off[s]):int(off[s + 1])] = True

    X = L2._stream(dec, "X.npy", keep).astype(np.float64)
    chosen = L2._stream(dec, "chosen.npy", keep)
    target = L2._stream(dec, "target.npy", keep).astype(np.float64)
    cact = L2._stream(dec, "cand_action.npy", keep)

    sizes = np.asarray([int(off[s + 1]) - int(off[s]) for s in states], int)
    row_state = np.repeat(np.arange(len(states)), sizes)
    dm = {int(a): decode(int(a)) for a in np.unique(cact)}
    dst = np.asarray([dm[int(a)][1] for a in cact], np.int16)
    disp = dst > 0
    X, chosen, target, row_state = X[disp], chosen[disp], target[disp], row_state[disp]

    gsz = np.bincount(row_state, minlength=len(states))
    ok = gsz >= 2
    if not ok.all():
        keeprow = ok[row_state]
        X, chosen, target = X[keeprow], chosen[keeprow], target[keeprow]
        row_state = row_state[keeprow]
        gsz = gsz[ok]
        states = states[ok]
    sig_sel = sig[states]
    _, dist_code = np.unique(sig_sel, return_inverse=True)
    return X, chosen, target, gsz, dist_code.astype(int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=160, help="시군구별 상태 표본 수")
    ap.add_argument("--seed", type=int, default=L2.SEED)
    ap.add_argument("--lgbm_jobs", type=int, default=8)
    args = ap.parse_args()

    base = [str(x) for x in L2._lazy(L2.DEC_NATIONAL, "feature_names")[0]]
    res = {"date": time.strftime("%Y-%m-%d %H:%M"),
           "tool": "tools/v22_link2_tree_sido.py",
           "space": "원본 43 특징 (파생 제외 — 시도 로그와 동일 공간)",
           "model": "LGBMRegressor G31 (stage_tree 와 동일 설정)",
           "split": "시군구 단위 5분할 중 1분할 홀드아웃",
           "weight_note": "캐시가 원본 row weight 미보유 — 균등 가중(gain 배분에만 영향)",
           "m_per_district": args.m, "seed": args.seed,
           "features": base, "teachers": {}}

    # 전국 — 기존 캐시를 43 열로 잘라 같은 공간에서 재적합
    t0 = time.time()
    d = L2.load_cache()
    cols = [d["names"].index(n) for n in base]
    g, rc, nte = _fit_tree(np.ascontiguousarray(d["X"][:, cols]), d["target"],
                           d["dist_code"], d["gsz"], d["chosen"], args.lgbm_jobs)
    res["teachers"]["전국"] = {"gain": g.tolist(), "holdout_recall": rc,
                               "n_sets": int(len(d["gsz"])), "n_test_sets": nte,
                               "wall_sec": round(time.time() - t0, 1)}
    print(f"  [전국] 집합 {len(d['gsz'])} recall={rc:.4f} ({time.time()-t0:.0f}s)", flush=True)
    del d

    for p in sorted(L2.CARDS.glob("dec_sido_*.npz")):
        name = p.name[len("dec_sido_"):-len(".npz")]
        t0 = time.time()
        X, chosen, target, gsz, dcode = _sido_arrays(p, args.m, args.seed)
        nd = int(dcode.max()) + 1
        g, rc, nte = _fit_tree(X, target, dcode, gsz, chosen, args.lgbm_jobs)
        res["teachers"][name] = {"gain": g.tolist(), "holdout_recall": rc,
                                 "n_sets": int(len(gsz)), "n_test_sets": nte,
                                 "n_districts": nd,
                                 "wall_sec": round(time.time() - t0, 1)}
        rs = "n/a" if rc is None else f"{rc:.4f}"
        print(f"  [{name}] 집합 {len(gsz)} 시군구 {nd} recall={rs} "
              f"({time.time()-t0:.0f}s)", flush=True)

    sido_ok = [k for k in res["teachers"] if k != "전국"]
    G = np.array([res["teachers"][k]["gain"] for k in sido_ok])
    nat = np.array(res["teachers"]["전국"]["gain"])
    order = np.argsort(-nat)
    res["summary"] = {
        "n_sido_fitted": len(sido_ok),
        "small_teachers": {k: res["teachers"][k]["n_districts"] for k in sido_ok
                           if res["teachers"][k]["n_districts"] < 5},
        "national_top10": [[base[i], float(nat[i])] for i in order[:10]],
        "sido_median_top10": [[base[i], float(np.median(G[:, i]))] for i in order[:10]],
        "top5_overlap_mean": float(np.mean([
            len(set(np.argsort(-G[r])[:5].tolist()) & set(order[:5].tolist()))
            for r in range(len(sido_ok))])),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "link2_tree_sido.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
