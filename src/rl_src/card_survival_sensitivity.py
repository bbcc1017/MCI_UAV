# -*- coding: utf-8 -*-
"""생존곡선 상한 민감도 — "황색 먼저"가 모델 파라미터의 부산물인지 검정한다.

현행 생존함수(``MCIEnvironment_gymnasium.getSurvProb``)는 상한이 적색 0.56 / 황색 0.81
이다. 카드 재시뮬에서 "황색 먼저"가 "적색 먼저"보다 우세했는데, 그 결론이 **상한 비율**
때문일 가능성이 있다(완벽히 처치해도 적색은 0.56밖에 못 살린다).

시뮬 코어는 수정하지 않는다. 워커 안에서 ``getSurvProb`` 만 런타임 몽키패치해 상한을
바꾸고, 등급 규칙만 다른 카드 세 종을 같은 좌표·시드로 재시뮬한다. ``preventable_woG``
도 같은 함수로 계산되므로 PDR 정규화는 각 셀 안에서 정합적으로 유지된다.

판정: 적색 상한을 올릴수록 "적색 먼저"가 "황색 먼저"를 언제 역전하는가.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "sim_src"))

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json"
DEV40 = REPO / "scoreboard/v11_ncrp_dev40_regions.txt"
CARD_DIR = REPO / "results/scoreboard/v16/card_policies"

# (red_ceiling, yellow_ceiling, 라벨)
CELLS = [
    (0.56, 0.81, "기준(현행)"),
    (0.65, 0.81, "적색상한↑ 0.65"),
    (0.75, 0.81, "적색상한↑ 0.75"),
    (0.85, 0.81, "적색상한↑ 0.85"),
    (0.95, 0.81, "적색상한↑ 0.95"),
    (0.56, 0.65, "황색상한↓ 0.65"),
    (0.56, 0.56, "상한 동일 0.56"),
]
CARDS = ["CARD_Y_T3_U6", "CARD_G_T3_U6", "CARD_R_T3_U6"]


def patch_surv(red_ceil: float, yellow_ceil: float) -> None:
    """생존함수 상한만 교체. 감쇠 지수/시정수는 원본 유지."""
    import MCIEnvironment_gymnasium as M

    def getSurvProb(self, t, p_class):
        if p_class == 0:
            return red_ceil / (math.pow((t / 91), 1.58) + 1)
        if p_class == 1:
            return yellow_ceil / (math.pow((t / 160), 2.41) + 1)
        if p_class == 2:
            return 1.0
        return 0.0

    M.MCIEnvironment_gym.getSurvProb = getSurvProb


def rollout(factory, policy, seed):
    env = factory(seed=seed)
    obs, _ = env.reset(seed=seed)
    done, reward, n = False, 0.0, 0
    while not done:
        mask = env.action_masks()
        action = policy(obs, mask, env.unwrapped)
        obs, _, term, trunc, info = env.step(action)
        reward += info.get("r_woG", 0.0)
        n += 1
        done = term or trunc
    prev = env.unwrapped.preventable_woG
    return reward, (1.0 - reward / prev if prev > 0 else float("nan")), n


def worker(job):
    region, cfg, red_ceil, yellow_ceil, label, n_eps, seed0 = job
    try:
        import torch as th

        th.set_num_threads(1)
        os.environ.update(MCI_CAP_GATE="occ", MCI_OBS_VARIANT="essential+load+valid",
                          MCI_H_PAD="47", MCI_REWARD_MODE="woG")
        from tree_distill_policy import load_tree_package, make_rank_tree_policy
        from viper_distill import _suppress_stdout, make_feature_env

        patch_surv(red_ceil, yellow_ceil)  # env 생성 전에 패치
        pkgs = [load_tree_package(str(CARD_DIR / f"{c}.pkl")) for c in CARDS]
        pols = [make_rank_tree_policy(p, h_pad=47) for p in pkgs]
        factory = make_feature_env(cfg, None)
        rows = []
        with _suppress_stdout():
            for ep in range(n_eps):
                seed = seed0 + ep
                for name, pol in zip(CARDS, pols):
                    r, pdr, n = rollout(factory, pol, seed)
                    rows.append({"region": region, "red_ceil": red_ceil,
                                 "yellow_ceil": yellow_ceil, "cell": label,
                                 "policy": name, "episode": ep, "seed": seed,
                                 "reward_woG": r, "pdr_woG": pdr, "n_decisions": n})
        return {"ok": True, "rows": rows, "region": region, "cell": label}
    except Exception as exc:
        import traceback
        return {"ok": False, "region": region, "cell": label,
                "err": (str(exc) + traceback.format_exc())[:1200]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_eps", type=int, default=5)
    p.add_argument("--seed0", type=int, default=8100)
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    regions = [x.strip() for x in open(DEV40, encoding="utf-8")
               if x.strip() and not x.startswith("#")]
    regions = [r for r in regions if r in manifest]
    jobs = [(r, manifest[r], rc, yc, lab, args.n_eps, args.seed0)
            for rc, yc, lab in CELLS for r in regions]
    print(f"[surv-sens] 셀 {len(CELLS)} × 지역 {len(regions)} × 카드 {len(CARDS)} "
          f"× {args.n_eps}ep = {len(jobs) * len(CARDS) * args.n_eps} 에피소드", flush=True)

    rows, bad, t0 = [], [], time.time()
    with Pool(processes=min(args.workers, len(jobs)), maxtasksperchild=1) as pool:
        for k, res in enumerate(pool.imap_unordered(worker, jobs), 1):
            if res["ok"]:
                rows.extend(res["rows"])
            else:
                bad.append(res)
                print(f"  !! {res['region']} {res['cell']}: {res['err'][:160]}", flush=True)
            if k % 40 == 0 or k == len(jobs):
                print(f"  [{k}/{len(jobs)}] rows={len(rows)} wall={time.time()-t0:.0f}s", flush=True)

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[surv-sens] 완료 rows={len(rows)} 실패={len(bad)} → {out}")


if __name__ == "__main__":
    main()
