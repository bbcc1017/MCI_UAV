# -*- coding: utf-8 -*-
"""Wave 1 스텝 예산 프로브 — 지역모델의 체크포인트를 그 지역의 **보류 좌표 p3** 에서 평가 (v18 E5).

지역모델은 3~4 좌표만 보므로 전국모델(1,000좌표 10M)의 스텝 예산을 그대로 쓸 이유가 없다.
좌표당 스텝이 250배가 되어 과적합이 기본값이다. 어디서 꺾이는지는 **학습에 안 쓴 좌표**
에서만 잴 수 있고, 평가좌표(대표점)로 예산을 고르면 누수다 → Wave 1 은 p0~p2 로 학습하고
p3 로 예산을 정한다.

⚠️ 정규화 통계 주의: `CheckpointCallback` 은 기본적으로 VecNormalize 를 저장하지 않는다
(Wave 1 이 그 상태로 돌았다). 체크포인트 옆에 `..._vecnormalize_*.pkl` 이 있으면 그것을
쓰고, 없으면 런의 최종 `vecnormalize.pkl` 을 **모든 체크포인트에 균일 적용**한다.
같은 통계를 전 체크포인트에 쓰므로 체크포인트 간 비교는 등가이고, 아주 이른 체크포인트의
절대수준만 근사다. 이후 런은 `--save_vecnormalize` 로 이 근사를 없앤다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(__file__))

REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "scenarios/manifests/sigungu250/_index.json"
TRAIN = REPO / "scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json"
_CK = re.compile(r"ppo_feature_(\d+)_steps\.zip$")
COLS = ["region", "steps", "episode", "seed", "pdr_woG", "reward_woG", "n_decisions"]


def worker(job):
    region, cfg, ckpt, steps, vecnorm, n_eps, seed0, obs_variant = job
    try:
        import numpy as np
        import torch as th

        th.set_num_threads(1)
        os.environ.update(MCI_CAP_GATE="occ", MCI_OBS_VARIANT=obs_variant,
                          MCI_H_PAD="47", MCI_REWARD_MODE="woG")
        from sb3_contrib import MaskablePPO
        from hospital_set_extractor import HospitalSetExtractor  # noqa: F401
        from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401
        import pad_vecnorm  # noqa: F401
        from evaluate import ppo_policy
        from viper_distill import _suppress_stdout, load_vecnorm, make_feature_env
        from v17_ppo_eval import rollout

        model = MaskablePPO.load(ckpt, device="cpu")
        norm = load_vecnorm(vecnorm) if vecnorm else None
        policy = ppo_policy(model)
        factory = make_feature_env(cfg, norm)
        rows = []
        with _suppress_stdout():
            for ep in range(n_eps):
                seed = seed0 + ep
                reward, pdr, _, n_dec, _ = rollout(factory, policy, seed)
                rows.append({"region": region, "steps": steps, "episode": ep, "seed": seed,
                             "pdr_woG": pdr, "reward_woG": reward, "n_decisions": n_dec})
        return {"ok": True, "key": f"{region}@{steps}", "rows": rows}
    except Exception as exc:
        import traceback
        return {"ok": False, "key": f"{region}@{steps}",
                "err": (str(exc) + traceback.format_exc())[:1200]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_root", default=str(REPO / "results/rl/sigungu250/wave1"))
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--train_manifest", default=str(TRAIN))
    ap.add_argument("--holdout", default="p3")
    ap.add_argument("--obs_variant", default="essential+load+valid")
    ap.add_argument("--n_eps", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--every", type=int, default=1, help="체크포인트 N개마다 1개만 평가")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    idx = json.load(open(a.index, encoding="utf-8"))["regions"]
    man = json.load(open(a.train_manifest, encoding="utf-8"))
    root = Path(a.model_root).resolve()

    jobs, skipped = [], []
    for region in sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "run_logs"):
        hold_key = f"{region}_{a.holdout}"
        if hold_key not in man:
            skipped.append(region); continue
        cks = sorted(((int(_CK.search(p.name).group(1)), p)
                      for p in (root / region / "checkpoints").glob("ppo_feature_*_steps.zip")),
                     key=lambda x: x[0])
        if not cks:
            skipped.append(region); continue
        final_vn = root / region / "vecnormalize.pkl"
        for i, (steps, ck) in enumerate(cks):
            if i % a.every:
                continue
            per_ck = list((root / region / "checkpoints").glob(
                f"ppo_feature_vecnormalize_{steps // 1}_steps.pkl"))
            vn = str(per_ck[0]) if per_ck else (str(final_vn) if final_vn.exists() else "")
            jobs.append((region, man[hold_key], str(ck), steps, vn,
                         a.n_eps, a.seed0, a.obs_variant))
        # 최종 모델도 한 점으로 넣는다
        fm = root / region / "final_model.zip"
        if fm.exists():
            jobs.append((region, man[hold_key], str(fm), -1,
                         str(final_vn) if final_vn.exists() else "",
                         a.n_eps, a.seed0, a.obs_variant))

    if skipped:
        print(f"[probe] 체크포인트/보류좌표 없어 제외: {len(skipped)}개 {skipped[:3]}")
    if not jobs:
        raise SystemExit("평가할 (지역, 체크포인트) 쌍이 없다")
    print(f"[probe] 잡 {len(jobs)} · 보류좌표 {a.holdout} · n_eps={a.n_eps} "
          f"seed={a.seed0}..{a.seed0+a.n_eps-1} workers={a.workers}", flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    t0, n_ok = time.time(), 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=COLS); wr.writeheader()
        with Pool(min(a.workers, len(jobs)), maxtasksperchild=1) as pool:
            for res in pool.imap_unordered(worker, jobs):
                if not res["ok"]:
                    print(f"  x {res['key']}: {res['err'][:160]}"); continue
                wr.writerows(res["rows"]); f.flush(); n_ok += 1
                if n_ok % 20 == 0:
                    print(f"  [{n_ok}/{len(jobs)}] wall={time.time()-t0:.0f}s", flush=True)
    print(f"[probe] 완료 {n_ok}/{len(jobs)} · {(time.time()-t0)/60:.1f}분 → {out}")


if __name__ == "__main__":
    main()
