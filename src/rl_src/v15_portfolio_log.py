# -*- coding: utf-8 -*-
"""선정된 v15 포트폴리오 정책의 결정별 설명 로그 수집.

분할 적합 모델(p0~p2)을 p3 좌표에서 실행해 GBDT 기준행동, PPO 행동,
최종 실행행동과 후보별 NCRP 가치를 함께 저장한다. LB-T는 사용하지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/rl_src"))

from tree_distill_policy import FEATURE_NAMES
from v15_portfolio_eval import (
    DEFAULT_CASES,
    MODEL_DIR,
    TRAIN_MANIFEST,
    _milp_proposer,
    _select_entries,
    _tree_fn,
)

DEFAULT_TREE_DIR = REPO / "results/scoreboard/v13/sota_distill/students_split750"
DEFAULT_OUT = REPO / "results/scoreboard/v15/explanation/portfolio_p3_decisions.csv"
BASE_COLS = [
    "region", "episode", "seed", "decision", "role", "action", "class", "destination",
    "mode", "source", "q_pdr", "switched", "dpdr", "planner_n_cand", "episode_pdr_woG",
]
COLS = BASE_COLS + FEATURE_NAMES


def _decode(action: int) -> tuple[int, int, int]:
    a = int(action)
    return a // 96, (a % 96) // 2, a % 2


def _worker(job: dict) -> dict:
    try:
        import torch as th

        th.set_num_threads(1)
        os.environ.update(
            MCI_CAP_GATE="occ", MCI_OBS_VARIANT="essential+load+valid",
            MCI_H_PAD="47", MCI_REWARD_MODE="woG",
        )
        from sb3_contrib import MaskablePPO
        from hospital_set_extractor import HospitalSetExtractor  # noqa: F401
        from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401
        from planner_policy import TruncatedRolloutPlanner
        from portfolio_policy import CompositeCandidateProposer, TreeCandidateProposer
        from tree_distill_policy import ActionFeatureBuilder, load_tree_package
        from viper_distill import _suppress_stdout, load_vecnorm, make_feature_env

        model = MaskablePPO.load(str(Path(job["model_dir"]) / "final_model.zip"), device="cpu")
        norm = load_vecnorm(str(Path(job["model_dir"]) / "vecnormalize.pkl"))
        packages = {name: load_tree_package(path) for name, path in job["tree_paths"].items()}
        tree_extra = TreeCandidateProposer(
            {x: packages[x] for x in ("G1", "G2", "G3")}, h_pad=47
        )
        extra = CompositeCandidateProposer([("MILP", _milp_proposer()), ("TREE", tree_extra)])
        planner = TruncatedRolloutPlanner(
            model, K=job["K"], h=job["h"], m=job["m"], leaf_fn=None, clairvoyant=False,
            reseed_base=777000, switch_margin=0.0, alloc="uniform", switch_z=0.0,
            extra_cand_fn=extra.propose, greedy_action_fn=_tree_fn(packages["G1"]),
        )
        factory = make_feature_env(job["cfg"], norm)
        builder = ActionFeatureBuilder(h_pad=47)
        rows = []
        with _suppress_stdout():
            for ep in job["episodes"]:
                seed = job["seed0"] + ep
                env = factory(seed=seed)
                obs, _ = env.reset(seed=seed)
                done, reward, dec = False, 0.0, 0
                episode_rows = []
                while not done:
                    mask = np.asarray(env.action_masks(), dtype=bool)
                    actions, X = builder.build(env.unwrapped, mask)
                    index = {int(a): i for i, a in enumerate(actions)}
                    extra.last_sources = {}
                    action = int(planner.act(env, ep_seed=seed, obs=obs))
                    info = planner.last_info
                    ppo = int(info.get("ppo_greedy_action", action))
                    base = int(info.get("greedy_action", action))
                    cand = tuple(int(x) for x in info.get("candidate_actions", (action,)))
                    q = tuple(float(x) for x in info.get("candidate_q_pdr", (np.nan,)))
                    qmap = dict(zip(cand, q))
                    ppo_top = set(map(int, info.get("ppo_candidate_actions", ())))
                    roles = [("GBDT_BASE", base), ("PPO_GREEDY", ppo), ("EXEC", action)]
                    for role, a in roles:
                        i = index[a]
                        c, d, m = _decode(a)
                        labels = list(extra.last_sources.get(a, ()))
                        if a in ppo_top:
                            labels.append("PPO_TOPK")
                        if a == base:
                            labels.append("G1_BASE")
                        rec = {
                            "region": job["region"], "episode": ep, "seed": seed,
                            "decision": dec, "role": role, "action": a,
                            "class": c, "destination": d, "mode": m,
                            "source": "+".join(dict.fromkeys(labels)) or "EXEC_ONLY",
                            "q_pdr": qmap.get(a, np.nan),
                            "switched": int(bool(info.get("switched", False))),
                            "dpdr": float(info.get("dpdr") or 0.0),
                            "planner_n_cand": int(info.get("n_cand", 0)),
                            "episode_pdr_woG": np.nan,
                        }
                        rec.update({name: float(X[i, j]) for j, name in enumerate(FEATURE_NAMES)})
                        episode_rows.append(rec)
                    obs, _, term, trunc, step_info = env.step(action)
                    reward += float(step_info.get("r_woG", 0.0))
                    done = term or trunc
                    dec += 1
                prev = float(env.unwrapped.preventable_woG)
                pdr = 1.0 - reward / prev if prev > 0 else 0.0
                for rec in episode_rows:
                    rec["episode_pdr_woG"] = pdr
                rows.extend(episode_rows)
        return {"ok": True, "region": job["region"], "rows": rows}
    except Exception as exc:
        import traceback
        return {"ok": False, "region": job.get("region", "?"),
                "err": (str(exc) + "\n" + traceback.format_exc())[:5000]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(TRAIN_MANIFEST))
    p.add_argument("--fold", default="p3")
    p.add_argument("--n_regions", type=int, default=250)
    p.add_argument("--n_eps", type=int, default=1)
    p.add_argument("--seed0", type=int, default=9200)
    p.add_argument("--model_dir", default=str(MODEL_DIR))
    p.add_argument("--tree_dir", default=str(DEFAULT_TREE_DIR))
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--h", type=int, default=20)
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    entries = _select_entries(Path(args.manifest).resolve(), args.fold, args.n_regions, set())
    tree_paths = {
        name: str(Path(args.tree_dir).resolve() / f"{case}.pkl")
        for name, case in DEFAULT_CASES.items() if name != "P1"
    }
    missing = [x for x in tree_paths.values() if not Path(x).exists()]
    if missing:
        raise FileNotFoundError(missing)
    out = Path(args.out).resolve()
    meta_path = Path(str(out) + ".meta.json")
    if meta_path.exists():
        raise FileExistsError(f"완료 설명 로그 보호: {meta_path}")
    out.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    complete: set[tuple[str, int]] = set()
    existing_rows = 0
    if out.exists():
        old = pd.read_csv(out)
        if list(old.columns) != COLS:
            raise ValueError("재개 설명 로그 스키마 불일치")
        if old.duplicated(["region", "seed", "decision", "role"]).any():
            raise ValueError("재개 설명 로그 복합키 중복")
        for (region, seed), g in old.groupby(["region", "seed"]):
            role_count = g.groupby("decision").role.nunique()
            decisions = sorted(map(int, g.decision.unique()))
            if (role_count == 3).all() and decisions == list(range(max(decisions) + 1)) \
                    and g.episode_pdr_woG.nunique() == 1:
                complete.add((str(region), int(seed)))
            else:
                raise ValueError(f"재개 설명 로그에 불완전 episode: {(region, seed)}")
        existing_rows = len(old)
    jobs = []
    for region, cfg in entries:
        remaining = [ep for ep in range(args.n_eps) if (region, args.seed0 + ep) not in complete]
        if remaining:
            jobs.append({
                **vars(args), "region": region, "cfg": cfg,
                "episodes": remaining, "tree_paths": tree_paths,
            })
    t0, n_rows = time.time(), existing_rows
    new_file = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        if new_file:
            writer.writeheader(); f.flush()
        with Pool(min(args.workers, max(len(jobs), 1)), maxtasksperchild=1) as pool:
            for i, result in enumerate(pool.imap_unordered(_worker, jobs), 1):
                if not result["ok"]:
                    raise RuntimeError(f"{result['region']} 실패: {result['err']}")
                writer.writerows(result["rows"]); f.flush()
                n_rows += len(result["rows"])
                print(f"[{i}/{len(jobs)}] {result['region']} rows={len(result['rows'])} wall={time.time()-t0:.0f}s", flush=True)
    # 한 결정에 세 역할 행이 정확히 존재해야 설명 비교가 가능하다.
    d = pd.read_csv(out)
    key = ["region", "seed", "decision", "role"]
    if d.duplicated(key).any() or len(d) != n_rows:
        raise RuntimeError("결정 로그 복합키/행수 불일치")
    counts = d.groupby(["region", "seed", "decision"]).role.nunique()
    if not (counts == 3).all() or d.isna().drop(columns=["q_pdr"]).any().any():
        raise RuntimeError("결정별 역할 3행 또는 유한값 검증 실패")
    cells = set(zip(d.region.astype(str), d.seed.astype(int)))
    expected_cells = {(region, args.seed0 + ep) for region, _ in entries for ep in range(args.n_eps)}
    if cells != expected_cells:
        raise RuntimeError(f"설명 로그 episode 완전격자 불일치 {len(cells)} != {len(expected_cells)}")
    meta = {
        "schema_version": 1, "policy": "V15_BASE_G1", "rows": len(d),
        "regions": len(entries), "n_eps": args.n_eps, "seed_start": args.seed0,
        "roles": ["GBDT_BASE", "PPO_GREEDY", "EXEC"],
        "fit_folds": ["p0", "p1", "p2"], "explanation_fold": args.fold,
        "lb_t_included": False, "wall_seconds": time.time() - t0,
        "planner": {"K": args.K, "h": args.h, "m": args.m},
        "resumed_episode_cells": len(complete),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"완료 → {out}")


if __name__ == "__main__":
    main()
