# -*- coding: utf-8 -*-
"""v15 PPO·MILP·증류정책 후보 포트폴리오의 paired 폐루프 평가.

정책 점수를 평균하지 않는다. 각 정책이 제안한 서로 다른 행동을 NCRP 후보집합에 넣고
동일 비천리안 CRN 미래로 직접 채점한다. LB-T는 독립 기준선이므로 이 파일에서 불러오지 않는다.

튜닝: random4의 지정 fold에서 sigungu code 순 균등간격 지역을 선택한다.
최종: 대표점250 manifest를 별도 명시하여 같은 seed 0..29로 실행한다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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

TRAIN_MANIFEST = REPO / "scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json"
MODEL_DIR = REPO / "results/rl/redesign/v10_random4_1000_pointer_s0"
TREE_DIR = REPO / "results/scoreboard/v13/sota_distill/students_full1000"
PPO_DISTILL_TREE = REPO / "results/scoreboard/v10/distill/students_parallel/I1_FIELD_GBDT_L31_SOFT.pkl"
DEFAULT_CASES = {
    "G1": "I3_CONNECTED_GBDT_L63_BASE",
    "G2": "I3_CONNECTED_GBDT_L31_BASE",
    "G3": "I1_FIELD_GBDT_L63_BASE",
    "E1": "I1_FIELD_EBM_I04",
    "C1": "I3_CONNECTED_CART_L384",
    "P1": "__PPO_DISTILL_I1_GBDT_L31_SOFT__",
}
DEFAULT_ARMS = (
    "PPO,PURE_G1,FINAL,ADD_G1,ADD_G3,ADD_FAMILY,BASE_G1,BASEFOLLOW_G1,"
    "ADD_P1,ADD_G1_P1,BASE_G1_P1,BASE_G1_CORE,BASE_G1_NOMILP"
)
COLS = [
    "region", "policy", "episode", "seed", "pdr_woG", "reward_woG",
    "n_decisions", "n_switch", "n_tree_offered", "n_tree_exec",
    "n_novel_tree_exec", "n_milp_exec", "ms_per_decision", "wall_seconds",
]


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _select_entries(path: Path, fold: str, k: int, regions: set[str]) -> list[tuple[str, str]]:
    manifest = json.load(open(path, encoding="utf-8"))
    items = list(manifest.items())
    if fold:
        items = [(key, cfg) for key, cfg in items if key.endswith(f"_{fold}")]
    if regions:
        items = [(key, cfg) for key, cfg in items if key in regions]
    def sigcd(item):
        toks = item[0].rsplit("_", 2)
        return toks[-2] if len(toks) == 3 and toks[-1].startswith("p") else toks[-1]
    items.sort(key=sigcd)
    if k and k < len(items):
        idx = np.unique(np.round(np.linspace(0, len(items) - 1, k)).astype(int))
        items = [items[i] for i in idx]
    return items


def _milp_proposer():
    from milp_policy import MilpProposer
    return MilpProposer(
        h_pad=47, n_propose=2, n_opp=3, topk_hosp=0,
        second_wave=False, future_patients=False, n_future_groups=2,
        force_dispatch=False, queue_model="fluid",
    )


def _tree_fn(package: dict):
    from portfolio_policy import TreeCandidateProposer
    prop = TreeCandidateProposer({"G1": package}, h_pad=47)
    return prop.action_fn


def _run_pure(factory, policy_fn, seed: int) -> dict:
    env = factory(seed=seed)
    obs, _ = env.reset(seed=seed)
    done, reward, n_dec, sec = False, 0.0, 0, 0.0
    while not done:
        mask = env.action_masks()
        t0 = time.perf_counter()
        action = int(policy_fn(obs, mask, env.unwrapped))
        sec += time.perf_counter() - t0
        obs, _, term, trunc, info = env.step(action)
        reward += float(info.get("r_woG", 0.0))
        n_dec += 1
        done = term or trunc
    prev = float(env.unwrapped.preventable_woG)
    return {
        "pdr_woG": 1.0 - reward / prev if prev > 0 else 0.0,
        "reward_woG": reward, "n_decisions": n_dec, "n_switch": 0,
        "n_tree_offered": 0, "n_tree_exec": 0, "n_novel_tree_exec": 0,
        "n_milp_exec": 0, "ms_per_decision": sec * 1000 / max(n_dec, 1),
    }


def _run_planner(factory, model, packages: dict[str, dict], arm: str, seed: int,
                 K: int, h: int, m: int, reseed_base: int) -> dict:
    from planner_policy import TruncatedRolloutPlanner
    from portfolio_policy import CompositeCandidateProposer, TreeCandidateProposer

    tree_names: list[str]
    greedy_fn = rollout_fn = None
    include_milp = arm != "BASE_G1_NOMILP"
    if arm in {"FINAL", "BASE_G1_CORE", "BASE_G1_NOMILP"}:
        tree_names = []
    elif arm == "ADD_G1":
        tree_names = ["G1"]
    elif arm == "ADD_P1":
        tree_names = ["P1"]
    elif arm == "ADD_G1_P1":
        tree_names = ["G1", "P1"]
    elif arm in {"ADD_G3", "BASE_G1", "BASEFOLLOW_G1"}:
        tree_names = ["G1", "G2", "G3"]
    elif arm == "BASE_G1_P1":
        tree_names = ["G1", "G2", "G3", "P1"]
    elif arm == "ADD_FAMILY":
        tree_names = ["G1", "E1", "C1"]
    else:
        raise ValueError(f"미지원 planner arm: {arm}")

    sources: list[tuple[str, object]] = []
    if include_milp:
        sources.append(("MILP", _milp_proposer()))
    if tree_names:
        tree_extra = TreeCandidateProposer({x: packages[x] for x in tree_names}, h_pad=47)
        sources.append(("TREE", tree_extra))
    extra = CompositeCandidateProposer(sources)
    if arm in {"BASE_G1", "BASEFOLLOW_G1", "BASE_G1_P1", "BASE_G1_CORE", "BASE_G1_NOMILP"}:
        greedy_fn = _tree_fn(packages["G1"])
    if arm == "BASEFOLLOW_G1":
        rollout_fn = _tree_fn(packages["G1"])

    planner = TruncatedRolloutPlanner(
        model, K=K, h=h, m=m, leaf_fn=None, clairvoyant=False,
        reseed_base=reseed_base, switch_margin=0.0, alloc="uniform", switch_z=0.0,
        extra_cand_fn=extra.propose, greedy_action_fn=greedy_fn,
        rollout_action_fn=rollout_fn,
    )
    env = factory(seed=seed)
    obs, _ = env.reset(seed=seed)
    done, reward, n_dec, n_switch = False, 0.0, 0, 0
    n_tree_offered = n_tree_exec = n_novel_tree_exec = n_milp_exec = 0
    policy_sec = 0.0
    while not done:
        t0 = time.perf_counter()
        extra.last_sources = {}  # 플래닝 불필요 조기반환 시 이전 결정 provenance 잔존 방지
        action = int(planner.act(env, ep_seed=seed, obs=obs))
        policy_sec += time.perf_counter() - t0
        li = planner.last_info
        ppo_cand = set(li.get("ppo_candidate_actions", ()))
        sources_now = extra.last_sources
        tree_actions = {a for a, labels in sources_now.items() if any(x.startswith("TREE:") for x in labels)}
        labels = sources_now.get(action, ())
        n_tree_offered += int(bool(tree_actions - ppo_cand))
        n_tree_exec += int(any(x.startswith("TREE:") for x in labels))
        n_novel_tree_exec += int(action in tree_actions and action not in ppo_cand and not any(x.startswith("MILP") for x in labels))
        n_milp_exec += int(any(x.startswith("MILP") for x in labels))
        n_switch += int(bool(li.get("switched", False)))
        n_dec += 1
        obs, _, term, trunc, info = env.step(action)
        reward += float(info.get("r_woG", 0.0))
        done = term or trunc
    prev = float(env.unwrapped.preventable_woG)
    return {
        "pdr_woG": 1.0 - reward / prev if prev > 0 else 0.0,
        "reward_woG": reward, "n_decisions": n_dec, "n_switch": n_switch,
        "n_tree_offered": n_tree_offered, "n_tree_exec": n_tree_exec,
        "n_novel_tree_exec": n_novel_tree_exec, "n_milp_exec": n_milp_exec,
        "ms_per_decision": policy_sec * 1000 / max(n_dec, 1),
    }


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
        from tree_distill_policy import load_tree_package, make_rank_tree_policy
        from viper_distill import _suppress_stdout, load_vecnorm, make_feature_env

        model = MaskablePPO.load(str(Path(job["model_dir"]) / "final_model.zip"), device="cpu")
        norm = load_vecnorm(str(Path(job["model_dir"]) / "vecnormalize.pkl"))
        packages = {name: load_tree_package(path) for name, path in job["tree_paths"].items()}
        pure_g1 = make_rank_tree_policy(packages["G1"], h_pad=47)
        factory = make_feature_env(job["cfg"], norm)
        rows = []
        with _suppress_stdout():
            for ep in job["episodes"]:
                seed = job["seed0"] + ep
                for arm in job["arms"]:
                    t0 = time.time()
                    if arm == "PPO":
                        def ppo_fn(obs, mask, env_unwrapped):
                            a, _ = model.predict(obs, action_masks=mask, deterministic=True)
                            return int(a)
                        result = _run_pure(factory, ppo_fn, seed)
                    elif arm == "PURE_G1":
                        result = _run_pure(factory, pure_g1, seed)
                    else:
                        result = _run_planner(
                            factory, model, packages, arm, seed,
                            job["K"], job["h"], job["m"], job["reseed_base"],
                        )
                    result.update(
                        region=job["region"], policy=arm, episode=ep, seed=seed,
                        wall_seconds=time.time() - t0,
                    )
                    rows.append(result)
        return {"ok": True, "region": job["region"], "rows": rows}
    except Exception as exc:
        import traceback
        return {"ok": False, "region": job.get("region", "?"),
                "err": (str(exc) + "\n" + traceback.format_exc())[:5000]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(TRAIN_MANIFEST))
    p.add_argument("--fold", default="p3", help="random4 튜닝 fold; 대표점 평가에서는 빈 문자열")
    p.add_argument("--n_regions", type=int, default=12)
    p.add_argument("--regions", default="")
    p.add_argument("--model_dir", default=str(MODEL_DIR))
    p.add_argument("--tree_dir", default=str(TREE_DIR))
    p.add_argument("--ppo_tree_path", default=str(PPO_DISTILL_TREE),
                   help="순수 PPO 증류 GBDT 패키지; 개발선정은 split750, 최종은 full1000")
    p.add_argument("--arms", default=DEFAULT_ARMS)
    p.add_argument("--n_eps", type=int, default=2)
    p.add_argument("--seed0", type=int, default=8000)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--h", type=int, default=20)
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--reseed_base", type=int, default=777000)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--chunk", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    arms = [x for x in args.arms.split(",") if x]
    allowed = set(DEFAULT_ARMS.split(","))
    if not arms or set(arms) - allowed:
        raise ValueError(f"arm 오류: {sorted(set(arms)-allowed)}")
    regions = {x for x in args.regions.split(",") if x}
    entries = _select_entries(Path(args.manifest).resolve(), args.fold, args.n_regions, regions)
    if not entries:
        raise ValueError("평가 지역 0개")
    tree_paths = {
        name: (str(Path(args.ppo_tree_path).resolve()) if name == "P1"
               else str(Path(args.tree_dir).resolve() / f"{case}.pkl"))
        for name, case in DEFAULT_CASES.items()
    }
    missing = [x for x in tree_paths.values() if not Path(x).exists()]
    if missing:
        raise FileNotFoundError(missing)

    out = Path(args.out).resolve()
    meta_path = Path(str(out) + ".meta.json")
    if meta_path.exists():
        raise FileExistsError(f"완료 산출물 보호: {meta_path}")
    out.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict] = []
    seen: set[tuple[str, str, int]] = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
        if existing_rows and set(existing_rows[0]) != set(COLS):
            raise ValueError("재개 CSV 스키마 불일치")
        for row in existing_rows:
            key = (row["region"], row["policy"], int(row["seed"]))
            if key in seen:
                raise ValueError(f"재개 CSV 중복: {key}")
            seen.add(key)
    jobs = []
    episodes = list(range(args.n_eps))
    for region, cfg in entries:
        for arm in arms:
            remaining = [ep for ep in episodes if (region, arm, args.seed0 + ep) not in seen]
            for start in range(0, len(remaining), args.chunk):
                # arm도 독립 job으로 나눠 느린 플래너들을 CPU 코어에 병렬 배치한다.
                jobs.append({
                    **vars(args), "region": region, "cfg": cfg, "arms": [arm],
                    "episodes": remaining[start:start + args.chunk], "tree_paths": tree_paths,
                })
    print(
        f"[v15-portfolio] regions={len(entries)} eps={args.n_eps} arms={arms} "
        f"K/h/m={args.K}/{args.h}/{args.m} existing={len(existing_rows)} "
        f"workers={min(args.workers,max(len(jobs),1))}", flush=True,
    )
    t0 = time.time()
    new_file = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new_file:
            w.writeheader(); f.flush()
        if jobs:
            with Pool(min(args.workers, len(jobs)), maxtasksperchild=1) as pool:
                for i, result in enumerate(pool.imap_unordered(_worker, jobs), 1):
                    if not result["ok"]:
                        raise RuntimeError(f"{result['region']} 실패: {result['err']}")
                    w.writerows(result["rows"]); f.flush()
                    print(f"  [{i}/{len(jobs)}] {result['region']} rows={len(result['rows'])} wall={time.time()-t0:.0f}s", flush=True)
    with open(out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expected = len(entries) * args.n_eps * len(arms)
    if len(rows) != expected or len({(r["region"], r["policy"], r["seed"]) for r in rows}) != expected:
        raise RuntimeError(f"출력 완전성 실패 {len(rows)} != {expected}")
    if not all(np.isfinite(float(r["pdr_woG"])) and 0 <= float(r["pdr_woG"]) <= 1 for r in rows):
        raise RuntimeError("PDR 범위/유한성 실패")
    meta = {
        "schema_version": 1, "protocol": "v15_policy_candidate_portfolio",
        "manifest": str(Path(args.manifest).resolve()), "manifest_sha256": _sha256(args.manifest),
        "fold": args.fold, "regions": [x for x, _ in entries], "n_regions": len(entries),
        "arms": arms, "n_eps": args.n_eps, "seed_start": args.seed0,
        "seed_end": args.seed0 + args.n_eps - 1, "planner": {"K": args.K, "h": args.h, "m": args.m,
        "reseed_base": args.reseed_base}, "tree_cases": DEFAULT_CASES,
        "tree_hashes": {name: _sha256(path) for name, path in tree_paths.items()},
        "model_hash": _sha256(Path(args.model_dir) / "final_model.zip"),
        "lb_t_included": False, "n_rows": len(rows), "resumed_rows": len(existing_rows),
        "wall_seconds": time.time() - t0,
    }
    Path(str(out) + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    means = {}
    for arm in arms:
        vals = [float(r["pdr_woG"]) for r in rows if r["policy"] == arm]
        means[arm] = float(np.mean(vals))
    print(json.dumps(means, ensure_ascii=False, indent=2))
    print(f"완료 → {out}")


if __name__ == "__main__":
    main()
