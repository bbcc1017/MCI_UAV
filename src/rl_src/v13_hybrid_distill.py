# -*- coding: utf-8 -*-
"""v13 최종 PPO+NCRP+MILP 교사의 결정별 증류 데이터 수집.

기존 ``v10_tree_distill.py collect``는 PPO masked probability를 교사점수로 저장한다.
이 모듈은 최종 실행 스택

``v10 PPO + NCRP(K8,h20,m16,비천리안 CRN) + MILP 후보 2개``

을 실제로 굴려 최종 실행행동을 hard label로 저장한다. 최종 플래너는 현재 관측에 없는
상상미래에도 의존하므로 이 데이터의 목적은 "완전 대체 가능"을 전제하는 것이 아니라,
반응형 학생이 회수할 수 있는 성능과 일반화 한계를 폐루프로 측정하는 것이다.

안전 원칙:
* 기존 ``results/scoreboard/v10/distill``을 절대 덮어쓰지 않는다.
* p0~p2(750좌표) 적합 / p3(250좌표) 내부검증으로 좌표를 분리한다.
* 대표점250은 이 수집기에서 읽지 않으며 최종 폐루프 평가에만 사용한다.
* 출력이 이미 있으면 실패한다(명시적 삭제 없이 재실행 금지).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(__file__))

from tree_distill_policy import ActionFeatureBuilder, FEATURE_NAMES

REPO = Path(__file__).resolve().parents[2]
TRAIN_MANIFEST = REPO / "scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json"
EVAL_MANIFEST = REPO / "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json"
MODEL_DIR = REPO / "results/rl/redesign/v10_random4_1000_pointer_s0"


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _coord(path: str) -> str:
    return Path(path).resolve().parent.name


def _manifest_entries(path: Path, folds: set[str]) -> list[tuple[str, str]]:
    manifest = json.load(open(path, encoding="utf-8"))
    eval_manifest = json.load(open(EVAL_MANIFEST, encoding="utf-8"))
    if len(manifest) != 1000:
        raise ValueError(f"train1000 manifest 필요: N={len(manifest)}")
    groups: dict[str, set[str]] = {}
    selected = []
    for key, cfg in manifest.items():
        toks = key.rsplit("_", 2)
        if len(toks) != 3 or not toks[-2].isdigit() or toks[-1] not in {"p0", "p1", "p2", "p3"}:
            raise ValueError(f"train1000 키 형식 오류: {key}")
        groups.setdefault(toks[-2], set()).add(toks[-1])
        if toks[-1] in folds:
            selected.append((key, cfg))
    bad = {k: v for k, v in groups.items() if v != {"p0", "p1", "p2", "p3"}}
    if len(groups) != 250 or bad:
        raise ValueError(f"train1000 그룹 오류: groups={len(groups)} bad={len(bad)}")
    overlap = {_coord(x) for _, x in selected} & {_coord(x) for x in eval_manifest.values()}
    if overlap:
        raise ValueError(f"수집/대표점250 좌표 중복 {len(overlap)}개")
    expected = 250 * len(folds)
    if len(selected) != expected:
        raise ValueError(f"fold 선택 행수 오류 {len(selected)} != {expected}")
    return selected


def _balanced_state_weights(chosen: np.ndarray) -> np.ndarray:
    """상태마다 양성/음성 총가중치를 0.5/0.5로 맞춘다."""
    n = len(chosen)
    if n == 1:
        return np.ones(1, dtype=np.float32)
    w = np.full(n, 0.5 / (n - 1), dtype=np.float32)
    w[chosen] = 0.5
    return w


def _collect_worker(job: dict) -> dict:
    try:
        import torch as th

        th.set_num_threads(1)
        os.environ.update(
            MCI_CAP_GATE="occ",
            MCI_OBS_VARIANT="essential+load+valid",
            MCI_H_PAD="47",
            MCI_REWARD_MODE="woG",
        )
        from sb3_contrib import MaskablePPO
        from hospital_set_extractor import HospitalSetExtractor  # noqa: F401
        from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401
        from milp_policy import MilpProposer
        from planner_policy import TruncatedRolloutPlanner
        from viper_distill import _masked_probs, _suppress_stdout, load_vecnorm, make_feature_env

        model_dir = job["model_dir"]
        model = MaskablePPO.load(os.path.join(model_dir, "final_model.zip"), device="cpu")
        norm = load_vecnorm(os.path.join(model_dir, "vecnormalize.pkl"))

        row_keys = ("X", "target", "weight", "chosen", "cand_action", "ppo_prob")
        state_keys = (
            "ncand", "teacher_action", "behavior_action", "ppo_action", "state_key",
            "state_seed", "decision_index", "teacher_switched", "teacher_in_milp",
            "planner_lookahead", "planner_dpdr", "planner_q_greedy", "planner_q_exec",
            "planner_n_cand", "planner_n_extra", "milp_action0", "milp_action1",
        )
        out: dict[str, list] = {k: [] for k in row_keys + state_keys}
        episodes = {k: [] for k in (
            "episode_key", "episode_seed_eval", "episode_reward_wog", "episode_pdr_wog",
            "episode_n_decisions", "episode_n_switch",
        )}

        with _suppress_stdout():
            for key, cfg in job["entries"]:
                fac = make_feature_env(cfg, norm)
                env = fac(seed=job["seed"])
                builder = ActionFeatureBuilder(h_pad=47)
                for ep in range(job["n_eps"]):
                    episode_seed = int(job["seed"] + ep)
                    milp_rec = {"actions": []}
                    proposer = MilpProposer(
                        h_pad=47, n_propose=job["milp_n_propose"],
                        n_opp=job["milp_n_opp"], topk_hosp=job["milp_topk"],
                        second_wave=job["milp_second_wave"],
                        future_patients=job["milp_future"],
                        n_future_groups=job["milp_future_groups"],
                        force_dispatch=job["milp_force_dispatch"],
                        queue_model=job["milp_queue_model"],
                    )

                    def extra_cand(env_unwrapped, mask):
                        acts = [int(x) for x in proposer.propose(env_unwrapped, mask)]
                        milp_rec["actions"] = acts
                        return acts

                    planner = TruncatedRolloutPlanner(
                        model, K=job["K"], h=job["h"], m=job["m"], leaf_fn=None,
                        clairvoyant=False, reseed_base=job["reseed_base"],
                        switch_margin=0.0, alloc="uniform", switch_z=0.0,
                        extra_cand_fn=extra_cand,
                    )
                    obs, _ = env.reset(seed=episode_seed)
                    done, dec, ep_reward, ep_switch = False, 0, 0.0, 0
                    while not done:
                        mask = np.asarray(env.action_masks(), dtype=bool)
                        actions, X = builder.build(env.unwrapped, mask)
                        prob_full = _masked_probs(model, obs, mask)
                        ppo_prob = np.asarray(prob_full[actions], dtype=np.float32)
                        ppo_action = int(np.argmax(prob_full))
                        milp_rec["actions"] = []
                        teacher = int(planner.act(env, ep_seed=episode_seed, obs=obs))
                        info = dict(planner.last_info)
                        chosen = actions == teacher
                        if int(chosen.sum()) != 1:
                            raise RuntimeError(f"교사행동이 유효후보에 없음: {key} seed={episode_seed} a={teacher}")

                        out["X"].append(X.astype(np.float32))
                        out["target"].append(chosen.astype(np.float32))
                        out["weight"].append(_balanced_state_weights(chosen))
                        out["chosen"].append(chosen)
                        out["cand_action"].append(actions.astype(np.int16))
                        out["ppo_prob"].append(ppo_prob)
                        out["ncand"].append(len(actions))
                        out["teacher_action"].append(teacher)
                        out["behavior_action"].append(teacher)
                        out["ppo_action"].append(ppo_action)
                        out["state_key"].append(key)
                        out["state_seed"].append(episode_seed)
                        out["decision_index"].append(dec)
                        out["teacher_switched"].append(teacher != ppo_action)
                        ep_switch += int(teacher != ppo_action)
                        out["teacher_in_milp"].append(teacher in milp_rec["actions"])
                        out["planner_lookahead"].append(bool(info.get("lookahead", False)))
                        out["planner_dpdr"].append(float(info.get("dpdr") or 0.0))
                        out["planner_q_greedy"].append(float(info.get("q_greedy") or 0.0))
                        out["planner_q_exec"].append(float(info.get("q_exec") or 0.0))
                        out["planner_n_cand"].append(int(info.get("n_cand", 0)))
                        out["planner_n_extra"].append(int(info.get("n_extra", 0)))
                        ma = milp_rec["actions"] + [-1, -1]
                        out["milp_action0"].append(ma[0])
                        out["milp_action1"].append(ma[1])

                        obs, _, term, trunc, step_info = env.step(teacher)
                        ep_reward += float(step_info.get("r_woG", 0.0))
                        done = term or trunc
                        dec += 1
                    preventable = float(env.unwrapped.preventable_woG)
                    episodes["episode_key"].append(key)
                    episodes["episode_seed_eval"].append(episode_seed)
                    episodes["episode_reward_wog"].append(ep_reward)
                    episodes["episode_pdr_wog"].append(
                        1.0 - ep_reward / preventable if preventable > 0 else 0.0
                    )
                    episodes["episode_n_decisions"].append(dec)
                    episodes["episode_n_switch"].append(ep_switch)

        if not out["ncand"]:
            raise RuntimeError("수집 상태 0개")
        packed = {
            "ok": True,
            "X": np.vstack(out["X"]),
            "target": np.concatenate(out["target"]),
            "weight": np.concatenate(out["weight"]),
            "chosen": np.concatenate(out["chosen"]),
            "cand_action": np.concatenate(out["cand_action"]),
            "ppo_prob": np.concatenate(out["ppo_prob"]),
        }
        dtypes = {
            "ncand": np.int16, "teacher_action": np.int16, "behavior_action": np.int16,
            "ppo_action": np.int16, "state_seed": np.int32, "decision_index": np.int16,
            "teacher_switched": bool, "teacher_in_milp": bool, "planner_lookahead": bool,
            "planner_dpdr": np.float32, "planner_q_greedy": np.float32,
            "planner_q_exec": np.float32, "planner_n_cand": np.int16,
            "planner_n_extra": np.int8, "milp_action0": np.int16, "milp_action1": np.int16,
        }
        for k in state_keys:
            packed[k] = np.asarray(out[k], dtype=dtypes.get(k))
        episode_dtypes = {
            "episode_seed_eval": np.int32, "episode_reward_wog": np.float32,
            "episode_pdr_wog": np.float32, "episode_n_decisions": np.int16,
            "episode_n_switch": np.int16,
        }
        for k, values in episodes.items():
            packed[k] = np.asarray(values, dtype=episode_dtypes.get(k))
        return packed
    except Exception as exc:
        import traceback
        return {"ok": False, "err": (str(exc) + "\n" + traceback.format_exc())[:5000]}


def collect(args) -> None:
    out_path = Path(args.out).resolve()
    if out_path.exists() or Path(str(out_path) + ".meta.json").exists():
        raise FileExistsError(f"기존 산출물 보호: {out_path}")
    folds = {x.strip() for x in args.folds.split(",") if x.strip()}
    if not folds or not folds <= {"p0", "p1", "p2", "p3"}:
        raise ValueError(f"folds 오류: {sorted(folds)}")
    entries = _manifest_entries(Path(args.manifest).resolve(), folds)
    if args.key:
        entries = [x for x in entries if x[0] == args.key]
        if not entries:
            raise ValueError(f"선택 fold에 없는 --key: {args.key}")
    if args.limit:
        entries = entries[:args.limit]
    chunks = [entries[i:i + args.chunk] for i in range(0, len(entries), args.chunk)]
    common = vars(args).copy()
    common["model_dir"] = str(Path(args.model_dir).resolve())
    jobs = [{**common, "entries": c} for c in chunks]
    n_workers = min(args.workers, len(jobs))
    print(
        f"[hybrid-collect] role={args.role} folds={sorted(folds)} points={len(entries)} "
        f"eps={args.n_eps} K/h/m={args.K}/{args.h}/{args.m} workers={n_workers}",
        flush=True,
    )

    packs, t0 = [], time.time()
    with Pool(n_workers, maxtasksperchild=1) as pool:
        for i, pack in enumerate(pool.imap_unordered(_collect_worker, jobs), 1):
            if not pack["ok"]:
                raise RuntimeError(f"수집 실패 [{i}/{len(jobs)}]: {pack['err']}")
            packs.append(pack)
            print(
                f"  [{i}/{len(jobs)}] states={len(pack['ncand'])} "
                f"switch={float(np.mean(pack['teacher_switched'])):.3f} wall={time.time()-t0:.0f}s",
                flush=True,
            )

    row_keys = ("X", "target", "weight", "chosen", "cand_action", "ppo_prob")
    state_keys = (
        "ncand", "teacher_action", "behavior_action", "ppo_action", "state_key",
        "state_seed", "decision_index", "teacher_switched", "teacher_in_milp",
        "planner_lookahead", "planner_dpdr", "planner_q_greedy", "planner_q_exec",
        "planner_n_cand", "planner_n_extra", "milp_action0", "milp_action1",
    )
    episode_keys = (
        "episode_key", "episode_seed_eval", "episode_reward_wog", "episode_pdr_wog",
        "episode_n_decisions", "episode_n_switch",
    )
    data = {k: np.concatenate([p[k] for p in packs]) for k in row_keys + state_keys + episode_keys}
    data["offsets"] = np.concatenate([[0], np.cumsum(data["ncand"], dtype=np.int64)])
    data["feature_names"] = np.asarray(FEATURE_NAMES)
    if data["offsets"][-1] != len(data["X"]):
        raise RuntimeError("후보행 offsets 불일치")
    if len(set(zip(data["state_key"], data["state_seed"], data["decision_index"]))) != len(data["ncand"]):
        raise RuntimeError("상태 복합키 중복")
    if not np.isfinite(data["X"]).all() or not np.isfinite(data["weight"]).all():
        raise RuntimeError("비유한 특징/가중치")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)

    teacher_id = (
        f"PPO_POINTER_V10_NCRP_K{args.K}_H{args.h}_M{args.m}_MILPINJ"
    )
    meta = {
        "schema_version": 1,
        "role": args.role,
        "teacher": teacher_id,
        "target_semantics": "hard final executed action; state-balanced one-vs-rest weights",
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "folds": sorted(folds),
        "excluded_final_eval_manifest": str(EVAL_MANIFEST),
        "coordinate_overlap": 0,
        "model_dir": str(Path(args.model_dir).resolve()),
        "model_sha256": _sha256(Path(args.model_dir) / "final_model.zip"),
        "seed_start": args.seed,
        "n_eps_per_point": args.n_eps,
        "n_points": len(entries),
        "n_states": int(len(data["ncand"])),
        "n_candidate_rows": int(len(data["X"])),
        "switch_rate_vs_ppo": float(np.mean(data["teacher_switched"])),
        "teacher_in_milp_rate": float(np.mean(data["teacher_in_milp"])),
        "lookahead_rate": float(np.mean(data["planner_lookahead"])),
        "episode_pdr_wog_mean": float(np.mean(data["episode_pdr_wog"])),
        "planner": {"K": args.K, "h": args.h, "m": args.m, "reseed_base": args.reseed_base},
        "milp": {
            "n_propose": args.milp_n_propose, "n_opp": args.milp_n_opp,
            "topk_hosp": args.milp_topk, "second_wave": args.milp_second_wave,
            "future_patients": args.milp_future, "force_dispatch": args.milp_force_dispatch,
            "queue_model": args.milp_queue_model,
        },
        "git_sha": _git_sha(),
        "output": str(out_path),
        "output_sha256": _sha256(out_path),
        "wall_seconds": time.time() - t0,
    }
    Path(str(out_path) + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[hybrid-collect] 완료 states={meta['n_states']:,} rows={meta['n_candidate_rows']:,} "
        f"switch={meta['switch_rate_vs_ppo']:.3f} → {out_path}", flush=True,
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(TRAIN_MANIFEST))
    p.add_argument("--model_dir", default=str(MODEL_DIR))
    p.add_argument("--folds", default="p0,p1,p2")
    p.add_argument("--role", choices=["train", "validation", "smoke"], default="train")
    p.add_argument("--n_eps", type=int, default=1)
    p.add_argument("--seed", type=int, default=5000)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--chunk", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--key", default="", help="정확한 manifest key 1개(재현 스모크용)")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--h", type=int, default=20)
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--reseed_base", type=int, default=777000)
    p.add_argument("--milp_n_propose", type=int, default=2)
    p.add_argument("--milp_n_opp", type=int, default=3)
    p.add_argument("--milp_topk", type=int, default=0)
    p.add_argument("--milp_second_wave", action="store_true")
    p.add_argument("--milp_future", action="store_true")
    p.add_argument("--milp_future_groups", type=int, default=2)
    p.add_argument("--milp_force_dispatch", action="store_true")
    p.add_argument("--milp_queue_model", choices=["fluid", "timed"], default="fluid")
    p.add_argument("--out", required=True)
    return p


if __name__ == "__main__":
    collect(parser().parse_args())
