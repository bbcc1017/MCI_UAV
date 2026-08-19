# -*- coding: utf-8 -*-
"""v17 특징증강 정합성 게이트.

G1 수집 재현   : 저장된 교사 데이터셋의 43특징을 같은 좌표·seed 로 라이브 재계산해 비교
G2 증강 일치   : 추론 경로(AugmentedFeatureBuilder)와 데이터셋 경로(augment_dataset) 동일
G3 구 트리 불변: feature_schema 없는 v10 패키지의 폐루프 PDR 이 기록값과 동일
G4 신 트리 구동: v17 증강 트리가 실제 환경에서 끝까지 돌고 유한한 PDR 을 낸다
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.update(
    MCI_CAP_GATE="occ",
    MCI_OBS_VARIANT="essential+load+valid",
    MCI_H_PAD="47",
    MCI_REWARD_MODE="woG",
)

sys.path.insert(0, os.path.dirname(__file__))

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "results/scoreboard/v10/distill/data/ppo_train1000_seed5000.npz"
TRAIN_MANIFEST = REPO / "scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json"
EVAL_MANIFEST = REPO / "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json"
MODEL_DIR = REPO / "results/rl/redesign/v10_random4_1000_pointer_s0"
OLD_TREE = REPO / "results/scoreboard/v10/distill/trees_final_prob/I3_CONNECTED_C4.pkl"
OLD_CSV = REPO / "results/scoreboard/v10/distill/tree_eval250_seed0_29.csv"


def main(new_tree: str | None) -> None:
    import torch as th

    th.set_num_threads(1)
    from sb3_contrib import MaskablePPO
    from hospital_set_extractor import HospitalSetExtractor  # noqa: F401
    from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401
    from viper_distill import _suppress_stdout, load_vecnorm, make_feature_env
    from tree_distill_policy import (
        ActionFeatureBuilder,
        load_tree_package,
        make_rank_tree_policy,
    )
    from v17_tree_features import AugmentedFeatureBuilder, augment_dataset, augment_state

    results: dict[str, object] = {}

    # ---------- G1 + G2 ----------
    z = np.load(DATA, allow_pickle=False)
    keys = np.asarray(z["state_key"])
    offsets = np.asarray(z["offsets"])
    train_manifest = json.load(open(TRAIN_MANIFEST, encoding="utf-8"))
    model = MaskablePPO.load(str(MODEL_DIR / "final_model.zip"), device="cpu")
    norm = load_vecnorm(str(MODEL_DIR / "vecnormalize.pkl"))

    probe_keys = [k for k in ("종로구_11110_p0", "강남구_11680_p0") if k in train_manifest]
    probe_keys = probe_keys or [sorted(set(keys.tolist()))[0]]
    g1_max, g2_max, n_states_checked = 0.0, 0.0, 0
    with _suppress_stdout():
        for key in probe_keys:
            sel = np.flatnonzero(keys == key)
            if sel.size == 0:
                continue
            fac = make_feature_env(train_manifest[key], norm)
            env = fac(seed=5000)
            base_builder = ActionFeatureBuilder(h_pad=47)
            aug_builder = AugmentedFeatureBuilder(h_pad=47)
            obs, _ = env.reset(seed=5000)
            done, step = False, 0
            while not done and step < sel.size:
                mask = np.asarray(env.action_masks(), dtype=bool)
                actions, X = base_builder.build(env.unwrapped, mask)
                _, X_aug = aug_builder.build(env.unwrapped, mask)
                s = int(sel[step])
                a, b = int(offsets[s]), int(offsets[s + 1])
                stored = np.asarray(z["X"][a:b], dtype=np.float32)
                if stored.shape != X.shape:
                    raise AssertionError(f"G1 shape {stored.shape} != {X.shape} at {key}#{step}")
                g1_max = max(g1_max, float(np.abs(stored - X).max()))
                g2_max = max(g2_max, float(np.abs(X_aug[:, 43:] - augment_state(X)).max()))
                pred, _ = model.predict(obs, action_masks=mask, deterministic=True)
                if int(pred) != int(z["teacher_action"][s]):
                    raise AssertionError(f"G1 교사 행동 불일치 {key}#{step}")
                obs, _, term, trunc, _ = env.step(int(pred))
                done = term or trunc
                step += 1
                n_states_checked += 1
    results["G1_collect_replay_max_abs_diff"] = g1_max
    results["G2_infer_vs_dataset_max_abs_diff"] = g2_max
    results["G1_states_checked"] = n_states_checked

    # 데이터셋 경로 자체의 벡터화/루프 일치(첫 200 state)
    n_probe = 200
    sub_off = offsets[: n_probe + 1]
    a_loop = np.vstack([augment_state(z["X"][int(sub_off[i]):int(sub_off[i + 1])])
                        for i in range(n_probe)])
    a_batch = augment_dataset(z["X"][: int(sub_off[-1])], sub_off)
    results["G2b_dataset_loop_vs_batch"] = float(np.abs(a_loop - a_batch).max())

    # ---------- G3 ----------
    eval_manifest = json.load(open(EVAL_MANIFEST, encoding="utf-8"))
    region = sorted(eval_manifest)[0]
    import csv

    ref = [
        float(r["pdr_woG"]) for r in csv.DictReader(open(OLD_CSV, encoding="utf-8"))
        if r["region"] == region and r["policy"] == "I3_CONNECTED_C4" and int(r["episode"]) < 3
    ]
    old_pkg = load_tree_package(str(OLD_TREE))
    old_policy = make_rank_tree_policy(old_pkg, h_pad=47)
    got = []
    with _suppress_stdout():
        fac = make_feature_env(eval_manifest[region], None)
        for seed in range(3):
            env = fac(seed=seed)
            obs, _ = env.reset(seed=seed)
            done, rew = False, 0.0
            while not done:
                mask = env.action_masks()
                obs, _, term, trunc, info = env.step(old_policy(obs, mask, env.unwrapped))
                rew += info.get("r_woG", 0.0)
                done = term or trunc
            prev = env.unwrapped.preventable_woG
            got.append(1.0 - rew / prev if prev > 0 else 0.0)
    results["G3_region"] = region
    results["G3_ref"] = ref
    results["G3_got"] = got
    results["G3_max_abs_diff"] = float(np.abs(np.array(ref) - np.array(got)).max()) if ref else None

    # ---------- G4 ----------
    if new_tree:
        pkg = load_tree_package(new_tree)
        if pkg.get("feature_schema") != "v17_aug68":
            raise AssertionError("G4 대상이 v17 증강 스키마가 아니다")
        pol = make_rank_tree_policy(pkg, h_pad=47)
        pdrs = []
        with _suppress_stdout():
            fac = make_feature_env(eval_manifest[region], None)
            for seed in range(3):
                env = fac(seed=seed)
                obs, _ = env.reset(seed=seed)
                done, rew = False, 0.0
                while not done:
                    mask = env.action_masks()
                    obs, _, term, trunc, info = env.step(pol(obs, mask, env.unwrapped))
                    rew += info.get("r_woG", 0.0)
                    done = term or trunc
                prev = env.unwrapped.preventable_woG
                pdrs.append(1.0 - rew / prev if prev > 0 else 0.0)
        results["G4_tree"] = Path(new_tree).stem
        results["G4_pdr"] = pdrs
        results["G4_finite"] = bool(np.isfinite(pdrs).all())

    ok = (
        results["G1_collect_replay_max_abs_diff"] == 0.0
        and results["G2_infer_vs_dataset_max_abs_diff"] == 0.0
        and results["G2b_dataset_loop_vs_batch"] == 0.0
        and (results["G3_max_abs_diff"] is None or results["G3_max_abs_diff"] == 0.0)
        and (not new_tree or results["G4_finite"])
    )
    results["verdict"] = "PASS" if ok else "FAIL"
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
