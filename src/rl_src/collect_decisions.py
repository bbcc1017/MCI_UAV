"""챔피언 MaskablePPO 의 의사결정을 (라벨된 reduced-obs, 디코드 액션, 반사실 휴리스틱)
형태로 로깅한다. 피드백 #2(로그→해석가능 정책)·#4(증류)의 입력 데이터 생성 단계.

설계 원칙:
- **src/sim_src 무수정.** evaluate.eval_policy 의 롤아웃 루프(evaluate.py:38-59)를 복제하되,
  wrapped env 핸들을 유지해 decode_action / action_masks 에 직접 접근한다.
- 매 결정 스텝마다 1행:
    * reduced obs 221차원 (wrapper._obs_keys 순서로 동적 라벨링 — gymnasium Dict 정렬 이슈 회피)
    * action_masks → 축별 자유선택 여부(valid≥2: 마스크가 강제한 결정과 구분)
    * RL action → decode [class, dest, mode] + dest 의 tier3/tier2/helipad 여부
    * 반사실 휴리스틱: 같은 상태에서 그 지역 best 룰의 결정 (HybridAMBHeurWrapper._dict_obs 패턴)
    * step reward / r_woG / sim time / region / episode / step

출력: results/analysis/decisions_<tag>{.npz, _meta.csv, _labels.json}
  - .npz       : obs 행렬 (float32, N×221) — 용량 효율
  - _meta.csv  : 결정별 메타 (region/ep/step/time/액션/플래그/reward)
  - _labels.json : obs 피처 라벨 + 부가 메타

사용:
  MCI_REDUCED_OBS=1 CUDA_VISIBLE_DEVICES="" \
    python src/rl_src/collect_decisions.py \
      --manifest scenarios/plan1nat_manifest.json \
      --model results/rl/plan1nat_f3/national/ppo/final_model.zip \
      --heur_csv results/plan1nat_f3_eval.csv \
      --n_episodes 50 --tag plan1nat_f3
"""
import argparse
import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from env_factory import make_base_env
from env_wrapper import FlattenAndDiscreteWrapper

SIM_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "sim_src"))
if SIM_SRC not in sys.path:
    sys.path.insert(0, SIM_SRC)
from RuleManager import Universal_Rule  # noqa: E402


@contextlib.contextmanager
def _suppress_stdout():
    """sim_src(EventManager 등)의 print 폭발을 /dev/null 로 우회 (stderr 는 유지)."""
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old


def parse_rule(rule_name: str) -> Universal_Rule:
    """'START, YellowNearest, Red Both_UAVFirst, Yellow Both_AMBFirst' → Universal_Rule."""
    parts = [p.strip() for p in rule_name.split(",")]
    priority = parts[0]
    hos_select = parts[1]
    mode_R = parts[2].replace("Red", "", 1).strip()
    mode_Y = parts[3].replace("Yellow", "", 1).strip()
    return Universal_Rule(priority, hos_select, mode_R, mode_Y)


def build_feature_labels(wrapped_env) -> list:
    """wrapper._obs_keys 순서대로 reduced-obs 플랫 벡터의 피처 라벨 생성."""
    sp = wrapped_env.env.observation_space.spaces  # AggregateObsWrapper 의 Dict
    keys = wrapped_env._obs_keys
    classes = ["Red", "Yellow", "Green", "Black"]
    stages = ["NotRescued", "AtSite", "InTransport", "AtHospital", "Done"]
    fleet_stats = ["avail", "busy", "minETA", "meanETA", "nCrit"]
    h_cols = ["idle", "queue", "occ"]
    site_cls = ["Red", "Yellow", "Green", "Black"]
    labels = []
    for k in keys:
        shape = sp[k].shape
        n = int(np.prod(shape))
        if k == "patient_agg":
            labels += [f"pa_{c}_{s}" for c in classes for s in stages]
        elif k == "vehicle_agg":
            labels += [f"ve_{fl}_{st}" for fl in ["amb", "uav"] for st in fleet_stats]
        elif k == "h_states":
            H = shape[0]
            # h_states 컬럼은 aggregate_obs 의 _h_keep 순서([idle,queue,occ] 부분집합).
            # MCI_OBS_VARIANT(idle/noqueue 등)로 컬럼 수가 줄면 shape[1] 만큼만 라벨링.
            n_cols = shape[1] if len(shape) > 1 else 1
            cols = h_cols[:n_cols] if n_cols <= len(h_cols) else [f"c{j}" for j in range(n_cols)]
            labels += [f"h{i}_{c}" for i in range(H) for c in cols]
        elif k == "p_sent":
            labels += [f"psent_{i}" for i in range(n)]
        elif k == "p_at_site":
            labels += [f"atsite_{site_cls[i]}" if i < 4 else f"atsite_{i}" for i in range(n)]
        elif k == "n_amb_at_site":
            labels += ["n_amb_at_site"]
        elif k == "n_uav_at_site":
            labels += ["n_uav_at_site"]
        elif k == "time":
            labels += ["time"]
        else:  # 미지의 키 → 일반 라벨
            labels += [f"{k}_{i}" for i in range(n)]
    return labels


def rollout_region(region, config_path, rule_name, model, n_episodes, seed_base,
                   hos_props_out, vn=None):
    """한 지역 N 에피소드 롤아웃. (obs_rows, meta_rows, mask_rows) 반환."""
    obs_rows, meta_rows, mask_rows = [], [], []
    for ep in range(n_episodes):
        seed = seed_base + ep
        with _suppress_stdout():
            base = make_base_env(config_path, seed=seed, rule_test=False, eval_mode=True)
            env = FlattenAndDiscreteWrapper(base)
            obs, _ = env.reset(seed=seed)
            u = env.unwrapped
            # 반사실 비교용 best 휴리스틱 룰 (지역별)
            rule = parse_rule(rule_name)
            rule.set_seed(np.random.default_rng(seed))
            rule.init_with_scenario({"EntityManager": u.en_manager})

            # 병원 메타 (tier/helipad) — 지역별 1회만 기록
            if region not in hos_props_out:
                hp = u.en_manager.en_properties["hospital"]
                hos_props_out[region] = {
                    "tier3_idx": list(map(int, np.atleast_1d(hp["hos_tier3_idx"]))),
                    "tier2_idx": list(map(int, np.atleast_1d(hp["hos_tier2_idx"]))),
                    "helipad_idx": list(map(int, np.atleast_1d(hp.get("hos_helipad_idx", [])))),
                    "hos_num": int(hp["hos_num"]),
                }
            tier3 = set(hos_props_out[region]["tier3_idx"])
            helipad = set(hos_props_out[region]["helipad_idx"])

            done = False
            step = 0
            while not done:
                mask = env.action_masks()
                obs_in = obs if vn is None else np.clip(
                    (np.asarray(obs, np.float32) - vn[0]) / vn[1], -vn[2], vn[2])
                rl_a, _ = model.predict(obs_in, action_masks=mask, deterministic=True)
                rl_a = int(rl_a)
                rl_c, rl_d, rl_m = env.decode_action(rl_a)

                # 반사실 휴리스틱 (동일 상태)
                dobs = u.en_manager.get_full_obs()
                dobs["time"] = u.ev_manager.time
                h_c, h_d, h_m = rule.select(dobs)

                # 축별 자유선택 플래그 (마스크 기반)
                valid = np.flatnonzero(mask)
                dec = np.array([env.decode_action(int(a)) for a in valid]) if valid.size else np.zeros((0, 3), int)
                n_class = len(set(dec[:, 0].tolist())) if dec.size else 0
                modes_c = set(dec[dec[:, 0] == rl_c, 2].tolist()) if dec.size else set()
                dests_cm = set(dec[(dec[:, 0] == rl_c) & (dec[:, 2] == rl_m), 1].tolist()) if dec.size else set()

                cur_time = float(np.asarray(obs)[-1]) if False else float(u.ev_manager.time)
                obs_rows.append(np.asarray(obs, dtype=np.float32))
                mask_rows.append(np.asarray(mask, dtype=bool))  # joint action mask (predict 에 쓴 그대로)
                meta_rows.append({
                    "region": region, "episode": ep, "step": step, "time": cur_time,
                    # RL 결정
                    "rl_class": rl_c, "rl_dest": rl_d, "rl_mode": rl_m,
                    "rl_is_stay": int(rl_d == 0),
                    "rl_dest_tier3": int((rl_d - 1) in tier3) if rl_d > 0 else -1,
                    "rl_dest_helipad": int((rl_d - 1) in helipad) if rl_d > 0 else -1,
                    # 반사실 휴리스틱 결정
                    "heur_class": int(h_c), "heur_dest": int(h_d), "heur_mode": int(h_m),
                    "heur_is_stay": int(h_d == 0 or h_c < 0),
                    # 일치
                    "agree_class": int(rl_c == h_c),
                    "agree_mode": int(rl_m == h_m),
                    "agree_full": int(rl_c == h_c and rl_d == h_d and rl_m == h_m),
                    # 자유선택 플래그
                    "n_valid": int(valid.size),
                    "n_class_choices": n_class, "free_class": int(n_class >= 2),
                    "n_mode_choices": len(modes_c), "free_mode": int(len(modes_c) >= 2),
                    "n_dest_choices": len(dests_cm), "free_dest": int(len(dests_cm) >= 2),
                })

                obs, r, term, trunc, info = env.step(rl_a)
                meta_rows[-1]["reward"] = float(r)
                meta_rows[-1]["r_woG"] = float(info.get("r_woG", 0.0))
                done = term or trunc
                step += 1
    return obs_rows, meta_rows, mask_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--heur_csv", required=True, help="지역별 best 휴리스틱 룰명 (plan1nat_f3_eval.csv)")
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--seed_base", type=int, default=1000)
    ap.add_argument("--regions", default=None, help="쉼표구분 지역 부분집합 (smoke 용)")
    ap.add_argument("--tag", default="plan1nat_f3")
    ap.add_argument("--out_dir", default="results/analysis")
    ap.add_argument("--vecnorm_path", default=None, help="VecNormalize pkl — predict 시 obs 표준화")
    args = ap.parse_args()

    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(args.model)
    vn = None
    if args.vecnorm_path:
        import pickle
        with open(args.vecnorm_path, "rb") as f:
            _v = pickle.load(f)
        vn = (_v.obs_rms.mean.astype(np.float32),
              np.sqrt(_v.obs_rms.var + _v.epsilon).astype(np.float32), float(_v.clip_obs))

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    heur = pd.read_csv(args.heur_csv)
    best_rule = dict(zip(heur["region"], heur["heuristic_rule"]))

    regions = list(manifest.keys())
    if args.regions:
        want = {r.strip() for r in args.regions.split(",")}
        regions = [r for r in regions if r in want]

    os.makedirs(args.out_dir, exist_ok=True)
    all_obs, all_meta, all_mask, hos_props = [], [], [], {}
    labels = None
    for ri, region in enumerate(regions):
        if region not in best_rule:
            print(f"[skip] {region}: heur_csv 에 없음", file=sys.stderr)
            continue
        sys.stderr.write(f"[{ri+1}/{len(regions)}] {region} (rule={best_rule[region]}) ...\n")
        sys.stderr.flush()
        obs_rows, meta_rows, mask_rows = rollout_region(
            region, manifest[region], best_rule[region], model,
            args.n_episodes, args.seed_base, hos_props, vn)
        if labels is None and obs_rows:
            # 라벨은 첫 지역 env 로 1회 생성
            with _suppress_stdout():
                base = make_base_env(manifest[region], seed=0, rule_test=False, eval_mode=True)
                wenv = FlattenAndDiscreteWrapper(base)
            labels = build_feature_labels(wenv)
            assert len(labels) == len(obs_rows[0]), f"라벨 {len(labels)} != obs {len(obs_rows[0])}"
        all_obs.extend(obs_rows)
        all_meta.extend(meta_rows)
        all_mask.extend(mask_rows)
        sys.stderr.write(f"    → {len(meta_rows)} decisions\n")

    obs_mat = np.asarray(all_obs, dtype=np.float32)
    mask_mat = np.asarray(all_mask, dtype=bool)  # N×(3*(H+1)*2) joint mask
    meta_df = pd.DataFrame(all_meta)
    base_out = os.path.join(args.out_dir, f"decisions_{args.tag}")
    np.savez_compressed(base_out + ".npz", obs=obs_mat, mask=mask_mat)
    meta_df.to_csv(base_out + "_meta.csv", index=False)
    with open(base_out + "_labels.json", "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "hospital_props": hos_props,
                   "n_decisions": int(len(meta_df)), "regions": regions,
                   "n_episodes": args.n_episodes}, f, ensure_ascii=False, indent=2)

    # 요약
    print(f"\n총 결정 {len(meta_df)} 행, obs {obs_mat.shape}", file=sys.stderr)
    if len(meta_df):
        print(f"  자유선택: class {meta_df['free_class'].mean():.2%}, "
              f"mode {meta_df['free_mode'].mean():.2%}, dest {meta_df['free_dest'].mean():.2%}",
              file=sys.stderr)
        print(f"  RL↔휴리스틱 일치: class {meta_df['agree_class'].mean():.2%}, "
              f"mode {meta_df['agree_mode'].mean():.2%}, full {meta_df['agree_full'].mean():.2%}",
              file=sys.stderr)
    print(f"저장: {base_out}{{.npz,_meta.csv,_labels.json}}", file=sys.stderr)


if __name__ == "__main__":
    main()
