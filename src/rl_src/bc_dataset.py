"""휴리스틱 best 룰로부터 Behavior Cloning 학습용 (obs, action, mask) 페어 수집.

피드백 5 — 아이디어 1: BC pretrain → PPO fine-tune.

사용 예 (스모크):
    MCI_REDUCED_OBS=1 CUDA_VISIBLE_DEVICES="" python src/rl_src/bc_dataset.py \
        --manifest scenarios/manifests/plan1nat_manifest.json \
        --eval_csv results/plan1nat_f3_eval.csv \
        --n_episodes 10 --out /tmp/bc_smoke.pkl

데이터셋 포맷 (pickle dict):
    {
      "obs":     np.float32 (N, obs_dim),
      "actions": np.int64  (N,),
      "masks":   np.bool_  (N, n_actions),
      "regions": list[str]  len=N,        # 디버깅용 — 어느 지역 샘플인지
      "obs_dim": int,
      "n_actions": int,
      "manifest": str,
      "eval_csv": str,
    }

설계 결정:
  - 학습 env(FlattenAndDiscreteWrapper) 와 정확히 같은 obs/action 공간에서 수집해야
    BC 모델이 PPO 와 호환된다 → 학습 시 쓰는 동일 wrapper 체인으로 env 구성.
  - 룰은 dict obs(rule_test=True 포맷) 를 받지만, 우리는 RL 모드 env(rule_test=False)
    위에서 학습 obs 를 받는다. 룰에게는 매 step 마다 en_manager.get_full_obs() +
    time 키를 합성해 전달 — HybridAMBHeurWrapper 의 패턴 그대로.
  - 룰의 STAY 출력 [-1, 0, -1] 은 flat 액션 0 (== [0,0,0])으로 매핑한다.
    이 액션은 action_masks_joint 에 의해 dest=0 일 때 항상 허용이라 안전.
"""
import argparse
import csv
import os
import pickle
import sys
import json
from typing import List, Tuple

import numpy as np

# rl_src on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_factory import make_base_env  # noqa: E402
from env_wrapper import FlattenAndDiscreteWrapper  # noqa: E402

# sim_src on path (RuleManager)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SIM_SRC = os.path.join(REPO_ROOT, "src", "sim_src")
if SIM_SRC not in sys.path:
    sys.path.insert(0, SIM_SRC)
from RuleManager import Universal_Rule  # noqa: E402


# heuristic_rule CSV 컬럼 토큰 → Universal_Rule 인자
_PRIORITY_SET = {"START", "ReSTART"}
_HOS_SELECT_SET = {"RedOnly", "YellowNearest", "YellowHalf"}
_MODE_SET = {"OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"}


def parse_rule_name(rule_str: str) -> Tuple[str, str, str, str]:
    """예: 'START, YellowNearest, Red Both_UAVFirst, Yellow Both_AMBFirst'
        → ('START', 'YellowNearest', 'Both_UAVFirst', 'Both_AMBFirst')
    """
    parts = [p.strip() for p in rule_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"휴리스틱 룰 이름 파싱 실패 — 4 토큰 필요: {rule_str!r}")
    priority, hos_select, red_token, yellow_token = parts
    if priority not in _PRIORITY_SET:
        raise ValueError(f"priority 토큰 비정상: {priority!r}")
    if hos_select not in _HOS_SELECT_SET:
        raise ValueError(f"hos_select 토큰 비정상: {hos_select!r}")
    if not red_token.startswith("Red "):
        raise ValueError(f"Red 토큰 비정상: {red_token!r}")
    if not yellow_token.startswith("Yellow "):
        raise ValueError(f"Yellow 토큰 비정상: {yellow_token!r}")
    mode_R = red_token[len("Red "):].strip()
    mode_Y = yellow_token[len("Yellow "):].strip()
    if mode_R not in _MODE_SET or mode_Y not in _MODE_SET:
        raise ValueError(f"mode 토큰 비정상: R={mode_R!r} Y={mode_Y!r}")
    return priority, hos_select, mode_R, mode_Y


def load_region_rules(eval_csv: str) -> dict:
    """eval_csv 에서 {region: (priority, hos_select, mode_R, mode_Y)} dict 반환."""
    out = {}
    with open(eval_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            region = row["region"].strip()
            out[region] = parse_rule_name(row["heuristic_rule"])
    return out


def _build_rule(args4):
    rule = Universal_Rule(*args4)
    return rule


def _dict_obs_for_rule(env_unwrapped):
    """rule.select() 가 기대하는 dict obs 합성."""
    obs = env_unwrapped.en_manager.get_full_obs()
    obs['time'] = env_unwrapped.ev_manager.time
    return obs


def collect_region(config_path: str, rule_args, n_episodes: int,
                   seed_base: int = 12345, region_label: str = "?"):
    """단일 지역 시나리오에서 N 에피소드 수집.

    Returns (obs_list, action_list, mask_list, n_actions, obs_dim).
    """
    obs_buf, act_buf, mask_buf = [], [], []
    n_actions = None
    obs_dim = None
    total_steps = 0
    for ep in range(n_episodes):
        base = make_base_env(config_path, seed=seed_base + ep,
                             rule_test=False, eval_mode=False)
        env = FlattenAndDiscreteWrapper(base)
        if n_actions is None:
            n_actions = int(env.action_space.n)
            obs_dim = int(env.observation_space.shape[0])

        rule = _build_rule(rule_args)
        rule.init_with_scenario({'EntityManager': env.unwrapped.en_manager})
        rule.set_seed(np.random.default_rng(seed_base + ep))

        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        ep_steps = 0
        while not done:
            mask = env.action_masks().astype(bool)
            rule_dict = _dict_obs_for_rule(env.unwrapped)
            decoded = rule.select(rule_dict)  # [c, d, m] 또는 STAY=[-1,0,-1]
            c, d, m = int(decoded[0]), int(decoded[1]), int(decoded[2])
            if c < 0 or m < 0:
                # STAY — canonical flat action 0 (= class=0, dest=0, mode=0/fixed)
                flat = 0
            else:
                try:
                    flat = env.encode_action([c, d, m])
                except Exception as e:
                    print(f"[{region_label} ep{ep}] encode 실패 {decoded}: {e} → flat=0")
                    flat = 0
            # mask 안 맞으면 STAY 로 폴백
            if not mask[flat]:
                # action_masks_joint 에서 (any c, dest=0, any m) 은 항상 허용 →
                # 가장 우선 valid 액션으로 폴백
                valid = np.flatnonzero(mask)
                if valid.size == 0:
                    print(f"[{region_label} ep{ep}] 모든 액션 invalid — 샘플 스킵")
                    break
                flat = int(valid[0])

            obs_buf.append(obs.astype(np.float32, copy=False))
            act_buf.append(int(flat))
            mask_buf.append(mask.copy())
            ep_steps += 1

            obs, _, term, trunc, _ = env.step(flat)
            done = term or trunc
        total_steps += ep_steps
        env.close()
    return obs_buf, act_buf, mask_buf, n_actions, obs_dim, total_steps


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True,
                   help="지역별 config_path JSON (plan1nat_manifest.json 등)")
    p.add_argument("--eval_csv", required=True,
                   help="region/heuristic_rule 컬럼이 있는 평가 CSV "
                        "(예: results/plan1nat_f3_eval.csv)")
    p.add_argument("--n_episodes", type=int, default=20,
                   help="지역당 에피소드 수")
    p.add_argument("--out", required=True,
                   help="출력 pickle 경로 (.pkl)")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--regions", nargs="*", default=None,
                   help="특정 지역만 수집 (생략 시 manifest 전체)")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    region_rules = load_region_rules(args.eval_csv)

    regions = list(manifest.keys()) if not args.regions else args.regions
    missing = [r for r in regions if r not in region_rules]
    if missing:
        raise KeyError(f"eval_csv 에 누락된 지역: {missing}")

    print(f"[bc_dataset] {len(regions)} 지역 × {args.n_episodes} ep 수집 시작 "
          f"(reduced_obs={os.environ.get('MCI_REDUCED_OBS', '0')})")
    all_obs, all_act, all_mask, all_region = [], [], [], []
    n_actions_ref, obs_dim_ref = None, None
    total_steps_all = 0
    for region in regions:
        cfg = manifest[region]
        if not os.path.exists(cfg):
            raise FileNotFoundError(f"[{region}] config 미발견: {cfg}")
        rule_args = region_rules[region]
        print(f"  [{region}] rule={rule_args}  config={os.path.basename(cfg)}")
        obs_b, act_b, mask_b, n_actions, obs_dim, steps = collect_region(
            cfg, rule_args, args.n_episodes,
            seed_base=args.seed, region_label=region,
        )
        if n_actions_ref is None:
            n_actions_ref, obs_dim_ref = n_actions, obs_dim
        else:
            if n_actions != n_actions_ref or obs_dim != obs_dim_ref:
                raise ValueError(f"[{region}] obs/action dim 불일치 — "
                                 f"obs {obs_dim} vs {obs_dim_ref}, "
                                 f"act {n_actions} vs {n_actions_ref}")
        all_obs.extend(obs_b)
        all_act.extend(act_b)
        all_mask.extend(mask_b)
        all_region.extend([region] * len(obs_b))
        total_steps_all += steps
        print(f"     → {steps} 스텝 수집 (누적 {total_steps_all})")

    obs_arr = np.stack(all_obs).astype(np.float32)
    act_arr = np.asarray(all_act, dtype=np.int64)
    mask_arr = np.stack(all_mask).astype(np.bool_)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    payload = {
        "obs": obs_arr,
        "actions": act_arr,
        "masks": mask_arr,
        "regions": all_region,
        "obs_dim": int(obs_dim_ref),
        "n_actions": int(n_actions_ref),
        "manifest": args.manifest,
        "eval_csv": args.eval_csv,
    }
    with open(args.out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[bc_dataset] 저장: {args.out}")
    print(f"  total (obs,a) = {len(act_arr)}  obs_dim={obs_dim_ref}  n_actions={n_actions_ref}")
    # 액션 분포 요약
    uniq, cnt = np.unique(act_arr, return_counts=True)
    top = sorted(zip(uniq.tolist(), cnt.tolist()), key=lambda x: -x[1])[:10]
    print(f"  top10 actions (action_id, count): {top}")


if __name__ == "__main__":
    main()
