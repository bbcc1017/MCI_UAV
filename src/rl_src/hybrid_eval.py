"""하이브리드 평가 (2안): AMB 결정은 휴리스틱 룰, UAV 결정은 RL.

매 step:
  1. RL 모델 → flat int action
  2. wrapper.decode_action → [class, dest, mode]
  3. mode==0(AMB) 이면 휴리스틱 룰의 결정으로 가공
       - strict: 룰이 [c,d,m] 전부 결정
       - loose : RL 의 c,d 유지, m 만 룰 결정 사용 (룰이 어떤 mode 를 권하는지)
  4. mode==1(UAV) 이면 RL 결정 그대로
  5. wrapper.encode_action 후 wrapper.step

비교를 위해 evaluate.eval_policy 와 동일한 통계 (mean R, PDR, time, 95% CI) 출력.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.stats import t as student_t

from env_factory import load_config
from evaluate import make_eval_env, ppo_policy, dqn_policy, reinforce_policy

SIM_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "sim_src"))
if SIM_SRC not in sys.path:
    sys.path.insert(0, SIM_SRC)
from RuleManager import Universal_Rule  # noqa: E402


def _dict_obs_for_rule(env_unwrapped):
    """RuleManager.Universal_Rule.select 가 기대하는 dict obs 구성."""
    obs = env_unwrapped.en_manager.get_full_obs()
    obs['time'] = env_unwrapped.ev_manager.time
    return obs


def hybrid_eval(env_factory, rl_model, rl_kind, rule, n_episodes,
                seed_base=1000, mode_split="strict"):
    """rl_kind in {ppo, dqn, reinforce}.
    return dict: mean_R, mean_R_woG, mean_PDR, mean_PDR_woG, mean_time,
                 ci_R, ci_R_woG"""
    if rl_kind == "ppo":
        rl_fn = ppo_policy(rl_model)
    elif rl_kind == "dqn":
        rl_fn = dqn_policy(rl_model)
    elif rl_kind == "reinforce":
        rl_fn = reinforce_policy(rl_model)
    else:
        raise ValueError(rl_kind)

    rewards, rewards_woG, pdrs, pdrs_woG, times = [], [], [], [], []
    for ep in range(n_episodes):
        env = env_factory(seed=seed_base + ep)
        unwrapped = env.unwrapped
        rule.init_with_scenario({'EntityManager': unwrapped.en_manager})
        rule.set_seed(np.random.default_rng(seed_base + ep))

        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        ep_r = 0.0
        ep_r_woG = 0.0
        info = {'time': 0.0, 'r_woG': 0.0}
        while not done:
            mask = env.action_masks()
            int_a = rl_fn(obs, mask, unwrapped)
            decoded = env.decode_action(int_a)  # [c, d, m]

            if decoded[2] == 0:  # RL 이 AMB 선택 → 룰로 가공
                rule_action = rule.select(_dict_obs_for_rule(unwrapped))
                if mode_split == "strict":
                    final = list(rule_action)
                elif mode_split == "loose":
                    final = [decoded[0], decoded[1], int(rule_action[2])]
                else:
                    raise ValueError(f"unknown mode_split={mode_split}")
            else:               # UAV → RL 그대로
                final = list(decoded)

            int_final = env.encode_action(final)
            obs, r, term, trunc, info = env.step(int_final)
            ep_r += r
            ep_r_woG += info.get('r_woG', 0.0)
            done = term or trunc

        preventable = unwrapped.preventable
        preventable_woG = unwrapped.preventable_woG
        pdr = 1 - ep_r / preventable if preventable > 0 else 0.0
        pdr_woG = 1 - ep_r_woG / preventable_woG if preventable_woG > 0 else 0.0
        rewards.append(ep_r)
        rewards_woG.append(ep_r_woG)
        pdrs.append(pdr)
        pdrs_woG.append(pdr_woG)
        times.append(info['time'])

    rewards = np.array(rewards); rewards_woG = np.array(rewards_woG)
    pdrs = np.array(pdrs); pdrs_woG = np.array(pdrs_woG); times = np.array(times)

    def _ci_half(x):
        if len(x) < 2: return 0.0
        std_err = np.std(x, ddof=1) / np.sqrt(len(x))
        ci = student_t.interval(0.95, df=len(x) - 1, loc=np.mean(x), scale=std_err)
        return (ci[1] - ci[0]) / 2

    return {
        "mean_R":      float(rewards.mean()),
        "mean_R_woG":  float(rewards_woG.mean()),
        "mean_PDR":    float(pdrs.mean()),
        "mean_PDR_woG": float(pdrs_woG.mean()),
        "mean_time":   float(times.mean()),
        "ci_R":        _ci_half(rewards),
        "ci_R_woG":    _ci_half(rewards_woG),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", required=True)
    p.add_argument("--ppo_path", default=None)
    p.add_argument("--dqn_path", default=None)
    p.add_argument("--reinforce_path", default=None)
    p.add_argument("--rule_priority", default="START", choices=["START", "ReSTART"])
    p.add_argument("--rule_hos_select", default="RedOnly", choices=["RedOnly", "YellowNearest"])
    p.add_argument("--rule_red_mode", default="Both_AMBFirst",
                   choices=["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"])
    p.add_argument("--rule_yellow_mode", default="Both_AMBFirst",
                   choices=["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"])
    p.add_argument("--mode_split", default="strict", choices=["strict", "loose"])
    p.add_argument("--n_episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    chosen = [x for x in (args.ppo_path, args.dqn_path, args.reinforce_path) if x]
    if len(chosen) != 1:
        raise SystemExit("--ppo_path / --dqn_path / --reinforce_path 중 정확히 하나만 지정해야 합니다.")

    rule = Universal_Rule(args.rule_priority, args.rule_hos_select,
                          args.rule_red_mode, args.rule_yellow_mode)
    env_factory = make_eval_env(args.config_path)

    if args.ppo_path:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(args.ppo_path)
        rl_kind = "ppo"; tag = os.path.basename(args.ppo_path)
    elif args.dqn_path:
        from stable_baselines3 import DQN
        model = DQN.load(args.dqn_path)
        rl_kind = "dqn"; tag = os.path.basename(args.dqn_path)
    else:
        from reinforce_agent import ReinforceAgent
        model = ReinforceAgent.load(args.reinforce_path)
        rl_kind = "reinforce"; tag = os.path.basename(args.reinforce_path)

    m = hybrid_eval(env_factory, model, rl_kind, rule,
                    args.n_episodes, args.seed, mode_split=args.mode_split)

    rule_tag = f"{args.rule_priority}/{args.rule_hos_select}/R={args.rule_red_mode}/Y={args.rule_yellow_mode}"
    name = f"Hybrid({rl_kind}={tag}, heur={rule_tag}, split={args.mode_split})"
    print(f"\n{'Policy':<90} {'mean_R':>8} {'R_woG':>8} {'PDR':>7} {'PDR_woG':>8} {'time':>8} {'95%CI':>8}")
    print("-" * 140)
    print(f"{name:<90} {m['mean_R']:>8.3f} {m['mean_R_woG']:>8.3f} "
          f"{m['mean_PDR']:>7.3f} {m['mean_PDR_woG']:>8.3f} {m['mean_time']:>8.1f} {m['ci_R']:>8.3f}")


if __name__ == "__main__":
    main()
