"""학습된 PPO/DQN 정책과 휴리스틱 룰을 비교 평가.

사용 예:
    python src/rl_src/evaluate.py \
        --config_path scenarios/.../config.yaml \
        --ppo_path results/rl/ppo_seoul/final_model.zip \
        --dqn_path results/rl/dqn_seoul/final_model.zip \
        --n_episodes 30
"""
import argparse
import os
import sys
from typing import Callable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.stats import t as student_t

from env_factory import make_base_env, load_config
from env_wrapper import FlattenAndDiscreteWrapper

# 룰 평가용 sim_src import
SIM_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "sim_src"))
if SIM_SRC not in sys.path:
    sys.path.insert(0, SIM_SRC)
from RuleManager import RuleManager  # noqa: E402


def eval_policy(env_factory: Callable, policy_fn: Callable, n_episodes: int,
                seed_base: int = 1000):
    """
    policy_fn(obs, mask, env_unwrapped) -> action(int).
    return dict: mean_R, mean_R_woG, mean_PDR, mean_PDR_woG, mean_time,
                 ci_R, ci_R_woG
    """
    rewards, rewards_woG, pdrs, pdrs_woG, times = [], [], [], [], []
    for ep in range(n_episodes):
        env = env_factory(seed=seed_base + ep)
        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        ep_r = 0.0
        ep_r_woG = 0.0
        while not done:
            mask = env.action_masks()
            a = policy_fn(obs, mask, env.unwrapped)
            obs, r, term, trunc, info = env.step(a)
            ep_r += r
            ep_r_woG += info.get('r_woG', 0.0)
            done = term or trunc
        preventable = env.unwrapped.preventable
        preventable_woG = env.unwrapped.preventable_woG
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


def make_eval_env(config_path: str):
    def _f(seed: int = 0):
        base = make_base_env(config_path, seed=seed, rule_test=False, eval_mode=True)
        return FlattenAndDiscreteWrapper(base)
    return _f


def make_rule_env(config_path: str):
    """rule_test=True 모드 base env (gym wrapper 안 씌움 — RuleManager 가 dict obs 직접 사용)."""
    def _f(seed: int = 0):
        return make_base_env(config_path, seed=seed, rule_test=True, eval_mode=True)
    return _f


def ppo_policy(model):
    def _fn(obs, mask, env_unwrapped):
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)
    return _fn


def dqn_policy(model):
    import torch
    from train_dqn import predict_with_mask  # noqa
    def _fn(obs, mask, env_unwrapped):
        return predict_with_mask(model, obs, mask)
    return _fn


def reinforce_policy(agent):
    """REINFORCE deterministic eval (argmax under mask)."""
    def _fn(obs, mask, env_unwrapped):
        action, _ = agent.act(obs, mask=mask, deterministic=True)
        return action
    return _fn


def random_policy():
    def _fn(obs, mask, env_unwrapped):
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            return 0
        return int(np.random.choice(valid))
    return _fn


def heuristic_eval(config_path: str, n_episodes: int, seed_base: int = 1000):
    """RuleManager 의 모든 룰을 한 번씩 평가. 결과 list of (rule_name, mean_r, mean_pdr, mean_time, ci_r)."""
    cfg = load_config(config_path)
    # rule list 한 번만 만들고 매 ep마다 새 ScenarioManager 생성 (seed 별로 시나리오 흔들기)
    base = make_base_env(config_path, seed=seed_base, rule_test=True, eval_mode=True)
    rules = RuleManager(cfg['rule_info'], scenario={'EntityManager': base.en_manager}, rng=np.random.default_rng(seed_base)).rules
    out = []
    for rule in rules:
        rs, ps, ts = [], [], []
        for ep in range(n_episodes):
            env = make_base_env(config_path, seed=seed_base + ep, rule_test=True, eval_mode=True)
            # rule 도 env 에 binding
            rule.init_with_scenario({'EntityManager': env.en_manager})
            rule.set_seed(np.random.default_rng(seed_base + ep))
            obs, _ = env.reset(seed=seed_base + ep)
            done = False; ep_r = 0.0
            while not done:
                action = rule.select(obs)
                obs, r, term, trunc, info = env.step(action)
                ep_r += r
                done = term or trunc
            rs.append(ep_r)
            preventable = env.preventable
            ps.append(1 - ep_r / preventable if preventable > 0 else 0.0)
            ts.append(info['time'])
        rs = np.array(rs); ps = np.array(ps); ts = np.array(ts)
        std_err = np.std(rs, ddof=1) / np.sqrt(len(rs)) if len(rs) > 1 else 0.0
        if len(rs) > 1:
            ci = student_t.interval(0.95, df=len(rs)-1, loc=np.mean(rs), scale=std_err)
            ci_half = (ci[1] - ci[0]) / 2
        else:
            ci_half = 0.0
        out.append((rule.rule_name, float(rs.mean()), float(ps.mean()), float(ts.mean()), float(ci_half)))
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", required=True)
    p.add_argument("--ppo_path", default=None)
    p.add_argument("--dqn_path", default=None)
    p.add_argument("--reinforce_path", default=None,
                   help="학습된 REINFORCE .pt 파일")
    p.add_argument("--include_random", action="store_true")
    p.add_argument("--include_heuristic", action="store_true")
    p.add_argument("--n_episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    rows = []

    rl_env_factory = make_eval_env(args.config_path)

    def _run(name, policy):
        m = eval_policy(rl_env_factory, policy, args.n_episodes, args.seed)
        rows.append((name, m["mean_R"], m["mean_R_woG"], m["mean_PDR"],
                     m["mean_PDR_woG"], m["mean_time"], m["ci_R"]))

    if args.ppo_path:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(args.ppo_path)
        _run(f"PPO ({os.path.basename(args.ppo_path)})", ppo_policy(model))

    if args.dqn_path:
        from stable_baselines3 import DQN
        model = DQN.load(args.dqn_path)
        _run(f"DQN ({os.path.basename(args.dqn_path)})", dqn_policy(model))

    if args.reinforce_path:
        from reinforce_agent import ReinforceAgent
        agent = ReinforceAgent.load(args.reinforce_path)
        _run(f"REINFORCE ({os.path.basename(args.reinforce_path)})", reinforce_policy(agent))

    if args.include_random:
        _run("Random (mask 적용)", random_policy())

    if args.include_heuristic:
        # heuristic_eval 은 woG 미포함 → 0 으로 채워서 호환
        for name, r, pdr, t, ci in heuristic_eval(args.config_path, args.n_episodes, args.seed):
            rows.append((name, r, 0.0, pdr, 0.0, t, ci))

    print(f"\n{'Policy':<60} {'mean_R':>8} {'R_woG':>8} {'PDR':>7} {'PDR_woG':>8} {'time':>8} {'95%CI':>8}")
    print("-" * 115)
    for name, r, r_woG, pdr, pdr_woG, t, ci in rows:
        print(f"{name:<60} {r:>8.3f} {r_woG:>8.3f} {pdr:>7.3f} {pdr_woG:>8.3f} {t:>8.1f} {ci:>8.3f}")


if __name__ == "__main__":
    main()
