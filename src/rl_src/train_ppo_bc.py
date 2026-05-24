"""BC pretrain + MaskablePPO fine-tune (피드백 5 — 아이디어 1).

bc_dataset.py 로 모은 휴리스틱 best (obs, action, mask) 데이터로 정책을 supervised
학습한 뒤, 동일 MaskablePPO 모델로 PPO fine-tune.

사용 예 (스모크):
    MCI_REDUCED_OBS=1 CUDA_VISIBLE_DEVICES="" python src/rl_src/train_ppo_bc.py \
        --config_path scenarios/plan1nat_manifest.json \
        --bc_data /tmp/bc_smoke.pkl \
        --bc_epochs 2 --total_timesteps 2000 \
        --log_dir /tmp/bc_smoke_train

본 학습 (예시 — 풀 학습은 외부 orchestration 에서 수행):
    MCI_REDUCED_OBS=1 CUDA_VISIBLE_DEVICES="" python src/rl_src/train_ppo_bc.py \
        --config_path scenarios/plan1nat_manifest.json \
        --bc_data results/rl/bc/bc_plan1nat.pkl \
        --bc_epochs 5 --total_timesteps 500000 \
        --log_dir results/rl/plan1nat_bc_ppo

설계 결정:
  - train_ppo.py 의 하이퍼파라미터(net_arch [256,256], ent_coef 0.01, n_steps 512,
    batch 128, lr 3e-4) 그대로 따른다 — 비교 공정성.
  - BC 단계: MaskableActorCriticPolicy.evaluate_actions(obs, action, mask) 로
    masked log_prob 계산 → 음수 평균이 BC 손실. policy 자체 optimizer 사용.
  - BC 후 model.learn() 호출 — 사전학습된 가중치 그대로 PPO rollout/update 진행.
  - 데이터셋의 obs_dim/n_actions 가 env 와 다르면 즉시 실패 (실수 방지).
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch as th

# rl_src on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy as masked_evaluate
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env_factory import make_base_env
from env_wrapper import FlattenAndDiscreteWrapper


def mask_fn(env):
    return env.action_masks()


def make_env_fn(config_path: str, seed: int = 0):
    def _f():
        if config_path.endswith(".json"):
            from multi_region_env import MultiRegionEnv
            env = MultiRegionEnv(config_path, seed=seed)
        else:
            base = make_base_env(config_path, seed=seed, rule_test=False, eval_mode=False)
            env = FlattenAndDiscreteWrapper(base)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    return _f


def load_bc_dataset(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f)
    required = {"obs", "actions", "masks", "obs_dim", "n_actions"}
    if not required.issubset(d.keys()):
        raise KeyError(f"bc_dataset payload 누락 키: {required - set(d.keys())}")
    return d


def bc_pretrain(model: MaskablePPO, dataset: dict, epochs: int,
                batch_size: int, device: str, verbose: bool = True):
    """policy.evaluate_actions 를 이용한 masked NLL 학습."""
    policy = model.policy
    policy.set_training_mode(True)

    obs_np = dataset["obs"].astype(np.float32)
    act_np = dataset["actions"].astype(np.int64)
    mask_np = dataset["masks"].astype(bool)
    N = obs_np.shape[0]
    rng = np.random.default_rng(0)

    optimizer = policy.optimizer  # MaskablePPO 가 만들어둔 Adam

    for ep in range(epochs):
        perm = rng.permutation(N)
        n_batches = max(N // batch_size, 1)
        ep_loss = 0.0
        ep_correct = 0
        ep_total = 0
        for bi in range(n_batches):
            idx = perm[bi * batch_size:(bi + 1) * batch_size]
            obs_b = th.as_tensor(obs_np[idx], device=device)
            act_b = th.as_tensor(act_np[idx], device=device)
            mask_b = th.as_tensor(mask_np[idx], device=device)

            optimizer.zero_grad()
            _, log_prob, _ = policy.evaluate_actions(obs_b, act_b, action_masks=mask_b)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()

            ep_loss += float(loss.detach().cpu().item()) * len(idx)
            with th.no_grad():
                dist = policy.get_distribution(obs_b, action_masks=mask_b)
                # MaskableCategorical distribution: argmax of probs
                logits = dist.distribution.logits  # already masked
                pred = th.argmax(logits, dim=-1)
                ep_correct += int((pred == act_b).sum().item())
                ep_total += int(act_b.shape[0])
        avg_loss = ep_loss / max(ep_total, 1)
        acc = ep_correct / max(ep_total, 1)
        if verbose:
            print(f"[bc] epoch {ep+1}/{epochs}  loss={avg_loss:.4f}  acc={acc:.3f}  "
                  f"(N={N}, batches={n_batches})")
    policy.set_training_mode(False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", required=True,
                   help="단일 config(.yaml) 또는 멀티지역 매니페스트(.json)")
    p.add_argument("--bc_data", required=True,
                   help="bc_dataset.py 가 만든 pickle")
    p.add_argument("--bc_epochs", type=int, default=5)
    p.add_argument("--bc_batch_size", type=int, default=256)
    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_dir", default="results/rl/ppo_bc")
    p.add_argument("--n_steps", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--checkpoint_freq", type=int, default=20_000)
    p.add_argument("--vec", choices=["dummy", "subproc"], default="dummy")
    p.add_argument("--skip_bc", action="store_true",
                   help="BC 단계 건너뛰고 PPO 만 실행 (베이스라인 비교용)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    # CPU 강제 (드라이버 구식)
    device = "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "" else "auto"

    env_fns = [make_env_fn(args.config_path, seed=args.seed + i)
               for i in range(args.n_envs)]
    vec_cls = SubprocVecEnv if args.vec == "subproc" else DummyVecEnv
    venv = vec_cls(env_fns)

    model = MaskablePPO(
        "MlpPolicy", venv,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=args.seed,
        tensorboard_log=os.path.join(args.log_dir, "tb"),
        device=device,
    )

    # ----- BC pretrain -----
    if not args.skip_bc:
        dataset = load_bc_dataset(args.bc_data)
        obs_dim_env = int(venv.observation_space.shape[0])
        n_actions_env = int(venv.action_space.n)
        if dataset["obs_dim"] != obs_dim_env or dataset["n_actions"] != n_actions_env:
            raise ValueError(
                "BC 데이터셋과 env 차원 불일치 — "
                f"obs {dataset['obs_dim']} vs {obs_dim_env}, "
                f"act {dataset['n_actions']} vs {n_actions_env}.  "
                "동일 MCI_REDUCED_OBS 설정 / 동일 매니페스트로 다시 수집하세요."
            )
        print(f"[bc] 데이터셋 로드: N={dataset['obs'].shape[0]}  "
              f"obs_dim={dataset['obs_dim']}  n_actions={dataset['n_actions']}")
        bc_pretrain(model, dataset, epochs=args.bc_epochs,
                    batch_size=args.bc_batch_size, device=model.device)
        bc_path = os.path.join(args.log_dir, "bc_pretrained.zip")
        model.save(bc_path)
        print(f"[bc] BC pretrained 모델 저장: {bc_path}")
    else:
        print("[bc] --skip_bc → BC 단계 생략, PPO 만 실행")

    # ----- PPO fine-tune -----
    ckpt_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=os.path.join(args.log_dir, "checkpoints"),
        name_prefix="ppo_bc",
    )
    model.learn(total_timesteps=args.total_timesteps, callback=ckpt_cb,
                tb_log_name="ppo_bc", progress_bar=False)
    final_path = os.path.join(args.log_dir, "final_model.zip")
    model.save(final_path)
    print(f"Saved: {final_path}")

    # 짧은 평가
    eval_env = make_env_fn(args.config_path, seed=args.seed + 999)()
    mean_r, std_r = masked_evaluate(model, eval_env, n_eval_episodes=5, use_masking=True)
    print(f"Eval mean reward: {mean_r:.3f} +/- {std_r:.3f}")


if __name__ == "__main__":
    main()
