"""추가 SB3 알고리즘 학습 (피드백 3 후속 — 알고리즘 확장 실험).

기존 3종(PPO=MaskablePPO / DQN / REINFORCE) 외에 stable-baselines3 계열
4종을 같은 환경·obs(MCI_REDUCED_OBS)에서 학습한다:

  a2c          stable_baselines3.A2C       — 경량 on-policy actor-critic
  trpo         sb3_contrib.TRPO            — 신뢰영역 정책 최적화
  qrdqn        sb3_contrib.QRDQN          — 분포형 DQN (DQN 개선 후보)
  recurrentppo sb3_contrib.RecurrentPPO    — LSTM 정책 (부분관측 대응)

이 4종은 sb3-contrib 의 action masking 을 지원하지 않는다. MCIEnvironment 는
없는 자원에 대한 dispatch 를 no-op 으로 처리하므로(env_wrapper 주석 참조)
DQN 과 동일하게 마스크 없이 학습 가능하다.

config_path 가 .json 매니페스트면 MultiRegionEnv(전국 단일 정책)로 분기한다.

예:
  MCI_REDUCED_OBS=1 python src/rl_src/train_sb3_extra.py --algo a2c \
    --config_path scenarios/manifests/plan1nat_manifest.json \
    --total_timesteps 500000 --log_dir results/rl/.../a2c
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env_factory import make_base_env
from env_wrapper import FlattenAndDiscreteWrapper
from learning_curve_plot import try_plot_learning_curve

# algo -> (모델 클래스, 정책 문자열, on-policy 여부)
_ALGOS = {
    "a2c":          ("stable_baselines3:A2C",    "MlpPolicy",     True),
    "trpo":         ("sb3_contrib:TRPO",         "MlpPolicy",     True),
    "qrdqn":        ("sb3_contrib:QRDQN",        "MlpPolicy",     False),
    "recurrentppo": ("sb3_contrib:RecurrentPPO", "MlpLstmPolicy", True),
}


def _resolve(spec):
    mod, cls = spec.split(":")
    return getattr(__import__(mod, fromlist=[cls]), cls)


def make_env_fn(config_path: str, seed: int = 0):
    def _f():
        if config_path.endswith(".json"):
            from multi_region_env import MultiRegionEnv
            env = MultiRegionEnv(config_path, seed=seed)
        else:
            base = make_base_env(config_path, seed=seed,
                                 rule_test=False, eval_mode=False)
            env = FlattenAndDiscreteWrapper(base)
        return Monitor(env)
    return _f


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=list(_ALGOS))
    p.add_argument("--config_path", required=True)
    p.add_argument("--total_timesteps", type=int, default=500_000)
    p.add_argument("--n_envs", type=int, default=4,
                   help="on-policy(a2c/trpo/recurrentppo) VecEnv 환경 수")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_dir", default="results/rl/sb3_extra")
    p.add_argument("--learning_rate", type=float, default=None,
                   help="미지정 시 알고리즘 기본값")
    p.add_argument("--n_quantiles", type=int, default=200,
                   help="qrdqn 전용 — 분위수 개수. SB3 기본 200 은 행동공간이 크면"
                        "(출력층 = n_actions×n_quantiles) CPU 에서 매우 느리다. "
                        "큰 행동공간엔 50 권장.")
    p.add_argument("--checkpoint_freq", type=int, default=50_000)
    p.add_argument("--vec", choices=["dummy", "subproc"], default="dummy")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    spec, policy, on_policy = _ALGOS[args.algo]
    Model = _resolve(spec)

    if on_policy:
        env_fns = [make_env_fn(args.config_path, seed=args.seed + i)
                   for i in range(args.n_envs)]
        vec_cls = SubprocVecEnv if args.vec == "subproc" else DummyVecEnv
        env = vec_cls(env_fns)
    else:
        # off-policy(qrdqn): 단일 env
        env = DummyVecEnv([make_env_fn(args.config_path, seed=args.seed)])

    # net_arch — recurrentppo 는 LSTM 정책이라 기본 구조 유지
    kwargs = dict(verbose=1, seed=args.seed,
                  tensorboard_log=os.path.join(args.log_dir, "tb"))
    if args.algo != "recurrentppo":
        kwargs["policy_kwargs"] = dict(net_arch=[256, 256])
    if args.algo == "qrdqn":
        # 출력층 = n_actions×n_quantiles → 행동공간이 크면 200 은 과중. 축소.
        kwargs["policy_kwargs"]["n_quantiles"] = args.n_quantiles
    if args.learning_rate is not None:
        kwargs["learning_rate"] = args.learning_rate

    model = Model(policy, env, **kwargs)
    print(f"[{args.algo}] policy={policy}  on_policy={on_policy}  "
          f"timesteps={args.total_timesteps}  log_dir={args.log_dir}")

    ckpt_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // (args.n_envs if on_policy else 1), 1),
        save_path=os.path.join(args.log_dir, "checkpoints"),
        name_prefix=args.algo)

    model.learn(total_timesteps=args.total_timesteps, callback=ckpt_cb,
                tb_log_name=args.algo, progress_bar=False)
    final_path = os.path.join(args.log_dir, "final_model.zip")
    model.save(final_path)
    print(f"Saved: {final_path}")
    try_plot_learning_curve(args.log_dir)


if __name__ == "__main__":
    main()
