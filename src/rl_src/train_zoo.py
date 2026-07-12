"""v5 공정비교 하네스 — 통합 트레이너(4 알고리즘: dqn / qrdqn / sacd / reinforce).

계획 §3.1 공정성 프로토콜을 코드로 고정: 전 알고리즘이 **동일 env 스택**(챔피언 래퍼체인
+ MaskInfoWrapper) · **동일 obs(essential+load 355)** · **동일 action(Discrete 192)** ·
**하드 마스킹** · **공유 obs 정규화(VecNormalize norm_obs)** 로 학습한다. 보상은 off-policy/
REINFORCE 는 pdrwog raw(0~1 유계 → norm_reward=False; replay 통계 stale 회피).

env 조립(inner→outer):
  yaml : make_base_env → RewardRedesignWrapper(MCI_REWARD_MODE) → HospitalFeatureWrapper
  .json: FeatureMultiRegionEnv(내부에서 위 2래퍼 적용) ← train_ppo_feature 재사용
  공통 : … → MaskInfoWrapper → ActionMasker(e.action_masks()) → Monitor
         → Dummy/SubprocVecEnv → VecNormalize(norm_obs=True, norm_reward=False, clip_obs=10)

저장: {log_dir}/{final_model.zip|final_model.pt} + vecnormalize.pkl + checkpoints/ + tb/ + meta.json.
meta.json = 평가 하네스(paired_eval_ladder)가 algo 를 자동 감지하는 근거.

⚠️ 이 트레이너는 stdout 을 리다이렉트하지 않는다(런처 책임 — 레포 관례: sim debug print 폭주는
`>/dev/null 2>run.err` 로 감싼다). SubprocVecEnv 병렬 시 스레드 핀(OMP/MKL/OPENBLAS_NUM_THREADS=1)
도 런처가 export 한다(loadavg 폭증 방지).
⚠️ masked_dqn/masked_qrdqn/masked_sac_discrete 는 지연 import(algo 분기 내부) — 부재 시
reinforce 경로는 영향 없이 동작하고, off-policy 경로만 명확한 에러를 낸다.

예(런처가 감싸는 형태):
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
  MCI_OBS_VARIANT=essential+load MCI_CAP_GATE=occ \\
  python src/rl_src/train_zoo.py --algo reinforce \\
    --config_path scenarios/manifests/sigungu_osrm_manifest.json \\
    --total_timesteps 10000000 --seed 0 --n_envs 8 --vec subproc \\
    --log_dir results/rl/zoo/reinforce_s0  >/dev/null 2>results/rl/zoo/reinforce_s0.err
"""
import argparse
import json
import os
import subprocess
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(__file__))
# venv.env_method("action_masks") 의 래퍼 위임 접근 UserWarning(무해) 억제 — viper_distill 선례.
warnings.filterwarnings("ignore", message=r".*action_masks.*")

import gymnasium as gym  # noqa: F401 (env 조립 의존)
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import numpy as np
import torch as th

from env_factory import make_base_env
from hospital_feature_wrapper import HospitalFeatureWrapper
from reward_redesign_wrapper import RewardRedesignWrapper
from mask_info_wrapper import MaskInfoWrapper
from pointer_policy import HospitalTokenExtractor
from train_ppo_feature import FeatureMultiRegionEnv

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_MANIFEST = os.path.join(REPO, "scenarios/manifests/sigungu_osrm_manifest.json")


def _mask_fn(env):
    return env.action_masks()


def make_env_fn(config_path: str, seed: int, rank: int, n_envs: int):
    """챔피언 체인 + MaskInfoWrapper 를 씌운 env 팩토리(SubprocVecEnv cloudpickle 호환 클로저).

    reward_mode 는 MCI_REWARD_MODE(main 이 os.environ 설정, 자식 프로세스 상속) 로 전달 —
    train_ppo_feature 와 동일 관례(FeatureMultiRegionEnv 는 mode 인자를 받지 않음).
    """
    def _f():
        # subproc 워커 프로세스에도 위임 경고 억제 적용(필터는 프로세스별)
        warnings.filterwarnings("ignore", message=r".*action_masks.*")
        if config_path.endswith(".json"):
            with open(config_path, encoding="utf-8") as f:
                n_regions = len(json.load(f))
            shard = (rank, n_envs) if n_regions > 500 else None
            env = FeatureMultiRegionEnv(config_path, seed=seed, shard=shard)
        else:
            base = make_base_env(config_path, seed=seed, rule_test=False, eval_mode=False)
            env = HospitalFeatureWrapper(RewardRedesignWrapper(base))
        env = MaskInfoWrapper(env)                       # next-mask/dt 를 info 로 주입(off-policy용)
        env = ActionMasker(env, _mask_fn)                # 하드 마스킹(전 알고 공통)
        env = Monitor(env)
        return env
    return _f


def build_venv(args):
    env_fns = [make_env_fn(args.config_path, seed=args.seed + i, rank=i, n_envs=args.n_envs)
               for i in range(args.n_envs)]
    vec_cls = SubprocVecEnv if args.vec == "subproc" else DummyVecEnv
    venv = vec_cls(env_fns)
    # off-policy/REINFORCE: pdrwog 는 0~1 유계 → 보상 정규화 금지(replay 통계 stale 회피, 계획 §3.1).
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=args.gamma)
    return venv


def build_policy_kwargs(net: str, H: int, g: int) -> dict:
    """net=pointer: 공유 torso(HospitalTokenExtractor wide) + Q/actor head 는 net_arch=[256].
    net=mlp256: 평탄 obs + [256,256]."""
    if net == "pointer":
        return dict(
            features_extractor_class=HospitalTokenExtractor,
            features_extractor_kwargs=dict(n_hospitals=H, entity_f=7, global_dim=g,
                                           embed_dim=64, ctx_dim=128),
            net_arch=[256],
        )
    return dict(net_arch=[256, 256])


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _resolve_hypers(args):
    """algo 별 기본값 해소(계획 §3.3). 명시 플래그가 있으면 그 값을 사용."""
    lr = args.lr
    if lr is None:
        lr = 3e-4 if args.algo in ("sacd", "reinforce") else 1e-4
    train_freq = args.train_freq
    if train_freq is None:
        train_freq = 1 if args.algo == "sacd" else 4
    return float(lr), int(train_freq)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=["dqn", "qrdqn", "sacd", "reinforce"])
    p.add_argument("--config_path", default=DEFAULT_MANIFEST, help="yaml 또는 .json 매니페스트")
    p.add_argument("--total_timesteps", type=int, default=10_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--vec", choices=["dummy", "subproc"], default="subproc")
    p.add_argument("--net", choices=["pointer", "mlp256"], default="pointer")
    p.add_argument("--reward_mode", choices=["raw", "woG", "pdrwog", "rywt"], default="pdrwog")
    p.add_argument("--log_dir", default=None, help="기본 results/rl/zoo/<algo>_s<seed>")
    p.add_argument("--checkpoint_freq", type=int, default=500_000, help="env-step 단위")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda")
    p.add_argument("--dry_run", action="store_true",
                   help="env 조립 + 알고 클래스 import + 구성 kwargs 출력까지만(학습 없음). "
                        "off-policy 파일 부재 검증용.")
    # ---- 공유(off-policy) ----
    p.add_argument("--lr", type=float, default=None, help="미지정=algo 기본(dqn/qrdqn 1e-4, sacd 3e-4)")
    p.add_argument("--buffer_size", type=int, default=500_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_starts", type=int, default=50_000)
    p.add_argument("--train_freq", type=int, default=None, help="미지정=algo 기본(sacd 1, 그외 4)")
    p.add_argument("--gamma", type=float, default=0.99)
    # ---- dqn/qrdqn ----
    p.add_argument("--target_update", type=int, default=10_000)
    p.add_argument("--eps_fraction", type=float, default=0.1)
    p.add_argument("--final_eps", type=float, default=0.05)
    p.add_argument("--smdp", action="store_true", help="dqn 전용: γ^Δt SMDP 할인(buffer dt 훅)")
    # ---- qrdqn ----
    p.add_argument("--n_quantiles", type=int, default=50)
    # ---- sacd ----
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--target_entropy_coef", type=float, default=0.5)
    # ---- reinforce ----
    p.add_argument("--batch_episodes", type=int, default=16)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=10.0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.log_dir is None:
        args.log_dir = os.path.join(REPO, f"results/rl/zoo/{args.algo}_s{args.seed}")
    os.makedirs(args.log_dir, exist_ok=True)

    # 공정 프로토콜 불변식: obs=essential+load, gate=occ (외부 설정이 있으면 존중). reward_mode 는 강제.
    os.environ.setdefault("MCI_OBS_VARIANT", "essential+load")
    os.environ.setdefault("MCI_CAP_GATE", "occ")
    os.environ["MCI_REWARD_MODE"] = args.reward_mode  # 자식 프로세스(Subproc) 상속

    if args.device == "auto":
        dev = "cuda" if th.cuda.is_available() else "cpu"
    else:
        dev = args.device
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    lr, train_freq = _resolve_hypers(args)
    print(f"[zoo] algo={args.algo} obs={os.environ['MCI_OBS_VARIANT']} "
          f"gate={os.environ['MCI_CAP_GATE']} reward={args.reward_mode} net={args.net} "
          f"lr={lr} train_freq={train_freq} device={dev} seed={args.seed} "
          f"config={os.path.basename(args.config_path)}", flush=True)

    venv = build_venv(args)
    obs_dim = int(venv.observation_space.shape[0])
    A = int(venv.action_space.n)
    assert A % 4 == 0, f"action dim {A} 가 2*(H+1)*2 형식 아님(uav=0 구성은 zoo 미지원)"
    H = A // 4 - 1                       # A = 2*(H+1)*2
    g = obs_dim - H * 7                  # essential+load entity_f=7
    assert 2 * (H + 1) * 2 == A and H * 7 + g == obs_dim
    policy_kwargs = build_policy_kwargs(args.net, H, g)
    print(f"[zoo] obs_dim={obs_dim} A={A} H={H} g={g}", flush=True)

    # 알고별 하이퍼(meta.json·로그 기록용)
    if args.algo == "reinforce":
        hypers = dict(lr=lr, gamma=args.gamma, ent_coef=args.ent_coef,
                      batch_episodes=args.batch_episodes, max_grad_norm=args.max_grad_norm)
    elif args.algo in ("dqn", "qrdqn"):
        hypers = dict(lr=lr, buffer_size=args.buffer_size, batch_size=args.batch_size,
                      train_freq=train_freq, target_update=args.target_update,
                      eps_fraction=args.eps_fraction, final_eps=args.final_eps,
                      learning_starts=args.learning_starts, gamma=args.gamma)
        if args.algo == "qrdqn":
            hypers["n_quantiles"] = args.n_quantiles
        if args.algo == "dqn":
            hypers["smdp"] = bool(args.smdp)
    else:  # sacd
        hypers = dict(lr=lr, buffer_size=args.buffer_size, batch_size=args.batch_size,
                      train_freq=train_freq, tau=args.tau,
                      target_entropy_coef=args.target_entropy_coef,
                      learning_starts=args.learning_starts, gamma=args.gamma)

    tb_dir = os.path.join(args.log_dir, "tb")
    t0 = time.time()

    if args.algo == "reinforce":
        from reinforce_vec import ReinforceVec
        agent = ReinforceVec(obs_dim, A, net=args.net, H=H, entity_f=7, global_dim=g,
                             embed_dim=64, ctx_dim=128, lr=lr, gamma=args.gamma,
                             ent_coef=args.ent_coef, max_grad_norm=args.max_grad_norm, device=dev)
        if args.dry_run:
            print(f"[zoo:dry_run] ReinforceVec 구성 완료 net={args.net} "
                  f"batch_episodes={args.batch_episodes} — 학습 생략", flush=True)
            venv.close()
            return
        stats = agent.train(venv, total_timesteps=args.total_timesteps,
                            batch_episodes=args.batch_episodes, lr=lr, ent_coef=args.ent_coef,
                            gamma=args.gamma, max_grad_norm=args.max_grad_norm,
                            log_dir=args.log_dir, checkpoint_freq=args.checkpoint_freq)
        final_path = os.path.join(args.log_dir, "final_model.pt")
        agent.save(final_path)
        print(f"[zoo] reinforce updates={stats['n_updates']} "
              f"num_ts={stats['num_timesteps']} mask_checked={stats['n_mask_checked']}", flush=True)
    else:
        # ---- off-policy(dqn/qrdqn/sacd): 클래스 지연 import ----
        try:
            if args.algo == "dqn":
                from masked_dqn import MaskedDQN as Algo
            elif args.algo == "qrdqn":
                from masked_qrdqn import MaskedQRDQN as Algo
            else:
                from masked_sac_discrete import SACDiscrete as Algo
        except ImportError as e:
            raise SystemExit(
                f"[zoo] '{args.algo}' 알고 클래스 import 실패: {e}\n"
                f"  → masked_{ {'dqn':'dqn','qrdqn':'qrdqn','sacd':'sac_discrete'}[args.algo] }.py "
                f"가 아직 없다(병렬 작성 중). reinforce 경로는 독립 동작한다.")

        # 공통 SB3 스타일 kwargs(계획 §3.3). ⚠️각 클래스 __init__ 시그니처는 병렬 작성물의 계약 —
        # 아래 키가 다르면 그 클래스에서 조정(통합 지점, 최종 보고에 명기).
        common = dict(
            learning_rate=lr, buffer_size=args.buffer_size, batch_size=args.batch_size,
            learning_starts=args.learning_starts, gamma=args.gamma, train_freq=train_freq,
            gradient_steps=1, policy_kwargs=policy_kwargs, verbose=1, seed=args.seed,
            tensorboard_log=tb_dir, device=dev,
        )
        if args.algo in ("dqn", "qrdqn"):
            common.update(target_update_interval=args.target_update,
                          exploration_fraction=args.eps_fraction,
                          exploration_final_eps=args.final_eps)
            if args.algo == "qrdqn":
                pk = dict(policy_kwargs); pk["n_quantiles"] = args.n_quantiles
                common["policy_kwargs"] = pk
            if args.algo == "dqn" and args.smdp:
                common["smdp"] = True
        else:  # sacd
            common.update(tau=args.tau, target_entropy_coef=args.target_entropy_coef)

        if args.dry_run:
            printable = {k: (v if not callable(v) else str(v)) for k, v in common.items()}
            printable["policy_kwargs"] = {kk: (str(vv) if callable(vv) else vv)
                                          for kk, vv in printable["policy_kwargs"].items()}
            print(f"[zoo:dry_run] {Algo.__module__}.{Algo.__name__} import OK. 구성 kwargs:\n"
                  f"  {json.dumps(printable, ensure_ascii=False, default=str)}\n"
                  f"  — 학습 생략(통합 스모크는 메인 수행)", flush=True)
            venv.close()
            return

        model = Algo("MlpPolicy", venv, **common)
        ckpt_cb = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=os.path.join(args.log_dir, "checkpoints"), name_prefix=args.algo)
        model.learn(total_timesteps=args.total_timesteps, callback=ckpt_cb,
                    tb_log_name=args.algo, progress_bar=False)
        final_path = os.path.join(args.log_dir, "final_model.zip")
        model.save(final_path)

    # ---- 공통 저장: vecnormalize.pkl + meta.json ----
    venv.save(os.path.join(args.log_dir, "vecnormalize.pkl"))
    meta = {
        "algo": args.algo,
        "obs_variant": os.environ.get("MCI_OBS_VARIANT"),
        "reward_mode": args.reward_mode,
        "cap_gate": os.environ.get("MCI_CAP_GATE"),
        "net": args.net,
        "hypers": hypers,
        "manifest": args.config_path,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "obs_dim": obs_dim, "n_actions": A, "H": H, "global_dim": g,
        "git_sha": _git_sha(),
    }
    with open(os.path.join(args.log_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[zoo] saved {args.log_dir} (final_model + vecnormalize.pkl + meta.json)", flush=True)
    print(f"[zoo] {args.algo} 완료: {time.time() - t0:.0f}s "
          f"({args.total_timesteps} steps)", file=sys.stderr)


if __name__ == "__main__":
    main()
