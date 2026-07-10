"""MaskablePPO 학습 스크립트 — Phase 3 HospitalFeatureWrapper 적용판.

train_ppo.py / train_ppo_enriched.py 의 파생본. env_wrapper.py / multi_region_env.py /
env_factory.py 는 수정하지 않고, 여기서만 HospitalFeatureWrapper 로 base env 를 감싼다.

차이점:
  * FlattenAndDiscreteWrapper 대신 HospitalFeatureWrapper (병원당 특징 엔티티 obs).
  * 정보수준 local/comms 는 env 변수 MCI_OBS_VARIANT 로 제어(래퍼 내장) — 인자 불필요.
  * 매니페스트(.json) 입력 시 각 지역 base env 를 HospitalFeatureWrapper 로 감싸는
    _FeatureMultiRegionEnv 자체 구현 사용 (multi_region_env.py 무수정).
  * --extractor {mlp,deepsets}: mlp(기본)=평탄 obs+MlpPolicy / deepsets=순열불변 인코더(3c).

주의: obs 차원이 기존과 달라 기존 가중치와 비호환 — 새로 학습할 것.
train/eval 시 MCI_OBS_VARIANT 를 동일하게 둘 것(obs 차원 일치).

예:
  python src/rl_src/train_ppo_feature.py \\
    --config_path scenarios/manifests/plan1nat_manifest.json --total_timesteps 200000 \\
    --n_envs 4 --log_dir results/rl/ppo_feature
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy as masked_evaluate
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from env_factory import make_base_env
from hospital_feature_wrapper import HospitalFeatureWrapper
from reward_redesign_wrapper import RewardRedesignWrapper
from learning_curve_plot import try_plot_learning_curve


# ---------- 매니페스트 → 멀티 지역 feature env (multi_region_env.py 무수정) ----------
class FeatureMultiRegionEnv(gym.Env):
    """MultiRegionEnv 의 HospitalFeatureWrapper 판. reset() 마다 무작위 지역 위임.

    전제: 모든 지역 H(병원 수) 동일(min_hos_num=H_max) — obs/action 차원 일치.
    """
    metadata = {"render_modes": []}

    def __init__(self, manifest_path: str, seed: int = 0, eval_mode: bool = False,
                 shard: "tuple[int, int] | None" = None,
                 weights_csv: "str | None" = None):
        super().__init__()
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        all_regions = list(manifest.keys())
        if not all_regions:
            raise ValueError(f"빈 manifest: {manifest_path}")

        # shard=(i,n): 워커 i 는 regions[i::n] 만 로드 — 대형(1000지역) 매니페스트의
        # 워커당 RSS 를 1/n 로 절감. None(기본)=전 지역 로드(기존 동작 불변).
        if shard is not None:
            si, sn = shard
            self.regions = all_regions[si::sn]
            if not self.regions:
                raise ValueError(f"shard {shard} 가 빈 지역 목록: 지역수 {len(all_regions)}")
        else:
            self.regions = all_regions

        self._envs = []
        for i, region in enumerate(self.regions):
            cfg = manifest[region]
            if not os.path.exists(cfg):
                raise FileNotFoundError(f"[{region}] config 미발견: {cfg}")
            base = make_base_env(cfg, seed=seed + i, rule_test=False, eval_mode=eval_mode)
            # 보상 변환(woG 등, 최내곽) → 그 위에 특징 obs 래퍼. info['r_woG'] 는 base 가 채움.
            self._envs.append(HospitalFeatureWrapper(RewardRedesignWrapper(base)))

        obs_shapes = {tuple(e.observation_space.shape) for e in self._envs}
        act_ns = {int(e.action_space.n) for e in self._envs}
        if len(obs_shapes) != 1 or len(act_ns) != 1:
            detail = "\n".join(f"  {r}: obs={e.observation_space.shape} act={e.action_space.n}"
                               for r, e in zip(self.regions, self._envs))
            raise ValueError("지역별 obs/action 차원 불일치 — min_hos_num 으로 재생성 필요.\n" + detail)

        self.observation_space = self._envs[0].observation_space
        self.action_space = self._envs[0].action_space
        # 3c 추출기용 엔티티 차원 노출
        self.n_hospitals = self._envs[0].H
        self.entity_f = self._envs[0]._F
        self.global_dim = self._envs[0]._flat_dim - self._envs[0].H * self._envs[0]._F
        # weights_csv(컬럼 region,weight): reset() 지역 샘플링을 균등 → 가중으로.
        # CSV 에 있는데 매니페스트에 없는 키는 에러(오타 침묵 방지). shard 시 shard 내 재정규화.
        self._p = None
        if weights_csv:
            w_by = {}
            with open(weights_csv, encoding="utf-8-sig") as f:  # 시군구 CSV 관례상 BOM 대응
                for row in csv.DictReader(f):
                    w_by[row["region"]] = float(row["weight"])
            unknown = sorted(set(w_by) - set(all_regions))
            if unknown:
                raise ValueError(f"weights_csv 에 매니페스트 밖 지역 키 {len(unknown)}개: "
                                 f"{unknown[:5]} ...")
            w = np.array([w_by.get(r, 0.0) for r in self.regions], dtype=np.float64)
            if (w < 0).any():
                raise ValueError("weights_csv 에 음수 가중치 존재")
            if w.sum() <= 0:
                raise ValueError(f"shard {shard} 내 가중치 합이 0 — CSV 커버리지 확인 필요")
            self._p = w / w.sum()

        self._rng = np.random.default_rng(seed)
        self._idx = 0
        self._cur = self._envs[0]

    @property
    def current_region(self) -> str:
        return self.regions[self._idx]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self._p is not None:
            self._idx = int(self._rng.choice(len(self._envs), p=self._p))  # 가중 샘플링
        else:
            self._idx = int(self._rng.integers(len(self._envs)))
        self._cur = self._envs[self._idx]
        return self._cur.reset(seed=seed, options=options)

    def step(self, action):
        return self._cur.step(action)

    def action_masks(self) -> np.ndarray:
        return self._cur.action_masks()

    @property
    def unwrapped(self):
        return self._cur.unwrapped

    def render(self):
        return None

    def close(self):
        for e in self._envs:
            e.close()


def mask_fn(env):
    return env.action_masks()


def make_env_fn(config_path: str, seed: int = 0, rank: int = 0, n_envs: int = 1,
                region_weights: "str | None" = None):
    """rank/n_envs: 매니페스트 지역수 > 500 일 때만 워커별 shard=(rank, n_envs) 활성
    (RSS 절감 훅). 기존 250 지역 매니페스트·단일 yaml 은 동작 완전 불변."""
    def _f():
        if config_path.endswith(".json"):
            with open(config_path, encoding="utf-8") as f:
                n_regions = len(json.load(f))
            shard = (rank, n_envs) if n_regions > 500 else None
            env = FeatureMultiRegionEnv(config_path, seed=seed, shard=shard,
                                        weights_csv=region_weights)
        else:
            base = make_base_env(config_path, seed=seed, rule_test=False, eval_mode=False)
            env = HospitalFeatureWrapper(RewardRedesignWrapper(base))
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    return _f


def _entity_dims(config_path: str, seed: int):
    """3c 추출기용 (H, F, global_dim) 산출 — probe env 1회 생성."""
    if config_path.endswith(".json"):
        e = FeatureMultiRegionEnv(config_path, seed=seed)
        dims = (e.n_hospitals, e.entity_f, e.global_dim)
        e.close()
        return dims
    base = make_base_env(config_path, seed=seed)
    w = HospitalFeatureWrapper(base)
    return (w.H, w._F, w._flat_dim - w.H * w._F)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", required=True)
    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_dir", default="results/rl/ppo_feature")
    p.add_argument("--n_steps", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--checkpoint_freq", type=int, default=20_000)
    p.add_argument("--vec", choices=["dummy", "subproc"], default="dummy")
    p.add_argument("--extractor", choices=["mlp", "deepsets", "pointer"], default="mlp",
                   help="mlp(기본): 평탄 obs+MlpPolicy / deepsets: 순열불변 인코더(3c) / "
                        "pointer: per-hospital 스코어링 head(pointer_policy, 랭킹 구조)")
    # ---- PPO 위생(플랜 v2 L1, 근거: docs/RL_재설계_설계노트_2026-07-04.md) ----
    p.add_argument("--lr_anneal", action="store_true", default=False,
                   help="learning_rate 를 진행률에 따라 →0 linear anneal(기본 off=고정 lr).")
    p.add_argument("--target_kl", type=float, default=None,
                   help="epoch 조기중단 KL 상한(권장 0.03). 미지정=SB3 기본(무제동).")
    p.add_argument("--n_epochs", type=int, default=None,
                   help="롤아웃 재사용 epoch 수(권장 4~6). 미지정=SB3 기본(10).")
    p.add_argument("--reward_mode", choices=["raw", "woG", "pdrwog", "rywt"], default="woG",
                   help="보상 변환(RewardRedesignWrapper). 기본 woG(Green 제외). "
                        "pdrwog=r_woG/preventable_woG(0~1 규모불변, --norm_reward 병용 권장).")
    p.add_argument("--norm_reward", action="store_true", default=False,
                   help="VecNormalize 보상 정규화(기본 off — woG 스케일 해석/휴리스틱 비교 유지).")
    p.add_argument("--resume_from", default=None,
                   help="기존 모델 디렉터리(또는 final_model.zip 경로). 주면 정책·옵티마이저·"
                        "num_timesteps·vecnormalize 통계를 복원해 이어학습(reset_num_timesteps=False). "
                        "이때 total_timesteps 는 '추가' 스텝 수(예: 5M→10M 이면 5_000_000).")
    # ---- 하이퍼 v3 (S1a): 할인/아키텍처 폭 스윕 ----
    p.add_argument("--gamma", type=float, default=0.99,
                   help="할인율(기본 0.99=SB3 기본). ⚠️VecNormalize 리턴 정규화에도 동기 전달됨.")
    p.add_argument("--gae_lambda", type=float, default=0.95,
                   help="GAE λ(기본 0.95=SB3 기본).")
    p.add_argument("--embed_dim", type=int, default=32,
                   help="병원 토큰 임베딩 폭(deepsets/pointer 추출기, 기본 32=구 아키텍처).")
    p.add_argument("--ctx_dim", type=int, default=64,
                   help="전역 ctx 폭(pointer 추출기 전용, 기본 64=구 아키텍처).")
    p.add_argument("--head_hidden", type=int, default=64,
                   help="PointerActionNet scorer 은닉폭(pointer 전용, 기본 64=구 아키텍처).")
    p.add_argument("--n_attn_blocks", type=int, default=1,
                   help="pointer 추출기 attention 블록 수(기본 1=구 아키텍처, ≥2 부터 "
                        "FFN 포함 블록 증축 — v4).")
    p.add_argument("--region_weights", default=None,
                   help="지역 샘플링 가중 CSV(컬럼 region,weight) — 매니페스트 학습 전용. "
                        "미지정(기본)=균등 샘플링(기존 동작).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    # RewardRedesignWrapper 는 MCI_REWARD_MODE 를 읽음 — CLI 값으로 강제(Subproc 자식에도 전파).
    os.environ["MCI_REWARD_MODE"] = args.reward_mode
    print(f"[feature] MCI_OBS_VARIANT={os.environ.get('MCI_OBS_VARIANT','(essential)')} "
          f"reward={args.reward_mode} norm_reward={args.norm_reward} extractor={args.extractor} "
          f"lr_anneal={args.lr_anneal} target_kl={args.target_kl} n_epochs={args.n_epochs} "
          f"gamma={args.gamma} gae_lambda={args.gae_lambda} "
          f"embed={args.embed_dim} ctx={args.ctx_dim} head_hidden={args.head_hidden} "
          f"n_attn_blocks={args.n_attn_blocks}")

    env_fns = [make_env_fn(args.config_path, seed=args.seed + i, rank=i, n_envs=args.n_envs,
                           region_weights=args.region_weights)
               for i in range(args.n_envs)]
    vec_cls = SubprocVecEnv if args.vec == "subproc" else DummyVecEnv
    venv = vec_cls(env_fns)

    # 추출기/정책 클래스는 (신규 정책생성 / resume 시 역직렬화) 양쪽에 import 되어 있어야 함.
    if args.extractor == "deepsets":
        from hospital_set_extractor import HospitalSetExtractor  # noqa: F401
    elif args.extractor == "pointer":
        from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401

    if args.resume_from:
        # ---- 이어학습: vecnorm 통계 + 정책/옵티마이저/num_timesteps 복원 ----
        model_zip = args.resume_from
        if os.path.isdir(model_zip):
            model_zip = os.path.join(model_zip, "final_model.zip")
        vn_path = os.path.join(os.path.dirname(model_zip), "vecnormalize.pkl")
        venv = VecNormalize.load(vn_path, venv)  # 동결 아님: training=True 로 obs 통계 계속 갱신
        venv.training = True
        venv.norm_reward = args.norm_reward
        model = MaskablePPO.load(model_zip, env=venv,
                                 tensorboard_log=os.path.join(args.log_dir, "tb"))
        print(f"[feature] resume from {model_zip}: num_timesteps={model.num_timesteps} "
              f"(+{args.total_timesteps} → {model.num_timesteps + args.total_timesteps})")
    else:
        # ---- 신규 학습 ----
        # obs 정규화 필수(ETA·cap_remain 스케일) / reward 정규화는 옵션. eval·VIPER 는 통계 동결 로드.
        # ⚠️gamma 동기화 필수: VecNormalize 의 리턴 추적(discounted return 분산)과 PPO 의
        # gamma 가 불일치하면 보상 정규화 스케일이 왜곡됨.
        venv = VecNormalize(venv, norm_obs=True, norm_reward=args.norm_reward, clip_obs=10.0,
                            gamma=args.gamma)

        policy_cls = "MlpPolicy"
        policy_kwargs = dict(net_arch=[256, 256])
        if args.extractor == "deepsets":
            H, F, gdim = _entity_dims(args.config_path, args.seed)
            policy_kwargs = dict(
                features_extractor_class=HospitalSetExtractor,
                features_extractor_kwargs=dict(n_hospitals=H, entity_f=F, global_dim=gdim,
                                               embed_dim=args.embed_dim),
                net_arch=[256, 256],
            )
            print(f"[feature] deepsets 추출기: H={H} F={F} global={gdim} embed={args.embed_dim}")
        elif args.extractor == "pointer":
            H, F, gdim = _entity_dims(args.config_path, args.seed)
            policy_cls = PointerMaskablePolicy  # net_arch 는 정책이 강제(pi=[], vf=[256,256])
            policy_kwargs = dict(
                features_extractor_class=HospitalTokenExtractor,
                features_extractor_kwargs=dict(n_hospitals=H, entity_f=F, global_dim=gdim,
                                               embed_dim=args.embed_dim, ctx_dim=args.ctx_dim,
                                               n_attn_blocks=args.n_attn_blocks),
                head_hidden=args.head_hidden,  # PointerMaskablePolicy.__init__ 로 전달
            )
            print(f"[feature] pointer 추출기+head: H={H} F={F} global={gdim} "
                  f"embed={args.embed_dim} ctx={args.ctx_dim} head_hidden={args.head_hidden} "
                  f"n_attn_blocks={args.n_attn_blocks}")

        # PPO 위생: lr anneal(진행률 p: 1→0 에 선형) / target_kl / n_epochs (미지정=SB3 기본)
        lr = (lambda p: args.learning_rate * p) if args.lr_anneal else args.learning_rate
        hygiene = {}
        if args.target_kl is not None:
            hygiene["target_kl"] = args.target_kl
        if args.n_epochs is not None:
            hygiene["n_epochs"] = args.n_epochs

        model = MaskablePPO(
            policy_cls, venv,
            learning_rate=lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            tensorboard_log=os.path.join(args.log_dir, "tb"),
            **hygiene,
        )

    ckpt_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=os.path.join(args.log_dir, "checkpoints"),
        name_prefix="ppo_feature",
    )

    # resume 시 reset_num_timesteps=False → total_timesteps 는 '추가' 스텝(이어서 카운트·체크포인트 번호 연속).
    model.learn(total_timesteps=args.total_timesteps, callback=ckpt_cb,
                tb_log_name="ppo_feature", progress_bar=False,
                reset_num_timesteps=(args.resume_from is None))
    final_path = os.path.join(args.log_dir, "final_model.zip")
    model.save(final_path)
    vecnorm_path = os.path.join(args.log_dir, "vecnormalize.pkl")
    venv.save(vecnorm_path)  # eval/VIPER 에서 VecNormalize.load 후 training=False 로 동결 적용 필수
    print(f"Saved: {final_path}\nSaved: {vecnorm_path}")
    try_plot_learning_curve(args.log_dir)

    eval_env = make_env_fn(args.config_path, seed=args.seed + 999)()
    mean_r, std_r = masked_evaluate(model, eval_env, n_eval_episodes=10, use_masking=True)
    print(f"Eval mean reward: {mean_r:.3f} +/- {std_r:.3f}")


if __name__ == "__main__":
    main()
