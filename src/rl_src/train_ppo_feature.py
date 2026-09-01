"""MaskablePPO 학습 스크립트 — Phase 3 HospitalFeatureWrapper 적용판.

train_ppo.py / train_ppo_enriched.py 의 파생본. env_wrapper.py / multi_region_env.py /
env_factory.py 는 수정하지 않고, 여기서만 HospitalFeatureWrapper 로 base env 를 감싼다.

차이점:
  * FlattenAndDiscreteWrapper 대신 HospitalFeatureWrapper (병원당 특징 엔티티 obs).
  * 정보수준 local/comms 는 env 변수 MCI_OBS_VARIANT 로 제어(래퍼 내장) — 인자 불필요.
  * 매니페스트(.json) 입력 시 각 지역 base env 를 HospitalFeatureWrapper 로 감싸는
    _FeatureMultiRegionEnv 자체 구현 사용 (multi_region_env.py 무수정).
  * --extractor pointer_rescm/pointer_resrank{1,2}: 기준 Pointer를 정확히 포함하는
    중증도×수단/저랭크 3원 residual head. --init_from 으로 기존 정책에서 안전하게 시작.

주의: obs 차원이 기존과 달라 기존 가중치와 비호환 — 새로 학습할 것.
train/eval 시 MCI_OBS_VARIANT 를 동일하게 둘 것(obs 차원 일치).

예:
  MCI_OBS_VARIANT=essential+load+valid MCI_H_PAD=47 MCI_CAP_GATE=occ \\
  python src/rl_src/train_ppo_feature.py \\
    --config_path scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json \\
    --extractor pointer --reward_mode pdrwog --norm_reward \\
    --n_envs 8 --vec subproc --total_timesteps 10000000 \\
    --log_dir results/rl/redesign/v10_random4_1000_pointer_s0
"""
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy as masked_evaluate
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from env_factory import make_base_env
from hospital_feature_wrapper import HospitalFeatureWrapper, _parse_variant
from reward_redesign_wrapper import RewardRedesignWrapper
from learning_curve_plot import try_plot_learning_curve
import pad_vecnorm  # noqa: F401 — PadAware VecNormalize(신규 학습 생성 + resume pickle 로드용)


# ---------- 매니페스트 → 멀티 지역 feature env (multi_region_env.py 무수정) ----------
# (v6 A5) 매니페스트 값 overrides 화이트리스트 — 자원·규모 노브. 네 노브 모두 sim 의
# ScenarioManager.__init__(=make_base_env 내부)에서만 1회 소비되므로, 빌드 직전 env 에
# 임시 주입하면 같은 config 로 규모·자원 변형 env 를 만들 수 있다(reset 마다 지역×변형 무작위
# 샘플 = 도메인 랜덤화). amb=0/uav=0 은 action(mode 축) 차원이 갈리므로 ≥1 강제. 그 외 키는
# 오타 침묵 방지로 명시 에러.
_OVERRIDE_ENV_KEYS = ("MCI_INCIDENT_SIZE", "MCI_AMB_NUM", "MCI_UAV_NUM", "MCI_CAPA_SCALE")
_RANDOM4_MANIFEST = "sigungu_osrm_train1000_random4_manifest.json"
_REPRESENTATIVE_MANIFEST = "sigungu_osrm_eval250_representative_manifest.json"
_RANDOM4_KEY_RE = re.compile(r"_(\d{5})_p([0-3])$")
_COORD_RE = re.compile(r"\(([-\d.]+),([-\d.]+)\)")


def _manifest_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _manifest_coords(manifest: dict) -> set:
    coords = set()
    for entry in manifest.values():
        path, _ = _parse_manifest_entry("(검증)", entry)
        match = _COORD_RE.search(path)
        if match is None:
            raise ValueError(f"매니페스트 경로에서 좌표 파싱 실패: {path}")
        coords.add((float(match.group(1)), float(match.group(2))))
    return coords


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _manifest_sampling_summary(manifest_path: str) -> str:
    """학습 시작 전에 멀티지역 reset 표본 구성을 요약하고 random4 정본은 엄격 검증한다."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    keys = list(manifest)
    if not keys:
        raise ValueError(f"빈 manifest: {manifest_path}")

    groups = {}
    unmatched = []
    for key in keys:
        match = _RANDOM4_KEY_RE.search(key)
        if match is None:
            unmatched.append(key)
            continue
        groups.setdefault(match.group(1), set()).add(int(match.group(2)))

    is_random4 = os.path.basename(manifest_path) == _RANDOM4_MANIFEST
    if is_random4:
        incomplete = {sigcd: sorted(pidx) for sigcd, pidx in groups.items()
                      if pidx != {0, 1, 2, 3}}
        if unmatched or len(keys) != 1000 or len(groups) != 250 or incomplete:
            raise ValueError(
                f"random4 학습 매니페스트 구조 오류: entries={len(keys)}, "
                f"sigungu={len(groups)}, unmatched={len(unmatched)}, "
                f"incomplete={list(incomplete.items())[:5]}")
        # 대표점 250은 최종 평가 전용이다. 키뿐 아니라 실제 config 경로와 경로 속 좌표까지
        # 교집합 0을 학습 시작 때 봉인해, 파일 조립 실수로 평가점이 학습에 섞이는 것을 막는다.
        eval_path = os.path.join(os.path.dirname(os.path.abspath(manifest_path)),
                                 _REPRESENTATIVE_MANIFEST)
        if not os.path.exists(eval_path):
            raise FileNotFoundError(f"평가 대표점 매니페스트 미발견: {eval_path}")
        with open(eval_path, encoding="utf-8") as f:
            eval_manifest = json.load(f)
        overlap_key = set(manifest) & set(eval_manifest)
        train_paths = {_parse_manifest_entry(k, v)[0] for k, v in manifest.items()}
        eval_paths = {_parse_manifest_entry(k, v)[0] for k, v in eval_manifest.items()}
        overlap_path = train_paths & eval_paths
        overlap_coord = _manifest_coords(manifest) & _manifest_coords(eval_manifest)
        if overlap_key or overlap_path or overlap_coord:
            raise ValueError(
                "학습 random4에 평가 대표점 혼입: "
                f"key={len(overlap_key)} path={len(overlap_path)} coord={len(overlap_coord)}")
        return "시군구 250 × 좌표 4 = 1,000개, reset마다 균등 무작위 샘플링"
    return f"지역/시나리오 {len(keys)}개, reset마다 균등 무작위 샘플링"


def _parse_manifest_entry(region: str, entry):
    """매니페스트 값 → (config 경로:str, overrides:dict[str,str]).

    구 스키마: entry=str(경로) → overrides={} (코드 경로 완전 불변).
    신 스키마: entry=dict{"path":str, "overrides":dict[str, str|int|float]} → 빌드시 env 주입.
    overrides 키는 화이트리스트만 허용(오타 침묵 방지), 값은 env 문자열로 정규화(ScenarioManager
    가 int/float 로 재파싱). MCI_AMB_NUM/MCI_UAV_NUM 은 ≥1 검증(action 차원 보존).
    """
    if isinstance(entry, str):
        return entry, {}
    if not isinstance(entry, dict):
        raise ValueError(f"[{region}] 매니페스트 값은 str|dict 여야 함 (got {type(entry).__name__})")
    if "path" not in entry:
        raise ValueError(f"[{region}] dict 엔트리에 'path' 키 없음: {entry!r}")
    raw = entry.get("overrides", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"[{region}] 'overrides' 는 dict 여야 함 (got {type(raw).__name__})")
    overrides = {}
    for k, v in raw.items():
        if k not in _OVERRIDE_ENV_KEYS:
            raise ValueError(f"[{region}] 허용 밖 overrides 키 {k!r} — 허용: "
                             f"{_OVERRIDE_ENV_KEYS} (오타 확인)")
        overrides[k] = str(v)  # env 는 문자열; ScenarioManager 가 int/float 로 파싱
    for k in ("MCI_AMB_NUM", "MCI_UAV_NUM"):
        if k in overrides and int(overrides[k]) < 1:
            raise ValueError(f"[{region}] {k}={overrides[k]} <1 금지 — "
                             f"amb=0/uav=0 은 action 차원(96) 이 갈림")
    return entry["path"], overrides


def _with_env_overrides(overrides: dict) -> dict:
    """overrides 를 os.environ 에 적용하고 원상복구 스냅샷(키→이전값|None)을 반환.
    순차 빌드(프로세스-로컬)라 안전. None=원래 없던 키(복구 시 삭제)."""
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    return saved


def _restore_env_overrides(saved: dict) -> None:
    for k, old in saved.items():
        if old is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = old


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
            cfg_path, overrides = _parse_manifest_entry(region, manifest[region])
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"[{region}] config 미발견: {cfg_path}")
            # (v6 A5) 자원·규모 노브는 make_base_env 내부(ScenarioManager.__init__)에서만 소비 →
            # 빌드 직전 env 주입 후 finally 원복(overrides 없으면 무동작 = 구 경로 불변).
            saved = _with_env_overrides(overrides)
            try:
                base = make_base_env(cfg_path, seed=seed + i, rule_test=False, eval_mode=eval_mode)
            finally:
                _restore_env_overrides(saved)
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


# 워커 샤딩 최소 지역수. 기본 501 = 구 동작(`n_regions > 500`). v18 시군구 24좌표 학습은
# `MCI_SHARD_MIN_REGIONS=2` 로 켜서 워커당 env 를 24→3 으로 줄인다(합집합은 24 불변).
_SHARD_MIN = int(os.environ.get("MCI_SHARD_MIN_REGIONS", "501"))


def make_env_fn(config_path: str, seed: int = 0, rank: int = 0, n_envs: int = 1,
                region_weights: "str | None" = None, shard_min_regions: int = 501):
    """rank/n_envs: 매니페스트 지역수 >= shard_min_regions 일 때 워커별 shard=(rank, n_envs).

    기본 501 = 구 동작(`n_regions > 500`)과 **완전 동일**. 기존 250/1000 지역 매니페스트·
    단일 yaml 경로는 불변.

    ⚠️v18 실측: 소형 매니페스트(시군구 24좌표)는 샤딩이 안 걸려 **워커 8개가 각각 24 env 를
    전부 빌드**한다(런당 192 env). 동시 118런이면 22,656 env 가 L3 를 두들겨 총 처리량이
    64런(4,352 steps/s)보다 오히려 낮아졌다(3,186). 샤딩하면 워커당 3 env 로 8배 줄고
    **워커 합집합은 여전히 24 전량**이라 커버리지가 같다 — `MCI_SHARD_MIN_REGIONS=2` 로 활성.
    """
    def _f():
        if config_path.endswith(".json"):
            with open(config_path, encoding="utf-8") as f:
                n_regions = len(json.load(f))
            shard = (rank, n_envs) if n_regions >= shard_min_regions else None
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
        # 차원은 전 지역 생성 시 이미 각 shard 내부에서 검증한다. 여기서는 대형 매니페스트
        # 1,000개를 중복 로드하지 않고 첫 엔트리 하나만 probe한다.
        with open(config_path, encoding="utf-8") as f:
            n_regions = len(json.load(f))
        e = FeatureMultiRegionEnv(config_path, seed=seed, shard=(0, n_regions))
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
    p.add_argument("--save_vecnormalize", action="store_true",
                   help="체크포인트마다 VecNormalize 통계도 저장(중간 체크포인트 평가용). "
                        "기본 False = 구 동작")
    p.add_argument("--vec", choices=["dummy", "subproc"], default="dummy")
    p.add_argument("--extractor",
                   choices=["mlp", "deepsets", "pointer", "pointer_joint3",
                            "pointer_rescm", "pointer_resrank1", "pointer_resrank2",
                            "gopt_bilinear"],
                   default="mlp",
                   help="mlp(기본): 평탄 obs+MlpPolicy / deepsets: 순열불변 인코더(3c) / "
                        "pointer: destination×mode 병원 랭킹(기준선) / pointer_joint3: "
                        "완전 3원 head / pointer_rescm: 기준+class×mode 잔차 / "
                        "pointer_resrank1,2: 기준+저랭크 3원 잔차 / "
                        "gopt_bilinear: GOPT식 수요(class×mode)토큰 × 목적지토큰 bilinear 채점 "
                        "(--n_gopt_blocks 로 크로스어텐션 증축 — v12)")
    # ---- PPO 위생(플랜 v2 L1, 근거: docs/RL_재설계_설계노트_2026-07-04.md) ----
    p.add_argument("--lr_anneal", action="store_true", default=False,
                   help="learning_rate 를 진행률에 따라 →0 linear anneal(기본 off=고정 lr).")
    p.add_argument("--target_kl", type=float, default=None,
                   help="epoch 조기중단 KL 상한(권장 0.03). 미지정=SB3 기본(무제동).")
    p.add_argument("--n_epochs", type=int, default=None,
                   help="롤아웃 재사용 epoch 수(권장 4~6). 미지정=SB3 기본(10).")
    p.add_argument("--reward_mode", choices=["raw", "woG", "pdrwog", "rywt", "pdrwog_da"], default="woG",
                   help="보상 변환(RewardRedesignWrapper). 기본 woG(Green 제외). "
                        "pdrwog=r_woG/preventable_woG(0~1 규모불변, --norm_reward 병용 권장). "
                        "pdrwog_da=결정귀속 재배치(합 보존, v5 P2).")
    p.add_argument("--norm_reward", action="store_true", default=False,
                   help="VecNormalize 보상 정규화(기본 off — woG 스케일 해석/휴리스틱 비교 유지).")
    p.add_argument("--resume_from", default=None,
                   help="기존 모델 디렉터리(또는 final_model.zip 경로). 주면 정책·옵티마이저·"
                        "num_timesteps·vecnormalize 통계를 복원해 이어학습(reset_num_timesteps=False). "
                        "이때 total_timesteps 는 '추가' 스텝 수(예: 5M→10M 이면 5_000_000).")
    p.add_argument("--init_from", default=None,
                   help="기존 모델 디렉터리(또는 zip)의 정책·vecnormalize만 새 모델에 이식. "
                        "optimizer/학습률/step은 새로 시작해 residual warm-start와 동일 예산 "
                        "control에 사용(--resume_from과 상호배타).")
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
                        "FFN 포함 블록 증축 — v4). 0 = 토큰 혼합 제거(deep-sets 하한, "
                        "attention 기여도 측정용 — v12).")
    # ---- v12 (GOPT 계열): 기본값이면 전부 기존 경로와 비트 동일 ----
    p.add_argument("--n_heads", type=int, default=4,
                   help="어텐션 헤드 수(기본 4=구 아키텍처).")
    p.add_argument("--n_gopt_blocks", type=int, default=0,
                   help="gopt_bilinear 전용: 수요↔목적지 크로스어텐션 블록 수"
                        "(self×2+cross×2/블록). 기본 0=인코더는 v10 그대로, head 만 bilinear.")
    p.add_argument("--ff_expansion", type=int, default=4,
                   help="GoptEncoderBlock FFN 확장배수(기본 4=GOPT 원문).")
    p.add_argument("--attn_dropout", type=float, default=0.0,
                   help="GoptEncoderBlock dropout(기본 0=GOPT 원문 기본).")
    p.add_argument("--pooled_critic", action="store_true", default=False,
                   help="critic 을 GOPT식 순열불변 합풀링 MLP 로 교체(기본 off=SB3 vf[256,256]). "
                        "⚠️ 고정 H 전제(패딩 마스크 미사용) — v12 X6 격리 팔.")
    p.add_argument("--region_weights", default=None,
                   help="지역 샘플링 가중 CSV(컬럼 region,weight) — 매니페스트 학습 전용. "
                        "미지정(기본)=균등 샘플링(기존 동작).")
    return p.parse_args()


def _write_run_meta(args, model, status: str) -> None:
    """결과 폴더만 보고도 scoreboard 방법·데이터·head를 식별할 수 있는 실행 명세 저장."""
    manifest_abs = os.path.abspath(args.config_path)
    manifest_name = os.path.basename(manifest_abs)
    is_random4 = manifest_name == _RANDOM4_MANIFEST
    fx = model.policy.features_extractor
    head = type(model.policy.action_net).__name__
    policy = type(model.policy).__name__
    extractor = type(fx).__name__
    fresh_start = args.resume_from is None and args.init_from is None
    # 정본 PPO_POINTER_V10 행은 v10 프로토콜이 고정한 **단일 구성**에만 부여한다
    # (기준 Pointer·attention 1블록·기준폭 64/128/128·seed 0). 아키텍처·시드 변형이
    # 같은 method_id 를 자칭하면 scoreboard 빌드가 정본 PPO 행으로 오인한다.
    _canon = {"extractor": "pointer", "n_attn_blocks": 1, "embed_dim": 64,
              "ctx_dim": 128, "head_hidden": 128, "seed": 0,
              "n_heads": 4, "n_gopt_blocks": 0, "pooled_critic": False}
    off_spec = [k for k, v in _canon.items() if getattr(args, k) != v]
    method_id = "PPO_POINTER_V10" if is_random4 and not off_spec else None
    meta = {
        "schema_version": 1,
        "status": status,
        "run_id": os.path.basename(os.path.normpath(args.log_dir)),
        "scoreboard_protocol": "v10_random4_train__representative250_eval",
        "scoreboard_method_id": method_id,
        "scoreboard_eligible": bool(method_id and fresh_start),
        # 정본 구성에서 벗어난 인자 목록(빈 리스트=정본). 변형 런의 provenance 근거.
        "scoreboard_off_spec_args": off_spec,
        "algorithm": "MaskablePPO",
        "extractor_arg": args.extractor,
        "policy_class": policy,
        "action_head_class": head,
        "feature_extractor_class": extractor,
        "pointer_spec": (
            "L(c,d,m)=f_class(c|ctx)+S(d,m|hospital,ctx)+g_mode(m|ctx)"
            if head == "PointerActionNet" else None),
        "action_structure": "[class,destination,mode] joint Discrete categorical",
        "train_manifest": manifest_abs,
        "train_manifest_sha256": _manifest_sha256(manifest_abs),
        "train_dataset_role": (
            "TRAIN_ONLY_RANDOM4_1000" if is_random4 else "OTHER_OR_LEGACY"),
        "eval_manifest": os.path.join(
            os.path.dirname(manifest_abs), _REPRESENTATIVE_MANIFEST),
        "eval_dataset_role": "FINAL_EVAL_ONLY_REPRESENTATIVE250",
        "train_eval_key_path_coord_overlap": [0, 0, 0] if is_random4 else None,
        "fresh_start": fresh_start,
        "resume_from": args.resume_from,
        "init_from": args.init_from,
        "seed": args.seed,
        "total_timesteps_requested": args.total_timesteps,
        "num_timesteps_current": int(model.num_timesteps),
        "obs_variant": os.environ.get("MCI_OBS_VARIANT", "essential"),
        "h_pad": os.environ.get("MCI_H_PAD"),
        "cap_gate": os.environ.get("MCI_CAP_GATE", "occ"),
        "reward_mode": args.reward_mode,
        "norm_reward": args.norm_reward,
        "n_envs": args.n_envs,
        "vec": args.vec,
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "lr_anneal": args.lr_anneal,
            "target_kl": args.target_kl,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "ent_coef": args.ent_coef,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "embed_dim": args.embed_dim,
            "ctx_dim": args.ctx_dim,
            "head_hidden": args.head_hidden,
            "n_attn_blocks": args.n_attn_blocks,
            # v12 (GOPT 계열) — 기본값이면 기존 경로와 동일
            "n_heads": args.n_heads,
            "n_gopt_blocks": args.n_gopt_blocks,
            "ff_expansion": args.ff_expansion,
            "attn_dropout": args.attn_dropout,
            "pooled_critic": args.pooled_critic,
        },
        # 학습 디바이스(저널 재현성 — 구 run 은 이 키가 없다)
        "device": str(getattr(model, "device", "unknown")),
        "n_parameters": int(sum(p.numel() for p in model.policy.parameters())),
        "obs_dim": int(model.observation_space.shape[0]),
        "n_actions": int(model.action_space.n),
        "n_hospitals": int(getattr(fx, "H", -1)),
        "entity_features": int(getattr(fx, "F", -1)),
        "git_sha": _git_sha(),
        "argv": sys.argv,
    }
    out = os.path.join(args.log_dir, "meta.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[feature] run metadata: {out} ({status}, head={head})")


def main():
    args = parse_args()
    if args.resume_from and args.init_from:
        raise ValueError("--resume_from과 --init_from은 동시에 사용할 수 없음")
    os.makedirs(args.log_dir, exist_ok=True)
    if args.config_path.endswith(".json"):
        print(f"[feature] manifest: {_manifest_sampling_summary(args.config_path)}")
    # RewardRedesignWrapper 는 MCI_REWARD_MODE 를 읽음 — CLI 값으로 강제(Subproc 자식에도 전파).
    os.environ["MCI_REWARD_MODE"] = args.reward_mode
    print(f"[feature] MCI_OBS_VARIANT={os.environ.get('MCI_OBS_VARIANT','(essential)')} "
          f"reward={args.reward_mode} norm_reward={args.norm_reward} extractor={args.extractor} "
          f"lr_anneal={args.lr_anneal} target_kl={args.target_kl} n_epochs={args.n_epochs} "
          f"gamma={args.gamma} gae_lambda={args.gae_lambda} "
          f"embed={args.embed_dim} ctx={args.ctx_dim} head_hidden={args.head_hidden} "
          f"n_attn_blocks={args.n_attn_blocks}")

    env_fns = [make_env_fn(args.config_path, seed=args.seed + i, rank=i, n_envs=args.n_envs,
                           shard_min_regions=_SHARD_MIN, region_weights=args.region_weights)
               for i in range(args.n_envs)]
    vec_cls = SubprocVecEnv if args.vec == "subproc" else DummyVecEnv
    venv = vec_cls(env_fns)

    # 추출기/정책 클래스는 (신규 정책생성 / resume 시 역직렬화) 양쪽에 import 되어 있어야 함.
    if args.extractor == "deepsets":
        from hospital_set_extractor import HospitalSetExtractor  # noqa: F401
    elif args.extractor.startswith("pointer"):
        from pointer_policy import (HospitalTokenExtractor, JointPointerMaskablePolicy,
                                    PointerMaskablePolicy,
                                    ResidualPointerMaskablePolicy)  # noqa: F401
        if args.pooled_critic:  # (v12 X6) v10 actor + 순열불변 pooled critic
            from gopt_policy import PointerPooledCriticMaskablePolicy  # noqa: F401
    elif args.extractor == "gopt_bilinear":
        from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401
        from gopt_policy import (GoptBilinearActionNet, GoptMaskablePolicy,
                                 GoptTokenExtractor)  # noqa: F401

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
        # (v6 A3) valid variant: valid 열 정규화 면제(PadAwareVecNormalize) 위해 H/F 를 먼저
        # 산출(exempt_idx 구성). deepsets 는 무마스크 mean pooling 이 패딩 행을 오염 → valid 배타.
        # 비-valid·mlp 경로는 H/F 산출을 건너뛰어 기존 동작 완전 보존(probe env 불생성).
        valid_variant = "valid" in _parse_variant()
        H = F = gdim = None
        if (args.extractor in ("deepsets", "gopt_bilinear")
                or args.extractor.startswith("pointer") or valid_variant):
            H, F, gdim = _entity_dims(args.config_path, args.seed)
        if valid_variant and args.extractor == "deepsets":
            raise ValueError("deepsets 추출기는 valid variant 미지원 — 무마스크 mean pooling 이 "
                             "패딩 행을 오염(pointer 사용). hospital_set_extractor 는 수정 금지 결정.")

        # obs 정규화 필수(ETA·cap_remain 스케일) / reward 정규화는 옵션. eval·VIPER 는 통계 동결 로드.
        # ⚠️gamma 동기화 필수: VecNormalize 의 리턴 추적(discounted return 분산)과 PPO 의
        # gamma 가 불일치하면 보상 정규화 스케일이 왜곡됨.
        if args.init_from:
            init_zip = args.init_from
            if os.path.isdir(init_zip):
                init_zip = os.path.join(init_zip, "final_model.zip")
            init_vn = os.path.join(os.path.dirname(init_zip), "vecnormalize.pkl")
            venv = VecNormalize.load(init_vn, venv)
            venv.training = True
            venv.norm_reward = args.norm_reward
            print(f"[feature] init vecnormalize from {init_vn} "
                  f"(optimizer/lr/step은 신규)")
        elif valid_variant:
            # valid 열(각 병원 flat idx i*F+(F-1))을 정규화 면제 — 0/1 보존(아핀변환 붕괴 방지).
            exempt = [i * F + (F - 1) for i in range(H)]
            venv = pad_vecnorm.PadAwareVecNormalize(
                venv, exempt_idx=exempt, norm_obs=True, norm_reward=args.norm_reward,
                clip_obs=10.0, gamma=args.gamma)
            print(f"[feature] PadAwareVecNormalize: valid_col={F - 1} exempt 열={len(exempt)}개 "
                  f"(H={H} F={F}) — valid 열 정규화 면제")
        else:
            venv = VecNormalize(venv, norm_obs=True, norm_reward=args.norm_reward, clip_obs=10.0,
                                gamma=args.gamma)

        policy_cls = "MlpPolicy"
        policy_kwargs = dict(net_arch=[256, 256])
        if args.extractor == "deepsets":
            policy_kwargs = dict(
                features_extractor_class=HospitalSetExtractor,
                features_extractor_kwargs=dict(n_hospitals=H, entity_f=F, global_dim=gdim,
                                               embed_dim=args.embed_dim),
                net_arch=[256, 256],
            )
            print(f"[feature] deepsets 추출기: H={H} F={F} global={gdim} embed={args.embed_dim}")
        elif args.extractor.startswith("pointer"):
            # 두 실험은 torso/critic/PPO 설정이 같고 action head 의 class 조건부 여부만 다르다.
            if args.extractor == "pointer":
                policy_cls = PointerMaskablePolicy
                residual_kwargs = {}
            elif args.extractor == "pointer_joint3":
                policy_cls = JointPointerMaskablePolicy
                residual_kwargs = {}
            else:
                policy_cls = ResidualPointerMaskablePolicy
                if args.extractor == "pointer_rescm":
                    residual_kwargs = dict(residual_kind="cm", residual_rank=1)
                else:
                    residual_kwargs = dict(
                        residual_kind="lowrank",
                        residual_rank=1 if args.extractor.endswith("rank1") else 2)
            if args.pooled_critic:  # (v12 X6) actor 는 v10 그대로, critic 만 교체
                policy_cls = PointerPooledCriticMaskablePolicy
            fe_kwargs = dict(n_hospitals=H, entity_f=F, global_dim=gdim,
                             embed_dim=args.embed_dim, ctx_dim=args.ctx_dim,
                             n_attn_blocks=args.n_attn_blocks, n_heads=args.n_heads)
            if valid_variant:
                fe_kwargs["valid_col"] = F - 1  # 마지막 열=valid → 마스크드 풀링 활성
            policy_kwargs = dict(
                features_extractor_class=HospitalTokenExtractor,
                features_extractor_kwargs=fe_kwargs,
                head_hidden=args.head_hidden,  # PointerMaskablePolicy.__init__ 로 전달
                **residual_kwargs,
            )
            print(f"[feature] {args.extractor} 추출기+head: H={H} F={F} global={gdim} "
                  f"embed={args.embed_dim} ctx={args.ctx_dim} head_hidden={args.head_hidden} "
                  f"n_attn_blocks={args.n_attn_blocks} n_heads={args.n_heads} "
                  f"pooled_critic={args.pooled_critic} valid_col={fe_kwargs.get('valid_col')}")
        elif args.extractor == "gopt_bilinear":
            # (v12) 수요(class×mode) 토큰 × 목적지(병원+stay) 토큰 bilinear 채점.
            # n_gopt_blocks=0 이면 인코더는 v10 과 동일 모듈 → head 효과만 격리(X1).
            policy_cls = GoptMaskablePolicy
            fe_kwargs = dict(n_hospitals=H, entity_f=F, global_dim=gdim,
                             embed_dim=args.embed_dim, ctx_dim=args.ctx_dim,
                             n_attn_blocks=args.n_attn_blocks, n_heads=args.n_heads,
                             n_gopt_blocks=args.n_gopt_blocks,
                             ff_expansion=args.ff_expansion, dropout=args.attn_dropout)
            if valid_variant:
                fe_kwargs["valid_col"] = F - 1
            policy_kwargs = dict(
                features_extractor_class=GoptTokenExtractor,
                features_extractor_kwargs=fe_kwargs,
                head_hidden=args.head_hidden,   # bilinear head 는 미사용(부모 계약상 전달)
                pooled_critic=args.pooled_critic,
            )
            print(f"[feature] gopt_bilinear: H={H} F={F} global={gdim} embed={args.embed_dim} "
                  f"ctx={args.ctx_dim} n_attn_blocks={args.n_attn_blocks} "
                  f"n_gopt_blocks={args.n_gopt_blocks} n_heads={args.n_heads} "
                  f"ff_expansion={args.ff_expansion} dropout={args.attn_dropout} "
                  f"pooled_critic={args.pooled_critic} valid_col={fe_kwargs.get('valid_col')}")

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
        if args.init_from:
            # 신규 residual 파라미터만 missing이어야 한다. 기준 torso/critic/head는 이름·shape가
            # 동일해 v6 state_dict를 그대로 이식한다. optimizer는 위에서 새로 생성된 상태 유지.
            source = MaskablePPO.load(init_zip, device="cpu")
            incompatible = model.policy.load_state_dict(source.policy.state_dict(), strict=False)
            allowed = ("action_net.r_cm.", "action_net.r_u.", "action_net.r_v.",
                       "action_net.r0.")
            bad_missing = [k for k in incompatible.missing_keys
                           if not k.startswith(allowed)]
            if bad_missing or incompatible.unexpected_keys:
                raise RuntimeError("init_from state_dict 불일치: "
                                   f"missing={incompatible.missing_keys} "
                                   f"unexpected={incompatible.unexpected_keys}")
            print(f"[feature] init policy from {init_zip}: "
                  f"신규 파라미터={incompatible.missing_keys or '(없음; control)'}")

    _write_run_meta(args, model, status="training")
    # save_vecnormalize 기본 False = 구 동작. 중간 체크포인트를 평가하려면 그 시점의
    # 정규화 통계가 함께 있어야 한다(최종 통계를 초기 체크포인트에 쓰면 불일치).
    ckpt_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=os.path.join(args.log_dir, "checkpoints"),
        name_prefix="ppo_feature",
        save_vecnormalize=bool(getattr(args, "save_vecnormalize", False)),
    )

    # resume 시 reset_num_timesteps=False → total_timesteps 는 '추가' 스텝(이어서 카운트·체크포인트 번호 연속).
    model.learn(total_timesteps=args.total_timesteps, callback=ckpt_cb,
                tb_log_name="ppo_feature", progress_bar=False,
                reset_num_timesteps=(args.resume_from is None))
    final_path = os.path.join(args.log_dir, "final_model.zip")
    model.save(final_path)
    vecnorm_path = os.path.join(args.log_dir, "vecnormalize.pkl")
    venv.save(vecnorm_path)  # eval/VIPER 에서 VecNormalize.load 후 training=False 로 동결 적용 필수
    _write_run_meta(args, model, status="complete")
    print(f"Saved: {final_path}\nSaved: {vecnorm_path}")
    try_plot_learning_curve(args.log_dir)

    eval_env = make_env_fn(args.config_path, seed=args.seed + 999)()
    mean_r, std_r = masked_evaluate(model, eval_env, n_eval_episodes=10, use_masking=True)
    print(f"Eval mean reward: {mean_r:.3f} +/- {std_r:.3f}")


if __name__ == "__main__":
    main()
