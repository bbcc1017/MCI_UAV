"""피드백 5 아이디어 6 — 휴리스틱 대비 advantage wrapper.

env_wrapper.py / sim_src 수정 금지. 신규 파일.

각 에피소드 종료 시 휴리스틱 baseline R 을 빼서 advantage 신호를 학습한다.
baseline 산출 방식 (mode 인자):

  mode="rollout"   — 매 reset 시 같은 seed 로 휴리스틱 1ep 시뮬을 돌려 heur_R 산출.
                     정확하지만 학습이 2배 가까이 느려질 수 있다.
                     (rule_test=True env 한 개를 캐시하고 reset 마다 재사용)

  mode="precomputed" — region → mean heur_R 표를 사전 로드 (CSV 기반).
                     매 에피소드 동일한 baseline 을 빼므로 빠르지만 분산은 무시.

terminal 적용 방식 (subtract_at 인자):
  "terminal" (기본)  — 마지막 step reward 에 -heur_R 가산.
                       gamma=1 가정 시 episode return = ep_R - heur_R.
  "replace"          — episode return 자체를 (ep_R - heur_R) 로 치환:
                       step 별 reward 를 모두 0 으로 만들고 terminal 에 (ep_R - heur_R) 부여.
                       sparse reward — gamma 사용시 신호가 크게 달라짐.

휴리스틱 룰은 Universal_Rule(priority, hos_select, mode_R, mode_Y) 4-튜플로 지정.
지정 안하면 rollout 모드: "START, YellowNearest, Both_UAVFirst, Both_UAVFirst" (national 기본),
precomputed 모드: CSV 의 heuristic_rule 컬럼 (지역별 best, 다만 baseline 값만 사용).

ENV 변수 hook:
    MCI_ADV_MODE           : rollout | precomputed
    MCI_ADV_SUBTRACT_AT    : terminal | replace
    MCI_ADV_CSV            : precomputed 모드 CSV 경로 (region/heuristic_R 컬럼)
    MCI_ADV_REGION         : precomputed 모드 region key
"""
import os
import sys
import copy

import gymnasium as gym
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir, "sim_src"))
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)

from RuleManager import Universal_Rule  # noqa: E402


_DEFAULT_RULE = ("START", "YellowNearest", "Both_UAVFirst", "Both_UAVFirst")


class _HeuristicRollout:
    """sub-env 한 개를 캐시하고 매 reset 마다 휴리스틱 1ep 의 ep_R 을 반환.

    sub-env 는 main env 와 독립된 ScenarioManager 인스턴스라야 한다 (랜덤성/state 격리).
    config_path 를 받아 매번 새로 만든다.
    """

    def __init__(self, config_path: str, rule: Universal_Rule, use_r_woG=False):
        self.config_path = config_path
        self.rule = rule
        self.use_r_woG = use_r_woG
        # 첫 호출 시 lazy 생성
        from env_factory import make_base_env  # 지연 import (순환 방지)
        self._make_env = make_base_env

    def evaluate(self, seed: int) -> float:
        env = self._make_env(self.config_path, seed=seed, rule_test=True, eval_mode=True)
        self.rule.init_with_scenario({'EntityManager': env.en_manager})
        self.rule.set_seed(np.random.default_rng(seed))
        obs, _ = env.reset(seed=seed)
        done = False
        ep_r = 0.0
        ep_r_woG = 0.0
        while not done:
            action = self.rule.select(obs)
            obs, r, te, tr, info = env.step(action)
            ep_r += r
            ep_r_woG += info.get('r_woG', 0.0)
            done = te or tr
        return ep_r_woG if self.use_r_woG else ep_r


class HeuristicAdvantageWrapper(gym.Wrapper):
    """휴리스틱 baseline 을 빼서 advantage 신호를 학습한다 (gym.Wrapper, reward 만 변환).

    Parameters
    ----------
    env : gym.Env
        wrapping 대상. 보통 RewardRedesignWrapper 외부 또는 base env.
    config_path : str
        rollout 모드 시 sub-env 생성용 yaml. 또는 precomputed 모드에서도 호환을 위해
        받아두지만 사용 안 함.
    mode : "rollout" | "precomputed"
    subtract_at : "terminal" | "replace"
    rule : tuple(priority, hos_select, mode_R, mode_Y)
        rollout 모드 휴리스틱 룰 4-튜플. None 이면 _DEFAULT_RULE.
    baseline_csv : str (precomputed)
        region/heuristic_R 컬럼을 가진 CSV. plan1nat_f3_eval.csv 호환.
    region : str (precomputed)
        baseline_csv 의 region 값 (예: '서울').
    use_r_woG : bool
        baseline 계산 시 woG 보상을 쓸지. 기본 False (raw R).
        RewardRedesignWrapper(mode='woG') 와 같이 쓸 때는 True 권장.
    """

    def __init__(self, env, config_path: str,
                 mode: str = None, *, subtract_at: str = None,
                 rule=None, baseline_csv: str = None, region: str = None,
                 use_r_woG: bool = False):
        super().__init__(env)
        self.config_path = config_path
        self.mode = mode or os.environ.get("MCI_ADV_MODE", "rollout")
        self.subtract_at = subtract_at or os.environ.get("MCI_ADV_SUBTRACT_AT", "terminal")
        self.use_r_woG = use_r_woG
        if self.mode not in ("rollout", "precomputed"):
            raise ValueError(f"mode must be rollout|precomputed, got {self.mode!r}")
        if self.subtract_at not in ("terminal", "replace"):
            raise ValueError(f"subtract_at must be terminal|replace, got {self.subtract_at!r}")

        if self.mode == "rollout":
            rule_tuple = tuple(rule) if rule is not None else _DEFAULT_RULE
            self._rule_obj = Universal_Rule(*rule_tuple)
            self._rollout = _HeuristicRollout(config_path, self._rule_obj, use_r_woG=use_r_woG)
            self._baseline_table = None
        else:
            csv_path = baseline_csv or os.environ.get("MCI_ADV_CSV")
            region_key = region or os.environ.get("MCI_ADV_REGION")
            if not csv_path or not region_key:
                raise ValueError("precomputed 모드: baseline_csv 와 region (또는 "
                                 "MCI_ADV_CSV / MCI_ADV_REGION) 필수")
            self._baseline_table = self._load_baseline_csv(csv_path, use_r_woG=use_r_woG)
            if region_key not in self._baseline_table:
                raise KeyError(f"region '{region_key}' 미수록 in {csv_path}")
            self._baseline_value = self._baseline_table[region_key]
            self._rule_obj = None
            self._rollout = None

        self._ep_seed = None       # 마지막 reset 시 seed (rollout 재현용)
        self._current_baseline = 0.0  # 이번 ep 의 baseline (reset 마다 갱신)
        self._ep_step_count = 0
        self._ep_return_so_far = 0.0  # replace 모드에서 누적

    # ---------- baseline 로딩 ----------
    @staticmethod
    def _load_baseline_csv(csv_path, use_r_woG=False):
        """CSV → {region: baseline}. use_r_woG=True 면 heuristic_R_woG 우선, 아니면 heuristic_R."""
        import pandas as pd
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if use_r_woG and "heuristic_R_woG" in df.columns:
            col = "heuristic_R_woG"
        else:
            col = "heuristic_R"
        return dict(zip(df["region"].astype(str), df[col].astype(float)))

    # ---------- gym API ----------
    def reset(self, **kwargs):
        seed = kwargs.get("seed", None)
        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))
            kwargs["seed"] = seed
        self._ep_seed = int(seed)
        self._ep_step_count = 0
        self._ep_return_so_far = 0.0

        if self.mode == "rollout":
            self._current_baseline = float(self._rollout.evaluate(self._ep_seed))
        else:
            self._current_baseline = float(self._baseline_value)

        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        info = dict(info) if info is not None else {}
        info["r_raw"] = info.get("r_raw", float(r))
        info["heuristic_baseline"] = self._current_baseline
        self._ep_step_count += 1
        self._ep_return_so_far += float(r)

        done = terminated or truncated

        if self.subtract_at == "terminal":
            if done:
                # 마지막 step 에 -baseline 가산 → episode return = ep_R - baseline
                r_out = r - self._current_baseline
                info["advantage"] = self._ep_return_so_far - self._current_baseline
            else:
                r_out = r
        else:  # "replace"
            if done:
                r_out = self._ep_return_so_far - self._current_baseline
                info["advantage"] = r_out
            else:
                r_out = 0.0

        return obs, r_out, terminated, truncated, info
