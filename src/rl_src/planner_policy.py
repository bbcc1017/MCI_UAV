"""P1 NCRP — 비천리안 제한 롤아웃 플래너 (계획 §4.1 표 #2).

기존 오라클(rollout_oracle.py)의 1-step lookahead 를 **배포 가능한 플래너**로 개조한다.
개조점 3개(계획 부록 A-3):
  ① 천리안 제거: copy.deepcopy 는 rng(np.random.Generator)까지 비트복제 → 롤아웃이 실제
     미래(lognormal 표본·큐 실현)를 내다보는 천리안 상한이 된다. clairvoyant=False 면 복제
     직후 `clone.unwrapped.ev_manager.set_seed(np.random.default_rng(...))` 로 재시드
     (EventManager.set_seed 는 rng 객체 교체뿐이라 미드에피소드 안전) — 미래 실현을 모르는
     몬테카를로 표본 m 개의 평균으로 후보 가치를 추정한다.
  ② h-결정 절단: 후보 액션 1결정 + champion greedy 최대 h−1 추가 결정에서 롤아웃을 끊고,
     미종결이면 학습된 리프 가치(leaf_value.load_leaf)로 부트스트랩. h<0 = 무한(종단까지).
  ③ 리프 가치 단위 환산: leaf_fn 은 **pdrwog 단위 suffix**(=Σr_woG/preventable_woG — 지역
     규모 불변이라 전국 단일 회귀가 성립, leaf_value.py 참조)를 예측하므로, 롤아웃이 누적하는
     r_woG(비정규화) suffix 에 더하기 전에 ×preventable_woG 로 환산한다.

재현성 앵커(계약): h=-1 + clairvoyant=True + leaf_fn=None + K=8 구성은 후보 선정(top-K·
stay dedup·greedy 포함)·엄격개선 스위치·롤아웃 누적을 rollout_oracle.lookahead_episode /
q_rollout 과 **부동소수 단위로 동일**하게 재현해야 한다(oracle_headroom CSV 재현이 합격선).
이를 위해 r_woG 누적은 오라클과 동일하게 캐스팅 없이 수행하고, clairvoyant 시 m 회 롤아웃이
비트 동일하므로 1회만 수행한다(평균=단일값, 수치 불변·비용 절약).

재사용: rollout_oracle(_dest_table·Cloner), viper_distill(_masked_probs), leaf_value(load_leaf).
사용처: planner_eval.py(판정 드라이버) — act() 는 wrapped env(정규화 obs)를 deepcopy 하므로
기존 정책 규약 fn(obs,mask,unwrapped)→int 와 달리 **wrapped env 자체**를 받는다.

원본 env 무접촉: 후보 평가는 전부 deepcopy 복제본에서 수행 — act() 전후 원본의 obs/mask/
ev_manager 상태 불변(스모크에서 검증). 재시드용 default_rng 생성은 전역 numpy 상태를 건드리지
않으므로 평가 CRN(reset(seed))도 오염되지 않는다(같은 명령 2회 → pdr_base 완전 동일).
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import gymnasium as gym
import warnings as _warnings
_warnings.filterwarnings("ignore", message=r".*action_masks.*")  # 래퍼 경유 접근 경고 억제
_warnings.filterwarnings("ignore", category=UserWarning, module=r"gymnasium.*")


def _current_obs(env):
    """wrapped env 의 '현재' obs 를 무접촉 재구성(폴백 — planner_eval 은 obs 를 직접 전달).
    체인: base._make_obs()(dict) → HospitalFeatureWrapper._flat_obs → ObservationWrapper.observation.
    상태를 바꾸지 않는 순수 조회만 사용한다."""
    chain, e = [], env
    while hasattr(e, "env"):
        chain.append(e)
        e = e.env
    obs = e._make_obs()
    for w in reversed(chain):
        if hasattr(w, "_flat_obs"):                  # HospitalFeatureWrapper
            obs = w._flat_obs(obs)
        elif isinstance(w, gym.ObservationWrapper):  # _NormObs 등
            obs = w.observation(obs)
    return obs


class TruncatedRolloutPlanner:
    """비천리안 제한 롤아웃 플래너.

    Args:
        model: 챔피언 MaskablePPO(greedy 후속정책 + 후보 확률 원천).
        K: masked-prob 상위 후보 수(stay dedup·greedy 포함 — 오라클 관례).
        h: 롤아웃 결정 지평(후보 1결정 + greedy h−1 결정). h<0 = 무한(종단까지).
        m: 비천리안 몬테카를로 롤아웃 수(clairvoyant=True 면 결정론이라 1회로 축약).
        leaf_fn: leaf_value.load_leaf 콜백((B,355)→(B,) pdrwog 단위) — None 이면 절단분 0.
        clairvoyant: True 면 재시드 생략(=기존 오라클, rng 비트복제 천리안).
        reseed_base: 비천리안 재시드 베이스(평가 CRN 11000·리프 20000 과 분리된 777000 대역).
        switch_margin: 스위치 마진 ε(pdrwog 단위) — 상상 미래 평균 개선이
            ε×preventable_woG 를 초과할 때만 greedy 에서 이탈. m 유한 MC 의 잔여
            노이즈가 한계 스위치를 만드는 것을 차단(0=기존 엄격개선).
    """

    def __init__(self, model, K=8, h=10, m=2, leaf_fn=None, clairvoyant=False,
                 reseed_base=777000, switch_margin=0.0):
        self.model = model
        self.K, self.h, self.m = int(K), int(h), int(m)
        self.leaf_fn = leaf_fn
        self.clairvoyant = bool(clairvoyant)
        self.reseed_base = int(reseed_base)
        self.switch_margin = float(switch_margin)
        self._dest_tab = None
        self._cloner = None
        # act() 부가정보: lookahead 수행여부·스위치여부·소요 ms·후보 수
        self.last_info = {"lookahead": False, "switched": False, "ms": 0.0, "n_cand": 0}

    # ---------------------------------------------------------------- 내부
    def _rollout(self, clone, action, preventable):
        """복제본에 후보 action 적용 후 champion greedy 로 최대 h−1 추가 결정(h<0=종단까지)
        진행하며 suffix r_woG 누적 — q_rollout(rollout_oracle)과 동일한 무캐스팅 누적으로
        앵커의 비트 동일성 보장. 지평 도달·미종결이면 leaf 부트스트랩(×preventable 환산)."""
        obs, _r, term, trunc, info = clone.step(int(action))
        w = info.get("r_woG", 0.0)
        done = term or trunc
        n_extra = 0
        while not done and (self.h < 0 or n_extra < self.h - 1):
            mask = clone.action_masks()
            a, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
            obs, _r, term, trunc, info = clone.step(int(a))
            w += info.get("r_woG", 0.0)
            done = term or trunc
            n_extra += 1
        if not done and self.leaf_fn is not None:
            # leaf 는 pdrwog(=r_woG/preventable) 단위 suffix 예측 → r_woG 단위로 환산해 합산
            w += float(self.leaf_fn(np.asarray(obs, dtype=np.float32))[0]) * preventable
        return w

    # ---------------------------------------------------------------- 공개 API
    def act(self, env, ep_seed, obs=None):
        """현 wrapped env 상태에서 플래닝 1회 → 실행할 flat action(int).

        env: planner_eval 이 만든 feature env(_NormObs 정규화 obs·action_masks 노출).
        ep_seed: 에피소드 시드(비천리안 재시드 스트림 유도용 — 원본 env 는 건드리지 않음).
        obs: 현재 정규화 obs(에피소드 루프가 보유한 값 — 생략 시 무접촉 재구성 폴백).
        부가정보는 self.last_info(dict)에 기록(ms=플래닝 소요, switched=greedy 이탈 여부)."""
        t0 = time.perf_counter()
        info = {"lookahead": False, "switched": False, "ms": 0.0, "n_cand": 0}

        mask = np.asarray(env.action_masks(), dtype=bool)
        valid = np.flatnonzero(mask)
        if valid.size <= 1:                      # 유효행동 ≤1 — 플래닝 불요(오라클 동일)
            a = int(valid[0]) if valid.size else 0
            info["ms"] = (time.perf_counter() - t0) * 1e3
            self.last_info = info
            return a

        if obs is None:
            obs = _current_obs(env)
        if self._dest_tab is None or len(self._dest_tab) != len(mask):
            from rollout_oracle import _dest_table
            self._dest_tab = _dest_table(len(mask), env.unwrapped.H)
        if self._cloner is None:
            from rollout_oracle import Cloner
            self._cloner = Cloner("deepcopy", None, None)  # 플래너는 deepcopy 전용

        # ---- 후보 선정(rollout_oracle.lookahead_episode 와 동일: top-K + stay dedup) ----
        from viper_distill import _masked_probs
        probs = _masked_probs(self.model, obs, mask)
        g = int(np.argmax(probs))                # deterministic greedy = masked argmax
        order = np.argsort(-probs)
        cand, seen_stay = [], False
        for x in order[:self.K]:
            x = int(x)
            if not mask[x] or probs[x] <= 0:
                continue
            if self._dest_tab[x] == 0:           # stay 는 (c,m) 무관 동일 no-op → 1개만
                if seen_stay:
                    continue
                seen_stay = True
            cand.append(x)
        if g not in cand:                        # 안전 가드(order[0]=g 라 항상 포함이긴 함)
            cand.append(g)
        if len(cand) <= 1:
            info["ms"] = (time.perf_counter() - t0) * 1e3
            self.last_info = info
            return g

        # ---- 후보별 m회 롤아웃(clairvoyant 는 결정론 → 1회로 축약: 평균=단일값) ----
        info["lookahead"] = True
        info["n_cand"] = len(cand)
        preventable = float(env.unwrapped.preventable_woG)
        m_eff = 1 if self.clairvoyant else max(1, self.m)
        # 비천리안 CRN(2026-07-13 수정): j번째 상상 미래 시드를 **후보 간 공유**(구현 1판은
        # 후보idx 를 시드에 포함 → 후보마다 다른 실현으로 Q 비교 = 랭킹이 실현 노이즈에
        # 오염되어 그리드 전 구성 악화·과잉 스위치 32회/ep). 진짜 미래는 여전히 미지
        # (비천리안 유지) — 같은 상상 미래 위 paired 비교로 분산만 소거. 결정마다 다른
        # 스트림(_n_dec 반영)이라 특정 실현 패턴에 고착되지 않음.
        self._n_dec = getattr(self, "_n_dec", 0) + 1
        seeds = [self.reseed_base + ep_seed * 97 + j * 13 + self._n_dec * 10007
                 for j in range(m_eff)]
        qs = []
        for a in cand:
            acc = 0.0
            for j in range(m_eff):
                clone = self._cloner.clone(env, ep_seed, None)
                if not self.clairvoyant:
                    # 비천리안 핵심: 복제 rng 를 미래-무지 스트림으로 교체(원본 무접촉).
                    clone.unwrapped.ev_manager.set_seed(np.random.default_rng(seeds[j]))
                acc += self._rollout(clone, a, preventable)
            qs.append(acc / m_eff)

        gi, bi = cand.index(g), int(np.argmax(qs))
        # 마진 초과 개선일 때만 스위치(동률·미세개선=greedy 유지 — margin=0 이면 기존 엄격개선)
        if qs[bi] > qs[gi] + self.switch_margin * preventable:
            a_exec = cand[bi]
            info["switched"] = (a_exec != g)
        else:
            a_exec = g
        info["ms"] = (time.perf_counter() - t0) * 1e3
        self.last_info = info
        return a_exec
