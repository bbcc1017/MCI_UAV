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
import math
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
                 reseed_base=777000, switch_margin=0.0, gamma=0.99,
                 alloc="uniform", switch_z=0.0, extra_cand_fn=None,
                 greedy_action_fn=None, rollout_action_fn=None):
        self.model = model
        self.K, self.h, self.m = int(K), int(h), int(m)
        self.leaf_fn = leaf_fn
        self.clairvoyant = bool(clairvoyant)
        self.reseed_base = int(reseed_base)
        self.switch_margin = float(switch_margin)
        # (v11) alloc: "uniform"=후보마다 m회(기존, 비트동일) / "sh"=successive halving
        #   같은 총 롤아웃 예산(K_cand×m)을 rung 마다 하위 절반 탈락시키며 재분배 →
        #   최종 생존 후보(greedy 포함 강제)의 유효 표본수가 uniform 보다 커진다.
        # (v11) switch_z: 스위치 판정을 고정 ε 대신 페어드 표준오차 z배로(0=기존 엄격개선).
        #   d_j = w_bj − w_gj 의 mean > z·SE 일 때만 이탈. SE 는 결정마다 달라 ε 보다 정합적.
        # (v11) extra_cand_fn(unwrapped, mask) -> [action,...]: 외부(MILP 등) 후보 주입 훅.
        #   주입 후보도 같은 롤아웃으로 검증되므로 greedy 대비 지배가 유지된다.
        self.alloc = str(alloc)
        self.switch_z = float(switch_z)
        self.extra_cand_fn = extra_cand_fn
        # (v15) 정책 포트폴리오: PPO 확률은 top-K 후보 생성에 그대로 사용하되,
        # 엄격개선 비교 기준행동과 롤아웃 후속정책만 외부 반응형 정책으로 교체할 수 있다.
        # 콜백은 (env_unwrapped, mask, obs) -> 유효 action. None이면 기존 PPO와 비트동일.
        self.greedy_action_fn = greedy_action_fn
        self.rollout_action_fn = rollout_action_fn
        # (v7) gamma: 할인 suffix q_*_disc 계산용(결정 스텝 단위, PPO gamma 정합 기본 0.99).
        # 스위치 판정은 무할인 qs 유지 → planner 성능 불변, 할인본은 value-target 학습용 노출만.
        self.gamma = float(gamma)
        self._dest_tab = None
        self._cloner = None
        # act() 부가정보: lookahead 수행여부·스위치여부·소요 ms·후보 수
        self.last_info = {"lookahead": False, "switched": False, "ms": 0.0, "n_cand": 0}

    # ---------------------------------------------------------------- 내부
    def _rollout(self, clone, action, preventable):
        """복제본에 후보 action 적용 후 champion greedy 로 최대 h−1 추가 결정(h<0=종단까지)
        진행하며 suffix r_woG 누적 — q_rollout(rollout_oracle)과 동일한 무캐스팅 누적으로
        앵커의 비트 동일성 보장. 지평 도달·미종결이면 leaf 부트스트랩(×preventable 환산).

        반환 (w, w_disc): w=무할인 누적(스위치 판정·앵커 재현용, 기존과 비트 동일),
        w_disc=결정 스텝 gamma 할인 누적(v7 value-target 학습용 — 무할인은 잔여 결정수
        비례 상태편향이라 크리틱 타깃엔 부적). k=결정 스텝 인덱스(후보 결정 k=0)."""
        obs, _r, term, trunc, info = clone.step(int(action))
        r0 = info.get("r_woG", 0.0)
        w = r0
        w_disc = r0
        done = term or trunc
        n_extra = 0
        k = 1
        while not done and (self.h < 0 or n_extra < self.h - 1):
            mask = clone.action_masks()
            if self.rollout_action_fn is None:
                a, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
            else:
                a = self.rollout_action_fn(clone.unwrapped, mask, obs)
            obs, _r, term, trunc, info = clone.step(int(a))
            r = info.get("r_woG", 0.0)
            w += r
            w_disc += (self.gamma ** k) * r
            done = term or trunc
            n_extra += 1
            k += 1
        if not done and self.leaf_fn is not None:
            # leaf 는 pdrwog(=r_woG/preventable) 단위 suffix 예측 → r_woG 단위로 환산해 합산
            lv = float(self.leaf_fn(np.asarray(obs, dtype=np.float32))[0]) * preventable
            w += lv
            w_disc += (self.gamma ** k) * lv
        return w, w_disc

    # ---------------------------------------------------------------- 공개 API
    def act(self, env, ep_seed, obs=None):
        """현 wrapped env 상태에서 플래닝 1회 → 실행할 flat action(int).

        env: planner_eval 이 만든 feature env(_NormObs 정규화 obs·action_masks 노출).
        ep_seed: 에피소드 시드(비천리안 재시드 스트림 유도용 — 원본 env 는 건드리지 않음).
        obs: 현재 정규화 obs(에피소드 루프가 보유한 값 — 생략 시 무접촉 재구성 폴백).
        부가정보는 self.last_info(dict)에 기록(ms=플래닝 소요, switched=greedy 이탈 여부)."""
        t0 = time.perf_counter()
        # q_* (v7 가치게이트): 후보 롤아웃 가치를 last_info 로 노출(pdrwog 단위 = r_woG/preventable).
        #   q_greedy=greedy 후보 가치, q_best=최선 후보 가치, dpdr=개선분(q_best−q_greedy).
        #   lookahead 미수행(유효≤1·후보≤1) 시 None. 기존 동작·반환값 불변(정보 추가만).
        info = {"lookahead": False, "switched": False, "ms": 0.0, "n_cand": 0,
                "q_greedy": None, "q_best": None, "q_exec": None, "dpdr": None,
                "q_greedy_disc": None, "q_best_disc": None, "q_exec_disc": None, "dpdr_disc": None,
                "n_rollout": 0, "n_extra": 0}

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
            # (v6) dest 테이블은 mask 레이아웃(H_pad)과 정합해야 한다: 패딩 env 는
            # len(mask)=2×(H_pad+1)×2 인데 unwrapped.H=실H → 오정렬(_codec_from_mask 예외).
            # gym.Wrapper 속성 위임으로 HospitalFeatureWrapper.H(레이아웃 H_pad)를 잡고,
            # 구 고정47 경로는 실H=H_pad=47 라 동일값(수치 무영향).
            H_layout = int(getattr(env, "H", env.unwrapped.H))
            self._dest_tab = _dest_table(len(mask), H_layout)
        if self._cloner is None:
            from rollout_oracle import Cloner
            self._cloner = Cloner("deepcopy", None, None)  # 플래너는 deepcopy 전용

        # ---- 후보 선정(rollout_oracle.lookahead_episode 와 동일: top-K + stay dedup) ----
        from viper_distill import _masked_probs
        probs = _masked_probs(self.model, obs, mask)
        ppo_g = int(np.argmax(probs))            # PPO 확률 top-K의 기준
        if self.greedy_action_fn is None:
            g = ppo_g
        else:
            g = int(self.greedy_action_fn(env.unwrapped, mask, obs))
            if g < 0 or g >= len(mask) or not mask[g]:
                raise ValueError(f"외부 기준정책이 무효행동을 반환: {g}")
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
        # 후보 출처 감사에서는 외부 기준정책 행동을 PPO top-K로 잘못 세지 않는다.
        info["ppo_candidate_actions"] = tuple(int(x) for x in cand)
        if g not in cand:                        # 외부 기준정책은 top-K 밖일 수 있다.
            cand.append(g)
        if self.extra_cand_fn is not None:       # (v11) 외부 후보 주입(MILP 등)
            for x in self.extra_cand_fn(env.unwrapped, mask):
                x = int(x)
                if not mask[x] or x in cand:
                    continue
                if self._dest_tab[x] == 0:       # stay 중복은 후보 관례대로 1개만
                    if seen_stay:
                        continue
                    seen_stay = True
                cand.append(x)
                info["n_extra"] += 1
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

        def _future_seed(j):
            return self.reseed_base + ep_seed * 97 + j * 13 + self._n_dec * 10007

        def _one(a, j):
            """후보 a 를 j번째 상상미래에서 1회 롤아웃 → (무할인, 할인) suffix."""
            clone = self._cloner.clone(env, ep_seed, None)
            if not self.clairvoyant:
                # 비천리안 핵심: 복제 rng 를 미래-무지 스트림으로 교체(원본 무접촉).
                clone.unwrapped.ev_manager.set_seed(np.random.default_rng(_future_seed(j)))
            return self._rollout(clone, a, preventable)

        w_paired = None       # (v11) 후보별 미래별 값 — z 스위치·SH 용(uniform 수치엔 무영향)
        if self.alloc == "sh" and not self.clairvoyant and len(cand) > 2 and m_eff > 1:
            # ---- (v11) successive halving: 같은 예산(len(cand)×m)을 rung 마다 재분배 ----
            budget = len(cand) * m_eff
            n_rungs = max(1, int(np.log2(max(len(cand), 2))) + 1)
            per_rung = budget / n_rungs
            W = [[] for _ in cand]
            surv = list(range(len(cand)))
            gi0 = cand.index(g)
            spent, guard = 0, 0
            while surv and spent < budget and guard < 12:
                guard += 1
                add = max(1, int(round(per_rung / len(surv))))
                add = min(add, max(1, int(np.ceil((budget - spent) / len(surv)))))
                for i in surv:
                    for j in range(len(W[i]), len(W[i]) + add):
                        W[i].append(_one(cand[i], j))
                    spent += add
                if len(surv) <= 2:
                    continue                       # 남은 예산은 최종 2후보(greedy 포함)에 투입
                means = {i: float(np.mean([x[0] for x in W[i]])) for i in surv}
                keep = sorted(surv, key=lambda i: -means[i])[:max(2, len(surv) // 2)]
                if gi0 not in keep:                # greedy 는 페어드 기준이라 항상 생존
                    keep = keep[:-1] + [gi0]
                surv = sorted(set(keep))
            info["n_rollout"] = int(spent)
            qs = [float(np.mean([x[0] for x in W[i]])) if W[i] else -np.inf
                  for i in range(len(cand))]
            qs_disc = [float(np.mean([x[1] for x in W[i]])) if W[i] else 0.0
                       for i in range(len(cand))]
            # 최종 argmax 는 생존 후보(표본수 동일)에서만 — 조기탈락 후보의 소표본 평균 배제
            gi = gi0
            bi = max(surv, key=lambda i: qs[i]) if surv else gi
            n_pair = min(len(W[gi]), len(W[bi]))
            if n_pair > 1 and bi != gi:
                w_paired = np.asarray([W[bi][j][0] - W[gi][j][0] for j in range(n_pair)])
        else:
            qs = []
            qs_disc = []   # (v7) 할인 suffix 평균 — value-target 학습용(스위치 판정엔 미사용)
            seeds = [_future_seed(j) for j in range(m_eff)]
            W = []
            for a in cand:
                acc = 0.0
                acc_disc = 0.0
                row = []
                for j in range(m_eff):
                    clone = self._cloner.clone(env, ep_seed, None)
                    if not self.clairvoyant:
                        clone.unwrapped.ev_manager.set_seed(np.random.default_rng(seeds[j]))
                    rw, rwd = self._rollout(clone, a, preventable)
                    acc += rw
                    acc_disc += rwd
                    row.append(rw)
                # 누적은 오라클과 동일한 순차합 유지(비트 동일성) — 평균도 같은 식.
                qs.append(acc / m_eff)
                qs_disc.append(acc_disc / m_eff)
                W.append(row)
            info["n_rollout"] = int(len(cand) * m_eff)
            # 스위치·bi 는 무할인 qs 로 판정(기존 성능 불변) — 할인본은 노출만.
            gi, bi = cand.index(g), int(np.argmax(qs))
            if m_eff > 1 and bi != gi:
                w_paired = np.asarray(W[bi]) - np.asarray(W[gi])

        # 마진 초과 개선일 때만 스위치(동률·미세개선=greedy 유지 — margin=0 이면 기존 엄격개선)
        thr = self.switch_margin * preventable
        if self.switch_z > 0.0 and w_paired is not None and w_paired.size > 1:
            # 페어드 SE 기반 마진: 유한 MC 잡음이 만든 한계 스위치를 결정별로 차단
            se = float(w_paired.std(ddof=1)) / math.sqrt(w_paired.size)
            thr = max(thr, self.switch_z * se)
        if qs[bi] > qs[gi] + thr:
            a_exec = cand[bi]
            info["switched"] = (a_exec != g)
        else:
            a_exec = g
        # (v7) 후보 가치를 pdrwog 단위로 노출 — 가치 예측 게이트·value-target 수집용
        pv = preventable if preventable > 0 else 1.0
        ai = cand.index(a_exec)
        info["q_greedy"] = float(qs[gi]) / pv
        info["q_best"] = float(qs[bi]) / pv
        info["q_exec"] = float(qs[ai]) / pv
        info["dpdr"] = (float(qs[bi]) - float(qs[gi])) / pv   # 개선분(≥0)
        # 할인본(value-target 학습용 — 무할인 상태편향 제거). bi/gi 는 무할인 기준 유지.
        info["q_greedy_disc"] = float(qs_disc[gi]) / pv
        info["q_best_disc"] = float(qs_disc[bi]) / pv
        info["q_exec_disc"] = float(qs_disc[ai]) / pv
        info["dpdr_disc"] = (float(qs_disc[bi]) - float(qs_disc[gi])) / pv
        # 후보 포트폴리오 감사용 진단값. 기존 평가 CSV에는 쓰지 않으므로 수치 동작 불변.
        info["ppo_greedy_action"] = int(ppo_g)
        info["greedy_action"] = int(g)
        info["exec_action"] = int(a_exec)
        info["candidate_actions"] = tuple(int(x) for x in cand)
        info["candidate_q_pdr"] = tuple(float(x) / pv for x in qs)
        info["ms"] = (time.perf_counter() - t0) * 1e3
        self.last_info = info
        return a_exec
