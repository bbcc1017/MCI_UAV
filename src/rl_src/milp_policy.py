# -*- coding: utf-8 -*-
"""v11 OR 기준선 — 롤링호라이즌 MILP 배차·수용 배정 정책.

NCRP(재시뮬 룩어헤드)와 대비되는 **최적화 기반** 결정정책이다. 매 결정시점에 현재 상태의
스냅샷으로 "어느 등급 환자를 어느 병원에 어느 수단으로" 배정할지 정수최적화로 풀고, 지금
당장 실행 가능한 배정 1건만 방출한 뒤 다음 결정에서 전부 다시 푼다(rolling horizon).

## 왜 선형 정수모형으로 표현되는가(시뮬 구조에서 유도)

* 보상은 **치료 개시(p_admit) 시각**의 생존확률이다 — Red `0.56/((t/91)^1.58+1)`,
  Yellow `0.81/((t/160)^2.41+1)` (`MCIEnvironment_gymnasium.getSurvProb`).
* 치료 개시는 병원 서버(수술실수 `hos_max_capa`, 실측 1~3개)가 idle 일 때만 일어나고
  서비스 시간은 **지수분포**(`EventManager.sample_service_time`)다. 지수분포는 무기억이라
  S개 서버가 모두 바쁠 때 다음 해방까지 기대시간이 `mean/S`, k번째 해방이 `k·mean/S` 로
  깔끔히 열거된다 → 병원 h의 "치료개시 기회" 상대시각
  `τ(h,k) = 0 (k ≤ n_idle)`, `(n_queue + j)·mean_h/S_h (그 외 j번째)`.
* 이송 도착 상대시각은 `release + leg[mode,h] + handover[mode]` 로 역시 상수.
  따라서 (트립슬롯 × 기회) 쌍마다 `치료개시 = max(도착, τ)` 가 **상수**이고, 배정 변수에
  대해 목적함수가 선형이 된다.

## 정보수준·제약의 공정성

적격 조합은 **`action_masks()` 에서만** 만든다(tier3·헬리패드·발송게이트·수단 가용이 이미
마스크에 인코딩). 병원 실시간 점유·큐를 읽으므로 정보수준은 증류 I3(병원연계)와 동급이며
`MCI_CAP_GATE=occ` 통신가정과 정합한다. 미래 구조시각 등 정책이 알 수 없는 정보는 쓰지 않는다.

## 사용법

* 단독 정책: `make_milp_policy()` → `fn(obs, mask, env_unwrapped) -> action`
  (`v10_tree_eval.rollout`·`evaluate` 정책 규약과 동일)
* 플래너 어댑터: `MilpPlanner(...)` → `act(env, ep_seed, obs)` (planner_eval 공용 규약)
* NCRP 후보 주입: `MilpProposer.propose(env_unwrapped, mask) -> [action, ...]`

재사용: `score_features.compute_static`(정적 ETA·raw분·tier·용량),
`tree_distill_policy.decode_action`(마스크 길이 유도 레이아웃), `EntityManager.in_flight_by_hospital`.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix

sys.path.insert(0, os.path.dirname(__file__))

from EntityManager import EntityManager
from score_features import compute_static
from tree_distill_policy import _layout, decode_action

_BIG_SERVICE = 1.0e4     # treat_mean 이 inf(영구점유)일 때의 대체 상수(분)
_FAR = 1.0e4             # 사용 불가 슬롯/기회 표시용


def surv_prob(t: float, p_class: int) -> float:
    """sim 과 동일한 생존확률(Red/Yellow 만 사용)."""
    if p_class == 0:
        return 0.56 / (math.pow(t / 91.0, 1.58) + 1.0)
    if p_class == 1:
        return 0.81 / (math.pow(t / 160.0, 2.41) + 1.0)
    return 0.0


def _surv_vec(t: np.ndarray, p_class: np.ndarray) -> np.ndarray:
    """벡터화 생존확률 — t(분, 절대), p_class(0=Red,1=Yellow)."""
    t = np.maximum(np.asarray(t, dtype=float), 0.0)
    red = 0.56 / (np.power(t / 91.0, 1.58) + 1.0)
    yel = 0.81 / (np.power(t / 160.0, 2.41) + 1.0)
    return np.where(np.asarray(p_class) == 0, red, yel)


class _Static:
    """en_manager 아이덴티티로 캐시되는 시나리오 상수."""

    __slots__ = ("manager_id", "H", "t_amb", "t_uav", "tier3", "helipad", "max_send",
                 "n_server", "max_queue", "treat_mean", "handover", "rescue_param")

    def __init__(self, env):
        props = env.en_manager.en_properties
        hp = props["hospital"]
        base = compute_static(env)
        H = int(base["H"])
        self.manager_id = id(env.en_manager)
        self.H = H
        self.t_amb = np.asarray(base["t_amb"], dtype=float)      # 현장→병원 기대 분(AMB)
        self.t_uav = np.asarray(base["t_uav"], dtype=float)      # 현장→병원 기대 분(UAV)
        self.tier3 = np.asarray(base["is_tier3"], dtype=bool)
        self.max_send = np.asarray(base["max_send"], dtype=float)
        self.n_server = np.maximum(
            np.asarray(hp["hos_max_capa"], dtype=float).reshape(-1)[:H], 1.0)
        self.max_queue = np.asarray(hp["hos_max_queue"], dtype=float).reshape(-1)[:H]
        helipad = np.zeros(H, dtype=bool)
        idx = np.asarray(hp.get("hos_helipad_idx", []), dtype=int).reshape(-1)
        helipad[idx[(idx >= 0) & (idx < H)]] = True
        self.helipad = helipad
        # 서비스 평균(분): treat_mean[class, tier3?] — 'inf' 문자열/np.inf 는 영구점유
        info = props["patient"]["patient_info"]
        tm = np.full((2, 2), _BIG_SERVICE, dtype=float)          # [class, 0=tier2,1=tier3]
        for c in (0, 1):
            for j, key in ((1, "treat_tier3_mean"), (0, "treat_tier2_mean")):
                try:
                    v = float(info[key][c])
                except (TypeError, ValueError):
                    v = _BIG_SERVICE
                tm[c, j] = _BIG_SERVICE if not np.isfinite(v) or v <= 0 else v
        self.treat_mean = tm
        self.handover = np.array([float(props["ambulance"].get("amb_handover_time", 0.0)),
                                 float(props["uav"].get("uav_handover_time", 0.0))])
        # 구조시간 Beta(α,β)·60 파라미터(R/Y) — 미래환자 예약 모형용(실현값 미사용)
        rp = []
        for c in (0, 1):
            try:
                rp.append((float(info["rescue_param_alpha"][c]), float(info["rescue_param_beta"][c])))
            except Exception:
                rp.append((0.0, 0.0))
        self.rescue_param = rp


class MilpAssignment:
    """결정시점 스냅샷 MILP 배정기.

    Args:
        h_pad: 마스크 레이아웃 병원 패딩(기본 47 — `MCI_H_PAD` 와 동일).
        n_opp: 병원별 '지연 치료개시 기회' 추가 개수(즉시분 외).
        slot_margin: 필요 트립슬롯 수 여유(대기환자 수 + margin 까지만 슬롯 생성).
        second_wave: True 면 현장 차량의 2차 왕복 슬롯 + 체인 제약 추가.
        topk_hosp: >0 이면 모드별 ETA 상위 k 병원으로 후보 제한(0=전체).
        time_limit: HiGHS 시간제한(초).
    """

    def __init__(self, h_pad: int = 47, n_opp: int = 3, slot_margin: int = 2,
                 second_wave: bool = False, topk_hosp: int = 0, time_limit: float = 2.0,
                 future_patients: bool = False, n_future_groups: int = 2,
                 future_cap: int = 8, force_dispatch: bool = False,
                 queue_model: str = "fluid"):
        self.h_pad = int(h_pad)
        self.n_opp = int(n_opp)
        self.slot_margin = int(slot_margin)
        self.second_wave = bool(second_wave)
        self.topk_hosp = int(topk_hosp)
        self.time_limit = float(time_limit)
        # 미래(미구조) 환자를 수요에 포함 → 임박 등급(Red)용 즉시 치료기회를 예약하게 된다.
        self.future_patients = bool(future_patients)
        self.n_future_groups = int(n_future_groups)
        self.future_cap = int(future_cap)
        # True 면 '지금 대기'(stay) 대신 현장 차량으로 최대가치 배정을 강제한다.
        self.force_dispatch = bool(force_dispatch)
        # 큐 모형: fluid=머릿수 기반(기존) | timed=이송중 도착시각 인식 이벤트 계산
        self.queue_model = str(queue_model)
        self._static: _Static | None = None
        self.last_info: dict = {}

    # ------------------------------------------------------------------ 내부
    def _get_static(self, env) -> _Static:
        st = self._static
        if st is None or st.manager_id != id(env.en_manager):
            st = self._static = _Static(env)
        return st

    def _slots(self, dobs, st, mode_pool, need):
        """(mode, release, cap) 목록 — 현장 대기 + 복귀예정 차량(+2차 왕복)."""
        rows = []
        for m, (key_state, key_wait, t_leg) in enumerate(
                (("amb_states", "amb_wait", st.t_amb), ("uav_states", "uav_wait", st.t_uav))):
            if not mode_pool[m]:
                continue
            states = np.asarray(dobs[key_state], dtype=float)
            wait = dobs.get(key_wait, [[]])
            n_site = len(wait[0]) if len(wait) else 0
            if n_site > 0:
                rows.append((m, 0.0, float(min(n_site, need))))
            if states.size == 0:
                continue
            busy = states[:, 1] > 1e-6
            if not busy.any():
                continue
            dest = states[busy, 0].astype(int)
            remain = states[busy, 1].astype(float)
            # 병원행 차량은 현장 복귀까지 복귀 leg 을 더한다(왕복 근사).
            back = np.zeros_like(remain)
            hop = (dest >= 1) & (dest <= st.H)
            if hop.any():
                back[hop] = t_leg[dest[hop] - 1]
            rel = np.sort(remain + back)
            take = max(1, int(need) - int(min(n_site, need)))   # 현장분으로 못 덮는 만큼만
            for r in rel[:take]:
                rows.append((m, float(r), 1.0))
            if self.second_wave and n_site > 0:
                # 현장 차량의 2차 왕복(nominal): 왕복 + handover. 체인 제약과 짝.
                nominal = 2.0 * float(np.median(t_leg[:st.H])) + st.handover[m]
                rows.append((m, nominal, float(min(n_site, need))))
        return rows

    def _groups(self, dobs, st, now, n_wait):
        """환자 수요 그룹 (class, 이송가능시각, 인원) — 현장 대기 + (옵션) 예상 구조환자.

        미래 그룹은 **실현 구조시각을 보지 않고** 정적 config 의 구조시간 분포
        Beta(α,β)·60(`EventManager.ev_onset`)를 `T > now` 로 절단한 조건분포의 분위수로
        추정한다(관측 가능한 미구조 인원수 + 교범상 알려진 분포만 사용).
        """
        groups = [(c, 0.0, float(n_wait[c])) for c in (0, 1) if n_wait[c] > 0]
        if not self.future_patients:
            return groups
        p_states = np.asarray(dobs["p_states"], dtype=float)
        info = st.rescue_param
        G = max(1, self.n_future_groups)
        for c in (0, 1):
            n_un = int(((p_states[:, 0] == c) & (p_states[:, 1] < 0.5)).sum())
            if n_un <= 0:
                continue
            n_un = min(n_un, self.future_cap)
            a, b = info[c]
            if a <= 0 or b <= 0:
                groups.append((c, 0.0, float(n_un)))
                continue
            from scipy.stats import beta as _beta
            f_now = float(_beta.cdf(min(max(now / 60.0, 0.0), 1.0 - 1e-9), a, b))
            per = n_un / G
            for j in range(G):
                q = f_now + (1.0 - f_now) * (j + 0.5) / G
                avail = float(_beta.ppf(min(q, 1.0 - 1e-9), a, b)) * 60.0
                groups.append((c, max(avail - now, 0.0), per))
        return groups

    def _inflight_times(self, dobs, st):
        """병원별 이송중 환자의 잔여 도착시간 목록 — 시각 인식 큐 모형용.

        `in_flight_by_hospital` 은 머릿수만 준다. 그런데 그 환자가 서버를 점유하는 시점은
        **도착 후**다. 머릿수를 즉시 대기열로 세면(fluid 모형) 장거리 병원이 과도하게
        불리해져 '먼 tier3 로 몰기' 같은 왜곡이 생긴다(농촌·도서에서 손실 관측).
        """
        out = [[] for _ in range(st.H)]
        for key in ("amb_states", "uav_states"):
            s = np.asarray(dobs.get(key, ()), dtype=float)
            if s.size == 0:
                continue
            sel = (s[:, 0] >= 1) & (s[:, 2] > 0)
            for dest, remain in zip(s[sel, 0].astype(int), s[sel, 1]):
                if 1 <= dest <= st.H:
                    out[dest - 1].append(float(remain))
        for v in out:
            v.sort()
        return out

    def _opps_timed(self, st, h_states, n_wait, inflight_t):
        """시각 인식 치료개시 기회 — 서버 해방시각과 선행 환자(대기+이송중) 도착시각을
        작은 결정론 이벤트 계산으로 합성한다. 반환 (h_idx, tau, cap)."""
        idle = h_states[:, 0].astype(float)
        queue = h_states[:, 1].astype(float)
        w = np.array([n_wait[0], n_wait[1]], dtype=float)
        wsum = float(w.sum()) or 1.0
        mu_all = np.where(st.tier3,
                          (w[0] * st.treat_mean[0, 1] + w[1] * st.treat_mean[1, 1]) / wsum,
                          (w[0] * st.treat_mean[0, 0] + w[1] * st.treat_mean[1, 0]) / wsum)
        hh, tau, cap = [], [], []
        for h in range(st.H):
            S = int(st.n_server[h])
            mu = float(mu_all[h])
            n_idle = int(min(idle[h], S))
            n_busy = S - n_idle
            # 서버 해방 예정시각: idle 은 0, busy 는 지수 잔여의 기대 순서통계 근사
            free = [0.0] * n_idle + [mu * (j + 1) / max(n_busy, 1) for j in range(n_busy)]
            free.sort()
            # 선행 환자: 현재 큐(즉시 가능) + 이송중(도착시각 = 잔여시간)
            ahead = [0.0] * int(queue[h]) + list(inflight_t[h])
            ahead.sort()
            for t_av in ahead:
                k = min(range(len(free)), key=lambda i: max(free[i], t_av))
                start = max(free[k], t_av)
                free[k] = start + mu
            # 내 환자용 기회: 가장 이른 서버부터 n_opp+1 개
            for _ in range(self.n_opp + 1):
                k = int(np.argmin(free))
                hh.append(h); tau.append(float(free[k])); cap.append(1.0)
                free[k] = free[k] + mu
        return np.asarray(hh, dtype=int), np.asarray(tau, dtype=float), np.asarray(cap, dtype=float)

    def _opps(self, st, h_states, n_wait, in_flight):
        """병원별 (상대시각, cap) 치료개시 기회 → (h_idx, tau, cap) 배열.

        ⚠️`in_flight`(이미 그 병원으로 이송 중인 환자)를 **서버 선점분으로 차감**해야 한다.
        같은 시각 dispatch burst 안에서 결정이 연속으로 들어오는데(EventManager.proceed_action
        repeat), n_idle 은 환자가 도착해 서비스를 시작할 때까지 줄지 않으므로 차감하지 않으면
        매 결정이 같은 병원의 '즉시 기회'를 중복 배정해 최근접 과집중이 재현된다.
        """
        idle = h_states[:, 0].astype(float)
        queue = h_states[:, 1].astype(float) + np.asarray(in_flight, dtype=float)
        w = np.array([n_wait[0], n_wait[1]], dtype=float)
        wsum = float(w.sum()) or 1.0
        # 병원 tier 에 맞는 R/Y 서비스 평균을 현장 대기 구성비로 가중(큐 배출률 근사)
        mu = np.where(st.tier3,
                      (w[0] * st.treat_mean[0, 1] + w[1] * st.treat_mean[1, 1]) / wsum,
                      (w[0] * st.treat_mean[0, 0] + w[1] * st.treat_mean[1, 0]) / wsum)
        step = mu / st.n_server
        hh, tau, cap = [], [], []
        free_now = np.maximum(idle - queue, 0.0)          # 선점분 차감 후 남은 즉시 서버
        rest = np.maximum(queue - idle, 0.0)              # 내 환자보다 앞선 대기·이송중 인원
        for h in range(st.H):
            if free_now[h] > 0:
                hh.append(h); tau.append(0.0); cap.append(free_now[h])
            for j in range(1, self.n_opp + 1):
                hh.append(h); tau.append(float((rest[h] + j) * step[h])); cap.append(1.0)
        return np.asarray(hh, dtype=int), np.asarray(tau, dtype=float), np.asarray(cap, dtype=float)

    # ------------------------------------------------------------------ 공개
    def solve(self, env, mask):
        """MILP 1회 해 → 실행가능 배정 목록.

        반환 dict: rows(list[(class, hospital, mode, value, arrival)] — release=0 슬롯 배정만,
        Red 우선·가치 내림차순), n_var, n_con, status, ms, fallback(bool).
        """
        t0 = time.perf_counter()
        st = self._get_static(env)
        mask = np.asarray(mask, dtype=bool)
        H_layout, n_mode = _layout(len(mask), self.h_pad)
        n_dest = H_layout + 1
        info = {"rows": [], "n_var": 0, "n_con": 0, "status": "empty", "ms": 0.0,
                "fallback": False}

        # ---- 적격 (class, hospital, mode) 를 마스크에서만 유도 ----
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            info["ms"] = (time.perf_counter() - t0) * 1e3
            self.last_info = info
            return info
        dec = np.asarray([decode_action(int(a), len(mask), self.h_pad) for a in valid], dtype=int)
        cls, dest, mode = dec[:, 0], dec[:, 1], dec[:, 2]
        move = (dest >= 1) & (dest <= st.H) & (cls <= 1)
        if not move.any():
            info["status"] = "stay_only"
            info["ms"] = (time.perf_counter() - t0) * 1e3
            self.last_info = info
            return info
        e_cls, e_h, e_mode = cls[move], dest[move] - 1, mode[move]

        dobs = env.en_manager.get_full_obs()
        now = float(env.ev_manager.time)
        h_states = np.asarray(dobs["h_states"], dtype=float)[:st.H]
        p_sent = np.asarray(dobs["p_sent"], dtype=float)[:st.H]
        in_flight = EntityManager.in_flight_by_hospital(dobs, st.H).astype(float)
        gate_psent = os.environ.get("MCI_CAP_GATE", "occ").strip().lower() == "psent"
        used = p_sent if gate_psent else (h_states[:, 2] + in_flight)
        cap_remain = np.maximum(st.max_send - used, 0.0)
        n_wait = [len(dobs["p_wait"][c][0]) for c in (0, 1)]
        need = int(n_wait[0] + n_wait[1])
        if need <= 0:
            info["status"] = "no_patient"
            info["ms"] = (time.perf_counter() - t0) * 1e3
            self.last_info = info
            return info

        mode_pool = [bool((e_mode == 0).any()), bool((e_mode == 1).any())]
        slots = self._slots(dobs, st, mode_pool, need + self.slot_margin)
        if not slots:
            info["status"] = "no_slot"
            info["ms"] = (time.perf_counter() - t0) * 1e3
            self.last_info = info
            return info
        s_mode = np.asarray([s[0] for s in slots], dtype=int)
        s_rel = np.asarray([s[1] for s in slots], dtype=float)
        s_cap = np.asarray([s[2] for s in slots], dtype=float)
        s_wave2 = np.asarray([self.second_wave and s[1] > 0.0 and s[2] > 1.0
                              for s in slots], dtype=bool)

        # ---- 후보 (class, hospital, mode) 축소: 발송여유 0 제외 + 선택적 topk ----
        keep = cap_remain[e_h] >= 1.0
        if self.topk_hosp > 0:
            for m in (0, 1):
                t_leg = st.t_amb if m == 0 else st.t_uav
                sel = np.flatnonzero(keep & (e_mode == m))
                if sel.size > self.topk_hosp:
                    order = np.argsort(t_leg[e_h[sel]])
                    drop = sel[order[self.topk_hosp:]]
                    keep[drop] = False
        e_cls, e_h, e_mode = e_cls[keep], e_h[keep], e_mode[keep]
        if e_cls.size == 0:
            info["status"] = "no_eligible"
            info["ms"] = (time.perf_counter() - t0) * 1e3
            self.last_info = info
            return info

        if self.queue_model == "timed":
            o_h, o_tau, o_cap = self._opps_timed(st, h_states, n_wait,
                                                 self._inflight_times(dobs, st))
        else:
            o_h, o_tau, o_cap = self._opps(st, h_states, n_wait, in_flight)
        groups = self._groups(dobs, st, now, n_wait)
        g_cls = np.asarray([g[0] for g in groups], dtype=int)
        g_avail = np.asarray([g[1] for g in groups], dtype=float)
        g_cap = np.asarray([g[2] for g in groups], dtype=float)

        # ---- 변수: (수요그룹 g) × (slot s, 수단일치) × (적격 e, class 일치) × (기회 o, 병원일치) ----
        S, E = len(slots), e_cls.size
        opp_by_h = [np.flatnonzero(o_h == h) for h in range(st.H)]
        cols_s, cols_e, cols_o, cols_g = [], [], [], []
        for e in range(E):
            oo = opp_by_h[e_h[e]]
            if oo.size == 0:
                continue
            ok_s = np.flatnonzero(s_mode == e_mode[e])
            if ok_s.size == 0:
                continue
            gg = np.flatnonzero(g_cls == e_cls[e])
            if gg.size == 0:
                continue
            ss, oos, ggs = np.meshgrid(ok_s, oo, gg, indexing="ij")
            cols_s.append(ss.ravel()); cols_o.append(oos.ravel()); cols_g.append(ggs.ravel())
            cols_e.append(np.full(ss.size, e, dtype=int))
        if not cols_s:
            info["status"] = "no_column"
            info["ms"] = (time.perf_counter() - t0) * 1e3
            self.last_info = info
            return info
        cs = np.concatenate(cols_s); ce = np.concatenate(cols_e)
        co = np.concatenate(cols_o); cg = np.concatenate(cols_g)
        n_var = cs.size

        t_leg = np.where(e_mode[ce] == 0, st.t_amb[e_h[ce]], st.t_uav[e_h[ce]])
        # 출발 = max(차량 가용, 환자 이송가능) → 도착 = 출발 + leg + handover
        arrive = np.maximum(s_rel[cs], g_avail[cg]) + t_leg + st.handover[e_mode[ce]]
        start = np.maximum(arrive, o_tau[co])
        value = _surv_vec(now + start, e_cls[ce])

        # ---- 제약 ----
        rows, cols, data, ub = [], [], [], []
        r = 0
        for s in range(S):                                    # 슬롯 대수
            idx = np.flatnonzero(cs == s)
            if idx.size:
                rows.append(np.full(idx.size, r)); cols.append(idx)
                data.append(np.ones(idx.size)); ub.append(s_cap[s]); r += 1
        for gidx in range(len(groups)):                       # 수요그룹 인원(현장/예상)
            idx = np.flatnonzero(cg == gidx)
            if idx.size:
                rows.append(np.full(idx.size, r)); cols.append(idx)
                data.append(np.ones(idx.size)); ub.append(float(g_cap[gidx])); r += 1
        for o in np.unique(co):                               # 치료개시 기회 cap
            idx = np.flatnonzero(co == o)
            rows.append(np.full(idx.size, r)); cols.append(idx)
            data.append(np.ones(idx.size)); ub.append(o_cap[o]); r += 1
        for h in np.unique(e_h[ce]):                          # 병원 발송여유
            idx = np.flatnonzero(e_h[ce] == h)
            rows.append(np.full(idx.size, r)); cols.append(idx)
            data.append(np.ones(idx.size)); ub.append(float(np.floor(cap_remain[h]))); r += 1
        if self.second_wave:                                  # 2차 왕복 체인(수단별)
            for m in (0, 1):
                i2 = np.flatnonzero(s_wave2[cs] & (s_mode[cs] == m))
                i1 = np.flatnonzero((~s_wave2[cs]) & (s_mode[cs] == m))
                if i2.size and i1.size:
                    rows.append(np.full(i2.size + i1.size, r))
                    cols.append(np.concatenate([i2, i1]))
                    data.append(np.concatenate([np.ones(i2.size), -np.ones(i1.size)]))
                    ub.append(0.0); r += 1
        A = coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(r, n_var)).tocsr()
        con = LinearConstraint(A, lb=-np.inf, ub=np.asarray(ub, dtype=float))

        res = milp(c=-value, constraints=con, integrality=np.ones(n_var),
                   bounds=Bounds(0, 1), options={"time_limit": self.time_limit,
                                                 "presolve": True})
        info["n_var"], info["n_con"] = int(n_var), int(r)
        if res.x is None:
            info["status"] = f"infeasible({res.status})"
            info["fallback"] = True
            info["ms"] = (time.perf_counter() - t0) * 1e3
            self.last_info = info
            return info
        x = np.asarray(res.x, dtype=float)
        picked = np.flatnonzero(x > 0.5)
        # release=0 슬롯(지금 현장에 있는 차량) 배정만 즉시 실행 대상
        exec_rows = []
        for j in picked:
            if s_rel[cs[j]] > 1e-9 or g_avail[cg[j]] > 1e-9:   # 지금 현장 차량 + 현장 환자만
                continue
            exec_rows.append((int(e_cls[ce[j]]), int(e_h[ce[j]]), int(e_mode[ce[j]]),
                              float(value[j]), float(arrive[j])))
        exec_rows.sort(key=lambda t: (t[0], -t[3]))
        info["rows"] = exec_rows
        # 지금-슬롯 배정이 비었을 때의 폴백 후보: 현장 차량·현장 환자 열 중 최대가치 1건.
        # (LP 가 '지금 대기 → 나중 복귀차량에 배정'을 택하면 차량이 유휴로 남는데, 평균시간
        #  근사가 과신되면 농촌·장거리에서 손실로 나타난다 — force_dispatch 로 차단 가능)
        now_cols = np.flatnonzero((s_rel[cs] <= 1e-9) & (g_avail[cg] <= 1e-9))
        if now_cols.size:
            jb = int(now_cols[int(np.argmax(value[now_cols]))])
            info["best_now"] = (int(e_cls[ce[jb]]), int(e_h[ce[jb]]), int(e_mode[ce[jb]]),
                                float(value[jb]))
        info["status"] = "ok"
        info["obj"] = float(-res.fun) if res.fun is not None else float("nan")
        info["n_assign"] = int(picked.size)
        info["ms"] = (time.perf_counter() - t0) * 1e3
        self.last_info = info
        return info

    # ------------------------------------------------------------- 액션 방출
    def actions(self, env, mask, max_n: int = 1):
        """MILP 해에서 즉시 실행 가능한 액션(평탄 int) 목록(최대 max_n, 중복 제거)."""
        mask = np.asarray(mask, dtype=bool)
        H_layout, n_mode = _layout(len(mask), self.h_pad)
        n_dest = H_layout + 1
        sol = self.solve(env, mask)
        out = []
        for c, h, m in ((r[0], r[1], r[2]) for r in sol["rows"]):
            a = c * (n_dest * n_mode) + (h + 1) * n_mode + (m if n_mode > 1 else 0)
            if 0 <= a < len(mask) and mask[a] and a not in out:
                out.append(int(a))
            if len(out) >= max_n:
                break
        return out

    def action(self, env, mask):
        """단독 정책용 — 액션 1개(없으면 stay, stay 도 없으면 최근접 폴백)."""
        mask = np.asarray(mask, dtype=bool)
        acts = self.actions(env, mask, max_n=1)
        if acts:
            return acts[0]
        H_layout, n_mode = _layout(len(mask), self.h_pad)
        n_dest = H_layout + 1
        if self.force_dispatch:
            bn = self.last_info.get("best_now")
            if bn is not None:
                c, h, m, _ = bn
                a = c * (n_dest * n_mode) + (h + 1) * n_mode + (m if n_mode > 1 else 0)
                if 0 <= a < len(mask) and mask[a]:
                    return int(a)
        # MILP 가 지금-슬롯을 비웠다 = 대기가 낫다 → stay
        for c in (0, 1):
            a = c * (n_dest * n_mode)
            if a < len(mask) and mask[a]:
                return int(a)
        # stay 가 마스크에 없으면 최근접·발송여유 폴백(Red 우선)
        st = self._get_static(env)
        valid = np.flatnonzero(mask)
        best, best_key = int(valid[0]), None
        for a in valid:
            c, d, m = decode_action(int(a), len(mask), self.h_pad)
            if d == 0 or d > st.H or c > 1:
                continue
            t = (st.t_amb if m == 0 else st.t_uav)[d - 1]
            key = (c, t)
            if best_key is None or key < best_key:
                best_key, best = key, int(a)
        return best


def make_milp_policy(h_pad: int = 47, **kw):
    """정책 규약 `fn(obs, mask, env_unwrapped) -> action`."""
    solver = MilpAssignment(h_pad=h_pad, **kw)
    stats = {"ms": [], "n_var": []}

    def fn(obs, mask, env_unwrapped):
        a = solver.action(env_unwrapped, mask)
        stats["ms"].append(solver.last_info.get("ms", 0.0))
        stats["n_var"].append(solver.last_info.get("n_var", 0))
        return a

    fn.solver = solver
    fn.stats = stats
    return fn


class MilpPlanner:
    """planner_eval 공용 규약(`act(env, ep_seed, obs)`)을 만족하는 MILP 단독 정책 어댑터.

    `switched` = MILP 액션이 기준정책(model) greedy 와 다른 비율(정책 이탈률) — model=None
    이면 항상 False. `lookahead` 는 MILP 를 실제로 푼 결정에서만 True(지연 통계 분모).
    """

    def __init__(self, model=None, h_pad: int = 47, **kw):
        self.model = model
        self.solver = MilpAssignment(h_pad=h_pad, **kw)
        self.last_info = {"lookahead": False, "switched": False, "ms": 0.0, "n_cand": 0}

    def act(self, env, ep_seed, obs=None):
        t0 = time.perf_counter()
        mask = np.asarray(env.action_masks(), dtype=bool)
        valid = np.flatnonzero(mask)
        info = {"lookahead": False, "switched": False, "ms": 0.0, "n_cand": int(valid.size)}
        if valid.size <= 1:
            info["ms"] = (time.perf_counter() - t0) * 1e3
            self.last_info = info
            return int(valid[0]) if valid.size else 0
        a = self.solver.action(env.unwrapped, mask)
        info["lookahead"] = True
        info["n_var"] = int(self.solver.last_info.get("n_var", 0))
        info["status"] = self.solver.last_info.get("status", "?")
        if self.model is not None and obs is not None:
            g, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
            info["switched"] = bool(int(g) != int(a))
        info["ms"] = (time.perf_counter() - t0) * 1e3
        self.last_info = info
        return int(a)


class MilpProposer:
    """NCRP 후보 주입기 — `propose(env_unwrapped, mask) -> [action, ...]`."""

    def __init__(self, h_pad: int = 47, n_propose: int = 2, **kw):
        self.solver = MilpAssignment(h_pad=h_pad, **kw)
        self.n_propose = int(n_propose)
        self.ms_total = 0.0
        self.n_call = 0

    def propose(self, env_unwrapped, mask):
        acts = self.solver.actions(env_unwrapped, mask, max_n=self.n_propose)
        self.ms_total += float(self.solver.last_info.get("ms", 0.0))
        self.n_call += 1
        return acts
