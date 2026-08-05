# -*- coding: utf-8 -*-
"""Shin and Lee (2020)의 규칙 기반 MCI 정책을 현행 시뮬레이터에 맞춘다.

논문:
  Emergency medical service resource allocation in a mass casualty incident
  by integrating patient prioritization and hospital selection problems

Shin and Lee (2020)가 비교한 규칙 3개와 제안한 규칙 1개:
  * Threshold: Jacobson 계열, 논문 식 (9)의 환자수 임계값으로 R/Y 결정
  * 2Step: Jacobson 계열, 논문 식 (10)의 현재+다음 이송가치로 R/Y 결정
  * PIH: Mills 계열을 논문이 수정한 식 (11), R 우선+병원 점수화
  * Integrated: Shin–Lee 제안 규칙, 식 (7)·(8)과 Figure 6으로 환자·병원 동시 결정

원 논문은 AMB만 고려한다. 현행 AMB+UAV 환경에서는 논문 식을 바꾸지 않고 먼저
mode 운용규칙으로 사용할 자원을 정한 다음, 해당 mode의 ETA·가용 병원으로 식을
평가한다. 따라서 ``OnlyAMB``가 원 논문에 가장 가까운 재현이고 나머지 세 변형은
본 연구의 수단 확장이다. 전국 다병원 환경에서는 Tier2/3·helipad·용량 hard mask를
공통 적용하고, 후보가 없거나 PIH 식이 특이점에 걸리면 가장 가까운 제약충족 병원으로
폴백한다. 그러므로 원 저자 코드의 바이트 단위 복제가 아니라 논문 수식의 도메인 적응
비교군으로 보고해야 한다.
"""
from __future__ import annotations

import math
import os

import numpy as np

from EntityManager import EntityManager


SHIN_METHODS = ("Threshold", "2Step", "PIH", "Integrated")
SHIN_MODE_RULES = ("OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB")

# 첨부 코드와 논문 실험에서 사용한 시간단위 위험률(/hour).
_ABANDONMENT = np.asarray([0.54, 0.36], dtype=np.float64)
_MAX_SURVIVAL = np.asarray([0.56, 0.81], dtype=np.float64)
# 첨부 PIH 구현이 지수 근사에 사용한 중증도별 파라미터(/hour).
_PIH_ALPHA = np.asarray([0.42, 0.30], dtype=np.float64)
_EPS = 1e-12


def _cap_gate_is_occ() -> bool:
    """기존 휴리스틱·RL mask와 같은 병원 용량 정보축을 사용한다."""
    return os.environ.get("MCI_CAP_GATE", "occ").strip().lower() != "psent"


def _finite_float(value, default=np.inf) -> float:
    """CSV의 ``inf`` 문자열까지 안전하게 실수로 바꾼다."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


class ShinHeuristicRule:
    """기존 ``RuleManager.Rule``과 같은 ``init/set_seed/select`` 계약."""

    def __init__(self, method: str, mode_rule: str):
        if method not in SHIN_METHODS:
            raise ValueError(f"지원하지 않는 Shin 휴리스틱: {method}")
        if mode_rule not in SHIN_MODE_RULES:
            raise ValueError(f"지원하지 않는 mode 규칙: {mode_rule}")
        self.method = method
        self.mode_rule = mode_rule
        self.rule_name = f"Shin {method}, Mode {mode_rule}"
        self.rng = np.random.default_rng()
        self.last_meta = None

    def set_seed(self, rng):
        self.rng = rng

    def init_with_scenario(self, scenario):
        props = scenario["EntityManager"].en_properties
        hp = props["hospital"]
        pp = props["patient"]["patient_info"]
        ap = props["ambulance"]
        up = props["uav"]

        self.hos_num = int(hp["hos_num"])
        self.hos_tier = np.asarray(hp["hos_tier"], dtype=np.int32)
        self.tier3_idx = np.asarray(hp["hos_tier3_idx"], dtype=np.int32)
        self.tier2_idx = np.asarray(hp["hos_tier2_idx"], dtype=np.int32)
        self.helipad_set = {
            int(x) for x in np.asarray(
                hp.get("hos_helipad_idx", np.array([]))
            ).reshape(-1)
        }
        self.hos_max_send = np.asarray(hp["hos_max_send"], dtype=np.float64)
        self.servers = np.maximum(
            np.asarray(hp["hos_max_capa"], dtype=np.float64), 1.0
        )
        self.fleet_size = {
            0: int(ap.get("amb_num", 0)),
            1: int(up.get("uav_num", 0)),
        }

        amb_eta = np.asarray(ap.get("amb_HtoS_t", (np.array([]),))[0], dtype=float)
        uav_eta = np.asarray(up.get("uav_HtoS_t", (np.array([]),))[0], dtype=float)
        if len(amb_eta) != self.hos_num:
            amb_eta = np.full(self.hos_num, np.inf, dtype=float)
        if len(uav_eta) != self.hos_num:
            uav_eta = np.full(self.hos_num, np.inf, dtype=float)
        # 원 논문의 "치료개시까지 시간"에 맞춰 현행 시뮬의 고정 인계시간을
        # ETA에 더한다. 원 코드에는 별도 인계시간이 없던 프로젝트 적응 항목이다.
        self.eta = {
            0: amb_eta + float(ap.get("amb_handover_time", 0.0)),
            1: uav_eta + float(up.get("uav_handover_time", 0.0)),
        }

        self.service_mean = np.full((2, self.hos_num), np.inf, dtype=float)
        t3 = pp["treat_tier3_mean"]
        t2 = pp["treat_tier2_mean"]
        for c in range(2):
            for h in range(self.hos_num):
                source = t3 if self.hos_tier[h] == 3 else t2
                self.service_mean[c, h] = _finite_float(source.iloc[c])

        # PIH는 중증도별 실제 치료시간, Integrated는 논문의 등급별 단일
        # 서비스율에 대응하도록 해당 시나리오 R/Y 구성의 평균 치료시간을 쓴다.
        # 이 매핑은 원 논문의 2등급·소수병원을 전국 다병원 자료로 확장한 부분이다.
        self.service_rate = np.zeros_like(self.service_mean)
        finite = np.isfinite(self.service_mean) & (self.service_mean > 0)
        self.service_rate[finite] = (
            self.servers[np.newaxis, :].repeat(2, axis=0)[finite]
            * 60.0
            / self.service_mean[finite]
        )

        ratio = np.asarray(pp["ratio"].iloc[:2], dtype=float)
        ratio = ratio / ratio.sum() if ratio.sum() > 0 else np.asarray([0.5, 0.5])
        self.integrated_service_rate = np.zeros(self.hos_num, dtype=float)
        for h in range(self.hos_num):
            if self.hos_tier[h] == 3:
                means = self.service_mean[:, h]
                valid = np.isfinite(means) & (means > 0)
                if valid.any():
                    local_w = ratio[valid]
                    local_w = local_w / local_w.sum()
                    mean_t = float(np.dot(local_w, means[valid]))
                else:
                    mean_t = np.inf
            else:
                # 현행 hard mask에서 Red는 Tier3만 가능하므로 하급병원은 Y 치료율.
                mean_t = self.service_mean[1, h]
            if np.isfinite(mean_t) and mean_t > 0:
                self.integrated_service_rate[h] = self.servers[h] * 60.0 / mean_t

    # ------------------------------------------------------------------ 공통 상태
    @staticmethod
    def _waiting_counts(obs) -> np.ndarray:
        return np.asarray(
            [len(obs["p_wait"][0][0]), len(obs["p_wait"][1][0])],
            dtype=np.int32,
        )

    def _load(self, obs) -> np.ndarray:
        if _cap_gate_is_occ():
            return (
                np.asarray(obs["h_states"][:, -1], dtype=float)
                + EntityManager.in_flight_by_hospital(obs, self.hos_num)
            )
        return np.asarray(obs["p_sent"], dtype=float)

    def _mode_order(self, obs) -> list[int]:
        available = {
            0: bool(obs["amb_wait"][0]),
            1: bool(obs["uav_wait"][0]),
        }
        if self.mode_rule == "OnlyAMB":
            order = [0]
        elif self.mode_rule == "OnlyUAV":
            order = [1]
        elif self.mode_rule == "Both_AMBFirst":
            order = [0, 1]
        else:
            order = [1, 0]
        return [m for m in order if available[m]]

    def _eligible(self, obs, mode: int, p_class: int) -> np.ndarray:
        load = self._load(obs)
        valid = np.isfinite(self.eta[mode]) & (load < self.hos_max_send)
        if mode == 1:
            valid &= np.asarray(
                [h in self.helipad_set for h in range(self.hos_num)], dtype=bool
            )
        if p_class == 0:
            valid &= self.hos_tier == 3
        return np.flatnonzero(valid)

    def _aggregate_travel_rate(self, mode: int, h: int) -> float:
        """논문의 N_A*mu를 현재 mode의 전체 대수와 ETA로 계산(/hour)."""
        eta = float(self.eta[mode][h])
        n_mode = self.fleet_size[mode]
        if n_mode <= 0 or not np.isfinite(eta) or eta <= 0:
            return 0.0
        return float(n_mode * 60.0 / eta)

    def _nearest(self, obs, mode: int, p_class: int, tier: int | None = None):
        candidates = self._eligible(obs, mode, p_class)
        if tier is not None:
            candidates = candidates[self.hos_tier[candidates] == tier]
        if not len(candidates):
            return None
        return int(candidates[np.argmin(self.eta[mode][candidates])])

    def _simple_destination(self, obs, mode: int, p_class: int):
        """Threshold/2-step의 논문상 simple 병원 선택을 재현한다."""
        if p_class == 0:
            return self._nearest(obs, mode, 0, tier=3)
        high = self._nearest(obs, mode, 1, tier=3)
        low = self._nearest(obs, mode, 1, tier=2)
        choices = [h for h in (high, low) if h is not None]
        if not choices:
            return None
        # 논문: D는 최근접 상·하급병원을 0.5 확률로 선택.
        return int(choices[int(self.rng.integers(0, len(choices)))])

    def _route_with_fallback(self, obs, mode: int, chosen: int, counts):
        order = [chosen, 1 - chosen]
        for p_class in order:
            if counts[p_class] <= 0:
                continue
            h = self._simple_destination(obs, mode, p_class)
            if h is not None:
                return p_class, h
        return None

    # ---------------------------------------------------------------- 환자우선 규칙
    def _class_rates(self, obs, mode: int):
        high = self._nearest(obs, mode, 0, tier=3)
        low = self._nearest(obs, mode, 1, tier=2)
        if low is None:
            low = self._nearest(obs, mode, 1, tier=3)
        return (
            self._aggregate_travel_rate(mode, high) if high is not None else 0.0,
            self._aggregate_travel_rate(mode, low) if low is not None else 0.0,
        )

    def _threshold(self, obs, mode: int, counts):
        if not np.all(counts > 0):
            chosen = 0 if counts[0] > 0 else 1
            return self._route_with_fallback(obs, mode, chosen, counts)

        rate_r, rate_y = self._class_rates(obs, mode)
        if rate_r <= 0 or rate_y <= 0:
            return self._route_with_fallback(obs, mode, 0, counts)
        r_r, r_y = _ABANDONMENT
        alpha_r = rate_r / (r_r + rate_r)
        alpha_y = rate_y / (r_y + rate_y)
        numerator_y = rate_r * (alpha_y * r_y - alpha_r * r_r)
        numerator_r = rate_y * (alpha_y * r_y - alpha_r * r_r)
        denominator_y = r_y * (
            r_r * (alpha_r - alpha_y) + alpha_y * (rate_r - rate_y)
        )
        denominator_r = r_r * (
            r_y * (alpha_r - alpha_y) + alpha_r * (rate_r - rate_y)
        )
        q_y = numerator_y / denominator_y if abs(denominator_y) > _EPS else np.nan
        q_r = numerator_r / denominator_r if abs(denominator_r) > _EPS else np.nan
        threshold = max(q_y, q_r) if np.isfinite([q_y, q_r]).all() else np.inf
        chosen = 0 if int(counts.sum()) < threshold else 1
        self.last_meta = {"threshold": float(threshold)}
        return self._route_with_fallback(obs, mode, chosen, counts)

    def _two_step(self, obs, mode: int, counts):
        if not np.all(counts > 0):
            chosen = 0 if counts[0] > 0 else 1
            return self._route_with_fallback(obs, mode, chosen, counts)
        rates = np.asarray(self._class_rates(obs, mode), dtype=float)
        if np.any(rates <= 0):
            return self._route_with_fallback(obs, mode, 0, counts)
        alpha = rates / (_ABANDONMENT + rates)
        common = counts[0] * _ABANDONMENT[0] + counts[1] * _ABANDONMENT[1]
        score = np.full(2, -np.inf, dtype=float)
        for c in range(2):
            # 식 (10)의 1{n_i>=2} a_i 항을 그대로 반영한다. 첨부 구현은
            # indicator를 생략했지만 논문 본문의 정의를 정본으로 삼는다.
            next_same = alpha[c] if counts[c] >= 2 else 0.0
            next_value = max(next_same, alpha[1 - c])
            denominator = rates[c] - _ABANDONMENT[c] + common
            if denominator > _EPS:
                score[c] = alpha[c] + rates[c] * next_value / denominator
        chosen = int(np.argmax(score)) if np.isfinite(score).any() else 0
        self.last_meta = {"two_step_scores": score.tolist()}
        return self._route_with_fallback(obs, mode, chosen, counts)

    # ---------------------------------------------------------------- 병원점수 규칙
    def _pih(self, obs, mode: int, counts):
        p_class = 0 if counts[0] > 0 else 1  # 논문 PIH 비교: START 우선순위
        candidates = self._eligible(obs, mode, p_class)
        if not len(candidates):
            if p_class == 0 and counts[1] > 0:
                p_class = 1
                candidates = self._eligible(obs, mode, p_class)
            if not len(candidates):
                return None

        travel_rate = np.asarray(
            [self._aggregate_travel_rate(mode, int(h)) for h in candidates],
            dtype=float,
        )
        service_rate = self.service_rate[p_class, candidates]
        useful = (travel_rate > 0) & (service_rate > 0)
        if not useful.any():
            h = int(candidates[np.argmin(self.eta[mode][candidates])])
            return p_class, h

        # 첨부 코드의 PIH 정적 분배확률(q_j) 구성: myopic value 순으로
        # 병원 안정용량(service/travel)만큼 차례로 배분한다.
        rank_value = np.full(len(candidates), -np.inf, dtype=float)
        rank_value[useful] = (
            travel_rate[useful]
            * _MAX_SURVIVAL[p_class]
            * np.exp(-_PIH_ALPHA[p_class] / travel_rate[useful])
        )
        q = np.zeros(len(candidates), dtype=float)
        remain = 1.0
        for pos in np.argsort(-rank_value):
            if remain <= _EPS or not useful[pos]:
                continue
            q[pos] = min(remain, service_rate[pos] / travel_rate[pos])
            remain -= q[pos]

        load = self._load(obs)[candidates]
        log_scores = np.full(len(candidates), -np.inf, dtype=float)
        for pos in range(len(candidates)):
            kappa = travel_rate[pos] * q[pos] / 2.0  # 첨부 PIH 구현과 동일
            service = service_rate[pos]
            if kappa <= _EPS or service <= _EPS:
                continue
            risk = _PIH_ALPHA[p_class]
            disc = max((service + kappa + risk) ** 2 - 4 * kappa * service, 0.0)
            g = math.sqrt(disc)
            denominator = service - kappa - risk - g
            if abs(denominator) <= _EPS:
                continue
            term1 = (service - kappa + risk - g) / denominator
            term2 = (service + kappa + risk - g) / (2 * kappa)
            if term1 <= 0 or term2 <= 0:
                continue
            log_scores[pos] = (
                math.log(service * _MAX_SURVIVAL[p_class] / kappa)
                + math.log(term1)
                + float(load[pos]) * math.log(term2)
            )

        if np.isfinite(log_scores).any():
            best = int(np.argmax(log_scores))
        else:
            best = int(np.argmin(self.eta[mode][candidates]))
        self.last_meta = {"pih_log_scores": log_scores.tolist()}
        return p_class, int(candidates[best])

    # --------------------------------------------------------------- 통합 규칙
    def _integrated_best(self, obs, mode: int, tier: int):
        # Tier3는 R 적격집합, Tier2는 Y 적격집합으로 구성한다.
        p_class = 0 if tier == 3 else 1
        candidates = self._eligible(obs, mode, p_class)
        candidates = candidates[self.hos_tier[candidates] == tier]
        if not len(candidates):
            return None
        load = self._load(obs)
        values = np.full(len(candidates), np.inf, dtype=float)
        for pos, h in enumerate(candidates):
            travel = self._aggregate_travel_rate(mode, int(h))
            service = self.integrated_service_rate[h]
            if travel > 0 and service > 0:
                values[pos] = 1.0 / travel + load[h] / service
        if not np.isfinite(values).any():
            return None
        pos = int(np.argmin(values))
        return int(candidates[pos]), float(values[pos])

    def _integrated(self, obs, mode: int, counts):
        high = self._integrated_best(obs, mode, tier=3)
        low = self._integrated_best(obs, mode, tier=2)
        if counts[0] > 0 and counts[1] == 0:
            return (0, high[0]) if high is not None else None
        if counts[0] == 0 and counts[1] > 0:
            candidates = [(1, x[0], x[1]) for x in (low, high) if x is not None]
            return (1, min(candidates, key=lambda x: x[2])[1]) if candidates else None

        # 두 등급이 모두 있을 때 후보 tier가 하나뿐이면 실현 가능한 보수적 폴백.
        if high is None:
            return (1, low[0]) if low is not None else None
        if low is None:
            return 0, high[0]

        h1, h2 = low[0], high[0]
        load = self._load(obs)
        mu1 = self._aggregate_travel_rate(mode, h1)
        mu2 = self._aggregate_travel_rate(mode, h2)
        w1 = self.integrated_service_rate[h1]
        w2 = self.integrated_service_rate[h2]
        if min(mu1, mu2, w1, w2) <= 0:
            return 0, h2

        line = (mu2 * w2) / (mu1 * w1) * load[h1]
        absolute = w2 / (_ABANDONMENT[0] * _ABANDONMENT[1])
        if line <= load[h2]:              # Region 1: Y → Tier2(하급)
            choice = (1, h1)
            region = 1
        elif load[h2] < absolute:         # Region 2: Y → Tier3(상급)
            choice = (1, h2)
            region = 2
        else:                             # Region 3: R → Tier3(상급)
            choice = (0, h2)
            region = 3
        self.last_meta = {
            "integrated_region": region,
            "line": float(line),
            "absolute": float(absolute),
        }
        return choice

    # ---------------------------------------------------------------- 공개 API
    def select(self, obs):
        counts = self._waiting_counts(obs)
        if not counts.any():
            self.last_meta = {"reason": "no_ry"}
            return [-1, 0, -1]

        for mode in self._mode_order(obs):
            self.last_meta = {}
            if self.method == "Threshold":
                picked = self._threshold(obs, mode, counts)
            elif self.method == "2Step":
                picked = self._two_step(obs, mode, counts)
            elif self.method == "PIH":
                picked = self._pih(obs, mode, counts)
            else:
                picked = self._integrated(obs, mode, counts)
            if picked is None:
                continue
            p_class, h_idx = picked
            meta = dict(self.last_meta or {})
            meta.update(
                {
                    "method": self.method,
                    "mode_rule": self.mode_rule,
                    "mode": mode,
                    "class": int(p_class),
                    "hospital": int(h_idx),
                }
            )
            self.last_meta = meta
            return [int(p_class), int(h_idx) + 1, int(mode)]

        self.last_meta = {"reason": "no_feasible_mode_or_hospital"}
        return [-1, 0, -1]
