"""환자/차량 행렬을 집계 통계로 압축하는 ObservationWrapper (피드백 3 — obs 차원 축소).

원본 obs 의 p_states(incident_size×5)·amb_states·uav_states 는 개별 엔티티
행렬이라 차원이 크다(전체 obs 의 대부분). 같은 등급 환자·같은 함대 차량은
교환 가능(exchangeable)하므로 정책 결정에 필요한 것은 '개별 환자 #37' 이 아니라
'어떤 상태의 환자/차량이 몇이나 있는가'의 집계뿐이다.

  p_states (incident_size,5) → patient_agg (20,)  = 4등급 × 5 생애단계 카운트
  amb_states / uav_states    → vehicle_agg (10,)  = 함대(amb,uav)별 5개 통계
  h_states (H,3) · p_sent (H) → 그대로 유지 (행동=병원 선택에 직접 필요)

obs flat 차원: 약 856 → 약 221 (H=46 기준). 특히 환자 부분이 incident_size 에
더는 비례하지 않아 사고 규모에 불변(scale-invariant)이 된다.

sim_src 무수정 — rl_src 레벨 gym.ObservationWrapper 로만 구현.

────────────────────────────────────────────────────────────────────────
obs_reduced v2 (2026-05-31, 환경변수 MCI_OBS_VARIANT) — obs ablation 후속:
  ablation 결과 (마스크 적용) — h_states 의 queue 44/46·occ 39/46 차원이 정책에
  거의 기여 안 함(<0.01). 반면 라우팅 최대 요인은 차량 ETA 인데 *병원별 목적지
  거리/ETA* 는 obs 에 없었다. 그래서 토큰으로 가지치기/추가를 조합한다:

  MCI_OBS_VARIANT 토큰 ('+' 또는 ',' 구분, 미설정/"base"=원본 221):
    - "noqueue" : h_states 에서 queue 컬럼 제거 (idle+occ 유지)
    - "idle"    : h_states 를 idle 컬럼만 (queue+occ 제거)
    - "eta"     : 병원별 amb_eta(H)+uav_eta(H) 추가 (en_properties 읽기, 분 단위)
    - "etanorm" : eta 를 그 시나리오 최소 ETA 로 정규화(상대값, 지역간 스케일 제거)
  예: "idle+eta", "noqueue+eta", "eta".

  ETA 는 시나리오 상수(시간 불변)라 wrapper 인스턴스당 1회 캐시.
  미설정 시 동작/차원 100% 동일 → 기존 f3 모델 평가 호환.
"""
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _parse_variant():
    raw = os.environ.get("MCI_OBS_VARIANT", "").strip().lower()
    if raw in ("", "base"):
        return set()
    return set(t for t in raw.replace(",", "+").split("+") if t)


def _comms_available():
    """통신 가용 여부 = MCI_CAP_GATE(occ=가용/psent=단절). 2026-07-03 통신축 재정의:
    단절 시 원격 차량의 실시간 잔여시간(텔레메트리)은 지득 불가 → _fleet_agg 서 0 처리.
    (가용수·운행수·수송 중증도는 현장에서 셀 수 있는 정보라 유지 — 출발은 현장이 시켰음.)"""
    return os.environ.get("MCI_CAP_GATE", "occ").strip().lower() != "psent"


def _cared_visible():
    """환자 '입원완료'(cared, 생애단계 4)를 obs 에서 보이게 둘지 — 통신 단절 실험 토글.

    기본=보임(병원과 실시간 통신 가능: 입원여부를 안다).
    MCI_CARED_OBS=0/off/hide/fold → 단계4를 단계3(병원도착)으로 흡수한다. 통신이 끊기면
    현장은 '차량이 환자를 전달'한 것까지만 알고 병원이 실제 입원시켰는지는 모르기 때문
    (보수적 현장-한정 정보). cap_remain 의 psent 게이트(MCI_CAP_GATE)와 짝이 되는 축.
    단계 수(5)는 유지 → 단계4 칸이 항상 0 이 될 뿐 obs 차원·모델 호환은 그대로."""
    v = os.environ.get("MCI_CARED_OBS", "1").strip().lower()
    return v not in ("0", "off", "hide", "fold", "false", "no")


class AggregateObsWrapper(gym.ObservationWrapper):
    """Dict obs 의 환자/차량 행렬을 집계 통계로 치환한다.

    p_states 컬럼 = [class, rescued, move, moved, cared]
    amb/uav_states 컬럼 = [destination, time_remaining, severity]
    h_states 컬럼 = [idle, queue, occ]
    """
    N_CLASS = 4   # Red/Yellow/Green/Black
    N_STAGE = 5   # not_rescued / at_site / in_transport / at_hospital / done

    def __init__(self, env):
        super().__init__(env)
        sp = env.observation_space.spaces
        big = float(np.iinfo(np.int32).max)
        H = int(sp["h_states"].shape[0])
        self.H = H

        toks = _parse_variant()
        self._drop_queue = ("noqueue" in toks) or ("idle" in toks)
        self._drop_occ = ("idle" in toks)
        self._add_eta = ("eta" in toks) or ("etanorm" in toks)
        self._eta_norm = ("etanorm" in toks)
        # 공격적 가지치기 토큰
        self._drop_black = ("noblack" in toks)   # patient_agg Black 등급 제거 (20→15)
        self._psent_mode = ("none" if "nopsent" in toks else
                            ("stat" if "psentstat" in toks else "full"))  # p_sent 제거/집계/원본
        self._variant_toks = toks
        # h_states 유지 컬럼 (정렬 [0=idle, 1=queue, 2=occ])
        keep = [0]
        if not self._drop_queue:
            keep.append(1)
        if not self._drop_occ:
            keep.append(2)
        self._h_keep = sorted(keep)
        self._eta_cache = None  # (amb_eta(H,), uav_eta(H,)) lazy
        self._add_route = ("routescore" in toks)  # 병원별 (가용용량/거리) 동적 라우팅 점수
        self._route_cache = None  # (max_send, d_road, d_euc) lazy

        pa_classes = (self.N_CLASS - 1) if self._drop_black else self.N_CLASS
        d = {
            "patient_agg":   spaces.Box(0.0, big, shape=(pa_classes * self.N_STAGE,),
                                        dtype=np.float32),
            "vehicle_agg":   spaces.Box(0.0, np.inf, shape=(10,), dtype=np.float32),
        }
        if len(self._h_keep) == 3:
            d["h_states"] = sp["h_states"]
        else:
            d["h_states"] = spaces.Box(0.0, big, shape=(H, len(self._h_keep)), dtype=np.float32)
        if self._psent_mode == "full":
            d["p_sent"] = sp["p_sent"]
        elif self._psent_mode == "stat":
            d["p_sent"] = spaces.Box(0.0, big, shape=(3,), dtype=np.float32)  # [sum, nnz, max]
        # "none": p_sent 키 제거 (병원 capa 는 action mask 가 처리)
        d["p_at_site"]     = sp["p_at_site"]
        d["n_amb_at_site"] = sp["n_amb_at_site"]
        d["n_uav_at_site"] = sp["n_uav_at_site"]
        d["time"]          = sp["time"]
        if self._add_eta:
            d["amb_eta"] = spaces.Box(0.0, np.inf, shape=(H,), dtype=np.float32)
            d["uav_eta"] = spaces.Box(0.0, np.inf, shape=(H,), dtype=np.float32)
        if self._add_route:
            d["route_amb"] = spaces.Box(0.0, np.inf, shape=(H,), dtype=np.float32)
            d["route_uav"] = spaces.Box(0.0, np.inf, shape=(H,), dtype=np.float32)
        self.observation_space = spaces.Dict(d)

    @classmethod
    def _patient_agg(cls, p_states: np.ndarray) -> np.ndarray:
        """(N,5) → (20,) : 4등급 × 5생애단계 카운트."""
        agg = np.zeros((cls.N_CLASS, cls.N_STAGE), dtype=np.float32)
        if p_states.shape[0] == 0:
            return agg.reshape(-1)
        c = p_states[:, 0].astype(int)
        rescued, move, moved, cared = (p_states[:, 1], p_states[:, 2],
                                       p_states[:, 3], p_states[:, 4])
        # 상호배타 생애단계: 0 미구조 / 1 현장대기 / 2 이송중 / 3 병원도착 / 4 완료
        stage = np.zeros(p_states.shape[0], dtype=int)
        stage[(rescued == 1) & (move == 0)] = 1
        stage[(move == 1) & (moved == 0)] = 2
        if _cared_visible():
            stage[(moved == 1) & (cared == 0)] = 3
            stage[cared == 1] = 4
        else:
            # 통신단절: 입원완료(cared)를 못 봄 → 병원도착(3)으로 흡수(단계4 항상 0)
            stage[moved == 1] = 3
        for ci in range(cls.N_CLASS):
            for si in range(cls.N_STAGE):
                agg[ci, si] = np.sum((c == ci) & (stage == si))
        return agg.reshape(-1)

    @staticmethod
    def _fleet_agg(states: np.ndarray) -> np.ndarray:
        """(n,3) → (5,) : [가용수, 운행수, 최단가용시간, 평균가용시간, 위중환자수송수]."""
        if states.shape[0] == 0:
            return np.zeros(5, dtype=np.float32)
        tr = states[:, 1].astype(np.float32)
        sev = states[:, 2].astype(np.float32)
        busy = tr > 1e-6
        n_busy = float(busy.sum())
        if _comms_available():
            min_t = float(tr[busy].min()) if n_busy > 0 else 0.0
            mean_t = float(tr[busy].mean()) if n_busy > 0 else 0.0
        else:
            # 통신단절: 운행중 차량의 실시간 잔여시간은 원격 텔레메트리 → 지득 불가(0)
            min_t = mean_t = 0.0
        n_crit = float(np.sum((sev == 1) | (sev == 2)))  # Red/Yellow 수송 중(현장이 태움=지득)
        return np.array([float((~busy).sum()), n_busy, min_t, mean_t, n_crit],
                        dtype=np.float32)

    def _get_eta(self):
        """병원별 amb/uav ETA(분) — 시나리오 상수, env.unwrapped 읽기 후 캐시."""
        if self._eta_cache is None:
            ep = self.env.unwrapped.en_manager.en_properties
            amb = np.asarray(ep["ambulance"]["amb_HtoS_t"][0], dtype=np.float32).reshape(-1)
            uav = np.asarray(ep["uav"]["uav_HtoS_t"][0], dtype=np.float32).reshape(-1)
            # 안전: 길이 H 보정
            amb = np.resize(amb, self.H).astype(np.float32)
            uav = np.resize(uav, self.H).astype(np.float32)
            if self._eta_norm:
                amb = amb / (amb[amb > 0].min() if np.any(amb > 0) else 1.0)
                uav = uav / (uav[uav > 0].min() if np.any(uav > 0) else 1.0)
            self._eta_cache = (amb, uav)
        return self._eta_cache

    def _get_route_static(self):
        """병원별 (max_send, 도로거리, 직선거리) — 시나리오 상수, 캐시. 거리는 +1 로 0division 방지."""
        if self._route_cache is None:
            ep = self.env.unwrapped.en_manager.en_properties["hospital"]
            ms = np.resize(np.asarray(ep["hos_max_send"], np.float32).reshape(-1), self.H)
            dr = np.resize(np.asarray(ep["d_HtoS_road"], np.float32).reshape(-1), self.H) + 1.0
            de = np.resize(np.asarray(ep["d_HtoS_euc"], np.float32).reshape(-1), self.H) + 1.0
            self._route_cache = (ms, dr, de)
        return self._route_cache

    def observation(self, obs: dict) -> dict:
        veh = np.concatenate([
            self._fleet_agg(np.asarray(obs["amb_states"])),
            self._fleet_agg(np.asarray(obs["uav_states"])),
        ])
        h = np.asarray(obs["h_states"], dtype=np.float32)
        if not _comms_available():
            # 통신단절: 병원 실시간 상태(idle/queue/occ)는 지득 불가 → 0 (차원 유지)
            h = np.zeros_like(h)
        h_out = h if len(self._h_keep) == 3 else h[:, self._h_keep]
        pa = self._patient_agg(np.asarray(obs["p_states"]))
        if self._drop_black:  # Black 등급(class 3) 제거 → (3,5)
            pa = pa.reshape(self.N_CLASS, self.N_STAGE)[:self.N_CLASS - 1].reshape(-1)
        out = {
            "patient_agg":   pa,
            "vehicle_agg":   veh,
            "h_states":      h_out,
            "p_at_site":     obs["p_at_site"],
            "n_amb_at_site": obs["n_amb_at_site"],
            "n_uav_at_site": obs["n_uav_at_site"],
            "time":          obs["time"],
        }
        if self._psent_mode == "full":
            out["p_sent"] = obs["p_sent"]
        elif self._psent_mode == "stat":
            ps = np.asarray(obs["p_sent"], dtype=np.float32).reshape(-1)
            out["p_sent"] = np.array(
                [ps.sum(), float((ps > 0).sum()), ps.max() if ps.size else 0.0],
                dtype=np.float32)
        # "none": p_sent 미포함
        if self._add_eta:
            amb_eta, uav_eta = self._get_eta()
            out["amb_eta"] = amb_eta
            out["uav_eta"] = uav_eta
        if self._add_route:
            ms, dr, de = self._get_route_static()
            ps = np.asarray(obs["p_sent"], dtype=np.float32).reshape(-1)
            avail = np.maximum(ms - ps, 0.0)  # 동적 가용용량 (p_sent 시변)
            out["route_amb"] = (avail / dr).astype(np.float32)  # 가깝고 여유 큰 병원일수록 높음
            out["route_uav"] = (avail / de).astype(np.float32)
        return out
