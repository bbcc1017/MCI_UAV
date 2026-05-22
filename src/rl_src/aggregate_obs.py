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
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AggregateObsWrapper(gym.ObservationWrapper):
    """Dict obs 의 환자/차량 행렬을 집계 통계로 치환한다.

    p_states 컬럼 = [class, rescued, move, moved, cared]
    amb/uav_states 컬럼 = [destination, time_remaining, severity]
    """
    N_CLASS = 4   # Red/Yellow/Green/Black
    N_STAGE = 5   # not_rescued / at_site / in_transport / at_hospital / done

    def __init__(self, env):
        super().__init__(env)
        sp = env.observation_space.spaces
        big = float(np.iinfo(np.int32).max)
        self.observation_space = spaces.Dict({
            "patient_agg":   spaces.Box(0.0, big, shape=(self.N_CLASS * self.N_STAGE,),
                                        dtype=np.float32),
            "vehicle_agg":   spaces.Box(0.0, np.inf, shape=(10,), dtype=np.float32),
            "h_states":      sp["h_states"],
            "p_sent":        sp["p_sent"],
            "p_at_site":     sp["p_at_site"],
            "n_amb_at_site": sp["n_amb_at_site"],
            "n_uav_at_site": sp["n_uav_at_site"],
            "time":          sp["time"],
        })

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
        stage[(moved == 1) & (cared == 0)] = 3
        stage[cared == 1] = 4
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
        min_t = float(tr[busy].min()) if n_busy > 0 else 0.0
        mean_t = float(tr[busy].mean()) if n_busy > 0 else 0.0
        n_crit = float(np.sum((sev == 1) | (sev == 2)))  # Red/Yellow 수송 중
        return np.array([float((~busy).sum()), n_busy, min_t, mean_t, n_crit],
                        dtype=np.float32)

    def observation(self, obs: dict) -> dict:
        veh = np.concatenate([
            self._fleet_agg(np.asarray(obs["amb_states"])),
            self._fleet_agg(np.asarray(obs["uav_states"])),
        ])
        return {
            "patient_agg":   self._patient_agg(np.asarray(obs["p_states"])),
            "vehicle_agg":   veh,
            "h_states":      obs["h_states"],
            "p_sent":        obs["p_sent"],
            "p_at_site":     obs["p_at_site"],
            "n_amb_at_site": obs["n_amb_at_site"],
            "n_uav_at_site": obs["n_uav_at_site"],
            "time":          obs["time"],
        }
