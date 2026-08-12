"""LB-T3 비교군 중 원 휴리스틱에 의존하지 않는 최소 부하분산 정책.

``LB3_AGNOSTIC_RR_FASTEST``의 의도는 세 축을 가능한 한 단순하게 만드는 것이다.

* class: R/Y가 모두 대기하면 round-robin(첫 동률만 R) — 중증도 가중 없음
* mode: 현재 가용한 AMB/UAV 중 선택 class의 최선 병원까지 실제 평균 ETA가 짧은 수단
* destination: ``p_sent < T``인 hard-mask 적격 병원 중 해당 수단 ETA 최소
* overflow: 모든 적격 병원이 T 이상이면 최소 p_sent, 그 안에서 ETA 최소

T는 기존 ``make_cap_policy``와 같은 soft balancing threshold다. 즉 T=3이면 병원당
세 번째 누적 발송까지 우선 허용하고, 전 병원 도달 뒤에는 최소발송 병원으로 계속 보낸다.
Red→Tier3, UAV→helipad, 용량 게이트는 재계산하지 않고 공통 action mask를 그대로 쓴다.
"""
from __future__ import annotations

import numpy as np

from loadbalance_heuristic import H_DEFAULT, _codec_from_mask


def _absolute_eta(env_unwrapped, H: int) -> tuple[np.ndarray, np.ndarray]:
    """수단간 비교가 가능한 분 단위 ETA(평균 이송시간+인계시간)."""
    props = env_unwrapped.en_manager.en_properties
    ap = props["ambulance"]
    up = props["uav"]

    amb = np.asarray(ap.get("amb_HtoS_t", (np.full(H, np.inf),))[0], dtype=float)
    uav = np.asarray(up.get("uav_HtoS_t", (np.full(H, np.inf),))[0], dtype=float)
    if len(amb) != H:
        amb = np.full(H, np.inf, dtype=float)
    if len(uav) != H:
        uav = np.full(H, np.inf, dtype=float)
    amb = amb + float(ap.get("amb_handover_time", 0.0))
    uav = uav + float(up.get("uav_handover_time", 0.0))
    return amb, uav


class _State:
    def __init__(self, H: int):
        self.H = int(H)
        self.en_manager = None
        self.encode = None
        self.eta = None
        self.next_class = 0

    def sync(self, env_unwrapped, mask_len: int) -> None:
        if self.en_manager is env_unwrapped.en_manager:
            return
        self.en_manager = env_unwrapped.en_manager
        self.encode = _codec_from_mask(mask_len, self.H)
        self.eta = _absolute_eta(env_unwrapped, self.H)
        self.next_class = 0


def make_agnostic_lb_policy(T: float = 3.0, H: int = H_DEFAULT):
    """원 HEUR64 규칙 없이 동작하는 severity/mode-agnostic LB 정책."""
    state = _State(H)

    def policy(ro, mask, env):
        mask = np.asarray(mask, dtype=bool)
        state.sync(env, len(mask))
        dobs = env.en_manager.get_full_obs()
        waits = [len(dobs["p_wait"][c][0]) for c in range(2)]

        if waits[0] and waits[1]:
            p_class = int(state.next_class)
            state.next_class = 1 - state.next_class
        elif waits[0]:
            p_class = 0
        elif waits[1]:
            p_class = 1
        else:
            valid = np.flatnonzero(mask)
            return int(valid[0]) if valid.size else 0

        p_sent = np.asarray(dobs["p_sent"], dtype=float)
        candidates = []
        for mode in (0, 1):
            eligible = np.asarray(
                [
                    h
                    for h in range(H)
                    if (lambda a: a < len(mask) and mask[a])(
                        state.encode(p_class, h + 1, mode)
                    )
                ],
                dtype=int,
            )
            if not len(eligible):
                continue
            under = eligible[p_sent[eligible] < float(T)]
            if len(under):
                pool = under
                overflow = False
            else:
                min_sent = float(p_sent[eligible].min())
                pool = eligible[p_sent[eligible] == min_sent]
                overflow = True
            eta = state.eta[mode]
            hospital = int(pool[np.argmin(eta[pool])])
            candidates.append((float(eta[hospital]), mode, hospital, overflow))

        if not candidates:
            valid = np.flatnonzero(mask)
            action = int(valid[0]) if valid.size else 0
            policy.last_meta = {
                "class": p_class,
                "fallback": True,
                "T": float(T),
            }
            return action

        eta, mode, hospital, overflow = min(candidates, key=lambda x: (x[0], x[1], x[2]))
        action = int(state.encode(p_class, hospital + 1, mode))
        if action >= len(mask) or not mask[action]:
            raise RuntimeError("LB3-AGN이 hard mask 밖 행동을 선택함")
        policy.last_meta = {
            "class": p_class,
            "mode": mode,
            "hospital": hospital,
            "eta_minutes": eta,
            "overflow": overflow,
            "fallback": False,
            "T": float(T),
        }
        return action

    policy.policy_name = "LB3_AGNOSTIC_RR_FASTEST"
    policy.last_meta = None
    return policy

