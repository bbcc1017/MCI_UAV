"""스코어 정책용 후보 특징 φ (플랜 v2 추출 트랙 B0).

최강 RL(포인터 head)의 dest 선택은 본질이 "적격 병원(과 모드)을 특징으로 점수화해 최선을
고르는" 랭킹 문제다(pointer_policy 의 S[d,m]). 이 모듈은 그 스코어를 **지역불변·해석가능한
선형 스코어** `score = w·φ(h,ctx)` 로 표현하기 위한 후보 특징 φ 를 만든다. φ 가 전부 지역불변
(정규화 ETA·상대 발송량·부하비 등)이라 argmax 가 순열등변 → **전국 단일 스코어 정책**이 성립.

설계 원칙(재사용 원천과 정합):
  * ⚠️ **평탄 obs 슬라이싱 금지**(F=4/7 오독 함정). ctx 피처는 전부 dict obs
    (`env.en_manager.get_full_obs()`)와 `en_properties` 에서 재계산한다.
  * ETA·raw 시간·tier·max_send 는 시나리오 상수 → `loadbalance_heuristic.get_static_eta`
    (최근접=1 정규화, MCI_ETA_CLIP)·`program_policy._raw_times`(raw 분) 를 그대로 재사용
    → LB/프로그램 정책과 **동일한 값**(쌍비교 불변식·selftest 등가의 근거).
  * ρ 는 `hospital_feature_wrapper` 글로벌 §정의식(R/Y 생애단계 0~2 / 잔여 유효용량,
    게이트 추종, 클립 MCI_RHO_CLIP)과 동일식으로 재계산(`AggregateObsWrapper._patient_agg`).
  * z-score(학습통계) 정규화 **금지** — 전부 물리단위·사전유계라 지역/규모 간 이식 가능.

φ 정의 (K=12, 0-based; 스케일·클립은 아래 표 그대로):
  ┌ idx  이름          정의식                                   단위/범위(클립)
  │  0   eta          모드별 정규화 ETA(get_static_eta)          최근접=1, ≤MCI_ETA_CLIP(10)
  │  1   eta_rank     적격 내 (자기보다 가까운 수)/n_elig        0(최근접)~<1
  │  2   is_nearest   적격 내 eta 최소면 1                        {0,1}
  │  3   p_sent_8     min(p_sent, MCI_PSENT_CLIP=32)/8           0~4  (내가 보낸 누적)
  │  4   p_sent_rel   (p_sent − 적격평균)/4                       상대 발송량
  │  5   in_flight_4  in_flight/4                                 0~   (그 병원행 이송중)
  │  6   occ_ratio    clip((occ+in_flight)/max_send,0,4)          실시간 부하비
  │  7   is_tier3     hos_tier==3                                 {0,1}
  │  8   rho_psent    rho × p_sent_8                              교호(부하×혼잡)
  │  9   dens_psent   p_sent_8 / max(n_elig,1) × 47               밀도 교호(H=47 앵커)
  │ 10   is_uav       모드=UAV 면 1                               {0,1}
  │ 11   dt_uav       (t_uav − t_amb)/30  (raw 분)                +면 UAV 가 느림
  └ (정규화 ETA 는 모드 간 비교 불가 → 모드 결정용 dt_uav 만 raw 분으로 유지)

  ※ φ 는 등급 c 에 직접 의존하지 않는다(적격/마스크가 이미 tier3·헬기장·게이트를 인코딩) —
    시그니처의 c 는 호출 규약 호환용(현재 미사용).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
# EntityManager(in_flight_by_hospital) 를 위해 sim_src 도 path 에 (evaluate import 전이라도 안전)
_SIM_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "sim_src"))
if _SIM_SRC not in sys.path:
    sys.path.insert(0, _SIM_SRC)

from loadbalance_heuristic import get_static_eta, H_DEFAULT  # noqa: E402
from program_policy import _raw_times                          # noqa: E402
from aggregate_obs import AggregateObsWrapper                  # noqa: E402
from EntityManager import EntityManager                        # noqa: E402

PHI_NAMES = [
    "eta", "eta_rank", "is_nearest", "p_sent_8", "p_sent_rel", "in_flight_4",
    "occ_ratio", "is_tier3", "rho_psent", "dens_psent", "is_uav", "dt_uav",
]
K_PHI = len(PHI_NAMES)


def _clip(name: str, default: str) -> float:
    """클립/스케일 노브 — 호출 시점에 env 조회(worker 가 import 후 env 를 세팅해도 반영)."""
    return float(os.environ.get(name, default))


# ---------------------------------------------------------------- ctx 헬퍼
def compute_static(env) -> dict:
    """시나리오 상수(1회 계산 후 정책이 en_manager 아이덴티티로 캐시) — ETA·raw시간·tier·용량.

    env 는 unwrapped env(정책 fn 이 받는 env 인자). get_static_eta/_raw_times 재사용."""
    hp = env.en_manager.en_properties['hospital']
    H = int(hp['hos_num'])
    is_tier3 = (np.asarray(hp['hos_tier']).reshape(-1) == 3).astype(np.float32)
    max_send = np.asarray(hp['hos_max_send'], float).reshape(-1)
    eta_amb, eta_uav = get_static_eta(env, H)          # 최근접=1 정규화 + MCI_ETA_CLIP
    t_amb, t_uav = _raw_times(env, H)                  # raw 분(모드 간 비교·게이트용)
    return {"H": H, "is_tier3": is_tier3, "max_send": max_send,
            "eta_amb": np.asarray(eta_amb, float), "eta_uav": np.asarray(eta_uav, float),
            "t_amb": np.asarray(t_amb, float), "t_uav": np.asarray(t_uav, float)}


def build_ctx(env, static: dict | None = None, dobs: dict | None = None) -> dict:
    """정책 결정 시점 컨텍스트 — 상수(static) + 동적(p_sent·in_flight·occ·cap_remain·rho).

    rho = hospital_feature_wrapper 글로벌 §정의식과 동일: 잔여 긴급부하(R/Y 생애단계 0~2)
    / (잔여 유효용량 합 + 1), 클립 MCI_RHO_CLIP. cap_remain 은 MCI_CAP_GATE(occ/psent) 추종.
    """
    if static is None:
        static = compute_static(env)
    H = static["H"]
    if dobs is None:
        dobs = env.en_manager.get_full_obs()
    p_sent = np.asarray(dobs['p_sent'], float).reshape(-1)
    h_states = np.asarray(dobs['h_states'], float)
    occ = h_states[:, 2].astype(float)                                    # 입원 census
    in_flight = EntityManager.in_flight_by_hospital(dobs, H).astype(float)  # 그 병원행 이송중
    comms = os.environ.get("MCI_CAP_GATE", "occ").strip().lower() != "psent"
    cap_used = (occ + in_flight) if comms else p_sent
    cap_remain = np.maximum(static["max_send"] - cap_used, 0.0)
    pa = AggregateObsWrapper._patient_agg(np.asarray(dobs['p_states']))[:10]  # R/Y × 5단계
    urgent = float(pa[0] + pa[1] + pa[2] + pa[5] + pa[6] + pa[7])            # 생애단계 0~2
    rho = min(urgent / (float(cap_remain.sum()) + 1.0), _clip("MCI_RHO_CLIP", "8"))
    ctx = dict(static)
    ctx.update(p_sent=p_sent, occ=occ, in_flight=in_flight,
               cap_remain=cap_remain, rho=float(rho), comms=comms)
    return ctx


def compute_rho(env, dobs: dict | None = None) -> float:
    """ρ(잔여 긴급부하/잔여 유효용량) 단일 스칼라 — T_lookup 등 컨텍스트 훅용."""
    return build_ctx(env, dobs=dobs)["rho"]


# ---------------------------------------------------------------- φ 빌더
def build_phi(env, c, m_or_None, elig_idx, ctx: dict | None = None) -> np.ndarray:
    """후보 특징 행렬 (n_cand, K).

    Parameters
    ----------
    env : unwrapped env (정책 fn 의 env 인자).
    c   : 등급(0R/1Y). φ 는 c 비의존이나 호출 규약 호환용으로 받음.
    m_or_None : int(0/1) 이면 고정 모드 — elig_idx=적격 병원 인덱스 배열, 후보=(h, m).
                None 이면 결합(joint) — elig_idx=(h, m) 쌍 배열, 후보 = 그 쌍들.
    elig_idx  : 위 규약에 따른 후보 목록.
    ctx : build_ctx 결과(재사용 주입). None 이면 내부 계산.

    상대 특징(eta_rank/is_nearest/p_sent_rel/dens_psent 의 n_elig)은 **넘어온 후보집합**
    기준으로 계산한다(호출측이 적격 전체를 넘겨야 '적격 내' 의미가 성립).
    """
    if ctx is None:
        ctx = build_ctx(env, static=compute_static(env))
    eta_amb, eta_uav = ctx["eta_amb"], ctx["eta_uav"]
    t_amb, t_uav = ctx["t_amb"], ctx["t_uav"]
    p_sent, in_flight = ctx["p_sent"], ctx["in_flight"]
    occ, max_send, is_tier3, rho = ctx["occ"], ctx["max_send"], ctx["is_tier3"], ctx["rho"]

    elig_idx = np.asarray(elig_idx)
    if m_or_None is None:                       # joint: (h, m) 쌍
        hs = elig_idx[:, 0].astype(int)
        ms = elig_idx[:, 1].astype(int)
    else:                                       # 고정 모드
        hs = elig_idx.astype(int).reshape(-1)
        ms = np.full(hs.shape[0], int(m_or_None), dtype=int)
    n = hs.shape[0]
    n_elig = max(n, 1)
    if n == 0:
        return np.zeros((0, K_PHI), dtype=np.float32)

    ps_clip = _clip("MCI_PSENT_CLIP", "32")
    or_clip = _clip("MCI_OCC_RATIO_CLIP", "4")

    eta = np.where(ms == 0, eta_amb[hs], eta_uav[hs]).astype(float)   # 모드별 정규화 ETA
    # eta_rank = (자기보다 엄격히 가까운 후보 수)/n_elig (0=최근접), 동률은 같은 값
    eta_rank = (eta[:, None] > eta[None, :]).sum(axis=1) / n_elig
    is_nearest = np.zeros(n, np.float32)
    is_nearest[int(np.argmin(eta))] = 1.0                            # 동률 시 첫 최소
    ps = np.minimum(p_sent[hs], ps_clip)
    ps8 = ps / 8.0
    ps_rel = (p_sent[hs] - float(p_sent[hs].mean())) / 4.0
    inf4 = in_flight[hs] / 4.0
    occ_ratio = np.clip((occ[hs] + in_flight[hs]) / np.maximum(max_send[hs], 1.0), 0.0, or_clip)
    t3 = is_tier3[hs].astype(float)
    rho_ps = rho * ps8
    dens_ps = ps8 / n_elig * 47.0                                    # H=47 앵커(밀도 교호)
    is_uav = ms.astype(float)
    dt_uav = (t_uav[hs] - t_amb[hs]) / 30.0                          # raw 분 시간차(모드 결정용)

    return np.stack([eta, eta_rank, is_nearest, ps8, ps_rel, inf4, occ_ratio,
                     t3, rho_ps, dens_ps, is_uav, dt_uav], axis=1).astype(np.float32)
