"""Phase 3a — 특징기반 병원 obs 래퍼 (Option A: 전체 H 병원, 슬롯축소 없음).

랩 피드백 #1(ETA=lognormal 평균)·#2(tier를 obs로)·#3(local/comms 정보수준)을 RL obs 로
해결한다. 인덱스 기반 h_states/p_sent 대신 **병원당 특징 엔티티 행렬 (H, F)** 로 표현 →
정책이 병원 "특징"을 읽어 일반화하고, VIPER 트리가 해석 가능한 특징으로 분기한다.

설계 원칙 (Phase 1/2 와 동일):
  * sim_src 동역학·env_wrapper.py 코어·multi_region_env.py **무수정**.
  * FlattenAndDiscreteWrapper 의 decode/encode/fixed-mode·tier 마스킹과 **동치** 동작.
  * action 은 Discrete(H+1=25) 유지(슬롯→idx 역매핑 없음 — H 고정이라 차원통일 이미 해결).
  * 자체적으로 compact obs 를 만들므로 MCI_REDUCED_OBS(AggregateObsWrapper) 불필요
    — 다만 글로벌 집계는 AggregateObsWrapper._patient_agg/_fleet_agg 를 재사용한다.

병원당 특징 F (MCI_OBS_VARIANT 로 토글):
  essential(기본): [is_tier3, cap_remain, eta_amb, eta_uav] — 중복 제거 최소핵심(F=4).
  essential+load+valid(v6 A3): essential+load(F=7) + valid(1=실병원/0=패딩) 8열 — MCI_H_PAD
  essential+load+valid+raw(v18 E6): 위와 **차원 402 동일**, 인코딩만 물리단위 —
    eta_amb/uav = 분/MCI_ETA_RAW_NORM(60) (좌표별 정규화 없음, 전역 상수만),
    occ_ratio → load_raw = clip(census+in_flight, 0, MCI_LOAD_CLIP=32) 명수.
  essential+load+valid+sat(v19): raw 의 ETA 축만 교체. eta = t/(t+MCI_ETA_SAT_T0(30)).
    raw 의 결함 = 헬기장 26칸(AMB 중위 264분)의 꼬리가 동적범위를 먹어 결정 밴드
    (교사 선택 p99 105분) 해상도가 0.086σ 로 구 정규화 0.156σ 보다도 나쁘다.
    포화형은 같은 밴드 0.620σ(7.2배)·동률 0%·꼬리 순위 보존. 부하열은 raw 와 동일.
  essential+load+valid+{raw|sat}+slot(v19): 용량축 교체(차원 402 불변).
    cap_remain_c → idle(빈 수술실 수) · load_raw → queue(대기 인원).
    근거: 실제 서비스 슬롯은 수술실수(중위 2)인데 obs 앵커는 max_send(중위 14)=7배 과대,
    diversion 게이트는 교사 37,000결정서 0.00% 발동(cap_remain 사실상 死열).
    부하항 제거 시 활동 병원 82% 가 n_idle==0(대기열 최대 15) → PDR 0.3028,
    CARD 켜면 36%(최대 3) → 0.1464. 부하항의 가치 = 수술실을 비워 두는 것.
    병원패딩 시 포인터 마스크드 풀링이 패딩 행을 식별(PadAwareVecNormalize 로 valid 열 정규화 면제).
  full(ablation):  [is_tier3, helipad, eta_amb, eta_uav, idle, queue, occ, cap_remain] (F=8).
  local/comms(ablation): 위 8열을 정적4/실시간4 로 분리.
  - 미설정/"essential" → essential. helipad 는 UAV 마스크가 강제(중복), idle/occ 는
    cap_remain 과 affine 중복, queue 는 ablation 무신호라 essential 에서 제외.
  - ETA = amb/uav_HtoS_t[0](=lognormal 평균=사전계산 deterministic, #1). 시나리오 최소
    ETA 로 정규화(최근접=1) 후 MCI_ETA_CLIP(기본 10배) 클립. 정적이라 1회 캐시.
  - cap_remain = max(hos_max_send - cap_used, 0). cap_used=occ(기본)|p_sent(psent게이트).

글로벌 특징 (병원 비의존): patient_agg(R/Y 2등급×5단계=10) + vehicle_agg(10) + time(1) = 21.
  - Green/Black 은 행동대상 아님(R/Y 소진+구조완료 시 sim 코어가 자동일괄 이송) → patient_agg
    에서 제거(R/Y 만). p_at_site·n_amb/uav_at_site 는 patient_agg stage1·vehicle_agg n_avail
    의 부분집합이라 제거(중복). raw h_states/p_sent 는 엔티티로 흡수.

행동 마스크: tier(Red→Tier3, MCI_TIER_MASK) + Green 이송 차단(MCI_GREEN_MASK) + helipad/capa
  (joint). train 스크립트에서 ActionMasker 와 함께 사용(FlattenAndDiscreteWrapper 대체).
"""
from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from aggregate_obs import AggregateObsWrapper  # _patient_agg / _fleet_agg 재사용

# 병원당 특징 열 정의 (순서 고정)
# essential(기본): 중복 제거 최소핵심. helipad(마스크중복)·idle/occ(cap_remain과 affine중복)·
#   queue(ablation 무신호) 제거 → capability+여유+AMB도달+UAV도달.
_ESSENTIAL_COLS = ["is_tier3", "cap_remain", "eta_amb", "eta_uav"]
_LOCAL_COLS = ["is_tier3", "helipad", "eta_amb", "eta_uav"]   # 정적 사전지식 (ablation)
_COMMS_COLS = ["idle", "queue", "occ", "cap_remain"]          # 실시간 동적 (ablation)
# essential+load (플랜 v2 L2, 2026-07-04): "LB 가 쓰는 신호를 RL 에게" — 승리한 발송상한
# 규칙의 결정 신호(p_sent 0~수십 스케일)와 in-flight·부하비를 결정 스케일 그대로 노출.
#   p_sent_c  = min(p_sent, MCI_PSENT_CLIP=32)           내가 보낸 누적(현장 지득 — psent 게이트에도 유지)
#   in_flight = 그 병원행 이송중 차량 수(출발은 현장이 시킴 = 지득)
#   occ_ratio = clip((census+in_flight)/max_send, 0, MCI_OCC_RATIO_CLIP=4)  실시간 부하비(통신 필요)
# + cap_remain 을 클립본(min(·, MCI_CAPREMAIN_CLIP=32))으로 교체 — max_send(≈670) 앵커로
#   VecNorm 분산이 압살되던 스케일 결함 해소(신규 학습 전용, 구 모델 비호환).
_LOAD_COLS = ["p_sent_c", "in_flight", "occ_ratio"]
# essential+load+ctx (v4 2026-07-10, 근거 docs/v4_알고리즘개선_설계_2026-07-10.md): 스코어 추출
# 지배항(순위·raw 분 모드비교)과 ExIt 천장의 결정론적 타이밍 성분을 관측화.
#   eta_rank_amb = argsort-rank/(H-1) ∈[0,1]          정적 — 절대 ETA 아닌 "순위"가 결정 신호
#   uav_timesave = clip((eta_amb−eta_uav)/eta_amb,±1) 정적 — 모드 비교는 raw 분 비율만 유효
#   arrive_min   = min(그 병원행 이송중 잔여시간)/MCI_ARRIVE_NORM(60), clip MCI_ARRIVE_CLIP(2);
#                  없으면 클립값 — in_flight "대수"에 없던 도착 "타이밍"(현장 발송기록=지득)
_CTX_COLS = ["eta_rank_amb", "uav_timesave", "arrive_min"]
# 글로벌: patient_agg(R/Y 2등급×5단계=10) + vehicle_agg(10) + time(1).
#   p_at_site·n_amb_at_site·n_uav_at_site 는 각각 patient_agg stage1·vehicle_agg n_avail 의
#   정확한 부분집합이라 제거(0손실). Green/Black 은 행동대상 아님(자동일괄 start_GB_transport)이라
#   patient_agg 에서 제거 — R/Y 만 유지.
_GLOBAL_DIM = 10 + 10 + 1
# load 글로벌 확장(+5): ρ(잔여 긴급부하/잔여 유효용량, 클립 MCI_RHO_CLIP=8) — 적응 T=f(ρ) 의
# 표현 근거 / 가용 AMB·UAV 비율 / uav_frac(=uav_num/MCI_UAV_MAX=26, UAV 대수축 신호) /
# t_norm(=min(time/MCI_TIME_NORM=240, 2)).
_LOAD_GLOBAL_EXTRA = 5
# ctx 글로벌 확장(+6, 전부 정적 지역 컨텍스트 — 지역 ID 아님·계산가능 특징이라 holdout 안전):
# [frac(eta≤2), frac(eta≤4)](근접권 밀도) + p90(eta)/clip(산포) + tier3_frac +
# uav_adv_frac(raw 분 기준 UAV 가 빠른 병원 비율) + min(최근접 raw ETA/MCI_ETA_MIN_NORM(30), 2)
# (정규화 eta 가 지운 절대 스케일=도농 신호 복원).
_CTX_GLOBAL_EXTRA = 6
# (v19) anchor 글로벌 확장(+2): 좌표별 정규화가 지운 **절대 스케일**만 되살린다.
# [최근접 AMB ETA(분)/MCI_ANCHOR_NORM(30), 최근접 UAV ETA(분)/동] clip MCI_ANCHOR_CLIP(4).
# 근거: 정규화 obs 의 "1.0"(최근접)이 좌표에 따라 1.4분~40.9분 = 29배로 달라, 전국 단일
# 정책이 "환자 1명 = N 분" 교환율을 표현할 수 없다(v17 진단). 다만 정규화 자체는
# 좌표별 적응 스케일러 역할도 하므로(E6 에서 전역 상수 나눗셈이 0.2109 로 대실패),
# 정규화는 그대로 두고 앵커 2개만 노출해 둘 다 취하는 것이 이 팔의 가설이다.
_ANCHOR_GLOBAL_EXTRA = 2


def _parse_variant():
    raw = os.environ.get("MCI_OBS_VARIANT", "").strip().lower()
    return set(t for t in raw.replace(",", "+").split("+") if t)


class HospitalFeatureWrapper(gym.Wrapper):
    """병원당 특징 엔티티 obs + Discrete action + 결합 마스크 (FlattenAndDiscreteWrapper 대체).

    Parameters
    ----------
    env : MCIEnvironment_gym
        ``make_base_env(cfg)`` 로 만든 raw base env (FlattenAndDiscreteWrapper 미적용).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # ---------- 1) action 차원 (FlattenAndDiscreteWrapper 동등 로직) ----------
        nvec = env.action_space.nvec.tolist()  # [3, H+1, 2]
        assert len(nvec) == 3, f"기대 형식 [class, dest, mode], got {nvec}"
        # (v6) MCI_H_PAD: 병원 슬롯 패딩 — 자연-H(가변 병원 수) 시나리오를 고정 차원 정책이
        # 소비하도록 obs/액션/마스크/코덱 레이아웃만 H_pad 기준으로 확장. sim 은 실 H(_H_real)
        # 로 돌고, 패딩 dest 는 action_masks 가 차단(step 에 방어 가드). 미설정 = 구 동작과
        # 비트동일(pad_smoke.py 로 봉인).
        self._sim_nvec = list(nvec)                     # sim 경계(joint mask reshape)용 실 차원
        self._H_real = nvec[1] - 1
        _pad = os.environ.get("MCI_H_PAD", "").strip()
        self.H = int(_pad) if _pad else self._H_real    # 레이아웃 H(외부 노출: n_hospitals 등)
        if self._H_real > self.H:
            raise ValueError(f"실 병원수 {self._H_real} > MCI_H_PAD {self.H} — "
                             f"H_pad 상향 또는 시나리오 max_hos_num cap 필요")
        self._orig_nvec = [nvec[0], self.H + 1, nvec[2]]  # 레이아웃 nvec(코덱/마스크 shape)

        u = env.unwrapped
        amb_num = int(getattr(u, "amb_num", 0))
        uav_num = int(getattr(u, "uav_num", 0))
        if amb_num == 0 and uav_num > 0:
            self._fixed_mode = 1
            self._effective_nvec = [self._orig_nvec[0], self._orig_nvec[1]]
            mode_label = "UAV-only (mode=1 고정)"
        elif uav_num == 0 and amb_num > 0:
            self._fixed_mode = 0
            self._effective_nvec = [self._orig_nvec[0], self._orig_nvec[1]]
            mode_label = "AMB-only (mode=0 고정)"
        else:
            self._fixed_mode = None
            self._effective_nvec = list(self._orig_nvec)
            mode_label = "AMB+UAV (mode 자유)"
        self._n_actions = int(np.prod(self._effective_nvec))
        self.action_space = spaces.Discrete(self._n_actions)

        # ---------- 2) 정적 병원 특징 (시나리오 상수, 1회 캐시) ----------
        hp = u.en_manager.en_properties['hospital']
        hos_tier = np.asarray(hp['hos_tier'], dtype=np.float32).reshape(-1)  # 3=Tier3, 2=그외
        self._is_tier3 = (hos_tier == 3).astype(np.float32)                  # (H,)
        helipad_idx = np.asarray(hp.get('hos_helipad_idx', np.array([])), dtype=int)
        self._helipad = np.zeros(self.H, dtype=np.float32)
        if helipad_idx.size > 0:
            self._helipad[helipad_idx] = 1.0
        self._max_send = np.asarray(hp['hos_max_send'], dtype=np.float32).reshape(-1)

        # ETA(분) = lognormal 평균(amb/uav_HtoS_t[0]) — 없으면 거리/속도로 폴백. (#1)
        ambp = u.en_manager.en_properties.get('ambulance', {})
        uavp = u.en_manager.en_properties.get('uav', {})
        d_road = np.asarray(hp.get('d_HtoS_road', hp.get('d_HtoS_euc', np.zeros(self._H_real))), dtype=np.float32)
        d_euc = np.asarray(hp.get('d_HtoS_euc', d_road), dtype=np.float32)
        amb_t = ambp.get('amb_HtoS_t', None)
        if amb_t is not None and len(amb_t[0]) == self._H_real:
            eta_amb = np.asarray(amb_t[0], dtype=np.float32)
        else:
            eta_amb = d_road * 60.0 / (float(ambp.get('amb_v', 40)) or 40.0)
        uav_t = uavp.get('uav_HtoS_t', None)
        if uav_t is not None and len(uav_t[0]) == self._H_real:
            eta_uav = np.asarray(uav_t[0], dtype=np.float32)
        else:
            eta_uav = d_euc * 60.0 / (float(uavp.get('uav_v', 80)) or 80.0)
        # 시나리오 최소 ETA 로 정규화(>0 기준, 최근접=1) → 지역간 스케일 제거.
        # + 외곽 병원 이상치 클립(최근접의 MCI_ETA_CLIP 배, 기본 10) → VecNorm std 왜곡 방지.
        #
        # ★ raw 토큰(v18 E6): 좌표별 정규화를 끄고 **전역 상수**로만 나눈다.
        #   v17 진단 — 좌표별 정규화가 거리의 절대 크기를 지워 전국 단일 정책이
        #   "환자 1명 = N km" 교환율을 표현할 수 없다(손실의 92%). 전역 상수 나눗셈은
        #   좌표 간 상대 스케일을 보존하므로 VecNormalize 의 러닝 통계가 전국 공통 축이 된다.
        #   열 구성·차원은 그대로라 아키텍처·dim 은 불변이고 인코딩만 바뀐다.
        # (v19) 절대 스케일 앵커 — 인코딩 전 **원시 분** 에서 계산한다.
        _an = float(os.environ.get("MCI_ANCHOR_NORM", "30.0"))
        _ac = float(os.environ.get("MCI_ANCHOR_CLIP", "4.0"))
        _pa, _pu = eta_amb[eta_amb > 0], eta_uav[eta_uav > 0]
        self._anchor_vec = np.array([
            min((float(_pa.min()) if _pa.size else 0.0) / _an, _ac),
            min((float(_pu.min()) if _pu.size else 0.0) / _an, _ac),
        ], dtype=np.float32)

        raw_eta = "raw" in _parse_variant()
        sat_eta = "sat" in _parse_variant()
        if sat_eta:
            # ★ v19 A4: 포화형 t/(t+T0). raw(분/60, clip 600분)의 결함 교정 —
            #   raw 는 좌표별 정규화는 없앴지만 헬기장 26칸의 200~900분 꼬리가
            #   동적범위를 먹어 **결정 밴드(교사 선택 p99=105분) 해상도가 0.086σ** 로
            #   구 정규화(0.156σ)보다도 나쁘다. t/(t+30)은 같은 밴드에서 0.620σ(7.2배)이고
            #   동률 0%·꼬리 순위 보존(264분 대 480분 = 0.176σ)이라 원거리 좌표서도 랭킹된다.
            #   유계 [0,1) 라 클립 불요, 좌표 간 스케일 보존은 raw 와 동일.
            t0 = float(os.environ.get("MCI_ETA_SAT_T0", "30.0"))
            eta_clip = 1.0                       # 패딩 병원 = 무한원거리 점근값
            self._eta_amb = (eta_amb / (eta_amb + t0)).astype(np.float32)
            self._eta_uav = (eta_uav / (eta_uav + t0)).astype(np.float32)
        elif raw_eta:
            eta_clip = float(os.environ.get("MCI_ETA_RAW_CLIP", "10.0"))
            div = float(os.environ.get("MCI_ETA_RAW_NORM", "60.0"))   # 분 → 시간 스케일
            self._eta_amb = np.minimum(eta_amb / div, eta_clip).astype(np.float32)
            self._eta_uav = np.minimum(eta_uav / div, eta_clip).astype(np.float32)
        else:
            eta_clip = float(os.environ.get("MCI_ETA_CLIP", "10.0"))
            self._eta_amb = np.minimum(self._norm_by_min(eta_amb), eta_clip).astype(np.float32)
            self._eta_uav = np.minimum(self._norm_by_min(eta_uav), eta_clip).astype(np.float32)
        # (v6) 정적 특징 패딩: 패딩 병원 = "무한 원거리 무용 병원"(tier3/용량 0, eta=클립상한
        # — eta=0 은 "초근접" 오독이라 금지). _helipad 는 이미 H_pad 사이즈 zeros 로 생성됨.
        if self.H > self._H_real:
            pn = self.H - self._H_real
            z = np.zeros(pn, dtype=np.float32)
            self._is_tier3 = np.concatenate([self._is_tier3, z])
            self._max_send = np.concatenate([self._max_send, z])
            self._eta_amb = np.concatenate([self._eta_amb, np.full(pn, eta_clip, dtype=np.float32)])
            self._eta_uav = np.concatenate([self._eta_uav, np.full(pn, eta_clip, dtype=np.float32)])

        # (v6 A3) valid 열 벡터: 실병원 1.0 / 패딩 0.0. 무조건 생성(값싸고 스모크·포인터
        # 마스크드 풀링 계약에 유용) — obs 에는 essential+load+valid variant 에서만 실린다.
        # 패딩 없으면 ones(H). PadAwareVecNormalize 가 이 열을 정규화 면제해 0/1 을 보존
        # (아핀변환이 all-zero 파생 패딩 식별을 뭉개는 것 방지, 설계 확정).
        self._valid_vec = np.concatenate([
            np.ones(self._H_real, dtype=np.float32),
            np.zeros(self.H - self._H_real, dtype=np.float32),
        ])

        # ---------- 3) MCI_OBS_VARIANT → 특징 열 선택 (local/comms/full/essential[+load]) ----------
        toks = _parse_variant()
        self._load = "load" in toks
        if "full" in toks:
            self._cols = _LOCAL_COLS + _COMMS_COLS
            var_label = "full(ablation)"
        elif "local" in toks and "comms" not in toks:
            self._cols = list(_LOCAL_COLS)
            var_label = "local(ablation)"
        elif "comms" in toks and "local" not in toks:
            self._cols = list(_COMMS_COLS)
            var_label = "comms(ablation)"
        else:  # 기본 = essential (essential 토큰 또는 미설정) [+load 확장]
            self._cols = list(_ESSENTIAL_COLS)
            var_label = "essential"
        if self._load:
            if var_label != "essential":
                raise ValueError(f"load 토큰은 essential 기반만 지원 (got MCI_OBS_VARIANT={toks})")
            # cap_remain → 클립본 교체 + 부하 신호 3열 추가 (F 4→7)
            self._cols = ["is_tier3", "cap_remain_c", "eta_amb", "eta_uav"] + list(_LOAD_COLS)
            var_label = "essential+load"
        self._ctx = "ctx" in toks
        if self._ctx:
            if not self._load:
                raise ValueError(f"ctx 토큰은 essential+load 기반만 지원 (got MCI_OBS_VARIANT={toks})")
            if self.H != self._H_real:
                raise ValueError("ctx variant 는 MCI_H_PAD 미지원(v4 기각 변형 — 패딩 배선 없음)")
            # v4: 순위·모드 시간절감·도착타이밍 3열 추가 (F 7→10)
            self._cols = self._cols + list(_CTX_COLS)
            var_label = "essential+load+ctx"
            self._eta_rank_amb = (np.argsort(np.argsort(eta_amb)).astype(np.float32)
                                  / max(self.H - 1, 1))
            self._uav_timesave = np.clip((eta_amb - eta_uav) / np.maximum(eta_amb, 1e-6),
                                         -1.0, 1.0).astype(np.float32)
            self._arrive_norm = float(os.environ.get("MCI_ARRIVE_NORM", "60"))
            self._arrive_clip = float(os.environ.get("MCI_ARRIVE_CLIP", "2.0"))
            ena, pos = self._eta_amb, eta_amb[eta_amb > 0]
            self._ctx_static = np.array([
                float((ena <= 2.0).mean()), float((ena <= 4.0).mean()),
                float(np.percentile(ena, 90)) / eta_clip,
                float(self._is_tier3.mean()),
                float((eta_uav < eta_amb).mean()),
                min((float(pos.min()) if pos.size else 0.0)
                    / float(os.environ.get("MCI_ETA_MIN_NORM", "30")), 2.0),
            ], dtype=np.float32)
        # (v6 A3) valid 열(마지막 열): 패딩 병원 명시 식별자 — 포인터 마스크드 풀링/
        # PadAwareVecNormalize 의 견고한 패딩 인지 근거. essential+load 필수·ctx 배타·
        # MCI_H_PAD 명시 필수(판정 하네스가 variant 문자열만으로 env 재현 원칙).
        self._valid = "valid" in toks
        if self._valid:
            if not self._load:
                raise ValueError(f"valid 토큰은 essential+load 기반만 지원 "
                                 f"(got MCI_OBS_VARIANT={toks})")
            if self._ctx:
                raise ValueError(f"valid 토큰은 ctx 와 동시 사용 불가 "
                                 f"(got MCI_OBS_VARIANT={toks})")
            if not os.environ.get("MCI_H_PAD", "").strip():
                raise ValueError("valid variant 는 MCI_H_PAD 명시 필수 — 판정 하네스가 "
                                 "variant 문자열만으로 env 재현(H_pad=실H 여도 명시 설정 요구)")
            self._cols = self._cols + ["valid"]
            var_label = "essential+load+valid"
        # (v18 E6) raw 토큰: 차원 402 불변, **인코딩만** 물리단위로.
        #   eta_amb/uav : 좌표별 정규화 → 분 / MCI_ETA_RAW_NORM(60)  ※ 전역 상수라 좌표 간 스케일 보존
        #   occ_ratio   → load_raw = clip(census + in_flight, 0, MCI_LOAD_CLIP=32)  ※ 명수
        # 두 번째 교체가 핵심이다 — 현장 규칙집이 이기는 수식이 `거리(km) + λ × 부하(명)` 인데
        # 비율형 occ_ratio 로는 정책이 "부하 몇 명"을 직접 만들 수 없다(max_send 가 obs 에 없어
        # cap_remain/(1-occ_ratio) 나눗셈이 필요하고 occ_ratio→1 에서 발산). load_raw 로 바꾸면
        # max_send = cap_remain + load 라 덧셈으로 복원되므로 정보가 오히려 늘어난다.
        self._raw_eta = raw_eta
        self._sat_eta = sat_eta
        if self._raw_eta and self._sat_eta:
            raise ValueError("raw 와 sat 는 같은 ETA 축의 배타 인코딩이다 — 하나만 지정")
        if self._raw_eta or self._sat_eta:
            tk = "raw" if self._raw_eta else "sat"
            if not self._load:
                raise ValueError(f"{tk} 토큰은 essential+load 기반만 지원 "
                                 f"(got MCI_OBS_VARIANT={toks})")
            if self._ctx:
                raise ValueError(f"{tk} 토큰은 ctx 와 동시 사용 불가 "
                                 f"(got MCI_OBS_VARIANT={toks})")
            # 부하열은 두 인코딩이 공통으로 물리단위(명수)를 쓴다 — occ_ratio 는 앵커가
            # max_send(중위 14)라 실제 병목(수술실수 중위 2)과 7배 어긋난다.
            self._cols = [("load_raw" if c == "occ_ratio" else c) for c in self._cols]
            var_label = var_label + "+" + tk
        # ---- (v19) slot 토큰: 용량축을 '실제 병목'으로 교체 ----
        # 실측 근거(v19 진단):
        #   * 서비스 슬롯 = 수술실수, 중위 **2개**(범위 1~3). 치료 지수분포 평균 Red 40분·
        #     Yellow 20분. 반면 obs 의 cap_remain/occ_ratio 앵커는 max_send=수술실수+병상수
        #     (중위 14) = **7배 과대**. 게다가 실제 diversion 게이트(occ>=max_send)는
        #     교사 결정 37,000건에서 **0.00%** 발동 = cap_remain 은 사실상 죽은 열이다.
        #   * 부하항을 끄면(lam=0) 활동중 병원의 **82%가 n_idle==0**, 대기열 최대 15명,
        #     PDR 0.3028. CARD(lam=12)를 켜면 36%·최대 3명·PDR 0.1464.
        #     즉 부하항의 가치 전부가 "수술실을 비워 두는 것"인데 그 상태변수가 obs 에 없다.
        #   * n_idle·n_queue 는 이미 sim obs 의 h_states[:,0:2] 에 있다(추가 계산 0).
        # 교체: cap_remain_c → idle(빈 수술실 수) **한 열만**. F·차원 불변(402).
        # ⚠️n_queue 는 넣지 않는다 — 스모크 실측에서 고유값 1(항상 0)이라 load_raw(고유값 17)
        #   를 갈아치우면 정보가 준다. 대기열은 이미 발생한 실패이고, 막아야 할 선행지표는
        #   idle(지금 빌 자리)과 in_flight(도착 예정, 부하의 68%)다.
        # ⚠️cap_remain_c 를 버려도 되는 근거: diversion 게이트(occ>=max_send)가 교사
        #   37,000결정에서 0.00% 발동 = 죽은 열. 반대로 idle 은 미사용 병원에서 n_rooms
        #   (1~3, 병원 서비스 규모)와 같아 정적 정보까지 겸한다.
        self._anchor = "anchor" in toks
        if self._anchor:
            if not self._load:
                raise ValueError(f"anchor 토큰은 essential+load 기반만 지원 "
                                 f"(got MCI_OBS_VARIANT={toks})")
            if self._ctx:
                raise ValueError(f"anchor 토큰은 ctx 와 동시 사용 불가(ctx 가 이미 포함) "
                                 f"(got MCI_OBS_VARIANT={toks})")
            var_label = var_label + "+anchor"
        self._slot = "slot" in toks
        if self._slot:
            if not self._load:
                raise ValueError(f"slot 토큰은 essential+load 기반만 지원 "
                                 f"(got MCI_OBS_VARIANT={toks})")
            if self._ctx:
                raise ValueError(f"slot 토큰은 ctx 와 동시 사용 불가 "
                                 f"(got MCI_OBS_VARIANT={toks})")
            self._cols = [("idle" if c == "cap_remain_c" else c) for c in self._cols]
            var_label = var_label + "+slot"
        self._F = len(self._cols)
        # load 스케일 노브(전 신규열 사전 유계 → VecNorm 러닝 std 유의미)
        self._ps_clip = float(os.environ.get("MCI_PSENT_CLIP", "32"))
        self._cr_clip = float(os.environ.get("MCI_CAPREMAIN_CLIP", "32"))
        self._or_clip = float(os.environ.get("MCI_OCC_RATIO_CLIP", "4"))
        self._rho_clip = float(os.environ.get("MCI_RHO_CLIP", "8"))
        self._t_norm_div = float(os.environ.get("MCI_TIME_NORM", "240"))
        self._uav_max = float(os.environ.get("MCI_UAV_MAX", "26"))
        self._amb_num = amb_num
        self._uav_num = uav_num

        # ---------- 4) obs space ----------
        self._gdim = (_GLOBAL_DIM + (_LOAD_GLOBAL_EXTRA if self._load else 0)
                      + (_CTX_GLOBAL_EXTRA if self._ctx else 0)
                      + (_ANCHOR_GLOBAL_EXTRA if self._anchor else 0))
        self._flat_dim = self.H * self._F + self._gdim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._flat_dim,), dtype=np.float32,
        )
        self._ct_cache = None  # 등급-tier 치료가능 마스크 (3, H)

        pad_note = f", H_pad={self.H}(실H {self._H_real})" if self.H != self._H_real else ""
        print(f"[HospitalFeatureWrapper] {mode_label}, action=Discrete({self._n_actions}), "
              f"obs={self._flat_dim} (entity {self.H}x{self._F} + global {self._gdim}), "
              f"variant={var_label}, helipad={int(self._helipad.sum())}/{self.H}{pad_note}")

    @staticmethod
    def _norm_by_min(eta: np.ndarray) -> np.ndarray:
        pos = eta[eta > 0]
        denom = float(pos.min()) if pos.size else 1.0
        return (eta / denom).astype(np.float32)

    # ---------- decode/encode (FlattenAndDiscreteWrapper 와 동치) ----------
    def _decode(self, action: int):
        a = int(action)
        if self._fixed_mode is not None:
            n_dest = self._effective_nvec[1]
            return [a // n_dest, a % n_dest, self._fixed_mode]
        n_dest, n_mode = self._orig_nvec[1], self._orig_nvec[2]
        c = a // (n_dest * n_mode)
        rem = a % (n_dest * n_mode)
        return [c, rem // n_mode, rem % n_mode]

    def _encode(self, decoded):
        c, d, m = int(decoded[0]), int(decoded[1]), int(decoded[2])
        if self._fixed_mode is not None:
            return c * self._effective_nvec[1] + d
        n_dest, n_mode = self._orig_nvec[1], self._orig_nvec[2]
        return c * (n_dest * n_mode) + d * n_mode + m

    decode_action = _decode
    encode_action = _encode

    # ---------- obs 구성 ----------
    def _dyn(self, obs: dict) -> dict:
        """동적 병원 신호 1회 계산 — _entity/_globals 공유 (중복 계산 방지)."""
        h = np.asarray(obs['h_states'], dtype=np.float32)  # (실H,3) = [idle, queue, occ]
        p_sent = np.asarray(obs['p_sent'], dtype=np.float32).reshape(-1)
        if self.H > self._H_real:  # (v6) 동적 신호 zero-pad — 패딩 병원 = 무활동
            pn = self.H - self._H_real
            h = np.vstack([h, np.zeros((pn, h.shape[1]), dtype=np.float32)])
            p_sent = np.concatenate([p_sent, np.zeros(pn, dtype=np.float32)])
        from EntityManager import EntityManager
        in_flight = EntityManager.in_flight_by_hospital(obs, self.H)  # 현장 지득(내가 보낸 이송중)
        # cap_remain 게이트 (2026-07-03 통신축 재정의, 마스크/RuleManager 와 동일 의미):
        #   occ(통신)  = census(occ) + in_flight(이송중=도착 예상) 차감
        #   psent(단절) = 보낸 만큼(p_sent) 차감 — 현장 지득 정보만
        comms = os.environ.get("MCI_CAP_GATE", "occ").strip().lower() != "psent"
        cap_used = (h[:, 2] + in_flight) if comms else p_sent
        cap_remain = np.maximum(self._max_send - cap_used, 0.0)
        d = {"h": h, "p_sent": p_sent, "in_flight": in_flight,
             "cap_remain": cap_remain, "comms": comms}
        if self._ctx:
            # 병원별 최근접 도착 타이밍 — dest 1..H(1-based)·severity>0 관례는
            # EntityManager.in_flight_by_hospital 과 동일. 이송중 없음 = 클립값(2.0).
            arr = np.full(self.H, self._arrive_clip, dtype=np.float32)
            for key in ('amb_states', 'uav_states'):
                st = np.asarray(obs.get(key, ()), dtype=np.float32)
                if st.size == 0:
                    continue
                m = (st[:, 0] >= 1) & (st[:, 2] > 0)
                dst = st[m, 0].astype(int) - 1
                t = np.clip(st[m, 1] / self._arrive_norm, 0.0, self._arrive_clip)
                ok = (dst >= 0) & (dst < self.H)
                np.minimum.at(arr, dst[ok], t[ok].astype(np.float32))
            d["arrive_min"] = arr
        return d

    def _entity(self, obs: dict, dyn: dict) -> np.ndarray:
        """병원당 특징 행렬 (H, F)."""
        h, comms = dyn["h"], dyn["comms"]
        # 통신단절(psent) 시 병원 실시간 컬럼(idle/queue/occ·occ_ratio)은 지득 불가 → 0 마스킹
        # (p_sent_c·in_flight 는 현장 발송 기록 = 지득 정보라 유지)
        z = np.zeros_like(h[:, 0])
        col_map = {
            "is_tier3": self._is_tier3,
            "helipad": self._helipad,
            "eta_amb": self._eta_amb,
            "eta_uav": self._eta_uav,
            "idle": h[:, 0] if comms else z,
            "queue": h[:, 1] if comms else z,
            "occ": h[:, 2] if comms else z,
            "cap_remain": dyn["cap_remain"],
            "cap_remain_c": np.minimum(dyn["cap_remain"], self._cr_clip),
            "p_sent_c": np.minimum(dyn["p_sent"], self._ps_clip),
            "in_flight": dyn["in_flight"].astype(np.float32),
            # occ_ratio 와 같은 통신 계층(census 포함)이라 psent 시 동일하게 0 마스킹한다.
            "load_raw": (np.minimum(h[:, 2] + dyn["in_flight"],
                                    float(os.environ.get("MCI_LOAD_CLIP", "32"))).astype(np.float32)
                         if comms else z),
            "occ_ratio": (np.clip((h[:, 2] + dyn["in_flight"]) / np.maximum(self._max_send, 1.0),
                                  0.0, self._or_clip) if comms else z),
            "valid": self._valid_vec,   # (v6 A3) 정적 패딩 식별자 — comms 무관 무조건 노출(1/0)
        }
        if self._ctx:
            col_map.update({"eta_rank_amb": self._eta_rank_amb,
                            "uav_timesave": self._uav_timesave,
                            "arrive_min": dyn["arrive_min"]})
        return np.stack([col_map[c] for c in self._cols], axis=1).astype(np.float32)  # (H, F)

    def _globals(self, obs: dict, dyn: dict) -> np.ndarray:
        # patient_agg 4등급×5단계(20) 중 R/Y(앞 2등급=10)만 — Green/Black 은 행동대상 아님(자동일괄).
        pa = AggregateObsWrapper._patient_agg(np.asarray(obs['p_states']))[:10]       # (10,) R/Y
        va = np.concatenate([
            AggregateObsWrapper._fleet_agg(np.asarray(obs['amb_states'])),
            AggregateObsWrapper._fleet_agg(np.asarray(obs['uav_states'])),
        ])                                                                            # (10,)
        # p_at_site/n_amb_at_site/n_uav_at_site 는 pa·va 의 부분집합이라 제거(중복 0손실).
        parts = [pa, va, np.asarray(obs['time'], dtype=np.float32).reshape(-1)]       # (1,)
        if self._ctx:
            parts.append(self._ctx_static)  # 정적 지역 컨텍스트 (+6) — reset 간 불변
        if self._load:
            # ρ = 잔여 긴급부하(R/Y 생애단계 0~2: 미구조+현장대기+이송중) / 잔여 유효용량(게이트 추종)
            urgent = float(pa[0] + pa[1] + pa[2] + pa[5] + pa[6] + pa[7])
            rho = min(urgent / (float(dyn["cap_remain"].sum()) + 1.0), self._rho_clip)
            t_norm = min(float(np.asarray(obs['time']).reshape(-1)[0]) / self._t_norm_div, 2.0)
            parts.append(np.array([
                rho,
                va[0] / max(self._amb_num, 1),          # 가용 AMB 비율
                va[5] / max(self._uav_num, 1),          # 가용 UAV 비율 (uav=0 이면 0/1=0)
                self._uav_num / self._uav_max,          # 함대 규모 신호(UAV 대수축)
                t_norm,
            ], dtype=np.float32))
        if self._anchor:
            parts.append(self._anchor_vec)   # (v19) 절대 스케일 2개 — reset 간 불변(정적)
        return np.concatenate(parts).astype(np.float32)

    def _flat_obs(self, obs: dict) -> np.ndarray:
        dyn = self._dyn(obs)
        return np.concatenate([self._entity(obs, dyn).reshape(-1),
                               self._globals(obs, dyn)]).astype(np.float32)

    # ---------- gym API ----------
    def step(self, action):
        decoded = self._decode(action)
        if decoded[1] > self._H_real:  # (v6) 방어: 패딩 dest 는 마스크가 차단 — 도달 시 버그
            raise RuntimeError(f"패딩 병원 dest={decoded[1]} 선택(실H={self._H_real}) — "
                               f"action mask 우회 의심")
        obs, reward, terminated, truncated, info = self.env.step(decoded)
        return self._flat_obs(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flat_obs(obs), info

    # ---------- action mask (env_wrapper.action_masks 와 동치: joint + tier) ----------
    def _can_treat_mask(self) -> np.ndarray:
        if self._ct_cache is None:
            ep = self.env.unwrapped.en_manager.en_properties
            hos_tier = np.asarray(ep['hospital']['hos_tier']).reshape(-1)
            pinfo = ep['patient']['patient_info']
            t3 = np.asarray(pinfo['treat_tier3']).astype(bool)
            t2 = np.asarray(pinfo['treat_tier2']).astype(bool)
            n_class = int(self._orig_nvec[0])  # 2 (R/Y — Green 은 action 차원서 제외)
            ct = np.zeros((n_class, self.H), dtype=bool)  # 패딩 열은 False 유지 (v6)
            for h in range(self._H_real):
                ht = int(hos_tier[h])
                col = t3 if ht == 3 else (t2 if ht == 2 else np.zeros(4, dtype=bool))
                ct[:, h] = col[:n_class]
            self._ct_cache = ct
        return self._ct_cache

    def action_masks(self) -> np.ndarray:
        full = self.env.unwrapped.action_masks_joint()
        full = full.reshape(self._sim_nvec[0], self._sim_nvec[1], self._sim_nvec[2]).copy()
        if self.H > self._H_real:  # (v6) 패딩 dest(실H+1..H_pad) 전부 False 로 확장
            padm = np.zeros((full.shape[0], self.H - self._H_real, full.shape[2]),
                            dtype=full.dtype)
            full = np.concatenate([full, padm], axis=1)
        if os.environ.get("MCI_TIER_MASK", "1") != "0":
            ct = self._can_treat_mask()           # (2, H) bool
            full[:, 1:, :] &= ct[:, :, None]      # dest 1..H 만 차단, stay(0) 유지
        # Green/Black 은 action 차원(class dim=2)에서 제외됨(2026-07-03) — 코어의
        # start_GB_transport 일괄이송에 위임. 구 MCI_GREEN_MASK 마스킹은 폐기.
        if self._fixed_mode is not None:
            return full[:, :, self._fixed_mode].reshape(-1)
        return full.reshape(-1)
