"""피드백 5 아이디어 5 — 보상 재설계 wrapper.

env_wrapper.py 와 src/sim_src 는 수정 금지. 신규 파일.

MCIEnvironment_gym 의 step() 은 (obs, r, term, trunc, info) 를 반환하고
info 에 'r_woG' (Green 제외 보상) 를 항상 채워줌. 이 wrapper 는 r 값을
mode 에 따라 재정의한다.

지원 모드 (mode 인자 또는 ENV `MCI_REWARD_MODE`):
  - "raw"   : 원본 보상 그대로 (no-op, sanity 용).
  - "woG"   : step reward 를 info['r_woG'] 로 치환 (Green 제외).
  - "pdrwog": step r_woG 를 에피소드 시작 시 확정되는 preventable_woG 로 정규화
             (r = r_woG / preventable_woG). 에피소드 누적합 = woG/preventable
             = 1 − PDR_woG 라 **사고규모가 달라도 스케일 불변**(0~1) — 규모 혼재
             학습(교수 지시 2026-07-04)의 표준 보상. preventable_woG 는 reset 의
             init_log(환자 실현)로 1회 확정·step 중 불변(MCIEnvironment reset 참조).
             스케일이 작으므로 학습 시 VecNormalize(norm_reward=True) 병용 권장.
  - "pdrwog_da": (v5 재난특화 P2, 2026-07-13) pdrwog 의 **결정귀속(Decision-Attributed)**
             재배치. 보상이 입원 이벤트 창에 계상되는 시간 불일치(행동↔보상 지연·확산)를,
             에피소드 합을 보존한 채 결정 스텝으로 이동:
                 r'_t = r̂(a_t) + [r_woG_t − (이번 창 성숙 예정 r̂ 합)]   (전부 /preventable_woG)
                 r̂(a_t) = getSurvProb(now + E[transport](m,d) + handover(m), c)
             E[transport] = en_properties amb/uav_HtoS_t[0](lognormal 평균, 분).
             성숙시각(now+E+handover) 큐를 유지, 종료 스텝에 미성숙분 일괄 정산 →
             **Σr' = Σ(r_woG/prev) 정확 보존**(return-equivalent, 등가성 테스트 필수).
             stay(d=0)·G/B 는 r̂=0. 실제 입원과의 개별 매칭 불요(집계 잔차) — 코어 무수정.
             ⚠️행동은 디코드된 [c,d,m] 로 도달해야 함(HospitalFeatureWrapper 안쪽 배선 전제).
  - "rywt"  : Red·Yellow 가중. logToReward(log) 가 호출되는 시점의
             log['p_admit']=[(time, p_class), ...] 를 가로채 가중합으로 재계산:
                 r = Σ over p_admit:
                     red_w   * getSurvProb(t, 0)   if p_class==0
                     yellow_w* getSurvProb(t, 1)   if p_class==1
                     green_w * 1.0                 if p_class==2 (green_w 기본 0)
             기본 가중치: red_w=2.0, yellow_w=1.0, green_w=0.0.
             sim_src 수정 불가 → env.unwrapped.logToReward 를 monkey-patch 하여
             마지막 log 를 저장하고, step 직후 wrapper 에서 가중합 재계산.

원본 보상은 항상 info['r_raw'] 에 보존된다.

사용 예:
    from env_factory import make_base_env
    from reward_redesign_wrapper import RewardRedesignWrapper
    base = make_base_env(cfg_path)
    env  = RewardRedesignWrapper(base, mode='woG')           # Green 제외
    env  = RewardRedesignWrapper(base, mode='rywt',
                                 red_w=2.0, yellow_w=1.0)    # Red 가중

ENV 변수 hook (인자 미지정 시 폴백):
    MCI_REWARD_MODE     : raw|woG|rywt
    MCI_REWARD_RED_W    : float (rywt 모드)
    MCI_REWARD_YELLOW_W : float (rywt 모드)
    MCI_REWARD_GREEN_W  : float (rywt 모드)
"""
import os
import gymnasium as gym


_VALID_MODES = ("raw", "woG", "pdrwog", "rywt", "pdrwog_da")


def _resolve_mode(mode):
    if mode is None:
        mode = os.environ.get("MCI_REWARD_MODE", "raw")
    if mode not in _VALID_MODES:
        raise ValueError(f"mode 는 {_VALID_MODES} 중 하나, got {mode!r}")
    return mode


def _resolve_float(value, env_key, default):
    if value is not None:
        return float(value)
    env_v = os.environ.get(env_key)
    if env_v is not None:
        return float(env_v)
    return float(default)


class RewardRedesignWrapper(gym.Wrapper):
    """reward 만 변환하는 wrapper. obs / action / done 은 통과."""

    def __init__(self, env, mode=None, *, red_w=None, yellow_w=None, green_w=None):
        super().__init__(env)
        self.mode = _resolve_mode(mode)
        self.red_w    = _resolve_float(red_w,    "MCI_REWARD_RED_W",    2.0)
        self.yellow_w = _resolve_float(yellow_w, "MCI_REWARD_YELLOW_W", 1.0)
        self.green_w  = _resolve_float(green_w,  "MCI_REWARD_GREEN_W",  0.0)

        self._unwrapped = env.unwrapped  # MCIEnvironment_gym
        self._last_log = None
        self._orig_logToReward = None
        self._pdr_denom = 1.0  # pdrwog(_da): preventable_woG 캐시 (reset 시 갱신)
        # pdrwog_da 상태: (성숙시각, r̂) 큐 + 정적 ETA/handover 캐시(reset 시 구축)
        self._da_queue = []
        self._da_eta = None       # (2, H) 분 — [mode][hos_idx]
        self._da_handover = None  # (2,) 분
        self._da_warned = False
        if self.mode == "rywt":
            self._install_log_hook()

    # ---------- monkey-patch hook (rywt 만) ----------
    def _install_log_hook(self):
        """env.unwrapped.logToReward 를 감싸 마지막 log 를 self._last_log 에 저장.

        원본 함수 동작은 그대로 유지 (반환값 보존) — info['r_raw'] 가 정상 채워짐.
        sim_src 파일은 건드리지 않고, 이미 import 된 객체의 메서드만 교체.
        """
        outer = self
        orig = self._unwrapped.logToReward
        self._orig_logToReward = orig

        def wrapped(log):
            outer._last_log = log
            return orig(log)

        self._unwrapped.logToReward = wrapped

    def _restore_log_hook(self):
        if self._orig_logToReward is not None:
            self._unwrapped.logToReward = self._orig_logToReward
            self._orig_logToReward = None

    def close(self):
        self._restore_log_hook()
        return super().close()

    # ---------- reward 변환 ----------
    def _rywt_reward(self):
        """self._last_log 의 p_admit 을 Red/Yellow/Green 가중합으로 변환."""
        if self._last_log is None:
            return 0.0
        log = self._last_log
        env = self._unwrapped
        p_admit = log.get("p_admit", []) if isinstance(log, dict) else []
        r = 0.0
        for (t, p_class) in p_admit:
            sp = env.getSurvProb(t, p_class)
            if p_class == 0:
                r += self.red_w * sp
            elif p_class == 1:
                r += self.yellow_w * sp
            elif p_class == 2:
                r += self.green_w * sp
        return r

    # ---------- pdrwog_da 보조 ----------
    def _da_build_static(self):
        """ETA 평균(분)·handover 캐시 — hospital_feature_wrapper 의 원천과 동일 폴백."""
        import numpy as np
        props = self._unwrapped.en_manager.en_properties
        hp = props['hospital']
        H = len(np.asarray(hp['hos_max_send']).reshape(-1))
        d_road = np.asarray(hp.get('d_HtoS_road', hp.get('d_HtoS_euc', np.zeros(H))), dtype=np.float64)
        d_euc = np.asarray(hp.get('d_HtoS_euc', d_road), dtype=np.float64)
        ambp = props.get('ambulance', {})
        uavp = props.get('uav', {})
        amb_t = ambp.get('amb_HtoS_t', None)
        eta_amb = (np.asarray(amb_t[0], dtype=np.float64) if amb_t is not None and len(amb_t[0]) == H
                   else d_road * 60.0 / (float(ambp.get('amb_v', 40)) or 40.0))
        uav_t = uavp.get('uav_HtoS_t', None)
        eta_uav = (np.asarray(uav_t[0], dtype=np.float64) if uav_t is not None and len(uav_t[0]) == H
                   else d_euc * 60.0 / (float(uavp.get('uav_v', 80)) or 80.0))
        self._da_eta = (eta_amb, eta_uav)
        self._da_handover = (float(ambp.get('amb_handover_time', 0.0)),
                             float(uavp.get('uav_handover_time', 0.0)))

    def _da_issue(self, action):
        """결정 스텝에서 r̂ 발행(큐 push) 후 r̂ 반환. dispatch 아니면 0."""
        import numpy as np
        try:
            a = np.asarray(action).reshape(-1)
            c, d, m = int(a[0]), int(a[1]), int(a[2])
        except Exception:
            if not self._da_warned:
                import warnings
                warnings.warn("pdrwog_da: 액션이 [c,d,m] 형태가 아님 — r̂ 미발행(pdrwog 로 퇴화). "
                              "HospitalFeatureWrapper 안쪽 배선인지 확인")
                self._da_warned = True
            return 0.0
        if d <= 0 or c not in (0, 1):
            return 0.0
        now = float(self._unwrapped.ev_manager.time)
        mature_t = now + float(self._da_eta[m][d - 1]) + self._da_handover[m]
        r_hat = float(self._unwrapped.getSurvProb(mature_t, c)) / self._pdr_denom
        self._da_queue.append((mature_t, r_hat))
        return r_hat

    # ---------- gym API ----------
    def step(self, action):
        r_issue = 0.0
        if self.mode == "pdrwog_da":
            r_issue = self._da_issue(action)  # step 전 시각(now) 기준 발행

        obs, r_raw, terminated, truncated, info = self.env.step(action)
        info = dict(info) if info is not None else {}
        info["r_raw"] = float(r_raw)

        if self.mode == "raw":
            r_new = r_raw
        elif self.mode == "woG":
            r_new = float(info.get("r_woG", 0.0))
        elif self.mode == "pdrwog":
            r_new = float(info.get("r_woG", 0.0)) / self._pdr_denom
        elif self.mode == "pdrwog_da":
            t_after = float(info.get("time", self._unwrapped.ev_manager.time))
            matured = 0.0
            if self._da_queue:
                keep = []
                for mt, rh in self._da_queue:
                    if mt <= t_after:
                        matured += rh
                    else:
                        keep.append((mt, rh))
                self._da_queue = keep
            r_new = r_issue + (float(info.get("r_woG", 0.0)) / self._pdr_denom) - matured
            if terminated or truncated:
                # 미성숙분 일괄 정산 → 에피소드 합 = Σ r_woG/prev 정확 보존
                r_new -= sum(rh for _, rh in self._da_queue)
                self._da_queue = []
        elif self.mode == "rywt":
            r_new = self._rywt_reward()
            # 다음 step 전 초기화 (terminal 후 leftover 방지)
            self._last_log = None
        else:
            r_new = r_raw  # 도달 불가능

        return obs, r_new, terminated, truncated, info

    def reset(self, **kwargs):
        self._last_log = None
        # reset 도중 logToReward 가 호출될 수 있음 (init_log → pending_terminal_reward).
        # hook 는 그대로 유지 — pending 처리에는 우리가 관여하지 않으므로 OK.
        out = self.env.reset(**kwargs)
        if self.mode in ("pdrwog", "pdrwog_da"):
            # preventable_woG 는 reset 의 init_log(환자 실현)로 확정 — 이 시점 캐시가 정확.
            # R/Y 가 0명인 극단 실현(preventable=0) 가드: 분모 1.0(그 에피소드 r_woG 도 전부 0).
            denom = float(getattr(self._unwrapped, "preventable_woG", 0.0))
            self._pdr_denom = denom if denom > 0.0 else 1.0
        if self.mode == "pdrwog_da":
            self._da_queue = []
            if self._da_eta is None:  # 정적 — 에피소드 간 불변(멀티리전은 지역별 env 라 1회면 충분)
                self._da_build_static()
        return out
