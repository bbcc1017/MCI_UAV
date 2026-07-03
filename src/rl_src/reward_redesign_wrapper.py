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


_VALID_MODES = ("raw", "woG", "pdrwog", "rywt")


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
        self._pdr_denom = 1.0  # pdrwog: preventable_woG 캐시 (reset 시 갱신)
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

    # ---------- gym API ----------
    def step(self, action):
        obs, r_raw, terminated, truncated, info = self.env.step(action)
        info = dict(info) if info is not None else {}
        info["r_raw"] = float(r_raw)

        if self.mode == "raw":
            r_new = r_raw
        elif self.mode == "woG":
            r_new = float(info.get("r_woG", 0.0))
        elif self.mode == "pdrwog":
            r_new = float(info.get("r_woG", 0.0)) / self._pdr_denom
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
        if self.mode == "pdrwog":
            # preventable_woG 는 reset 의 init_log(환자 실현)로 확정 — 이 시점 캐시가 정확.
            # R/Y 가 0명인 극단 실현(preventable=0) 가드: 분모 1.0(그 에피소드 r_woG 도 전부 0).
            denom = float(getattr(self._unwrapped, "preventable_woG", 0.0))
            self._pdr_denom = denom if denom > 0.0 else 1.0
        return out
