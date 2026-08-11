import os
import gymnasium as gym
import numpy as np
import math

from gymnasium import spaces


def _cap_gate_is_occ():
    """용량 게이트 기준. occ(기본)=실시간 점유(휴리스틱·sim 입원게이트와 일치, 병원 실시간 통신 가정),
    psent=누적 발송수(현장중심 제한정보 — 보낸 만큼만 알고 안 줄어듦). MCI_CAP_GATE 로 토글."""
    return os.environ.get("MCI_CAP_GATE", "occ").strip().lower() != "psent"


class MCIEnvironment_gym(gym.Env):
    """
    Gymnasium env for MCI Triage simulation.

    obs (RL용 dict, 고정 shape):
        - p_states:        (incident_size, 5) int32
        - h_states:        (H, 3) int32
        - p_sent:          (H,) int32
        - amb_states:      (amb_num, 3) float32  (UAV-only면 (0,3))
        - uav_states:      (uav_num, 3) float32
        - p_at_site:       (4,) int32     # R/Y/G/B 현장 대기 환자 수
        - n_amb_at_site:   (1,) int32
        - n_uav_at_site:   (1,) int32
        - time:            (1,) float32

    action: MultiDiscrete([2, H+1, 2])
        - [0] p_class: 0=Red, 1=Yellow
          (Green/Black 은 action 차원에서 제외 — 재난 대응 원칙상 R/Y 소진 후
           sim 코어(EventManager.start_GB_transport)가 일괄 자동이송. 2026-07-03
           기존 MCI_GREEN_MASK 마스킹을 차원 자체 제거로 대체.)
        - [1] destination: 0=stay, 1..H=hospital
        - [2] mode: 0=AMB, 1=UAV
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario, rng=None, default_rule=None,
                 max_steps=2000, rule_test=False, eval_mode=False, **kwargs):
        super().__init__()
        self.scenario_decoder(scenario)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.default_rule = default_rule
        self.max_steps = max_steps
        self.rule_test = rule_test
        self.eval_mode = eval_mode
        self.pen_size = 1.0

        # 시나리오 메타에서 차원 결정
        en_props = self.en_manager.en_properties
        self.incident_size = int(en_props['patient']['incident_size'])
        self.H = int(en_props['hospital']['hos_num'])
        self.amb_num = int(en_props['ambulance']['amb_num'])
        self.uav_num = int(en_props['uav']['uav_num'])

        # gymnasium spaces
        big = np.iinfo(np.int32).max
        self.observation_space = spaces.Dict({
            "p_states":      spaces.Box(0, big, shape=(self.incident_size, 5), dtype=np.int32),
            "h_states":      spaces.Box(0, big, shape=(self.H, 3),             dtype=np.int32),
            "p_sent":        spaces.Box(0, big, shape=(self.H,),               dtype=np.int32),
            "amb_states":    spaces.Box(0.0, np.inf, shape=(self.amb_num, 3),  dtype=np.float32),
            "uav_states":    spaces.Box(0.0, np.inf, shape=(self.uav_num, 3),  dtype=np.float32),
            "p_at_site":     spaces.Box(0, big, shape=(4,),                    dtype=np.int32),
            "n_amb_at_site": spaces.Box(0, big, shape=(1,),                    dtype=np.int32),
            "n_uav_at_site": spaces.Box(0, big, shape=(1,),                    dtype=np.int32),
            "time":          spaces.Box(0.0, np.inf, shape=(1,),               dtype=np.float32),
        })
        self.action_space = spaces.MultiDiscrete([2, self.H + 1, 2])

        # gym 표준: 사용자가 첫 step 전에 reset()을 호출하도록 강제.
        # 생성자에서 시뮬을 자동 시작하면 main.py가 첫 reset을 무시하면서
        # 한 번 더 ev_onset 이 트리거되어 헛 시뮬 로그가 남는다.
        # 이하 attribute 만 초기화 (실제 상태는 reset() 호출 시 채워짐).
        self.pending_terminal_reward = 0.0
        self.pending_terminal_reward_woG = 0.0
        self.n_step = 0
        self.preventable = 0.0
        self.preventable_woG = 0.0

    # ---------- helpers ----------
    def set_seed(self, rng):
        self.rng = rng

    def scenario_decoder(self, scenario):
        self.en_manager = scenario['EntityManager']
        self.ev_manager = scenario['EventManager']

    def _to_action_list(self, action):
        """np.ndarray / tuple / list → [int, int, int] (EventManager가 기대하는 형식)."""
        if isinstance(action, (list, tuple)):
            return [int(a) for a in action]
        arr = np.asarray(action).reshape(-1).tolist()
        return [int(a) for a in arr]

    def _rl_obs(self):
        """EntityManager 상태를 RL용 고정 shape dict로 가공."""
        full = self.en_manager.get_full_obs()
        p_at_site = np.zeros(4, dtype=np.int32)
        for c in range(4):
            p_at_site[c] = len(full['p_wait'][c][0])
        return {
            "p_states":      full['p_states'].astype(np.int32, copy=False),
            "h_states":      full['h_states'].astype(np.int32, copy=False),
            "p_sent":        full['p_sent'].astype(np.int32, copy=False),
            "amb_states":    full['amb_states'].astype(np.float32, copy=False),
            "uav_states":    full['uav_states'].astype(np.float32, copy=False),
            "p_at_site":     p_at_site,
            "n_amb_at_site": np.array([len(full['amb_wait'][0])], dtype=np.int32),
            "n_uav_at_site": np.array([len(full['uav_wait'][0])], dtype=np.int32),
            "time":          np.array([self.ev_manager.time], dtype=np.float32),
        }

    def _make_obs(self):
        if self.rule_test:
            obs = self.en_manager.get_full_obs()
            obs['time'] = self.ev_manager.time
            return obs
        return self._rl_obs()

    # ---------- gym API ----------
    def step(self, action):
        info = {}
        if self.ev_manager.check_termination():
            reward = self.pending_terminal_reward
            reward_woG = self.pending_terminal_reward_woG
            self.pending_terminal_reward = 0.0
            self.pending_terminal_reward_woG = 0.0
            obs = self._make_obs()
            info['time'] = self.ev_manager.time
            info['r_woG'] = reward_woG
            return obs, reward, True, False, info

        self.n_step += 1
        if self.n_step > self.max_steps:
            print("OVERTIME")
            info['time'] = self.ev_manager.time
            info['r_woG'] = -self.pen_size
            # 시간초과는 truncation(테르미널 아님) — terminated=True 로 주면 SB3 가
            # 종단으로 취급해 가치 부트스트랩을 생략(편향). 2026-07-03 정합성 수정.
            return self._make_obs(), -self.pen_size, False, True, info

        action_list = self._to_action_list(action)
        log, terminated = self.ev_manager.run_next(action_list)
        reward = self.logToReward(log)
        reward_woG = self.logToReward_woG(log)
        obs = self._make_obs()
        info['time'] = self.ev_manager.time
        info['r_woG'] = reward_woG
        return obs, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            # sim 동역학 rng(EventManager — 환자 multinomial·이송 lognormal 등)까지
            # 재시드해야 reset(seed) 재현성 계약이 성립. 기존엔 self.rng(미사용 dead
            # 변수)만 갈아끼워 env 재사용 시 same-seed 페어드 비교가 조용히 깨졌다.
            # (2026-07-03 정합성 수정. fresh-env 패턴은 생성 시 시드라 결과 불변.)
            self.rng = np.random.default_rng(seed)
            self.ev_manager.set_seed(self.rng)
        self.pending_terminal_reward = 0.0
        self.pending_terminal_reward_woG = 0.0
        self.n_step = 0

        self.en_manager.init_en_status()
        init_log = self.ev_manager.start()
        self.preventable = self.computePreventable(init_log)
        self.preventable_woG = self.computePreventable_woG(init_log)
        if self.ev_manager.check_termination():
            self.pending_terminal_reward = self.logToReward(init_log)
            self.pending_terminal_reward_woG = self.logToReward_woG(init_log)

        obs = self._make_obs()
        return obs, {}

    # ---------- reward ----------
    def computePreventable(self, log):
        val = 0.0
        for p_class, times in enumerate(log['rescue_times']):
            for t in times:
                val += self.getSurvProb(t, p_class)
        return val

    def computePreventable_woG(self, log):
        val = 0.0
        for p_class, times in enumerate(log['rescue_times']):
            if p_class == 2:
                continue
            for t in times:
                val += self.getSurvProb(t, p_class)
        return val

    def logToReward(self, log):
        reward = 0.0
        for x in log['p_admit']:
            reward += self.getReward(x[0], x[1])
        return reward

    def logToReward_woG(self, log):
        reward = 0.0
        for x in log['p_admit']:
            if x[1] == 2:
                continue
            reward += self.getReward(x[0], x[1])
        return reward

    def getReward(self, time, p_class):
        return self.getSurvProb(time, p_class)

    def getSurvProb(self, time, p_class):
        if p_class == 0:    # Red
            return 0.56 / (math.pow((time / 91), 1.58) + 1)
        elif p_class == 1:  # Yellow
            return 0.81 / (math.pow((time / 160), 2.41) + 1)
        elif p_class == 2:  # Green
            return 1.0
        else:               # Black
            return 0.0

    # ---------- action masking ----------
    def action_masks(self):
        """
        Per-sub-action 평탄 mask: length = 3 + (H+1) + 2.
        SB3-contrib MaskablePPO MultiDiscrete 호환 형식.

        주의: 이 mask는 차원별 독립이라 결합 제약은 보장 못 함.
              - 예: "Red인데 UAV=0" 같은 (class, mode) 결합 제약
              - 예: "UAV(mode=1) 은 helipad 보유 병원만" 같은 (dest, mode) 결합 제약
              → 모두 joint mask (action_masks_joint) 에서만 정확히 표현된다.
              실제 RL 학습/평가 경로(env_wrapper.FlattenAndDiscreteWrapper)는 joint mask 만
              사용하므로 per-dim 은 보수적으로 모든 병원을 열어둔다. mask 우회 알고리즘
              방어는 EventManager.proceed_action() 의 NO HELIPAD 가드가 담당한다.
        """
        full = self.en_manager.get_full_obs()
        H = self.H

        # dim 0: p_class (R/Y 만 — Green 은 action 차원에서 제외, 코어 일괄이송)
        m_class = np.zeros(2, dtype=bool)
        for c in range(2):
            m_class[c] = len(full['p_wait'][c][0]) > 0

        # dim 1: destination (0=stay 항상 허용, 1..H = 가용 모드 + 모드별 가능여부)
        m_dest = np.zeros(H + 1, dtype=bool)
        m_dest[0] = True
        any_amb = len(full['amb_wait'][0]) > 0
        any_uav = len(full['uav_wait'][0]) > 0
        if any_amb or any_uav:
            # occ(통신)=입원 census+이송중 in-flight(도착 예상) / psent(단절)=누적 발송
            if _cap_gate_is_occ():
                cap_used_arr = full['h_states'][:, -1] + self._in_flight_vec(full, H)  # [고속화 S2-7]
            else:
                cap_used_arr = full['p_sent']
            for h in range(H):
                # 보낼 곳 capa 여유 있으면 허용 (occ 기본 / psent 토글)
                max_send = self.en_manager.en_properties['hospital']['hos_max_send'][h]
                if cap_used_arr[h] < max_send:
                    m_dest[h + 1] = True

        # dim 2: mode
        m_mode = np.zeros(2, dtype=bool)
        m_mode[0] = any_amb
        m_mode[1] = any_uav

        # 어떤 환자도 없거나 어떤 자원도 없으면 stay가 유일한 합법 액션 →
        # 정책이 어쨌든 destination=0 선택할 수 있게 dim 0/2는 모두 True 로 풀어줌
        if not m_class.any() or not (any_amb or any_uav):
            m_class[:] = True
            m_mode[:] = True
            m_dest[:] = False
            m_dest[0] = True

        return np.concatenate([m_class, m_dest, m_mode])

    def _in_flight_vec(self, full, H):
        """[고속화 S2-7] 이송중 대수 — EventManager 증분 카운터가 있으면 그걸 쓴다.

        카운터는 `EventManager.start()` 이후에만 존재하므로(그 전 호출·구 시나리오 객체)
        없으면 원본과 같은 전수 계산으로 되돌아간다. 값은 어느 쪽이든 동일하다.
        """
        cnt = getattr(self.ev_manager, "_in_flight_cnt", None)
        if cnt is not None and len(cnt) == H:
            return cnt
        return self.en_manager.in_flight_by_hospital(full, H)

    def action_masks_joint(self):
        """
        Discrete(2*(H+1)*2) 형식의 결합 mask. env_wrapper.SB3DiscreteWrapper 가 사용.
        - stay (dest=0)는 항상 허용
        - dest!=0 은 (해당 class 환자 존재) AND (해당 mode 자원 존재) AND (해당 병원 capa 여유)
        - mode=1 (UAV) 은 helipad 보유 병원에만 허용 (도메인 기본 제약)
        """
        full = self.en_manager.get_full_obs()
        H = self.H
        any_amb = len(full['amb_wait'][0]) > 0
        any_uav = len(full['uav_wait'][0]) > 0
        hos_props = self.en_manager.en_properties['hospital']
        max_send = hos_props['hos_max_send']
        # 용량 게이트 (2026-07-03 통신축 재정의): occ(통신 가용)=입원 census+이송중
        # in-flight(도착 예상, 수술완료 시 census 감소=완료 확인) | psent(통신 단절)=
        # 현장이 보낸 누적 발송. _cap_gate_is_occ/RuleManager:253 과 동일 정의(쌍비교 불변식).
        if _cap_gate_is_occ():
            cap_used = full['h_states'][:, -1] + self._in_flight_vec(full, H)
        else:
            cap_used = full['p_sent']

        # UAV → helipad 보유 병원만. helipad_idx 가 비어있으면 UAV는 어떤 병원도 못 감.
        # [고속화 S2-8] 병원 축을 벡터화. 원본은 2×(H+1)×2 파이썬 3중 루프였다.
        #   허용 조건은 원본과 글자 그대로 같다:
        #     AMB : cap_used[h] < max_send[h]
        #     UAV : cap_used[h] < max_send[h] AND h 가 헬기장 보유
        #   범위를 벗어난 helipad 인덱스는 원본에서도 `h in range(H)` 와 안 만나 무시된다.
        helipad_idx = np.asarray(hos_props.get('hos_helipad_idx', np.array([]))).reshape(-1)
        helipad_ok = np.zeros(H, dtype=bool)
        if helipad_idx.size:
            hp = helipad_idx.astype(np.intp)
            hp = hp[(hp >= 0) & (hp < H)]
            helipad_ok[hp] = True

        cap_ok = np.asarray(cap_used)[:H] < np.asarray(max_send)[:H]
        allow = (cap_ok, cap_ok & helipad_ok)          # (AMB, UAV)
        mode_avail = (any_amb, any_uav)

        mask = np.zeros((2, H + 1, 2), dtype=bool)
        mask[:, 0, :] = True                            # stay 는 항상 허용
        for c in range(2):
            if len(full['p_wait'][c][0]) == 0:
                continue
            for m in range(2):
                if mode_avail[m]:
                    mask[c, 1:, m] = allow[m]
        return mask.reshape(-1)
