import gymnasium as gym
import numpy as np
import math

from gymnasium import spaces


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

    action: MultiDiscrete([3, H+1, 2])
        - [0] p_class: 0=Red, 1=Yellow, 2=Green
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
        self.action_space = spaces.MultiDiscrete([3, self.H + 1, 2])

        # gym 표준: 사용자가 첫 step 전에 reset()을 호출하도록 강제.
        # 생성자에서 시뮬을 자동 시작하면 main.py가 첫 reset을 무시하면서
        # 한 번 더 ev_onset 이 트리거되어 헛 시뮬 로그가 남는다.
        # 이하 attribute 만 초기화 (실제 상태는 reset() 호출 시 채워짐).
        self.pending_terminal_reward = 0.0
        self.n_step = 0
        self.preventable = 0.0

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
            self.pending_terminal_reward = 0.0
            obs = self._make_obs()
            info['time'] = self.ev_manager.time
            return obs, reward, True, False, info

        self.n_step += 1
        if self.n_step > self.max_steps:
            print("OVERTIME")
            return self._make_obs(), -self.pen_size, True, True, info

        action_list = self._to_action_list(action)
        log, terminated = self.ev_manager.run_next(action_list)
        reward = self.logToReward(log)
        obs = self._make_obs()
        info['time'] = self.ev_manager.time
        return obs, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pending_terminal_reward = 0.0
        self.n_step = 0

        self.en_manager.init_en_status()
        init_log = self.ev_manager.start()
        self.preventable = self.computePreventable(init_log)
        if self.ev_manager.check_termination():
            self.pending_terminal_reward = self.logToReward(init_log)

        obs = self._make_obs()
        return obs, {}

    # ---------- reward ----------
    def computePreventable(self, log):
        val = 0.0
        for p_class, times in enumerate(log['rescue_times']):
            for t in times:
                val += self.getSurvProb(t, p_class)
        return val

    def logToReward(self, log):
        reward = 0.0
        for x in log['p_admit']:
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

        주의: 이 mask는 차원별 독립이라 결합 제약(예: Red인데 UAV=0)은 보장 못 함.
              joint mask는 env_wrapper에서 별도 구성.
        """
        full = self.en_manager.get_full_obs()
        H = self.H

        # dim 0: p_class
        m_class = np.zeros(3, dtype=bool)
        for c in range(3):
            m_class[c] = len(full['p_wait'][c][0]) > 0

        # dim 1: destination (0=stay 항상 허용, 1..H = 가용 모드 + 모드별 가능여부)
        m_dest = np.zeros(H + 1, dtype=bool)
        m_dest[0] = True
        any_amb = len(full['amb_wait'][0]) > 0
        any_uav = len(full['uav_wait'][0]) > 0
        if any_amb or any_uav:
            for h in range(H):
                # 보낼 곳 capa 여유 있으면 허용
                max_send = self.en_manager.en_properties['hospital']['hos_max_send'][h]
                p_sent = full['p_sent'][h]
                if p_sent < max_send:
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

    def action_masks_joint(self):
        """
        Discrete(3*(H+1)*2) 형식의 결합 mask. env_wrapper.SB3DiscreteWrapper 가 사용.
        - stay (dest=0)는 항상 허용
        - dest!=0 은 (해당 class 환자 존재) AND (해당 mode 자원 존재) AND (해당 병원 capa 여유)
        """
        full = self.en_manager.get_full_obs()
        H = self.H
        any_amb = len(full['amb_wait'][0]) > 0
        any_uav = len(full['uav_wait'][0]) > 0
        max_send = self.en_manager.en_properties['hospital']['hos_max_send']
        p_sent = full['p_sent']

        mask = np.zeros((3, H + 1, 2), dtype=bool)
        for c in range(3):
            class_avail = len(full['p_wait'][c][0]) > 0
            for m, mode_avail in enumerate([any_amb, any_uav]):
                # stay 는 항상 허용
                mask[c, 0, m] = True
                if class_avail and mode_avail:
                    for h in range(H):
                        if p_sent[h] < max_send[h]:
                            mask[c, h + 1, m] = True
        return mask.reshape(-1)
