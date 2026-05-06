import gymnasium as gym
import numpy as np
import math

# from gym import spaces
# import pandas as pd

class MCIEnvironment_gym(gym.Env):
    def __init__(self, scenario, rng=None, default_rule = None, max_steps=2000, rule_test=False, eval_mode=False, **kwargs):
        # max_steps: 최대 decision 횟수 (episode 종료 없이 무한 반복 방지)
        self.scenario_decoder(scenario)
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()
        self.default_rule = default_rule
        self.max_steps = max_steps
        self.rule_test = rule_test
        self.eval_mode = eval_mode
        self.pen_size = 1.0  # Default: 1.0
        self.reset()

    def set_seed(self, rng):
        self.rng = rng

    def scenario_decoder(self, scenario):
        # Entity 관리
        self.en_manager = scenario['EntityManager']

        # Event 관리
        self.ev_manager = scenario['EventManager']

    def step(self, action):
        info = {}
        if self.ev_manager.check_termination():
            reward = self.pending_terminal_reward
            self.pending_terminal_reward = 0.0
            if self.rule_test:
                obs = self.en_manager.get_full_obs()
            else:
                obs = self.en_manager.get_obs()
            obs['time'] = self.ev_manager.time
            info['time'] = self.ev_manager.time
            return obs, reward, True, False, info
        self.n_step += 1
        if self.n_step > self.max_steps: # 최대 결정 회수 넘어갈 시 강제 종료
            print("OVERTIME")
            return self.en_manager.get_obs(), -self.pen_size, True, True, info
        log, terminated = self.ev_manager.run_next(action)
        reward = self.logToReward(log)
        if self.rule_test:
            obs = self.en_manager.get_full_obs()
        else:
            obs = self.en_manager.get_obs()
        # Resource 상태 외의 요소 추가
        obs['time'] = self.ev_manager.time
        info['time'] = self.ev_manager.time
        return obs, reward, terminated, False, info

    def reset(self, seed = None):
        self.pending_terminal_reward = 0.0
        info = {}
        self.n_step = 0 # Decision 횟수

        # 자원 초기화
        self.en_manager.init_en_status()
        # EventManager 초기화
        init_log = self.ev_manager.start()
        self.preventable = self.computePreventable(init_log)
        if self.ev_manager.check_termination():
            self.pending_terminal_reward = self.logToReward(init_log)
        # observation 생성
        if self.rule_test:
            obs = self.en_manager.get_full_obs()
        else:
            obs = self.en_manager.get_obs()
        obs['time'] = self.ev_manager.time
        return obs, info

    def computePreventable(self, log):
        val = 0
        for p_class, times in enumerate(log['rescue_times']):
            for t in times:
                val += self.getSurvProb(t, p_class)
        return val

    def logToReward(self, log):
        reward = 0
        for x in log['p_admit']:
            reward += self.getReward(x[0], x[1])
        return reward

    def getReward(self, time, p_class):
        if self.eval_mode:
            val = self.getSurvProb(time, p_class)
        else:
            # 1. SurvProb
            val = self.getSurvProb(time, p_class)
            # # 2. PDR w.o. Green
            # if p_class != 2:
            #     val = self.getSurvProb(time, p_class)/ self.preventable
            # else: # Green은 학습 시 제외
            #     val = 0
        return val

    def getSurvProb(self, time, p_class):
        # read survival probability function
        if (p_class == 0):  # immediate (Red)
            val = 0.56 / (math.pow((time / 91), 1.58) + 1)  # Scenario 5
        elif (p_class == 1):  # delayed (Yellow)
            val = 0.81 / (math.pow((time / 160), 2.41) + 1)  # Scenario 5
        elif (p_class == 2):  # Green
            val = 1.0
            # val = 0.0
        elif (p_class == 3): # Black
            val = 0.0
        # else:
        #     print(self.state, p_class)
        return val


    # def set_seed(self, seed):
    #     # 초기 seed 고정
    #     super().reset(seed)
