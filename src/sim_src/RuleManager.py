import os
import numpy as np

from EntityManager import EntityManager
from ShinHeuristics import SHIN_METHODS, SHIN_MODE_RULES, ShinHeuristicRule


def _cap_gate_is_occ():
    """발송(현장→병원) 용량 게이트 신호 선택 (2026-07-03 통신축 재정의).
    occ(기본)=통신 가용: 병원 입원 census(수술완료 시 감소=완료 확인) + 이송중
    in-flight(도착 예상)를 앎. psent=통신 단절: 현장이 보낸 누적(p_sent)만 앎.
    MCI_CAP_GATE 로 토글하며 RL(MCIEnvironment_gymnasium._cap_gate_is_occ)·
    휴리스틱(여기)·obs(hospital_feature_wrapper cap_remain)가 같은 env 변수를 공유한다.
    ⚠️ 이건 '발송 결정' 게이트일 뿐 — 병원의 실제 입원/diversion 은 sim 이 항상 occ
    (n_occupied<max_capa, 퇴원 시 occ-=1, 꽉 차면 diversion)로 처리한다(불변)."""
    return os.environ.get("MCI_CAP_GATE", "occ").strip().lower() != "psent"


class RuleManager():
    def __init__(self, configs, scenario, rng=None):
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        self.scenario = scenario
        self.rules = []
        self.rule_names = []
        include_standard = configs.get('include_standard', True)
        if include_standard:
            if configs['isFullFactorial']:
                priorities = ["START", "ReSTART"]
                hospital_rules = ["RedOnly", "YellowNearest"]
                red_modes = ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]
                yellow_modes = ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]
            else:
                priorities = configs['priority_rule']
                hospital_rules = configs['hos_select_rule']
                red_modes = configs['red_mode_rule']
                yellow_modes = configs['yellow_mode_rule']
            for priority in priorities:
                for hos_select in hospital_rules:
                    for mode_R in red_modes:
                        for mode_Y in yellow_modes:
                            self._append(Universal_Rule(priority, hos_select, mode_R, mode_Y))

        # 기존 시나리오·scoreboard의 64룰 개수는 기본적으로 그대로 유지한다.
        # 명시적으로 include_shin=true일 때만 4개 방법×4개 mode=16개를 추가한다.
        if configs.get('include_shin', False):
            methods = tuple(configs.get('shin_methods', SHIN_METHODS))
            modes = tuple(configs.get('shin_mode_rules', SHIN_MODE_RULES))
            unknown_methods = sorted(set(methods) - set(SHIN_METHODS))
            unknown_modes = sorted(set(modes) - set(SHIN_MODE_RULES))
            if unknown_methods or unknown_modes:
                raise ValueError(
                    f"Shin 규칙 설정 오류 methods={unknown_methods}, modes={unknown_modes}"
                )
            for method in methods:
                for mode in modes:
                    self._append(ShinHeuristicRule(method, mode))

        self.rule_names = [rule.rule_name for rule in self.rules]

    def _append(self, rule):
        rule.set_seed(self.rng)
        rule.init_with_scenario(self.scenario)
        self.rules.append(rule)

    def set_seed(self, rng):
        self.rng = rng

class Rule:
    def __init__(self):
        self.name = "Undefined Rule"
    def init_with_scenario(self, scenario):
        en_properties = scenario['EntityManager'].en_properties

        
    #     # 가까운 세 개 병원 왕복이동시간 평균으로 theta 값 계산
    #     self.theta_amb = np.mean(en_properties['ambulance']['amb_HtoS_t'][0][0:3]) * 2
    #     self.theta_uav = np.mean(en_properties['uav']['uav_HtoS_t'][0][0:3]) * 2
    #     self.K_amb = en_properties['ambulance']['amb_num']
    #     self.K_uav = en_properties['uav']['uav_num']
    
    # UAV 대수 먼저 확인
        self.K_uav = en_properties['uav']['uav_num']
        self.K_amb = en_properties['ambulance']['amb_num']

        self.hos_num = en_properties['hospital']['hos_num']
        self.tier3_idx = en_properties['hospital']['hos_tier3_idx'] # Tier3 = 상급종합병원, Tier2 = 나머지
        self.tier2_idx = en_properties['hospital']['hos_tier2_idx'] # Tier3 = 상급종합병원, Tier2 = 나머지
        self.helipad_idx = en_properties['hospital'].get('hos_helipad_idx', np.array([]))
        self.hos_max_send = en_properties['hospital']['hos_max_send'] # 최대 보낼 수 있는 환자수 (목표치)

        # ScenarioManager가 is_use_time을 이미 반영한 실제 평균 ETA를 단일 기준으로 사용한다.
        # AMB: API duration 또는 거리/속도, UAV: 유클리드 거리/속도.
        self.amb_eta = np.asarray(en_properties['ambulance']['amb_HtoS_t'][0], dtype=float)
        self.uav_eta = np.asarray(en_properties['uav']['uav_HtoS_t'][0], dtype=float)

        def nearest_roundtrip_mean(eta, candidates=None):
            if candidates is not None:
                candidates = np.asarray(candidates, dtype=int)
                eta = eta[candidates] if candidates.size else np.array([], dtype=float)
            if eta.size == 0:
                return 0.0
            n_nearest = min(3, eta.size)
            return float(np.mean(np.sort(eta, kind='stable')[:n_nearest]) * 2)

        # ReSTART theta는 병원 배열 앞 3개가 아니라 실제 ETA가 가장 짧은 3개 왕복 평균.
        self.theta_amb = nearest_roundtrip_mean(self.amb_eta)

    # UAV theta 계산 (UAV=0 대응, 헬기장 병원만 실제 후보)
        if self.K_uav == 0:
            self.theta_uav = 0  # UAV가 없으면 0으로 설정
        else:
            self.theta_uav = nearest_roundtrip_mean(self.uav_eta, self.helipad_idx)

        self.expected_R = en_properties['patient']['incident_size'] * en_properties['patient']['patient_info']['ratio'][0]
        self.expected_Y = en_properties['patient']['incident_size'] * en_properties['patient']['patient_info']['ratio'][1]

    def _ordered_hospital_indices(self, candidates, mode):
        """현재 이송수단의 실제 평균 ETA 오름차순으로 병원 후보를 반환."""
        candidates = np.asarray(list(candidates), dtype=int)
        if candidates.size == 0:
            return candidates
        eta = self.uav_eta if int(mode) == 1 else self.amb_eta
        if eta.size != self.hos_num:
            return candidates
        order = np.argsort(eta[candidates], kind='stable')
        return candidates[order]


    def set_seed(self, rng):
        self.rng = rng

    def select(self, obs):
        """
        :param obs: 현재 상태 정보
        :return: 선택할 결정(액션)
        """
        raise NotImplementedError

# 기본 룰

class Universal_Rule(Rule):
    def __init__(self, priority, hos_select, mode_R, mode_Y):
        assert priority in ["START", "ReSTART"]
        # assert hos_select in ["RedOnly", "YellowHalf"]
        assert hos_select in ["RedOnly", "YellowNearest"]
        assert mode_R in ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]
        assert mode_Y in ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]

        self.priority = priority
        self.hos_select = hos_select
        self.mode_R = mode_R
        self.mode_Y = mode_Y

        self.rule_name = f"{priority}, {hos_select}, Red {mode_R}, Yellow {mode_Y}"

    def select(self, obs):
        self.obs = obs
        action = [0, 0, 0] # STAY action if not changed at all

        # For ReSTART
        if self.priority == "ReSTART":
            mask_red = self.obs['p_states'][:,0] == 0
            mask_yellow = self.obs['p_states'][:,0] == 1
            # red_move = self.obs['p_states'][mask_red][:, 1:].sum()
            # yellow_move = self.obs['p_states'][mask_yellow][:, 1:].sum()

            # p_states[:,0]=class, [:,2]=> move(이송 시작으로 수정)
            red_move = self.obs['p_states'][mask_red][:, 2].sum()
            yellow_move = self.obs['p_states'][mask_yellow][:, 2].sum()


            num_D = max(self.expected_Y - yellow_move,0) # yellow 환자 발생 예상 환자 수, yellow_count = yellow 환자 이송 수
            num_I = max(self.expected_R - red_move,0)
            # K_amb=0(UAV-only) 또는 K_uav=0 어느 쪽이든 ZeroDivisionError 방지
            amb_term = (self.theta_amb / self.K_amb) if self.K_amb > 0 else 0.0
            uav_term = (self.theta_uav / self.K_uav) if self.K_uav > 0 else 0.0
            self.tau = 71 - (0.5 * num_D * (amb_term + uav_term))
            

        red_exist = self.obs['p_wait'][0][0]
        yellow_exist = self.obs['p_wait'][1][0]
        if not red_exist and not yellow_exist:
            return action

        # 1. Prioirty selection
        if self.priority == "START":
            if red_exist:  # Red 환자 있는 경우
                action[0] = 0  # Red
            elif yellow_exist:  # Yellow 환자 있는 경우
                action[0] = 1
        elif self.priority == "ReSTART":
            if self.tau <= 0:  # 모든 yellow 우선 red다음
                if yellow_exist:  # Yellow 환자 있는 경우
                    action[0] = 1
                elif red_exist:  # Red 환자 있는 경우
                    action[0] = 0  # Red
            # K_amb=0(UAV-only) 또는 K_uav=0 어느 쪽이든 안전하게 처리
            elif self.tau >= num_I * (amb_term + uav_term):
                if red_exist:  # Red 환자 있는 경우
                    action[0] = 0  # Red
                elif yellow_exist:  # Yellow 환자 있는 경우
                    action[0] = 1
            else:  # red환자를 시간이 self.tau가 될때까지 먼저 보내거나 현장에 red가 없을때까지 먼저 보내다가 yellow 다 보낸후 남은 인원 보내기
                if self.obs['time'] <= self.tau:
                    if red_exist:  # Red 환자 있는 경우
                        action[0] = 0  # Red
                    elif yellow_exist:  # Yellow 환자 있는 경우
                        action[0] = 1
                else:
                    if yellow_exist:  # Yellow 환자 있는 경우
                        action[0] = 1
                    elif red_exist:  # Red 환자 있는 경우
                        action[0] = 0  # Red
        


            

        # 3. Mode selection
        available_UAV = self.obs['uav_wait'][0]
        available_Amb = self.obs['amb_wait'][0]
        if not available_UAV and not available_Amb:
            return action
        isSTAY = False
        if action[0] == 0: # Red selected
            if self.mode_R == "OnlyUAV":
                # if available_UAV: action[2] = 1  # UAV
                # else: action[1] = 0 # STAY
                if available_UAV: action[2] = 1  # UAV
                elif available_Amb:
                    if yellow_exist and self.mode_Y != "OnlyUAV": # Send yellow via AMB
                        action[0] = 1 # Yellow
                        action[2] = 0 # AMB
                    else:
                        isSTAY = True # STAY
                else: print("Error in Transition", action, self.obs)
            elif self.mode_R == "Both_UAVFirst":
                if available_UAV: action[2] = 1  # UAV
                elif available_Amb: action[2] = 0  # AMB
                else: print("Error in Transition", action, self.obs)
            elif self.mode_R == "Both_AMBFirst":
                if available_Amb: action[2] = 0  # AMB
                elif available_UAV: action[2] = 1  # UAV
                else: print("Error in Transition", action, self.obs)
            elif self.mode_R == "OnlyAMB":
                # if available_Amb: action[2] = 0  # AMB
                # else: action[1] = 0  # STAY
                if available_Amb: action[2] = 0  # AMB
                elif available_UAV:
                    if yellow_exist and self.mode_Y != "OnlyAMB": # Send yellow via UAV
                        action[0] = 1 # Yellow
                        action[2] = 1 # UAV
                    else:
                        isSTAY = True # STAY
                else: print("Error in Transition", action, self.obs)
        elif action[0] == 1: # Yellow selected
            if self.mode_Y == "OnlyUAV":
                # if available_UAV: action[2] = 1  # UAV
                # else: action[1] = 0 # STAY

                if available_UAV: action[2] = 1  # UAV
                elif available_Amb:
                    if red_exist and self.mode_R != "OnlyUAV": # Send red via AMB
                        action[0] = 0 # Red
                        action[2] = 0 # AMB
                    else:
                        isSTAY = True # STAY
                else: print("Error in Transition", action, self.obs)

            elif self.mode_Y == "Both_UAVFirst":
                if available_UAV: action[2] = 1  # UAV
                elif available_Amb: action[2] = 0  # AMB
                else: print("Error in Transition", action, self.obs)
            elif self.mode_Y == "Both_AMBFirst":
                if available_Amb: action[2] = 0  # AMB
                elif available_UAV: action[2] = 1  # UAV
                else: print("Error in Transition", action, self.obs)
            elif self.mode_Y == "OnlyAMB":
                # if available_Amb: action[2] = 0  # AMB
                # else: action[1] = 0  # STAY

                if available_Amb: action[2] = 0  # AMB
                elif available_UAV:
                    if red_exist and self.mode_R != "OnlyAMB": # Send red via UAV
                        action[0] = 0 # Red
                        action[2] = 1 # UAV
                    else:
                        isSTAY = True # STAY
                else: print("Error in Transition", action, self.obs)
        # 2. Hospital selection
        if not isSTAY:
            # 발송 게이트 신호 (2026-07-03 통신축 재정의):
            #   occ(통신 가용)  = n_occupied(입원 census, 수술완료 시 감소=완료 정보)
            #                    + in_flight(그 병원으로 이송중=도착 예상 정보)
            #   psent(통신 단절) = p_sent(현장이 보낸 누적) — 현장 지득 정보만
            if _cap_gate_is_occ():
                cap_used = (self.obs['h_states'][:, -1]
                            + EntityManager.in_flight_by_hospital(self.obs, self.hos_num))
            else:
                cap_used = self.obs['p_sent']
            if self.hos_select == "RedOnly":
                if action[0] == 0: # Red selected
                    for i in self._ordered_hospital_indices(self.tier3_idx, action[2]):
                        # ★ 헬기장 체크 추가 (UAV 선택 시)
                        if action[2] == 1 and i not in self.helipad_idx:
                            continue
                        if self.hos_max_send[i] > cap_used[i]: # max_send > n_occupied
                            action[1] = i + 1
                            break
                elif action[0] == 1: # Yellow selected
                    for i in self._ordered_hospital_indices(range(self.hos_num), action[2]):
                        if i in self.tier3_idx:
                            continue
                        # ★ 헬기장 체크 추가 (UAV 선택 시)
                        if action[2] == 1 and i not in self.helipad_idx:
                            continue
                        if self.hos_max_send[i] > cap_used[i]: # max_send > n_occupied
                            action[1] = i + 1
                            break
            # elif self.hos_select == "YellowHalf":
            #     if action[0] == 0:  # Red selected
            #         for i in self.tier3_idx:
            #             # ★ 헬기장 체크 추가 (UAV 선택 시)
            #             if action[2] == 1 and i not in self.helipad_idx:
            #                 continue
            #             if self.hos_max_send[i] > cap_used[i]: # max_send > n_occupied
            #                 action[1] = i + 1
            #                 break
            #     elif action[0] == 1: # Yellow selected
            #         # Get random number from environment for seed control
            #         r = self.rng.random()
            #         if r > 0.5: # Send to tier2
            #             for i in range(self.hos_num):
            #                 if i in self.tier3_idx:  # Tier3 skip → Tier2만 사용
            #                     continue
            #                 # ★ 헬기장 체크 추가 (UAV 선택 시)
            #                 if action[2] == 1 and i not in self.helipad_idx:
            #                     continue
            #                 if self.hos_max_send[i] > cap_used[i]: # max_send > n_occupied
            #                     action[1] = i + 1
            #                     break
            #         else: # Send to tier3
            #             for i in self.tier3_idx:  # Tier3만 사용
            #                 # ★ 헬기장 체크 추가 (UAV 선택 시)
            #                 if action[2] == 1 and i not in self.helipad_idx:
            #                     continue
            #                 if self.hos_max_send[i] > cap_used[i]: # max_send > n_occupied
            #                     action[1] = i + 1
            #                     break
            elif self.hos_select == "YellowNearest":
                if action[0] == 0:  # Red selected
                    for i in self._ordered_hospital_indices(self.tier3_idx, action[2]):
                        # ★ 헬기장 체크 추가 (UAV 선택 시)
                        if action[2] == 1 and i not in self.helipad_idx:
                            continue
                        if self.hos_max_send[i] > cap_used[i]: # max_send > n_occupied
                            action[1] = i + 1
                            break
                elif action[0] == 1: # Yellow selected - 거리순으로 tier 구분 없이 선택
                    for i in self._ordered_hospital_indices(range(self.hos_num), action[2]):
                        # ★ 헬기장 체크 추가 (UAV 선택 시)
                        if action[2] == 1 and i not in self.helipad_idx:
                            continue
                        if self.hos_max_send[i] > cap_used[i]: # max_send > n_occupied
                            action[1] = i + 1
                            break
            

        if isSTAY: # STAY
            action[0], action[2] = -1, -1 # To make redundant

        return action

# class Proposed(Rule):
#     def __init__(self, priority, hos_select, mode_R, mode_Y):
#         assert priority in ["START", "ReSTART"]
#         assert hos_select in ["RedOnly", "YellowHalf"]
#         assert mode_R in ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]
#         assert mode_Y in ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]
#
#         self.priority = priority
#         self.hos_select = hos_select
#         self.mode_R = mode_R
#         self.mode_Y = mode_Y
#
#         self.rule_name = f"Proposed"
#
#
#     def select(self, obs):
#         self.obs = obs
#         action = [0, 0, 0] # STAY action if not changed at all
#         amb_idx = np.where(np.isclose(self.env.state['time_amb'], 0))
#         uav_idx = np.where(np.isclose(self.env.state['time_uav'], 0))
#         # For ReSTART
#         num_D = self.env.yellow_check - self.env.yellow_count
#         num_I = self.env.red_check - self.env.red_count
#         if num_D <= 0:
#             num_D = 0
#         if num_I <= 0:
#             num_I = 0
#         self.tau = 71 - (0.5 * num_D * (self.theta_amb/self.K_amb + self.theta_uav/self.K_uav))
#
#         # 1. Prioirty selection
#         if self.priority == "START":
#             if self.obs['p_at_sites'][0] != 0:  # Red 환자 있는 경우
#                 action[0] = 0  # Red
#             elif self.obs['p_at_sites'][1] != 0:  # Yellow 환자 있는 경우
#                 action[0] = 1
#         elif self.priority == "ReSTART":
#             if self.tau <= 0:  # 모든 yellow 우선 red다음
#                 if self.obs['p_at_sites'][1] != 0:  # Yellow 환자 있는 경우
#                     action[0] = 1
#                 elif self.obs['p_at_sites'][0] != 0:  # Red 환자 있는 경우
#                     action[0] = 0  # Red
#             elif self.tau >= num_I * (self.theta_amb / self.K_amb + self.theta_uav / self.K_uav):  # 모든 red 보내고 yellow
#                 if self.obs['p_at_sites'][0] != 0:  # Red 환자 있는 경우
#                     action[0] = 0  # Red
#                 elif self.obs['p_at_sites'][1] != 0:  # Yellow 환자 있는 경우
#                     action[0] = 1
#             else:  # red환자를 시간이 self.tau가 될때까지 먼저 보내거나 현장에 red가 없을때까지 먼저 보내다가 yellow 다 보낸후 남은 인원 보내기
#                 if self.env.time <= self.tau:
#                     if self.obs['p_at_sites'][0] != 0:  # Red 환자 있는 경우
#                         action[0] = 0  # Red
#                     elif self.obs['p_at_sites'][1] != 0:  # Yellow 환자 있는 경우
#                         action[0] = 1
#                 else:
#                     if self.obs['p_at_sites'][1] != 0:  # Yellow 환자 있는 경우
#                         action[0] = 1
#                     elif self.obs['p_at_sites'][0] != 0:  # Red 환자 있는 경우
#                         action[0] = 0  # Red
#
#         # 3. Mode selection
#         isUAV = bool(len(uav_idx[0]))
#         isAmb = bool(len(amb_idx[0]))
#         isSTAY = False
#         if action[0] == 0: # Red selected
#             if self.mode_R == "OnlyUAV":
#                 # if isUAV: action[2] = 1  # UAV
#                 # else: action[1] = 0 # STAY
#                 if isUAV: action[2] = 1  # UAV
#                 elif isAmb:
#                     if self.obs['p_at_sites'][1] != 0 and self.mode_Y != "OnlyUAV": # Send yellow via AMB
#                         action[0] = 1 # Yellow
#                         action[2] = 0 # AMB
#                     else:
#                         isSTAY = True # STAY
#                 else: print("Error in Transition", action, self.env.state)
#             elif self.mode_R == "Both_UAVFirst":
#                 if isUAV: action[2] = 1  # UAV
#                 elif isAmb: action[2] = 0  # AMB
#                 else: print("Error in Transition", action, self.env.state)
#             elif self.mode_R == "Both_AMBFirst":
#                 if isAmb: action[2] = 0  # AMB
#                 elif isUAV: action[2] = 1  # UAV
#                 else: print("Error in Transition", action, self.env.state)
#             elif self.mode_R == "OnlyAMB":
#                 # if isAmb: action[2] = 0  # AMB
#                 # else: action[1] = 0  # STAY
#                 if isAmb: action[2] = 0  # AMB
#                 elif isUAV:
#                     if self.obs['p_at_sites'][1] != 0 and self.mode_Y != "OnlyAMB": # Send yellow via UAV
#                         action[0] = 1 # Yellow
#                         action[2] = 1 # UAV
#                     else:
#                         isSTAY = True # STAY
#                 else: print("Error in Transition", action, self.env.state)
#         elif action[0] == 1: # Yellow selected
#             if self.mode_Y == "OnlyUAV":
#                 # if isUAV: action[2] = 1  # UAV
#                 # else: action[1] = 0 # STAY
#
#                 if isUAV: action[2] = 1  # UAV
#                 elif isAmb:
#                     if self.obs['p_at_sites'][0] != 0 and self.mode_R != "OnlyUAV": # Send red via AMB
#                         action[0] = 0 # Red
#                         action[2] = 0 # AMB
#                     else:
#                         isSTAY = True # STAY
#                 else: print("Error in Transition", action, self.env.state)
#
#             elif self.mode_Y == "Both_UAVFirst":
#                 if isUAV: action[2] = 1  # UAV
#                 elif isAmb: action[2] = 0  # AMB
#                 else: print("Error in Transition", action, self.env.state)
#             elif self.mode_Y == "Both_AMBFirst":
#                 if isAmb: action[2] = 0  # AMB
#                 elif isUAV: action[2] = 1  # UAV
#                 else: print("Error in Transition", action, self.env.state)
#             elif self.mode_Y == "OnlyAMB":
#                 # if isAmb: action[2] = 0  # AMB
#                 # else: action[1] = 0  # STAY
#
#                 if isAmb: action[2] = 0  # AMB
#                 elif isUAV:
#                     if self.obs['p_at_sites'][0] != 0 and self.mode_R != "OnlyAMB": # Send red via UAV
#                         action[0] = 0 # Red
#                         action[2] = 1 # UAV
#                     else:
#                         isSTAY = True # STAY
#                 else: print("Error in Transition", action, self.env.state)
#         # 2. Hospital selection
#         if not isSTAY:
#             if self.hos_select == "RedOnly":
#                 if action[0] == 0: # Red selected
#                     for i in self.env.tier2_idx:
#                         if self.env.capa_scale[i] > 0:
#                             action[1] = i + 1
#                             break
#                 elif action[0] == 1: # Yellow selected
#                     for i in range(self.env.numH):
#                         if i in self.env.tier2_idx:
#                             continue
#                         if self.env.capa_scale[i] > 0:
#                             action[1] = i + 1
#                             break
#             elif self.hos_select == "YellowHalf":
#                 if action[0] == 0:  # Red selected
#                     for i in self.env.tier2_idx:
#                         if self.env.capa_scale[i] > 0:
#                             action[1] = i + 1
#                             break
#                 elif action[0] == 1: # Yellow selected
#                     # Get random number from environment for seed control
#                     r = self.env.random_gen.rand()
#                     if r > 0.5: # Send to tier2
#                         for i in self.env.tier2_idx:
#                             if self.env.capa_scale[i] > 0:
#                                 action[1] = i + 1
#                                 break
#                     else: # Send to tier3
#                         for i in range(self.env.numH):
#                             if i in self.env.tier2_idx:
#                                 continue
#                             if self.env.capa_scale[i] > 0:
#                                 action[1] = i + 1
#                                 break
#
#         row_amb = len(self.env.ed_data_amb) - 1
#         row_uav = len(self.env.ed_data_uav) - 1
#         if action[2] == 0: # AMB 선택 시
#             # # Version 1: UAV 수보다 적을 때 시작
#             # if self.env.rescue_finish: # 구조 종료
#             #     remainedRY = self.env.totalN - self.env.moved_N - self.obs['p_at_sites'][2]
#             #     if remainedRY < self.env.uav_num: # 남은 환자 수가 UAV 수보다 적음.
#             #         if len(self.obs['time_uav'][self.obs['dest_uav'] == 0]) != 0:
#             #             min_time_uav = min(self.obs['time_uav'][self.obs['dest_uav'] == 0])
#             #         else:
#             #             min_time_uav = np.infty
#             #         time_amb = float(self.env.ed_data_amb.iloc[row_amb, action[1]]) * 60 / self.env.v_amb
#             #         time_uav = float(self.env.ed_data_uav.iloc[row_uav, action[1]]) * 60 / self.env.v_uav
#             #         if min_time_uav < time_amb - time_uav:
#             #             action[1] = 0  # STAY (AMB)
#             # # Version 2: AMB + UAV 수보다 적을 때 시작
#             # if self.env.rescue_finish: # 구조 종료
#             #     remainedRY = self.env.totalN - self.env.moved_N - self.obs['p_at_sites'][2]
#             #     if remainedRY < self.env.uav_num + self.env.amb_num: # 남은 환자 수가 UAV + AMB 수보다 적음.
#             #         if len(self.obs['time_uav'][self.obs['dest_uav'] == 0]) != 0:
#             #             min_time_uav = min(self.obs['time_uav'][self.obs['dest_uav'] == 0])
#             #         else:
#             #             min_time_uav = np.infty
#             #         time_amb = float(self.env.ed_data_amb.iloc[row_amb, action[1]]) * 60 / self.env.v_amb
#             #         time_uav = float(self.env.ed_data_uav.iloc[row_uav, action[1]]) * 60 / self.env.v_uav
#             #         if min_time_uav < time_amb - time_uav:
#             #             action[1] = 0  # STAY (AMB)
#             # # Version 3: 돌아오는 시간 계산
#             # if self.env.rescue_finish: # 구조 종료
#             #     remainedRY = self.env.totalN - self.env.moved_N - self.obs['p_at_sites'][2]
#             #     if remainedRY < self.env.uav_num: # 남은 환자 수가 UAV 수보다 적음.
#             #         min_time_uav = np.infty
#             #         for i in range(self.env.uav_num):
#             #             tmp = 0
#             #             if self.obs['dest_uav'][i] == 0:
#             #                 tmp = self.obs['time_uav'][i]
#             #             else:
#             #                 tmp = self.obs['time_uav'][i] + self.env.uav_d_mean[0, self.obs['dest_uav'][i] - 1] * 60 / self.env.v_uav
#             #             if tmp < min_time_uav:
#             #                 min_time_uav = tmp
#             #         time_amb = float(self.env.ed_data_amb.iloc[row_amb, action[1]]) * 60 / self.env.v_amb
#             #         time_uav = float(self.env.ed_data_uav.iloc[row_uav, action[1]]) * 60 / self.env.v_uav
#             #         if min_time_uav < time_amb - time_uav:
#             #             action[1] = 0  # STAY (AMB)
#             # Version 4: proposition 활용
#             # if self.env.rescue_finish:  # 구조 종료
#             uav_return_times = sorted(self.env.uav_expected_return) # UAV마다 현장 도착 예상 시간
#             stay_amb = True
#             for idx, record in enumerate(self.env.slack_times):
#                 if record[0] < uav_return_times[0]: # slack vs expected UAV return time
#                     stay_amb = False
#                     break
#                 else:
#                     uav_return_times[0] += 2 * self.env.uav_d_mean[0, record[2]]
#                 uav_return_times.sort()
#             if stay_amb:
#                 isSTAY = True # STAY (AMB)
#             else:  # AMB 써야하는 상황
#                 # Check whether flip priority
#                 uav_return_times = sorted(self.env.uav_expected_return)
#                 isFlip = True
#                 for idx, record in enumerate(self.env.slack_times):
#                     if record[1] == action[0]:
#                         if record[0] < uav_return_times[0]:  # slack vs expected UAV return time
#                             isFlip = False
#                             break
#                         else:
#                             uav_return_times[0] += 2 * self.env.uav_d_mean[0, record[2]]
#                         uav_return_times.sort()
#                 if isFlip: # Priority change
#                     if self.obs['p_at_sites'][abs(action[0] - 1)] != 0:
#                         action[0] = abs(action[0] - 1)
#                         action[2] = 0  # AMB
#                         for idx, record in enumerate(self.env.slack_times):
#                             if record[1] == action[0]:
#                                 action[1] = record[2] + 1
#                                 break
#                 # Version 1
#                 # min_uav_return = min(self.env.uav_expected_return)
#                 # isFlip = False
#                 # for idx, record in enumerate(self.env.slack_times):
#                 #     if record[1] == action[0]:
#                 #         if record[0] > min_uav_return:
#                 #             isFlip = True
#                 #         break
#                 # if isFlip: # Priority change
#                 #     if self.obs['p_at_sites'][abs(action[0] - 1)] != 0:
#                 #         action[0] = abs(action[0] - 1)
#                 #         action[2] = 0  # AMB
#                 #         for idx, record in enumerate(self.env.slack_times):
#                 #             if record[1] == action[0]:
#                 #                 action[1] = record[2] + 1
#                 #                 break
#         if isSTAY: # STAY
#             action[0], action[2] = -1, -1 # To make redundant
#             action[1] = 0
#
#         return action


# class Universial_Rule_for_RL(Rule):
#     def __init__(self):
#         self.priority_list = ["START", "ReSTART"]
#         self.hos_select_list = ["RedOnly", "YellowHalf"]
#         self.mode_R_list = ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]
#         self.mode_Y_list = ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]
#
#     def select_with_args(self, obs, priority, hos_select, mode_R, mode_Y):
#         self.priority = self.priority_list[priority]
#         self.hos_select = self.hos_select_list[hos_select]
#         self.mode_R = self.mode_R_list[mode_R]
#         self.mode_Y = self.mode_Y_list[mode_Y]
#
#         self.obs = obs
#         action = [0, 0, 0] # STAY action if not changed at all
#         amb_idx = np.where(np.isclose(self.env.state['time_amb'], 0))
#         uav_idx = np.where(np.isclose(self.env.state['time_uav'], 0))
#         # For ReSTART
#         num_D = self.env.yellow_check - self.env.yellow_count
#         num_I = self.env.red_check - self.env.red_count
#         if num_D <= 0:
#             num_D = 0
#         if num_I <= 0:
#             num_I = 0
#         self.tau = 71 - (0.5 * num_D * (self.theta_amb/self.K_amb + self.theta_uav/self.K_uav))
#
#         # 1. Prioirty selection
#         if self.priority == "START":
#             if self.obs['p_at_sites'][0] != 0:  # Red 환자 있는 경우
#                 action[0] = 0  # Red
#             elif self.obs['p_at_sites'][1] != 0:  # Yellow 환자 있는 경우
#                 action[0] = 1
#         elif self.priority == "ReSTART":
#             if self.tau <= 0:  # 모든 yellow 우선 red다음
#                 if self.obs['p_at_sites'][1] != 0:  # Yellow 환자 있는 경우
#                     action[0] = 1
#                 elif self.obs['p_at_sites'][0] != 0:  # Red 환자 있는 경우
#                     action[0] = 0  # Red
#             elif self.tau >= num_I * (self.theta_amb / self.K_amb + self.theta_uav / self.K_uav):  # 모든 red 보내고 yellow
#                 if self.obs['p_at_sites'][0] != 0:  # Red 환자 있는 경우
#                     action[0] = 0  # Red
#                 elif self.obs['p_at_sites'][1] != 0:  # Yellow 환자 있는 경우
#                     action[0] = 1
#             else:  # red환자를 시간이 self.tau가 될때까지 먼저 보내거나 현장에 red가 없을때까지 먼저 보내다가 yellow 다 보낸후 남은 인원 보내기
#                 if self.env.time <= self.tau:
#                     if self.obs['p_at_sites'][0] != 0:  # Red 환자 있는 경우
#                         action[0] = 0  # Red
#                     elif self.obs['p_at_sites'][1] != 0:  # Yellow 환자 있는 경우
#                         action[0] = 1
#                 else:
#                     if self.obs['p_at_sites'][1] != 0:  # Yellow 환자 있는 경우
#                         action[0] = 1
#                     elif self.obs['p_at_sites'][0] != 0:  # Red 환자 있는 경우
#                         action[0] = 0  # Red
#
#         # 2. Hospital selection
#         if self.hos_select == "RedOnly":
#             if action[0] == 0: # Red selected
#                 for i in self.env.tier2_idx:
#                     if self.env.capa_scale[i] > 0:
#                         action[1] = i + 1
#                         break
#             elif action[0] == 1: # Yellow selected
#                 for i in range(self.env.numH):
#                     if i in self.env.tier2_idx:
#                         continue
#                     if self.env.capa_scale[i] > 0:
#                         action[1] = i + 1
#                         break
#         elif self.hos_select == "YellowHalf":
#             if action[0] == 0:  # Red selected
#                 for i in self.env.tier2_idx:
#                     if self.env.capa_scale[i] > 0:
#                         action[1] = i + 1
#                         break
#             elif action[0] == 1: # Yellow selected
#                 # Get random number from environment for seed control
#                 r = self.env.random_gen.rand()
#                 if r > 0.5: # Send to tier2
#                     for i in self.env.tier2_idx:
#                         if self.env.capa_scale[i] > 0:
#                             action[1] = i + 1
#                             break
#                 else: # Send to tier3
#                     for i in range(self.env.numH):
#                         if i in self.env.tier2_idx:
#                             continue
#                         if self.env.capa_scale[i] > 0:
#                             action[1] = i + 1
#                             break
#
#         # 3. Mode selection
#         isUAV = bool(len(uav_idx[0]))
#         isAmb = bool(len(amb_idx[0]))
#         if action[0] == 0: # Red selected
#             if self.mode_R == "OnlyUAV":
#                 if isUAV: action[2] = 1  # UAV
#                 else: action[1] = 0 # STAY
#             elif self.mode_R == "Both_UAVFirst":
#                 if isUAV: action[2] = 1  # UAV
#                 elif isAmb: action[2] = 0  # AMB
#                 else: print("Error in Transition", action, self.env.state)
#             elif self.mode_R == "Both_AMBFirst":
#                 if isAmb: action[2] = 0  # AMB
#                 elif isUAV: action[2] = 1  # UAV
#                 else: print("Error in Transition", action, self.env.state)
#             elif self.mode_R == "OnlyAMB":
#                 if isAmb: action[2] = 0  # AMB
#                 else: action[1] = 0  # STAY
#         elif action[0] == 1: # Yellow selected
#             if self.mode_Y == "OnlyUAV":
#                 if isUAV: action[2] = 1  # UAV
#                 else: action[1] = 0 # STAY
#             elif self.mode_Y == "Both_UAVFirst":
#                 if isUAV: action[2] = 1  # UAV
#                 elif isAmb: action[2] = 0  # AMB
#                 else: print("Error in Transition", action, self.env.state)
#             elif self.mode_Y == "Both_AMBFirst":
#                 if isAmb: action[2] = 0  # AMB
#                 elif isUAV: action[2] = 1  # UAV
#                 else: print("Error in Transition", action, self.env.state)
#             elif self.mode_Y == "OnlyAMB":
#                 if isAmb: action[2] = 0  # AMB
#                 else: action[1] = 0  # STAY
#
#         if action[1] == 0: # STAY
#             action[0], action[2] = -1, -1 # To make redundant
#         return action
