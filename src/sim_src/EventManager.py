import heapq
import numpy as np

class EventManager():
    def __init__(self, ev_info, en_manager, rng=None, enable_trace=False):
        self.ev_info = ev_info
        self.en_manager = en_manager
        self.properties = self.en_manager.en_properties
        self.enable_trace = enable_trace
        self.trace_log = []  # per-event trace records

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()
        self.event_queue = []

    def set_seed(self, rng):
        self.rng = rng

    def start(self):
        self.e_ID = 0 # event 생성 번호 초기화
        self.time = 0 # event clock 초기화
        self.status = self.en_manager.en_status # entity 상태 불러오기
        self.rescue_finish = False # 구조 완료 여부 확인
        self.event_queue = [] # event queue 초기화
        self.trace_log = []  # reset trace
        # 사고 발생
        init_log = {}
        init_log, _ = self.ev_onset(init_log, None)
        # 첫 decision epoch까지 실행
        auto_log, _ = self.run_next(action=None)
        init_log['p_admit'] = auto_log.get('p_admit', [])
        return init_log

    def run_next(self, action=None):
        log = {'p_admit':[]} # 시뮬레이션 수행되면서 기록할 지표
        if action is not None:
            normal_action, repeat = self.proceed_action(action)
            log['normal_action'] = normal_action
            if repeat: # 추가 액션 필요
                return log, False

        terminated = False
        while True:
            # 1. 가장 빠른 event 탐색
            if not self.event_queue:

                # 이벤트가 더 이상 없으면 종료 (안전 가드)

                h_idle_que_occ = self.status['hospital']['h_states'].copy() if 'hospital' in self.status else None

                if h_idle_que_occ is None:
                    print(f'[EventQueueEmpty] t={self.time} h_states=None')
                else:
                    # numpy가 ... 로 줄이지 않도록 옵션
                    with np.printoptions(threshold=np.inf, linewidth=200, suppress=True):
                        print(f'[EventQueueEmpty] t={self.time}  h_states.shape={h_idle_que_occ.shape}\n{h_idle_que_occ}')

                return log, True

            c_event = heapq.heappop(self.event_queue)  # event = (event_time, e_ID, ev_name, entity_idx)
            print(c_event)

            time_interval = c_event[0] - self.time
            self.time = c_event[0]
            # 시간 경과에 따른 상태 업데이트
            self.status['ambulance']['amb_states'][:,1] -= time_interval
            np.maximum(self.status['ambulance']['amb_states'][:,1], 0, out=self.status['ambulance']['amb_states'][:,1])
            self.status['uav']['uav_states'][:,1] -= time_interval
            np.maximum(self.status['uav']['uav_states'][:,1], 0, out=self.status['uav']['uav_states'][:,1])
            log, stop_condition = getattr(self, "ev_" + c_event[2])(log, c_event[3])
            if stop_condition: # 2. decision 내려야 하는 시점까지 진행
                break
            terminated = self.check_termination() # 3. 더 이상 의사결정 필요 없으면 남은 시뮬레이션 진행 후 종료
            if terminated:
                break
        return log, terminated

    def proceed_action(self, action):
        print("Action:", action)
        # action[0]: Red = 0, Yellow = 1, Green = 2
        # action[1]: 0: 현장, 1번 병원 ~ N번 병원; 병원 10개일 때 0은 현장, 1번 병원 ~ 9번 병원
        # action[2]: 0: Amb, 1: UAV
        # return: normal_action, repeat
        p_class, destination, mode = action # destination = 병원 index + 1
        if destination == 0:  # 현장에 머무르기
            return True, False
        else:  # 어딘가로 이송
            # 0. Penalize wrong mode selection & terminate
            if mode == 1 and not self.status['uav']['uav_wait'][0]:  # UAV 없는데 다른 곳 이송 명령
                print("NO UAV")
                return False, False
            elif mode == 0 and not self.status['ambulance']['amb_wait'][0]:  # amb 없는데 다른 곳 이송 명령
                print("NO AMB")
                return False, False

            # 0b. UAV는 헬기장 보유 병원에만 이송 가능 (도메인 기본 제약).
            # mask 우회/제거 알고리즘이 non-helipad 행동을 선택해도 여기서 차단한다.
            if mode == 1:
                helipad_idx = self.properties['hospital'].get('hos_helipad_idx', np.array([]))
                if (destination - 1) not in helipad_idx:
                    print("NO HELIPAD")
                    return False, False

            # 1. 현장 환자 수 변경
            try:
                p_idx = self.status['patient']['p_wait'][p_class][0].pop()
                # # TO-DO: Rule에 사용 시 UPDATE
                # if p_class == 0 or p_class == 1:
                #     for idx, record in enumerate(self.slack_times):
                #         if record[1] == action[0]:
                #             del self.slack_times[idx]
                #             break
            except IndexError:
                print("NO PATIENT")
                return False, False

            # 2. 이송 수단에 따른 목적지 및 도달 시간 변경
            h_idx = destination - 1
            tranportation_t = self.sample_transportation_time(mode=mode, origination=0, destination=destination)
            if mode == 0: # amb
                a_idx = self.status['ambulance']['amb_wait'][0].pop()
                elapsed_time = tranportation_t + self.properties['ambulance']['amb_handover_time'] # 환자 넘기는 시간
                # 상태 변경
                self.status['ambulance']['amb_states'][a_idx] = (destination, elapsed_time, p_class+1) # destination, time, severity
                self.status['patient']['p_states'][p_idx, 2] = 1  # move
                self.status['patient']['p_sent'][h_idx] += 1 # sent record
                self.add_event(elapsed_time, 'amb_arrival_hospital', (p_idx, a_idx, h_idx))
                self._record_trace("transport_start", patient_id=int(p_idx), vehicle="AMB",
                                   vehicle_id=int(a_idx), hospital_id=int(h_idx), severity=int(p_class))
            elif mode == 1: # uav
                u_idx = self.status['uav']['uav_wait'][0].pop()
                elapsed_time = tranportation_t + self.properties['uav']['uav_handover_time'] # 환자 넘기는 시간
                # 상태 변경
                self.status['uav']['uav_states'][u_idx] = (destination, elapsed_time, p_class+1) # destination, time, severity
                self.status['patient']['p_states'][p_idx, 2] = 1  # move
                self.status['patient']['p_sent'][h_idx] += 1  # sent record
                self.add_event(elapsed_time, 'uav_arrival_hospital', (p_idx, u_idx, h_idx))
                self._record_trace("transport_start", patient_id=int(p_idx), vehicle="UAV",
                                   vehicle_id=int(u_idx), hospital_id=int(h_idx), severity=int(p_class))

            # 3. 현장에 R, Y 있고 mode 아직 더 있는 경우 추가 결정
            hasAvailableMode = bool(self.status['ambulance']['amb_wait'][0] or self.status['uav']['uav_wait'][0])
            hasRY = bool(self.status['patient']['p_wait'][0][0] or self.status['patient']['p_wait'][1][0])
            if hasAvailableMode and hasRY:
                return True, True # 한 번 더 결정
            else:
                return True, False

    def check_termination(self):
        # 환자 모두 처치 끝나면 terminated (불필요한 iteration 막을 수 있음.)
        terminated = np.all(self.status['patient']['p_states'][:,-1] == 1)
        # or event_queue length로 확인 가능
        return terminated

    def sample_transportation_time(self, mode, origination, destination):
        # Note: 데이터에서 병원 idx 0부터 시작, destination과 origination 병원 idx는 1부터 시작
        # origination=0: 현장, destination=0: 현장 (복귀)
        # 1. 이송 시간 샘플
        log_mean, log_std = None, None
        if origination == 0:  # Site → Hospital
            if mode == 0:  # ambulance
                log_mean = self.properties['ambulance']['amb_HtoS_t'][1][destination - 1]
                log_std = self.properties['ambulance']['amb_HtoS_t'][2][destination - 1]
            elif mode == 1:  # UAV
                log_mean = self.properties['uav']['uav_HtoS_t'][1][destination - 1]
                log_std = self.properties['uav']['uav_HtoS_t'][2][destination - 1]
        elif destination == 0:  # Hospital → Site (복귀)
            if mode == 0:  # ambulance
                log_mean = self.properties['ambulance']['amb_HtoS_t'][1][origination - 1]
                log_std = self.properties['ambulance']['amb_HtoS_t'][2][origination - 1]
            elif mode == 1:  # UAV
                log_mean = self.properties['uav']['uav_HtoS_t'][1][origination - 1]
                log_std = self.properties['uav']['uav_HtoS_t'][2][origination - 1]
        else:  # Hospital → Hospital
            if mode == 0:  # ambulance
                log_mean = self.properties['ambulance']['amb_HtoH_t'][1][origination-1, destination - 1]
                log_std = self.properties['ambulance']['amb_HtoH_t'][2][origination-1, destination - 1]
            elif mode == 1:  # UAV
                log_mean = self.properties['uav']['uav_HtoH_t'][1][origination-1, destination - 1]
                log_std = self.properties['uav']['uav_HtoH_t'][2][origination-1, destination - 1]
        tranportation_t = self.rng.lognormal(log_mean, log_std)
        return tranportation_t

    def start_GB_transport(self, log):
        # 현장에 있는 Green, Black을 현장에 있는 이송 수단 모두 사용해서 이송
        # 1. UAV 먼저 사용
        while self.status['uav']['uav_wait'][0]:
            u_idx = self.status['uav']['uav_wait'][0].pop()
            if self.status['patient']['p_wait'][2][0]:  # 현장 대기 중인 Green 이송
                p_class = 2
                p_idx = self.status['patient']['p_wait'][p_class][0].pop()
            elif self.status['patient']['p_wait'][3][0]:  # 현장 대기 중인 Black 이송
                p_class = 3
                p_idx = self.status['patient']['p_wait'][3][0].pop()
            else:
                break
            destination = self.default_transportation_GB(mode=1)
            tranportation_t = self.sample_transportation_time(mode=1, origination=0, destination=destination)
            elapsed_time = tranportation_t + self.properties['uav']['uav_handover_time']  # 환자 넘기는 시간
            # 상태 변경
            self.status['uav']['uav_states'][u_idx] = (destination, elapsed_time, p_class + 1)  # destination, time, severity
            self.status['patient']['p_states'][p_idx, 2] = 1  # move
            self.status['patient']['p_sent'][destination-1] += 1  # sent record
            self.add_event(elapsed_time, 'uav_arrival_hospital', (p_idx, u_idx, destination - 1))
        # 2. AMB 사용
        while self.status['ambulance']['amb_wait'][0]:
            a_idx = self.status['ambulance']['amb_wait'][0].pop()
            if self.status['patient']['p_wait'][2][0]:  # 현장 대기 중인 Green 이송
                p_class = 2
                p_idx = self.status['patient']['p_wait'][p_class][0].pop()
            elif self.status['patient']['p_wait'][3][0]:  # 현장 대기 중인 Black 이송
                p_class = 3
                p_idx = self.status['patient']['p_wait'][3][0].pop()
            else:
                break
            destination = self.default_transportation_GB(mode=0)
            tranportation_t = self.sample_transportation_time(mode=0, origination=0, destination=destination)
            elapsed_time = tranportation_t + self.properties['ambulance']['amb_handover_time']  # 환자 넘기는 시간
            # 상태 변경
            self.status['ambulance']['amb_states'][a_idx] = (destination, elapsed_time, p_class + 1)  # destination, time, severity
            self.status['patient']['p_states'][p_idx, 2] = 1  # move
            self.status['patient']['p_sent'][destination-1] += 1  # sent record
            self.add_event(elapsed_time, 'amb_arrival_hospital', (p_idx, a_idx, destination - 1))

        return log

    def default_transportation_GB(self, mode):
        # Rule1: Ver250724
        # 1. Tier3(상급종합) 병원은 제외
        # 2. 가까운 순서대로 이송 (현장에서부터 거리순으로 hospital index 지정됨을 가정)
        # 3. max_send - p_sent > 0 인 경우에만 이송 (최대 보내겠다고 생각했던 환자 수 - 실제 보낸 환자 수)
        # 4. 만족하는 병원 없으면 등급 상관 없이 가장 가까운 병원으로 이송
        # 5. UAV(mode=1)이면 헬기장 있는 병원에만 이송

        destination = None
        idle_capa = self.properties['hospital']['hos_max_send'] - self.status['patient']['p_sent']
        helipad_idx = self.properties['hospital'].get('hos_helipad_idx', np.array([]))
        for h_idx in self.properties['hospital']['hos_tier2_idx']:
            if mode == 1 and h_idx not in helipad_idx:
                continue
            if idle_capa[h_idx] > 0:
                destination = h_idx + 1
                break
        if destination is None:
            for h_idx in self.properties['hospital']['hos_tier3_idx']:
                if mode == 1 and h_idx not in helipad_idx:
                    continue
                if idle_capa[h_idx] > 0:
                    destination = h_idx + 1
                    break
        if destination is None:
            raise RuntimeError("No hospital with remaining send capacity for Green/Black transport.")
        return destination

    def diversion_rule(self, c_hos, pass_to_tier3, pass_to_tier2, mode):
        # Rule1: Ver250724
        # 1. 보낼 수 있는 병원 등급 중 가까운 순서대로 이송
        # 2. max_send - p_sent > 0 인 경우에만 이송 (최대 보내겠다고 생각했던 환자 수 - 실제 보낸 환자 수)
        # 3. 만족하는 병원 없으면 에러 메시지 발생
        # 4. UAV(mode=1)이면 헬기장 있는 병원에만 이송

        d_to_H = self.properties['hospital']['d_HtoH_road'][c_hos] if mode==0 else self.properties['hospital']['d_HtoH_euc'][c_hos]
        destination = None
        idle_capa = self.properties['hospital']['hos_max_send'] - self.status['patient']['p_sent']
        helipad_idx = self.properties['hospital'].get('hos_helipad_idx', np.array([]))
        max_capa_arr = self.properties['hospital']['hos_max_capa'] + self.properties['hospital']['hos_max_queue']
        n_occupied_arr = self.status['hospital']['h_states'][:, -1]

        sorted_h = np.argsort(d_to_H)
        for h_idx in sorted_h:
            if mode == 1 and h_idx not in helipad_idx:
                continue
            h_tier = self.properties['hospital']['hos_tier'][h_idx]
            can_admit = (pass_to_tier3 and h_tier==3) or (pass_to_tier2 and h_tier==2)
            if can_admit and idle_capa[h_idx] > 0 and n_occupied_arr[h_idx] < max_capa_arr[h_idx]:
                destination = h_idx + 1
                break
        if destination is None: # 전원 가능 병원 없음
            raise Exception("Impossible to divert")

        return destination

    def sample_service_time(self, h_tier, p_class):
        #   if service time 9999이면 n_idle -= 1, definite cared로 변경, 추가 event 생성 없음
        if h_tier == 3:
            service_mean = self.properties['patient']['patient_info']['treat_tier3_mean'][p_class]
        elif h_tier == 2:
            service_mean = self.properties['patient']['patient_info']['treat_tier2_mean'][p_class]
        if isinstance(service_mean, str):
            service_time = np.inf
        else:
            service_time = self.rng.exponential(service_mean)
        return service_time

    def ev_onset(self, log, entity_idx):
        """
        사고 발생 이벤트
        :return:
        """
        rescue_times = []
        # 1. 환자 구조 이벤트 생성
        p_param = self.properties['patient']
        p_num = self.rng.multinomial(p_param['incident_size'],
                                     pvals=p_param['patient_info']['ratio'])
        self.status['patient']['p_states'][:,0] = np.repeat([0,1,2,3], p_num)

        rescue_max_time = 60

        for p_class in range(4):
            alpha, beta = p_param['patient_info']['rescue_param_alpha'][p_class], p_param['patient_info']['rescue_param_beta'][p_class]
            if alpha != 0 and beta != 0:
                sampled = self.rng.beta(alpha, beta, size = p_num[p_class]) * rescue_max_time
            else:
                sampled = np.zeros(p_num[p_class])
            rescue_times.append(sampled)
        p_idx = 0
        for p_class, event_times in enumerate(rescue_times):
            for t in event_times:
                self.add_event(elapsed_time=t, ev_name='p_rescue', entity_idx=(p_idx,))
                p_idx += 1
        # 2. amb, uav 현장 도착 (출동) 이벤트 생성
        amb_response_param = self.properties['ambulance']['amb_response_t']
        time_amb = self.rng.lognormal(amb_response_param[1], amb_response_param[2])
        for a_idx, t in enumerate(time_amb):
            self.add_event(elapsed_time=t, ev_name='amb_arrival_site', entity_idx=(a_idx,))
        self.status['ambulance']['amb_states'][:,1] = time_amb
        # 3. uav 현장 도착 (출동) 이벤트 생성
        uav_response_param = self.properties['uav']['uav_response_t']
        time_uav = self.rng.lognormal(uav_response_param[1], uav_response_param[2])
        for u_idx, t in enumerate(time_uav):
            self.add_event(elapsed_time=t, ev_name='uav_arrival_site', entity_idx=(u_idx,))
        self.status['uav']['uav_states'][:,1] = time_uav

        # Trace: record patient generation
        self._record_trace("onset", n_patients=int(sum(p_num)),
                           severity_dist={i: int(p_num[i]) for i in range(4)})

        log = {'rescue_times': rescue_times}
        return log, False

    def ev_p_rescue(self, log, entity_idx):
        """
        환자 구조 이벤트
        :param log: 이벤트 내용 기록할 dictionary
        :param entity_idx: tuple(환자 idx, )
        :return:
        log
        stop_condition
        """
        p_idx = entity_idx[0]
        p_class = self.status['patient']['p_states'][p_idx, 0]
        self.status['patient']['p_states'][p_idx, 1] = 1 # rescued
        self.status['patient']['p_wait'][p_class][0].append(p_idx)
        self._record_trace("rescue", patient_id=int(p_idx), severity=int(p_class))
        self.rescue_finish = np.all(self.status['patient']['p_states'][:, 1] == 1) # 최소값이 0이면 아직 다 구조 안 된 경우

        hasAvailableMode = bool(self.status['ambulance']['amb_wait'][0] or self.status['uav']['uav_wait'][0])
        if not hasAvailableMode: # 이송 수단 없으면 next event 수행
            return log, False
        else:
            hasRY = bool(self.status['patient']['p_wait'][0][0] or self.status['patient']['p_wait'][1][0])
            if hasRY: # 이송 수단 있고, 현장에 R, Y 있을 때 decision 호출
                return log, True
            else:
                if self.rescue_finish: # 이송 수단 있고, 현장에 R, Y 없고 모든 환자 구조했을 때, GB 이송 시작
                    log = self.start_GB_transport(log)
                return log, False

    def ev_amb_arrival_site(self, log, entity_idx):
        """
        Ambulance 현장 도착 이벤트
        :param log: 이벤트 내용 기록할 dictionary
        :param entity_idx: tuple(ambulance idx, )
        :return:
        """
        a_idx = entity_idx[0]
        self.status['ambulance']['amb_wait'][0].append(a_idx)

        hasRY = bool(self.status['patient']['p_wait'][0][0] or self.status['patient']['p_wait'][1][0])
        if hasRY: # 1. Red나 Yellow 환자가 현장에 있으면 decision
            return log, True
        else: # 2. Red나 Yellow 환자가 현장에 없는 경우
            if self.rescue_finish: # 구조 끝났으면, Green -> Black 환자 이송 시작
                log = self.start_GB_transport(log)
                return log, False
            else: # 미구조 환자 존재 --> 현장 대기
                return log, False

    def ev_uav_arrival_site(self, log, entity_idx):
        """
        UAV 현장 도착 이벤트
        :param log: 이벤트 내용 기록할 dictionary
        :param entity_idx: tuple(uav idx, )
        :return:
        """
        u_idx = entity_idx[0]
        self.status['uav']['uav_wait'][0].append(u_idx)

        hasRY = bool(self.status['patient']['p_wait'][0][0] or self.status['patient']['p_wait'][1][0])
        if hasRY: # 1. Red나 Yellow 환자가 현장에 있으면 decision
            return log, True
        else: # 2. Red나 Yellow 환자가 현장에 없는 경우
            if self.rescue_finish: # 구조 끝났으면, Green -> Black 환자 이송 시작
                log = self.start_GB_transport(log)
                return log, False
            else: # 미구조 환자 존재 --> 현장 대기
                return log, False

    def ev_p_care_ready(self, log, entity_idx):
        """
        병원 도착 후 handover / triage 후 초기처치 완료 이벤트
        :param log:
        :param entity_idx:
        :return:
        """
        p_idx, h_idx = entity_idx
        p_class = self.status['patient']['p_states'][p_idx, 0]
        n_idle, n_queue = self.status['hospital']['h_states'][h_idx][0:2]
        if n_idle > 0: # 서비스 시작
            h_tier = self.properties['hospital']['hos_tier'][h_idx]
            service_time = self.sample_service_time(h_tier=h_tier, p_class=p_class)
            log['p_admit'].append((self.time, p_class))
            self._record_trace("care_start", patient_id=int(p_idx), hospital_id=int(h_idx), severity=int(p_class))
            # 병원, 환자 상태 업데이트
            self.status['hospital']['h_states'][h_idx, 0] -= 1  # n_idle -= 1
            # 이벤트 추가
            if service_time == np.inf: # capa 끝까지 점유
                self.status['patient']['p_states'][p_idx, -1] = 1
            else: # 서비스 종료 이벤트 존재
                self.add_event(service_time, 'p_def_care', (p_idx, h_idx))
        else: # 대기 시작 (선행 조건에서 max_queue 넘지 않도록 이벤트 추가했었음)
            # 병원, 환자 상태 업데이트
            self.status['hospital']['h_states'][h_idx, 1] += 1  # n_queue += 1
            self.status['patient']['p_wait'][p_class][h_idx+1].append(p_idx) # 환자 대기 시작
        return log, False

    def _can_treat_patient(self, h_idx, p_class):
        h_tier = self.properties['hospital']['hos_tier'][h_idx]
        p_info = self.properties['patient']['patient_info']
        if h_tier == 3:
            return bool(p_info['treat_tier3'][p_class])
        if h_tier == 2:
            return bool(p_info['treat_tier2'][p_class])
        return False

    def ev_amb_arrival_hospital(self, log, entity_idx):
        """
        Ambulance hospital-arrival event.
        :param log: event log dictionary
        :param entity_idx: tuple(patient idx, ambulance idx, hospital idx)
        :return:
        """
        p_idx, a_idx, h_idx = entity_idx
        p_class = self.status['patient']['p_states'][p_idx, 0]
        p_info = self.properties['patient']['patient_info']
        destination = 0
        handover_time = 0

        if not self._can_treat_patient(h_idx, p_class):
            destination = self.diversion_rule(h_idx,
                                              pass_to_tier3=p_info['treat_tier3'][p_class],
                                              pass_to_tier2=p_info['treat_tier2'][p_class],
                                              mode=0)
            self.status['patient']['p_sent'][h_idx] -= 1
            self.status['patient']['p_sent'][destination-1] += 1
            transportation_t = self.sample_transportation_time(mode=0, origination=h_idx + 1, destination=destination)
            self.status['ambulance']['amb_states'][a_idx] = (destination, transportation_t, p_class + 1)
            self.add_event(transportation_t, 'amb_arrival_hospital', (p_idx, a_idx, destination - 1))
            return log, False

        max_capa = self.properties['hospital']['hos_max_capa'][h_idx] + self.properties['hospital']['hos_max_queue'][h_idx]
        n_occupied = self.status['hospital']['h_states'][h_idx, -1]
        if n_occupied < max_capa:
            self.status['patient']['p_states'][p_idx, 3] = 1
            self.status['hospital']['h_states'][h_idx, -1] += 1
            handover_time = self.properties['ambulance']['amb_handover_time']
            self.add_event(handover_time, 'p_care_ready', (p_idx, h_idx))
            self._record_trace("hospital_arrival", patient_id=int(p_idx), vehicle="AMB",
                               hospital_id=int(h_idx), severity=int(p_class), admitted=True)
        else:
            destination = self.diversion_rule(h_idx,
                                              pass_to_tier3=p_info['treat_tier3'][p_class],
                                              pass_to_tier2=p_info['treat_tier2'][p_class],
                                              mode=0)
            self.status['patient']['p_sent'][h_idx] -= 1
            self.status['patient']['p_sent'][destination-1] += 1
            self._record_trace("diversion", patient_id=int(p_idx), vehicle="AMB",
                               from_hospital=int(h_idx), to_hospital=int(destination-1), severity=int(p_class))

        transportation_t = self.sample_transportation_time(mode=0, origination=h_idx + 1, destination=destination)
        if destination == 0:
            self.status['ambulance']['amb_states'][a_idx] = (destination, transportation_t, 0)
            self.add_event(transportation_t + handover_time, 'amb_arrival_site', (a_idx,))
        else:
            self.status['ambulance']['amb_states'][a_idx] = (destination, transportation_t, p_class + 1)
            self.add_event(transportation_t + handover_time, 'amb_arrival_hospital', (p_idx, a_idx, destination - 1))
        return log, False

    def ev_uav_arrival_hospital(self, log, entity_idx):
        """
        UAV hospital-arrival event.
        :param log: event log dictionary
        :param entity_idx: tuple(patient idx, uav idx, hospital idx)
        :return:
        """
        p_idx, u_idx, h_idx = entity_idx
        p_class = self.status['patient']['p_states'][p_idx, 0]
        p_info = self.properties['patient']['patient_info']
        destination = 0
        handover_time = 0

        if not self._can_treat_patient(h_idx, p_class):
            destination = self.diversion_rule(h_idx,
                                              pass_to_tier3=p_info['treat_tier3'][p_class],
                                              pass_to_tier2=p_info['treat_tier2'][p_class],
                                              mode=1)
            self.status['patient']['p_sent'][h_idx] -= 1
            self.status['patient']['p_sent'][destination-1] += 1
            transportation_t = self.sample_transportation_time(mode=1, origination=h_idx + 1, destination=destination)
            self.status['uav']['uav_states'][u_idx] = (destination, transportation_t, p_class + 1)
            self.add_event(transportation_t, 'uav_arrival_hospital', (p_idx, u_idx, destination - 1))
            return log, False

        max_capa = self.properties['hospital']['hos_max_capa'][h_idx] + self.properties['hospital']['hos_max_queue'][h_idx]
        n_occupied = self.status['hospital']['h_states'][h_idx, -1]
        if n_occupied < max_capa:
            self.status['patient']['p_states'][p_idx, 3] = 1
            self.status['hospital']['h_states'][h_idx, -1] += 1
            handover_time = self.properties['uav']['uav_handover_time']
            self.add_event(handover_time, 'p_care_ready', (p_idx, h_idx))
            self._record_trace("hospital_arrival", patient_id=int(p_idx), vehicle="UAV",
                               hospital_id=int(h_idx), severity=int(p_class), admitted=True)
        else:
            destination = self.diversion_rule(h_idx,
                                              pass_to_tier3=p_info['treat_tier3'][p_class],
                                              pass_to_tier2=p_info['treat_tier2'][p_class],
                                              mode=1)
            self.status['patient']['p_sent'][h_idx] -= 1
            self.status['patient']['p_sent'][destination-1] += 1
            self._record_trace("diversion", patient_id=int(p_idx), vehicle="UAV",
                               from_hospital=int(h_idx), to_hospital=int(destination-1), severity=int(p_class))

        transportation_t = self.sample_transportation_time(mode=1, origination=h_idx + 1, destination=destination)
        if destination == 0:
            self.status['uav']['uav_states'][u_idx] = (destination, transportation_t, 0)
            self.add_event(transportation_t + handover_time, 'uav_arrival_site', (u_idx,))
        else:
            self.status['uav']['uav_states'][u_idx] = (destination, transportation_t, p_class + 1)
            self.add_event(transportation_t + handover_time, 'uav_arrival_hospital', (p_idx, u_idx, destination - 1))
        return log, False

    def ev_p_def_care(self, log, entity_idx):
        """
        병원에서 환자처치 종료 이벤트
        :param log: 이벤트 내용 기록할 dictionary
        :param entity_idx: tuple(patient idx, hospital idx)
        :return:
        """
        p_idx, h_idx = entity_idx
        # 처치 완료 환자 상태 변경
        self.status['patient']['p_states'][p_idx, -1] = 1
        self._record_trace("care_complete", patient_id=int(p_idx), hospital_id=int(h_idx))

        n_idle, n_queue = self.status['hospital']['h_states'][h_idx][0:2]
        # 새로운 처치 시작
        if n_queue > 0:
            h_tier = self.properties['hospital']['hos_tier'][h_idx]
            # Red, Yellow, Green, Black 순으로 처치
            for p_class in range(4):
                if self.status['patient']['p_wait'][p_class][h_idx+1]:
                    new_p_idx = self.status['patient']['p_wait'][p_class][h_idx+1].pop()
                    break
            service_time = self.sample_service_time(h_tier=h_tier, p_class=p_class)
            log['p_admit'].append((self.time, p_class))
            # 병원, 환자 상태 업데이트
            self.status['hospital']['h_states'][h_idx, 1] -= 1  # n_queue -= 1
            # 이벤트 추가
            if service_time == np.inf: # capa 끝까지 점유
                self.status['patient']['p_states'][new_p_idx, -1] = 1
            else: # 서비스 종료 이벤트 존재
                self.add_event(service_time, 'p_def_care', (new_p_idx, h_idx))
        else:
            self.status['hospital']['h_states'][h_idx, 0] += 1  # n_idle += 1
        return log, False

    def _record_trace(self, event_type, **kwargs):
        """Record a trace event if tracing is enabled."""
        if self.enable_trace:
            record = {"time": self.time, "event": event_type}
            record.update(kwargs)
            self.trace_log.append(record)

    def get_trace(self):
        """Return the accumulated trace log."""
        return list(self.trace_log)

    def add_event(self, elapsed_time, ev_name, entity_idx):
        self.e_ID += 1
        heapq.heappush(self.event_queue, (elapsed_time + self.time, self.e_ID, ev_name, entity_idx)) # event = (event_time, e_ID, ev_name, entity_idx)

    def ev_template(self, log, entity_idx):
        """
        :param log: 이벤트 내용 기록할 dictionary
        :param entity_idx: 이벤트 참여 entity마다의 index
        :return:
        stop_condition: decision 받기 위해 멈출 여부
        """
        stop_condition = False
        return log, stop_condition

