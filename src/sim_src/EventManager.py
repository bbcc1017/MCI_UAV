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
        self.e_ID = 0 # event ?앹꽦 踰덊샇 珥덇린??
        self.time = 0 # event clock 珥덇린??
        self.status = self.en_manager.en_status # entity ?곹깭 遺덈윭?ㅺ린
        self.rescue_finish = False # 援ъ” ?꾨즺 ?щ? ?뺤씤
        self.event_queue = [] # event queue 珥덇린??
        self.trace_log = []  # reset trace
        # ?ш퀬 諛쒖깮
        init_log = {}
        init_log, _ = self.ev_onset(init_log, None)
        # 泥?decision epoch源뚯? ?ㅽ뻾
        auto_log, _ = self.run_next(action=None)
        init_log['p_admit'] = auto_log.get('p_admit', [])
        return init_log

    def run_next(self, action=None):
        log = {'p_admit':[]} # ?쒕??덉씠???섑뻾?섎㈃??湲곕줉??吏??
        if action is not None:
            normal_action, repeat = self.proceed_action(action)
            log['normal_action'] = normal_action
            if repeat: # 異붽? ?≪뀡 ?꾩슂
                return log, False

        terminated = False
        while True:
            # 1. 媛??鍮좊Ⅸ event ?뚯븙
            if not self.event_queue:

                # ?대깽?멸? ???댁긽 ?놁쑝硫?醫낅즺(?덉쟾 媛??

                h_idle_que_occ = self.status['hospital']['h_states'].copy() if 'hospital' in self.status else None

                if h_idle_que_occ is None:
                    print(f'[EventQueueEmpty] t={self.time} h_states=None')
                else:
                    # numpy媛 ... 濡?以꾩씠吏 ?딅룄濡??듭뀡
                    with np.printoptions(threshold=np.inf, linewidth=200, suppress=True):
                        print(f'[EventQueueEmpty] t={self.time}  h_states.shape={h_idle_que_occ.shape}\n{h_idle_que_occ}')

                return log, True

            c_event = heapq.heappop(self.event_queue)  # event = (event_time, e_ID, ev_name, entity_idx)
            print(c_event)

            time_interval = c_event[0] - self.time
            self.time = c_event[0]
            # ?쒓컙 寃쎄낵???곕Ⅸ ?곹깭 ?낅뜲?댄듃
            self.status['ambulance']['amb_states'][:,1] -= time_interval
            np.maximum(self.status['ambulance']['amb_states'][:,1], 0, out=self.status['ambulance']['amb_states'][:,1])
            self.status['uav']['uav_states'][:,1] -= time_interval
            np.maximum(self.status['uav']['uav_states'][:,1], 0, out=self.status['uav']['uav_states'][:,1])
            log, stop_condition = getattr(self, "ev_" + c_event[2])(log, c_event[3])
            if stop_condition: # 2. decision ?대젮?쇳븯???쒖젏源뚯? 吏꾪뻾
                break
            terminated = self.check_termination() # 3. ???댁긽 ?섏궗寃곗젙 ?꾩슂 ?놁쑝硫??⑥? ?쒕??덉씠??吏꾪뻾 ??醫낅즺
            if terminated:
                break
        return log, terminated

    def proceed_action(self, action):
        print("Action:", action)
        # action[0]: Red = 0, Yellow = 1, Green = 2
        # action[1]: 0: ?꾩옣, 1踰?蹂묒썝 ~ N踰?蹂묒썝; 蹂묒썝 10媛??? 0? ?꾩옣, 1踰?蹂묒썝 ~ 9踰?蹂묒썝
        # action[2]: 0: Amb, 1: UAV
        # return: normal_action, repeat
        p_class, destination, mode = action # destination = 蹂묒썝 index + 1
        if destination == 0:  # ?꾩옣??癒몃Ъ湲?
            return True, False
        else:  # ?대뵖媛濡??댁넚
            # 0. Penalize wrong mode selection & terminate
            if mode == 1 and not self.status['uav']['uav_wait'][0]:  # UAV ?녿뒗???ㅻⅨ 怨??댁넚 紐낅졊
                print("NO UAV")
                return False, False
            elif mode == 0 and not self.status['ambulance']['amb_wait'][0]:  # amb ?녿뒗???ㅻⅨ 怨??댁넚 紐낅졊
                print("NO AMB")
                return False, False

            # 1. ?꾩옣 ?섏옄 ??蹂寃?
            try:
                p_idx = self.status['patient']['p_wait'][p_class][0].pop()
                # # TO-DO: Rule???ъ슜 ??UPDATE
                # if p_class == 0 or p_class == 1:
                #     for idx, record in enumerate(self.slack_times):
                #         if record[1] == action[0]:
                #             del self.slack_times[idx]
                #             break
            except IndexError:
                print("NO PATIENT")
                return False, False

            # 2. ?댁넚 ?섎떒???곕Ⅸ 紐⑹쟻吏 諛??꾨떖 ?쒓컙 蹂寃?
            h_idx = destination - 1
            tranportation_t = self.sample_transportation_time(mode=mode, origination=0, destination=destination)
            if mode == 0: # amb
                a_idx = self.status['ambulance']['amb_wait'][0].pop()
                elapsed_time = tranportation_t + self.properties['ambulance']['amb_handover_time'] # ?섏옄 ?ｋ뒗 ?쒓컙
                # ?곹깭 蹂寃?
                self.status['ambulance']['amb_states'][a_idx] = (destination, elapsed_time, p_class+1) # destination, time, severity
                self.status['patient']['p_states'][p_idx, 2] = 1  # move
                self.status['patient']['p_sent'][h_idx] += 1 # sent record
                self.add_event(elapsed_time, 'amb_arrival_hospital', (p_idx, a_idx, h_idx))
                self._record_trace("transport_start", patient_id=int(p_idx), vehicle="AMB",
                                   vehicle_id=int(a_idx), hospital_id=int(h_idx), severity=int(p_class))
            elif mode == 1: # uav
                u_idx = self.status['uav']['uav_wait'][0].pop()
                elapsed_time = tranportation_t + self.properties['uav']['uav_handover_time'] # ?섏옄 ?ｋ뒗 ?쒓컙
                # ?곹깭 蹂寃?
                self.status['uav']['uav_states'][u_idx] = (destination, elapsed_time, p_class+1) # destination, time, severity
                self.status['patient']['p_states'][p_idx, 2] = 1  # move
                self.status['patient']['p_sent'][h_idx] += 1  # sent record
                self.add_event(elapsed_time, 'uav_arrival_hospital', (p_idx, u_idx, h_idx))
                self._record_trace("transport_start", patient_id=int(p_idx), vehicle="UAV",
                                   vehicle_id=int(u_idx), hospital_id=int(h_idx), severity=int(p_class))

            # 3. ?꾩옣??R, Y ?덇퀬 mode ?꾩쭅 ???덈뒗 寃쎌슦 異붽? 寃곗젙
            hasAvailableMode = bool(self.status['ambulance']['amb_wait'][0] or self.status['uav']['uav_wait'][0])
            hasRY = bool(self.status['patient']['p_wait'][0][0] or self.status['patient']['p_wait'][1][0])
            if hasAvailableMode and hasRY:
                return True, True # ??踰???寃곗젙
            else:
                return True, False

    def check_termination(self):
        # ?섏옄 紐⑤몢 泥섏튂 ?앸굹硫?terminated (遺덊븘?뷀븳 iteration ?⑥쓣 ???덉쓬.)
        terminated = np.all(self.status['patient']['p_states'][:,-1] == 1)
        # or event_queue length濡??뺤씤 媛??
        return terminated

    def sample_transportation_time(self, mode, origination, destination):
        # Note: ?곗씠?곗뿉??蹂묒썝 idx 0遺???쒖옉, destination怨?origination 蹂묒썝 idx??1遺???쒖옉
        # origination=0: ?꾩옣, destination=0: ?꾩옣 (蹂듦?)
        # 1. ?댁넚 ?쒓컙 ?섑뵆
        log_mean, log_std = None, None
        if origination == 0:  # Site ??Hospital
            if mode == 0:  # ambulance
                log_mean = self.properties['ambulance']['amb_HtoS_t'][1][destination - 1]
                log_std = self.properties['ambulance']['amb_HtoS_t'][2][destination - 1]
            elif mode == 1:  # UAV
                log_mean = self.properties['uav']['uav_HtoS_t'][1][destination - 1]
                log_std = self.properties['uav']['uav_HtoS_t'][2][destination - 1]
        elif destination == 0:  # Hospital ??Site (蹂듦?)
            if mode == 0:  # ambulance
                log_mean = self.properties['ambulance']['amb_HtoS_t'][1][origination - 1]
                log_std = self.properties['ambulance']['amb_HtoS_t'][2][origination - 1]
            elif mode == 1:  # UAV
                log_mean = self.properties['uav']['uav_HtoS_t'][1][origination - 1]
                log_std = self.properties['uav']['uav_HtoS_t'][2][origination - 1]
        else:  # Hospital ??Hospital
            if mode == 0:  # ambulance
                log_mean = self.properties['ambulance']['amb_HtoH_t'][1][origination-1, destination - 1]
                log_std = self.properties['ambulance']['amb_HtoH_t'][2][origination-1, destination - 1]
            elif mode == 1:  # UAV
                log_mean = self.properties['uav']['uav_HtoH_t'][1][origination-1, destination - 1]
                log_std = self.properties['uav']['uav_HtoH_t'][2][origination-1, destination - 1]
        tranportation_t = self.rng.lognormal(log_mean, log_std)
        return tranportation_t

    def start_GB_transport(self, log):
        # ?꾩옣???덈뒗 Green, Black???꾩옣???덈뒗 ?댁넚 ?섎떒 紐⑤몢 ?ъ슜?댁꽌 ?댁넚
        # 1. UAV 癒쇱? ?댁슜
        while self.status['uav']['uav_wait'][0]:
            u_idx = self.status['uav']['uav_wait'][0].pop()
            if self.status['patient']['p_wait'][2][0]:  # ?꾩옣 ?湲?以묒씤 Green ?댁넚
                p_class = 2
                p_idx = self.status['patient']['p_wait'][p_class][0].pop()
            elif self.status['patient']['p_wait'][3][0]:  # ?꾩옣 ?湲?以묒씤 Black ?댁넚
                p_class = 3
                p_idx = self.status['patient']['p_wait'][3][0].pop()
            else:
                break
            destination = self.default_transportation_GB(mode=1)
            tranportation_t = self.sample_transportation_time(mode=1, origination=0, destination=destination)
            elapsed_time = tranportation_t + self.properties['uav']['uav_handover_time']  # ?섏옄 ?ｋ뒗 ?쒓컙
            # ?곹깭 蹂寃?
            self.status['uav']['uav_states'][u_idx] = (destination, elapsed_time, p_class + 1)  # destination, time, severity
            self.status['patient']['p_states'][p_idx, 2] = 1  # move
            self.status['patient']['p_sent'][destination-1] += 1  # sent record
            self.add_event(elapsed_time, 'uav_arrival_hospital', (p_idx, u_idx, destination - 1))
        # 2. AMB ?댁슜
        while self.status['ambulance']['amb_wait'][0]:
            a_idx = self.status['ambulance']['amb_wait'][0].pop()
            if self.status['patient']['p_wait'][2][0]:  # ?꾩옣 ?湲?以묒씤 Green ?댁넚
                p_class = 2
                p_idx = self.status['patient']['p_wait'][p_class][0].pop()
            elif self.status['patient']['p_wait'][3][0]:  # ?꾩옣 ?湲?以묒씤 Black ?댁넚
                p_class = 3
                p_idx = self.status['patient']['p_wait'][3][0].pop()
            else:
                break
            destination = self.default_transportation_GB(mode=0)
            tranportation_t = self.sample_transportation_time(mode=0, origination=0, destination=destination)
            elapsed_time = tranportation_t + self.properties['ambulance']['amb_handover_time']  # ?섏옄 ?ｋ뒗 ?쒓컙
            # ?곹깭 蹂寃?
            self.status['ambulance']['amb_states'][a_idx] = (destination, elapsed_time, p_class + 1)  # destination, time, severity
            self.status['patient']['p_states'][p_idx, 2] = 1  # move
            self.status['patient']['p_sent'][destination-1] += 1  # sent record
            self.add_event(elapsed_time, 'amb_arrival_hospital', (p_idx, a_idx, destination - 1))

        return log

    def default_transportation_GB(self, mode):
        # Rule1: Ver250724
        # 1. Tier3(?곴툒醫낇빀) 蹂묒썝? ?쒖쇅
        # 2. 媛源뚯슫 ?쒖꽌?濡??댁넚 (?꾩옣?먯꽌遺??嫄곕━?쒖쑝濡?hospital index 吏?뺣맖??媛??
        # 3. max_send - p_sent > 0??寃쎌슦?먮쭔 ?댁넚 (理쒕? 蹂대궡?ㅺ퀬 ?앷컖?덈뜕 ?섏옄 ??- ?ㅼ젣 蹂대궦 ?섏옄 ??
        # 4. 留뚯”?섎뒗 蹂묒썝 ?놁쑝硫??깃툒 ?곴? ?놁씠 媛??媛源뚯슫 蹂묒썝?쇰줈 ?댁넚
        # 5. UAV(mode=1)?대㈃ ?ш린???덈뒗 蹂묒썝?먮쭔 ?댁넚

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
        # 1. 蹂대궪 ???덈뒗 蹂묒썝 ?깃툒 以?媛源뚯슫 ?쒖꽌?濡??댁넚
        # 2. max_send - p_sent > 0??寃쎌슦?먮쭔 ?댁넚 (理쒕? 蹂대궡?ㅺ퀬 ?앷컖?덈뜕 ?섏옄 ??- ?ㅼ젣 蹂대궦 ?섏옄 ??
        # 3. 留뚯”?섎뒗 蹂묒썝 ?놁쑝硫??먮윭 硫붿꽭吏 諛쒖깮
        # 4. UAV(mode=1)?대㈃ ?ш린???덈뒗 蹂묒썝?먮쭔 ?댁넚

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
        if destination is None: # ?꾩썝 媛??蹂묒썝 ?놁쓬
            raise Exception("Impossible to divert")

        return destination

    def sample_service_time(self, h_tier, p_class):
        #   if service time 9999?대㈃ n_idle -= 1, definite cared濡?蹂寃? 異붽? event ?앹꽦 ?놁쓬
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
        ?ш퀬 諛쒖깮 ?대깽??
        :return:
        """
        rescue_times = []
        # 1. ?섏옄 援ъ“ ?대깽???앹꽦
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
        # 2. amb, uav ?꾩옣 ?꾩갑 (異쒕룞) ?대깽???앹꽦
        amb_response_param = self.properties['ambulance']['amb_response_t']
        time_amb = self.rng.lognormal(amb_response_param[1], amb_response_param[2])
        for a_idx, t in enumerate(time_amb):
            self.add_event(elapsed_time=t, ev_name='amb_arrival_site', entity_idx=(a_idx,))
        self.status['ambulance']['amb_states'][:,1] = time_amb
        # 3. uav ?꾩옣 ?꾩갑 (異쒕룞) ?대깽???앹꽦
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
        ?섏옄 援ъ“ ?대깽??
        :param log: ?대깽???댁슜 湲곕줉??dictionary
        :param entity_idx: tuple(?섏옄 idx, )
        :return:
        log
        stop_condition
        """
        p_idx = entity_idx[0]
        p_class = self.status['patient']['p_states'][p_idx, 0]
        self.status['patient']['p_states'][p_idx, 1] = 1 # rescued
        self.status['patient']['p_wait'][p_class][0].append(p_idx)
        self._record_trace("rescue", patient_id=int(p_idx), severity=int(p_class))
        self.rescue_finish = np.all(self.status['patient']['p_states'][:, 1] == 1) # 理쒖냼媛믪씠 0?대㈃ ?꾩쭅 ??援ъ“??寃쎌슦

        hasAvailableMode = bool(self.status['ambulance']['amb_wait'][0] or self.status['uav']['uav_wait'][0])
        if not hasAvailableMode: # ?댁넚 ?섎떒 ?놁쑝硫?next event ?섑뻾
            return log, False
        else:
            hasRY = bool(self.status['patient']['p_wait'][0][0] or self.status['patient']['p_wait'][1][0])
            if hasRY: # ?댁넚 ?섎떒 ?덇퀬, ?꾩옣??R, Y ?덉쓣 ?? decision ?몄텧
                return log, True
            else:
                if self.rescue_finish: # ?댁넚 ?섎떒 ?덇퀬, ?꾩옣??R, Y ?녾퀬 紐⑤뱺 ?섏옄 援ъ“?섏뿀???? GB ?댁넚 ?쒖옉
                    log = self.start_GB_transport(log)
                return log, False

    def ev_amb_arrival_site(self, log, entity_idx):
        """
        Ambulance ?꾩옣 ?꾩갑 ?대깽??
        :param log: ?대깽???댁슜 湲곕줉??dictionary
        :param entity_idx: tuple(ambulance idx, )
        :return:
        """
        a_idx = entity_idx[0]
        self.status['ambulance']['amb_wait'][0].append(a_idx)

        hasRY = bool(self.status['patient']['p_wait'][0][0] or self.status['patient']['p_wait'][1][0])
        if hasRY: # 1. Red??Yellow ?섏옄媛 ?꾩옣???덉쑝硫?decision
            return log, True
        else: # 2. Red??Yellow ?섏옄媛 ?꾩옣???녿뒗 寃쎌슦
            if self.rescue_finish: # 援ъ“ ?앸궗?쇰㈃, Green -> Black ?섏옄 ?댁넚 ?쒖옉
                log = self.start_GB_transport(log)
                return log, False
            else: # 誘멸뎄議??섏옄 議댁옱 --> ?꾩옣 ?湲?
                return log, False

    def ev_uav_arrival_site(self, log, entity_idx):
        """
        UAV ?꾩옣 ?꾩갑 ?대깽??
        :param log: ?대깽???댁슜 湲곕줉??dictionary
        :param entity_idx: tuple(uav idx, )
        :return:
        """
        u_idx = entity_idx[0]
        self.status['uav']['uav_wait'][0].append(u_idx)

        hasRY = bool(self.status['patient']['p_wait'][0][0] or self.status['patient']['p_wait'][1][0])
        if hasRY: # 1. Red??Yellow ?섏옄媛 ?꾩옣???덉쑝硫?decision
            return log, True
        else: # 2. Red??Yellow ?섏옄媛 ?꾩옣???녿뒗 寃쎌슦
            if self.rescue_finish: # 援ъ“ ?앸궗?쇰㈃, Green -> Black ?섏옄 ?댁넚 ?쒖옉
                log = self.start_GB_transport(log)
                return log, False
            else: # 誘멸뎄議??섏옄 議댁옱 --> ?꾩옣 ?湲?
                return log, False

    def ev_p_care_ready(self, log, entity_idx):
        """
        蹂묒썝 ?꾩갑 ??handover / triage ??珥덇린泥섏튂 ?꾨즺 ?대깽??
        :param log:
        :param entity_idx:
        :return:
        """
        p_idx, h_idx = entity_idx
        p_class = self.status['patient']['p_states'][p_idx, 0]
        n_idle, n_queue = self.status['hospital']['h_states'][h_idx][0:2]
        if n_idle > 0: # ?쒕퉬???쒖옉
            h_tier = self.properties['hospital']['hos_tier'][h_idx]
            service_time = self.sample_service_time(h_tier=h_tier, p_class=p_class)
            log['p_admit'].append((self.time, p_class))
            self._record_trace("care_start", patient_id=int(p_idx), hospital_id=int(h_idx), severity=int(p_class))
            # 蹂묒썝, ?섏옄 ?곹깭 ?낅뜲?댄듃
            self.status['hospital']['h_states'][h_idx, 0] -= 1  # n_idle -= 1
            # ?대깽??異붽?
            if service_time == np.inf: # capa ?앷퉴吏 ?먯쑀
                self.status['patient']['p_states'][p_idx, -1] = 1
            else: # ?쒕퉬??醫낅즺 ?대깽??議댁옱
                self.add_event(service_time, 'p_def_care', (p_idx, h_idx))
        else: # ?湲??쒖옉 (?좏뻾 議곌굔?먯꽌 max_queue ?섏튂吏 ?딅룄濡??대깽??異붽??덉뿀??
            # 蹂묒썝, ?섏옄 ?곹깭 ?낅뜲?댄듃
            self.status['hospital']['h_states'][h_idx, 1] += 1  # n_queue += 1
            self.status['patient']['p_wait'][p_class][h_idx+1].append(p_idx) # ?섏옄 ?湲??쒖옉
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
        蹂묒썝???섏옄泥섏튂 醫낅즺 ?대깽??
        :param log: ?대깽???댁슜 湲곕줉??dictionary
        :param entity_idx: tuple(patient idx, hospital idx)
        :return:
        """
        p_idx, h_idx = entity_idx
        # 泥섏튂 ?꾨즺 ?섏옄 ?곹깭 蹂寃?
        self.status['patient']['p_states'][p_idx, -1] = 1
        self._record_trace("care_complete", patient_id=int(p_idx), hospital_id=int(h_idx))

        n_idle, n_queue = self.status['hospital']['h_states'][h_idx][0:2]
        # ?덈줈??泥섏튂 ?쒖옉
        if n_queue > 0:
            h_tier = self.properties['hospital']['hos_tier'][h_idx]
            # Red, Yellow, Green, Black ?쒖쑝濡?泥섏튂
            for p_class in range(4):
                if self.status['patient']['p_wait'][p_class][h_idx+1]:
                    new_p_idx = self.status['patient']['p_wait'][p_class][h_idx+1].pop()
                    break
            service_time = self.sample_service_time(h_tier=h_tier, p_class=p_class)
            log['p_admit'].append((self.time, p_class))
            # 蹂묒썝, ?섏옄 ?곹깭 ?낅뜲?댄듃
            self.status['hospital']['h_states'][h_idx, 1] -= 1  # n_queue -= 1
            # ?대깽??異붽?
            if service_time == np.inf: # capa ?앷퉴吏 ?먯쑀
                self.status['patient']['p_states'][new_p_idx, -1] = 1
            else: # ?쒕퉬??醫낅즺 ?대깽??議댁옱
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
        :param log: ?대깽???댁슜 湲곕줉??dictionary
        :param entity_idx: ?대깽??李몄뿬 entity留덈떎??index
        :return:
        stop_condition: decision 諛쏄린 ?꾪빐 硫덉텧 ?щ?
        """
        stop_condition = False
        return log, stop_condition


