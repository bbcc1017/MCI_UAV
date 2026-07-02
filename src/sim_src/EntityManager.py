import numpy as np
class EntityManager():
    def __init__(self, en_types):
        self.en_types = en_types # entity 이름; dict_keys()
        self.en_properties = dict.fromkeys(en_types, None)  # entity type마다 공동 활용될 특성 기록해둔 dictionary (e.g. velocity, distanceMat)
    def en_register(self, en_name, en_prop):
        self.en_properties[en_name] = en_prop

    def init_en_status(self):
        self.en_status = dict.fromkeys(self.en_types, None) # entity마다 상태 기록할 dictionary
        hos_num = self.en_properties['hospital']['hos_num']
        for en_name in self.en_status:
            self.en_status[en_name] = {}
            base = self.en_status[en_name]
            if en_name == "patient":
                totalN = self.en_properties[en_name]['incident_size']
                base['p_states'] = np.zeros(shape=(totalN, 5), dtype=np.int32) # class, rescued(0/1), move(0/1), moved(0/1), cared(0/1)
                base['p_wait'] = [[[] for i in range(hos_num+1)] for p_class in range(4)] # red_wait, yellow_wait, green_wait, black_wait
                base['p_sent'] = np.zeros(shape=(hos_num,), dtype=np.int32)
            elif en_name == "hospital":
                base['h_states'] = np.zeros(shape=(hos_num, 3), dtype=np.int32) # (n_idle, n_queue, n_occupied)
                base['h_states'][:,0] = np.copy(self.en_properties[en_name]['hos_max_capa'])
            elif en_name == "ambulance":
                amb_num = self.en_properties[en_name]['amb_num']
                base['amb_states'] = np.zeros(shape=(amb_num,3), dtype=np.float32) # (dest, time, severity); severity in {empty = 0, red = 1, yellow = 2, green = 3}
                base['amb_wait'] = [[] for i in range(hos_num + 1)]
            elif en_name == "uav":
                uav_num = self.en_properties[en_name]['uav_num']
                base['uav_states'] = np.zeros(shape=(uav_num,3), dtype=np.float32) # (dest, time, severity); severity in {empty = 0, red = 1, yellow = 2, green = 3}
                base['uav_wait'] = [[] for i in range(hos_num + 1)]
            else:
                raise NotImplementedError(f"{en_name}은 아직 구현되지 않은 개체입니다.")

            # if en_name == "patient":
            #     totalN = self.en_properties[en_name]['incident_size']
            #     base['p_class'] = np.ones(totalN, dtype=np.int32)*-1  # (Red, Yellow, Green, Black); 환자별 중증도
            #     base['p_not_rescued'] = np.zeros(4, dtype=np.int32)  # (Red, Yellow, Green, Black); 이 시점에는 사고 발생 전
            #     base['p_at_sites'] = np.zeros(4, dtype=np.int32) # (Red, Yellow, Green, Black); 현장 환자수
            #     base['p_admitted'] = np.zeros(4, dtype=np.int32) # (Red, Yellow, Green, Black); 병원에 수용된 환자수
            #     base['p_definite_cared'] = np.zeros(4, dtype=np.int32) # (Red, Yellow, Green, Black); 처치 완료 환자수
            # elif en_name == "hospital":
            #     base['capa'] = np.copy(self.en_properties[en_name]['hos_max_capa'])
            # elif en_name == "ambulance":
            #     amb_num = self.en_properties[en_name]['amb_num']
            #     base['dest_amb'] = np.zeros(amb_num, dtype=np.int32)
            #     base['time_amb'] = np.zeros(amb_num, dtype=np.float32)
            #     base['severity_amb'] = np.zeros(amb_num, dtype=np.int32) # severity in {empty = 0, red = 1, yellow = 2, green = 3}
            #     base['amb_idx_at_site'] = []
            # elif en_name == "uav":
            #     uav_num = self.en_properties[en_name]['uav_num']
            #     base['dest_uav'] = np.zeros(uav_num, dtype=np.int32)
            #     base['time_uav'] = np.zeros(uav_num, dtype=np.float32)
            #     base['severity_uav'] = np.zeros(uav_num, dtype=np.int32) # severity in {empty = 0, red = 1, yellow = 2, green = 3}
            #     base['uav_idx_at_site'] = []
            # else:
            #     raise NotImplementedError(f"{en_name}은 아직 구현되지 않은 개체입니다.")

    def get_obs(self):
        """
        강화학습용으로 entity마다 상태 하나로 통합해서 return
        :return:
        """
        obs = {}
        for v in self.en_status.values():
            obs |= v  # dict 병합 연산 (3.9 이상)
        return obs

    def get_full_obs(self):
        """
        heuristic rule이 사용 가능하도록 full observation return
        :return:
        """
        full_obs = {}
        for v in self.en_status.values():
            full_obs |= v  # dict 병합 연산 (3.9 이상)
        return full_obs

    @staticmethod
    def in_flight_by_hospital(obs, hos_num):
        """병원별 이송중(발송 후 미도착) 환자 수 — 통신가용(occ) 게이트의 '도착 예상' 신호.

        amb/uav_states = (dest, time, severity): dest 1..H(1-based 병원행), severity>0
        = 환자 탑승. 복귀 leg 는 dest=0/severity=0 으로 재설정되므로 자동 제외.
        occ(입원 census, 수술완료 시 감소)에 아직 안 잡힌 예약 부하를 나타낸다.
        (2026-07-03 통신축 재정의: occ 게이트 = n_occupied + in_flight < max_send)
        """
        inflight = np.zeros(hos_num, dtype=np.int32)
        for key in ('amb_states', 'uav_states'):
            st = obs.get(key)
            if st is None or len(st) == 0:
                continue
            st = np.asarray(st)
            carrying = (st[:, 0] >= 1) & (st[:, 2] > 0)
            for d in st[carrying, 0].astype(int):
                if 1 <= d <= hos_num:
                    inflight[d - 1] += 1
        return inflight