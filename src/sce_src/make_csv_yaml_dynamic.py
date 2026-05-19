# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import yaml
import argparse
import pandas as pd
import numpy as np
import requests
from haversine import haversine
# [ADD] ──────────────────────────────────────────────────────────────
import re
from datetime import timezone, timedelta, datetime
from typing import Optional

KST = timezone(timedelta(hours=9))

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def slugify(name: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^\w\-\s]", "", str(name))
    s = re.sub(r"\s+", "_", s).strip("_")
    return (s[:maxlen] or "noname")

def save_route_json(meta: dict, payload: Optional[dict], out_path: str):
    ensure_dir(os.path.dirname(out_path))
    data = {"meta": meta, "payload": {"naver_response": payload} if payload else None}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
# [ADD END] ─────────────────────────────────────────────────────────


def str2bool(v):
    """argparse용 문자열→bool 변환. True/False가 아닌 값은 ArgumentTypeError."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f", ""):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v!r}")


def parse_util_map(text: str):
    """
    "1:0.90,11:0.75,etc:0.60" -> {1:0.9, 11:0.75, "etc":0.6}
    """
    if not text:
        return None
    m = {}
    for part in str(text).split(","):
        if not part.strip():
            continue
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        try:
            val = float(v)
        except Exception:
            continue
        if k.lower() == "etc":
            m["etc"] = val
        else:
            try:
                m[int(k)] = val
            except Exception:
                pass
    return m if m else None

class ScenarioGenerator:
    """동적 파라미터 기반 시나리오 생성 클래스 (크로스 환경 호환)"""

    def __init__(self, base_path, experiment_id=None, kakao_api_key=None, departure_time=None,
                 osrm_url=None, road_provider=None, is_use_time=True, fixed_hos_num=None):
        # 실제 도로 API 호출 횟수 카운터 (Kakao + OSRM 공통)
        self.api_call_count = 0
        # 프로젝트 경로 절대화
        self.base_path = os.path.abspath(base_path)

        # 도로 데이터 공급자 결정 (experiment_id 접미사 결정에 필요)
        # 우선순위: 명시적 road_provider > is_use_time 플래그
        is_use_time = bool(is_use_time) if not isinstance(is_use_time, str) else str2bool(is_use_time)
        if road_provider is None:
            road_provider = "kakao" if is_use_time else "osrm"
        self.road_provider = road_provider

        # experiment_id 생성:
        #   - kakao 모드 + departure_time → exp_<base>_dep_<YYYYMMDDHHMM>
        #   - osrm  모드                  → exp_<base>_osrm
        #   - 이미 적절한 접미사가 붙어 있으면 그대로 존중 (idempotent)
        if experiment_id:
            base_exp_id = str(experiment_id).strip()
            if base_exp_id.startswith("exp_"):
                base_exp_id = base_exp_id[4:]
            # 공백만 언더스코어로 정규화하고, 기존 언더스코어는 유지
            base_exp_id = re.sub(r"\s+", "_", base_exp_id).strip("_")
            if not base_exp_id:
                base_exp_id = datetime.now().strftime("%Y%m%d%H%M")
        else:
            base_exp_id = datetime.now().strftime("%Y%m%d%H%M")

        if self.road_provider == "osrm":
            # 동일 base가 카카오용 _dep_ 접미사를 달고 있다면 제거 후 _osrm 부착
            base_exp_id = re.sub(r"_dep_\d{12}$", "", base_exp_id)
            if base_exp_id.endswith("_osrm"):
                self.experiment_id = f"exp_{base_exp_id}"
            else:
                self.experiment_id = f"exp_{base_exp_id}_osrm"
        else:  # kakao
            # 반대로 _osrm 접미사가 붙어 있으면 제거하고 _dep_ 부착
            base_exp_id = re.sub(r"_osrm$", "", base_exp_id)
            if departure_time and f"_dep_{departure_time}" not in base_exp_id:
                self.experiment_id = f"exp_{base_exp_id}_dep_{departure_time}"
            else:
                self.experiment_id = f"exp_{base_exp_id}"

        # 카카오 API 키 설정 (인자 미지정 시 ENV("KAKAO_API_KEY") 자동 사용)
        self.kakao_api_key = kakao_api_key or os.environ.get("KAKAO_API_KEY")
        self.departure_time = departure_time  # YYYYMMDDHHMM 형식

        # OSRM 백엔드 설정 (is_use_time=False일 때 사용)
        self.osrm_url = (osrm_url
                         or os.environ.get("MCI_OSRM_URL", "https://router.project-osrm.org"))
        
        # 데이터 파일 경로들 (절대경로로 설정)
        self.scenarios_path = os.path.join(self.base_path, "scenarios")
        self.fire_data_path = os.path.join(self.scenarios_path, "안전센터와 소방서.csv")
        self.hospital_data_path = os.path.join(self.scenarios_path, "엑셀 결합 데이터.xlsx")
        
        # 파일 존재성 검증
        self._validate_data_files()

        # Patient 정보 (하드코딩)
        self.patient_config = {
            "ratio": {"Red": 0.1, "Yellow": 0.3, "Green": 0.5, "Black": 0.1},
            "rescue_param": {"Red": (6, 5), "Yellow": (2, 13), "Green": (1, 22), "Black": (0, 0)},
            "treat_tier3": {"Red": True, "Yellow": True, "Green": True, "Black": True},
            "treat_tier2": {"Red": False, "Yellow": True, "Green": True, "Black": True},
            "treat_tier3_mean": {"Red": 40, "Yellow": 20, "Green": 10, "Black": 0},
            "treat_tier2_mean": {"Red": float('inf'), "Yellow": 30, "Green": 15, "Black": 0}
        }
        
        # 후보군 확장 배수 (AMB road distance 호출 수 완화)
        self.multiplier = 1.5

        # --- ENV 주입(PS에서 전달) ---
        # util_by_tier: 예) "1:0.656,11:0.461,etc:0.461"
        env_util = parse_util_map(os.environ.get("MCI_UTIL_BY_TIER", ""))
        self.util_by_tier = env_util or {1: 0.656, 11: 0.461, "etc": 0.461}

        # queue_policy: "0" | "capa/2" | "0.5" 등
        # self.queue_policy = os.environ.get("MCI_QUEUE_POLICY", "0")

        # buffer_ratio: float
        try:
            self.buffer_ratio = float(os.environ.get("MCI_BUFFER_RATIO", "1.5"))
        except Exception:
            self.buffer_ratio = 1.5
        
        # (추가) max_send_coeff 기본 입력경로: ENV → 기본값
        self.max_send_coeff_text = os.environ.get("MCI_MAX_SEND_COEFF", "1,1")
        
        # 모든 좌표에서 hos_num 을 동일하게 강제 (RL 학습/평가 obs 차원 일치용)
        self.fixed_hos_num = int(fixed_hos_num) if fixed_hos_num else None

        print(f"📁 프로젝트 경로: {self.base_path}")
        print(f"🆔 실험 ID: {self.experiment_id}")
        print(f"buffer_ratio={self.buffer_ratio}")
        if self.fixed_hos_num:
            print(f"fixed_hos_num={self.fixed_hos_num} (보장 룰 후 가까운 N개로 cap)")

    def _validate_data_files(self):
        """필수 데이터 파일들의 존재성 검증"""
        required_files = [
            (self.fire_data_path, "소방서 데이터"),
            (self.hospital_data_path, "병원 데이터")
        ]
        missing_files = []
        for file_path, description in required_files:
            if not os.path.exists(file_path):
                missing_files.append(f"{description}: {file_path}")
        if missing_files:
            print("❌ 다음 필수 파일들이 없습니다:")
            for missing in missing_files:
                print(f"   • {missing}")
            raise FileNotFoundError("필수 데이터 파일들을 확인해주세요.")
        print("✅ 모든 필수 데이터 파일 확인 완료")

    def get_road_distance(self, start, end, **kwargs):
        """도로 거리/시간 디스패처. self.road_provider에 따라 kakao/osrm으로 위임.

        Returns:
            (distance_km, duration_min) 튜플
        """
        provider = (self.road_provider or "kakao").lower()
        if provider == "osrm":
            return self.get_road_distance_osrm(start, end, **kwargs)
        return self.get_road_distance_kakao(start, end, **kwargs)

    def get_road_distance_kakao(self, start, end, max_retries=3, save_json_dir=None, route_type=None, source_index=None, name=None, start_label="start", goal_label="goal"):
        """카카오 모빌리티 API를 사용한 도로 거리 및 시간 계산 (재시도 로직 포함)

        Args:
            start: (lat, lon) 튜플
            end: (lat, lon) 튜플
            save_json_dir: JSON 저장 디렉토리
            route_type: "center2site" 또는 "hos2site"

        Returns:
            (distance_km, duration_min) 튜플 - 거리(km)와 이송시간(분)
        """
        if not self.kakao_api_key:
            raise RuntimeError(
                "카카오 API 키가 없습니다 (is_use_time=True 모드). "
                "키가 없다면 is_use_time=False로 OSRM 백엔드를 사용하세요."
            )

        url = "https://apis-navi.kakaomobility.com/v1/future/directions"
        headers = {
            "Authorization": f"KakaoAK {self.kakao_api_key}",
            "Content-Type": "application/json"
        }
        params = {
            "origin": f"{start[1]},{start[0]}",  # lon,lat 순서
            "destination": f"{end[1]},{end[0]}",
            "priority": "TIME",  # 최단시간 우선
            "car_fuel": "GASOLINE",
            "car_hipass": "false",
            "alternatives": "false",
            "road_details": "false"
        }

        # departure_time 파라미터 추가 (실시간 또는 미래시간)
        if self.departure_time:
            params["departure_time"] = self.departure_time

        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()

                    # 카카오 API 응답 구조: routes[0].summary
                    if not data.get("routes") or len(data["routes"]) == 0:
                        raise RuntimeError(
                            f"카카오 API 경로 없음 ({start} → {end}): "
                            f"해상·섬 좌표이거나 도로가 연결되지 않는 구간입니다."
                        )

                    route = data["routes"][0]
                    result_code = route.get("result_code", 0)
                    if result_code != 0:
                        _rc_msg = {
                            101: "경유지 주변 도로 탐색 불가",
                            102: "출발지 주변 도로 탐색 불가",
                            103: "도착지 주변 도로 탐색 불가",
                            104: "출발지와 도착지가 5m 이내",
                            105: "출발지 주변 도로에 교통 장애(유고 정보) 존재",
                        }.get(result_code, "알 수 없는 오류")
                        raise RuntimeError(
                            f"카카오 API 경로 없음 (result_code={result_code}, {start} → {end}): "
                            f"{_rc_msg}"
                        )
                    summary = route.get("summary", {})

                    # 거리(m) → km 변환
                    distance_km = summary.get("distance", 0) / 1000.0

                    # 시간(초) → 분 변환
                    duration_sec = summary.get("duration", 0)
                    duration_min = duration_sec / 60.0

                    # JSON 저장
                    if save_json_dir:
                        now = datetime.now(KST).isoformat()
                        meta = {
                            "api_provider": "kakao",
                            "route_type": route_type,
                            "source_index": source_index,
                            "name": name,
                            # 좌표는 [lon, lat] 형식으로 저장
                            start_label: [start[1], start[0]],
                            goal_label: [end[1], end[0]],
                            "departure_time": self.departure_time or "realtime",
                            "priority": params.get("priority"),
                            "saved_at": now,
                            # 요약 필드
                            "distance_km": round(distance_km, 3),
                            "duration_min": round(duration_min, 2),
                            "duration_sec": duration_sec,
                            "toll_fare": summary.get("fare", {}).get("toll", 0),
                            "taxi_fare": summary.get("fare", {}).get("taxi", 0),
                            "direction_note": f"{start_label}->{goal_label}"
                        }
                        fname = f"{(source_index if source_index is not None else 0):03d}_{slugify(name)}.json"
                        out_path = os.path.join(save_json_dir, fname)

                        # 카카오 응답 저장
                        ensure_dir(os.path.dirname(out_path))
                        json_data = {
                            "meta": meta,
                            "payload": {"kakao_response": data}
                        }
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(json_data, f, ensure_ascii=False, indent=2)

                        print(f"  📦 [{route_type}] idx={source_index:03d} {name} → {distance_km:.2f}km, {duration_min:.1f}min")

                    self.api_call_count += 1
                    return distance_km, duration_min

                elif response.status_code == 401:
                    raise RuntimeError(
                        f"카카오 API 인증 실패 (401): API 키를 확인하세요."
                    )
                elif response.status_code == 429:
                    print(f"  ⚠️ API 호출 한도 초과 (429): 3초 대기 중...")
                    time.sleep(3)
                else:
                    raise RuntimeError(
                        f"카카오 API 호출 실패 (status {response.status_code}): {start} → {end}"
                    )

            except RuntimeError:
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  ⚠️ API 호출 중 오류 ({attempt+1}/{max_retries}): {e}")
                    time.sleep(2)
                else:
                    raise RuntimeError(
                        f"카카오 API 호출 실패 ({max_retries}회 재시도 초과): {e}"
                    ) from e

        raise RuntimeError(
            f"카카오 API 호출 한도 초과 (429): {max_retries}회 재시도 후에도 실패. "
            f"일일 할당량이 소진되었습니다."
        )

    def get_road_distance_osrm(self, start, end, max_retries=3, save_json_dir=None,
                                route_type=None, source_index=None, name=None,
                                start_label="start", goal_label="goal"):
        """OSRM HTTP API(/route/v1/driving)를 사용한 도로 거리 및 시간 계산.

        카카오 함수와 동일한 시그니처/반환값/JSON 저장 스키마를 사용한다.
        OSRM은 정적 도로 그래프 기반이라 실시간 혼잡도/요금 정보가 없다.

        Args:
            start: (lat, lon) 튜플
            end: (lat, lon) 튜플

        Returns:
            (distance_km, duration_min)
        """
        base = (self.osrm_url or "https://router.project-osrm.org").rstrip("/")
        url = f"{base}/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}"
        params = {
            "overview": "full",
            "geometries": "geojson",
            "steps": "false",
            "annotations": "false",
            "alternatives": "false",
        }
        headers = {"Accept": "application/json"}

        last_err = None
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()

                    code = data.get("code")
                    routes = data.get("routes") or []
                    if code != "Ok" or not routes:
                        raise RuntimeError(
                            f"OSRM 경로 없음 ({start} → {end}, code={code}): "
                            f"도로가 연결되지 않는 구간이거나 OSRM 서버가 해당 영역을 커버하지 않습니다."
                        )

                    route = routes[0]
                    distance_m = float(route.get("distance", 0))
                    duration_s = float(route.get("duration", 0))
                    distance_km = distance_m / 1000.0
                    duration_min = duration_s / 60.0
                    duration_sec = int(round(duration_s))

                    if save_json_dir:
                        now = datetime.now(KST).isoformat()
                        meta = {
                            "api_provider": "osrm",
                            "route_type": route_type,
                            "source_index": source_index,
                            "name": name,
                            # 좌표는 카카오와 동일하게 [lon, lat] 형식
                            start_label: [start[1], start[0]],
                            goal_label: [end[1], end[0]],
                            "departure_time": "static",   # OSRM은 시각 개념 없음
                            "priority": "shortest_time",
                            "saved_at": now,
                            "distance_km": round(distance_km, 3),
                            "duration_min": round(duration_min, 2),
                            "duration_sec": duration_sec,
                            "toll_fare": 0,                # OSRM 미제공
                            "taxi_fare": 0,                # OSRM 미제공
                            "direction_note": f"{start_label}->{goal_label}",
                        }
                        fname = f"{(source_index if source_index is not None else 0):03d}_{slugify(name)}.json"
                        out_path = os.path.join(save_json_dir, fname)
                        ensure_dir(os.path.dirname(out_path))
                        json_data = {"meta": meta, "payload": {"osrm_response": data}}
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(json_data, f, ensure_ascii=False, indent=2)

                        print(f"  📦 [{route_type}] idx={(source_index or 0):03d} {name} → "
                              f"{distance_km:.2f}km, {duration_min:.1f}min (OSRM)")

                    self.api_call_count += 1
                    return distance_km, duration_min

                elif response.status_code == 429:
                    print(f"  ⚠️ OSRM 호출 한도 초과 (429): 3초 대기 중...")
                    time.sleep(3)
                else:
                    raise RuntimeError(
                        f"OSRM 호출 실패 (status {response.status_code}): {start} → {end}"
                    )

            except RuntimeError:
                raise
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    print(f"  ⚠️ OSRM 호출 중 오류 ({attempt+1}/{max_retries}): {e}")
                    time.sleep(2)
                else:
                    raise RuntimeError(
                        f"OSRM 호출 실패 ({max_retries}회 재시도 초과): {e}"
                    ) from e

        raise RuntimeError(
            f"OSRM 호출 실패 ({max_retries}회 재시도 후에도 실패): last_err={last_err}"
        )

    def make_amb_info(self, latitude, longitude, incident_size, amb_count, save_folder):
        """구급차 정보 생성"""
        print(f"  🚑 구급차 정보 생성 중...")
        try:
            df = pd.read_csv(self.fire_data_path, encoding="cp949")
        except Exception as e:
            print(f"❌ 소방서 데이터 로드 실패: {e}")
            return
        
        # '수량' 기반으로 센터 행 복제 (구급차 개체 수 반영)
        # ------------------------------------------------------------
        if "수량" in df.columns:
            df["수량"] = pd.to_numeric(df["수량"], errors="coerce").fillna(1).astype(int)
            df.loc[df["수량"] < 1, "수량"] = 1
        else:
            df["수량"] = 1
        df["보유대수"] = df["수량"]

        # 센터를 수량만큼 복제
        df = df.loc[df.index.repeat(df["수량"])].copy()
        # ------------------------------------------------------------
        coords = list(zip(df["y좌표"], df["x좌표"]))
        euc_distances = [haversine(coord, (latitude, longitude)) for coord in coords]
        df["euclidean_distance"] = euc_distances

        # EUC 저장
        # df_sorted_euc = df.sort_values("euclidean_distance").head(incident_size).copy()
        df_sorted_euc = df.sort_values("euclidean_distance").head(amb_count).copy()
        df_sorted_euc = df_sorted_euc.rename(columns={
            "euclidean_distance": "init_distance",
            "기관명": "안전센터/소방서이름"
        })
        df_sorted_euc = df_sorted_euc.reset_index(drop=True)
        df_sorted_euc = df_sorted_euc[["init_distance", "안전센터/소방서이름", "보유대수"]]
        euc_save_path = os.path.join(save_folder, "amb_info_euc.csv")
        df_sorted_euc.to_csv(euc_save_path, index=True, index_label="Index", encoding="utf-8-sig")
        
        # [ADD] center2site 저장 폴더
        routes_dir = os.path.join(save_folder, "routes", "center2site")
        ensure_dir(routes_dir)

        # 후보군 확장 및 도로 거리/시간 계산 (카카오 API)
        df_candidates = df.sort_values("euclidean_distance").head(int(incident_size * self.multiplier)).copy()
        road_distances = []
        road_durations = []

        # for j, (_, row) in enumerate(df_candidates.iterrows()):
        #     coord = (row["y좌표"], row["x좌표"])  # (lat, lon) of center
        #     dist_km, duration_min = self.get_road_distance_kakao(
        #         start=coord, end=(latitude, longitude),  # center → site
        #         save_json_dir=routes_dir, route_type="center2site",
        #         source_index=j, name=row.get("기관명", f"center_{j}"),
        #         start_label="center", goal_label="site"
        #     )
        cache = {}  # key: (center_lat, center_lon) -> (dist_km, duration_min)
        for j, (_, row) in enumerate(df_candidates.iterrows()):
            coord = (row["y좌표"], row["x좌표"])  # (lat, lon)
            key = coord
            if key in cache:
                dist_km, duration_min = cache[key]
            else:
                dist_km, duration_min = self.get_road_distance(
                    start=coord, end=(latitude, longitude),  # center → site
                    save_json_dir=routes_dir, route_type="center2site",
                    source_index=j, name=row.get("기관명", f"center_{j}"),
                    start_label="center", goal_label="site"
                )
                cache[key] = (dist_km, duration_min)
            road_distances.append(dist_km)
            road_durations.append(duration_min)
            time.sleep(0.05)

        df_candidates["road_distance"] = road_distances
        df_candidates["road_duration"] = road_durations

        # ROAD 저장 (duration 기준으로 정렬 후 상위 incident_size개 선택)
        # df_sorted_road = df_candidates.sort_values("road_duration").head(incident_size).copy()
        df_sorted_road = df_candidates.sort_values("road_duration").head(amb_count).copy()
        df_sorted_road = df_sorted_road.rename(columns={
            "road_distance": "init_distance",
            "road_duration": "duration",
            "기관명": "안전센터/소방서이름"
        })
        df_sorted_road = df_sorted_road.reset_index(drop=True)
        df_sorted_road = df_sorted_road[["init_distance", "duration", "안전센터/소방서이름", "보유대수"]]
        road_save_path = os.path.join(save_folder, "amb_info_road.csv")
        df_sorted_road.to_csv(road_save_path, index=True, index_label="Index", encoding="utf-8-sig")
        
        print(f"  ✅ 구급차 정보 생성 완료")

    def make_hospital_info(self, latitude, longitude, incident_size, save_folder, uav_count=0):
        """병원 정보 생성 (기존 로직 유지 + 최소 조건 추가 보장)

        Args:
            latitude: 사고지점 위도
            longitude: 사고지점 경도
            incident_size: 환자 수
            save_folder: 저장 폴더
            uav_count: UAV 대수 (헬기장 병원 최소 보장에 사용)
        """
        print(f"  🏥 병원 정보 생성 중...")
        
        # ---------- (0) 데이터 로드 ----------
        try:
            df_full = pd.read_excel(self.hospital_data_path, engine='openpyxl')
        except Exception as e:
            print(f"❌ 병원 데이터 로드 실패: {e}")
            return

        # 필요한 열만 사용 (이름 유지)
        cols_needed = ["요양기관명", "종별코드", "응급실병상수", "x좌표", "y좌표"]
        for c in cols_needed:
            if c not in df_full.columns:
                raise KeyError(f"필수 컬럼 누락: {c}")
        df = df_full[cols_needed].copy()

        # ★ 헬기장 여부 컬럼 추가 (있으면 포함, 없으면 0으로 채움)
        if "헬기장 여부" in df_full.columns:
            df["헬기장 여부"] = df_full["헬기장 여부"].fillna(0).astype(int)
        else:
            df["헬기장 여부"] = 0  # 헬기장 정보 없으면 모두 0

        # ---------- (1) 유클리드 거리 계산 ----------
        coords = list(zip(df["y좌표"], df["x좌표"]))  # (lat, lon)
        df["euclidean_distance"] = [haversine((lat, lon), (latitude, longitude)) for (lat, lon) in coords]

        # ---------- (2) 파라미터 ----------
        util_by_tier = getattr(self, "util_by_tier", {1: 0.656, 11: 0.461, "etc": 0.461})
        # queue_policy = str(getattr(self, "queue_policy", "0")).strip()
        try:
            buffer_ratio = float(getattr(self, "buffer_ratio", 1.5))
        except Exception:
            buffer_ratio = 1.5

        ratio = self.patient_config.get("ratio", {"Red":0.1,"Yellow":0.3,"Green":0.5,"Black":0.1})
        U = int(round(incident_size * float(ratio.get("Red", 0))))
        N = int(incident_size)
        
        import math
        def _get_util(code):
            try:
                icode = int(code)
                return util_by_tier.get(icode, util_by_tier.get("etc", 0.461))
            except Exception:
                return util_by_tier.get("etc", 0.461)
            
        df["util"] = df["종별코드"].apply(_get_util)
        df["capa"] = (df["응급실병상수"] * (1 - df["util"])).apply(lambda x: int(max(0, math.floor(x))))
        # 수술실 수 종별코드별 고정
        conditions = [df['종별코드'] == 1, df['종별코드'] == 11]; values = [3, 2]
        df['operating_rooms'] = np.select(conditions, values, default=1)
        df["eff"] = df["operating_rooms"] + df["capa"]
        df["is_tier3"] = (df["종별코드"].astype(str).astype(float).astype(int) == 1).astype(int)
        
        # ---------- (3) 전역 상급 용량 점검 (불가능 사전 감지) ----------
        total_tier3_capa_all = int(df.loc[df["is_tier3"]==1, "capa"].sum())
        total_capa_all = int(df["capa"].sum())
        if total_tier3_capa_all < U:
            print(f"  ⚠️ 전역 상급 용량 부족: Tier3_capa_all={total_tier3_capa_all} < U={U}. 최선 선택으로 진행(전원 실패 가능).")
        
        # --- (4) 후보군 확장: 기존 코드와 동일 ---
        # 가까운 병원들을 포함한 넉넉한 후보군(df_cand)
        df_sorted = df.sort_values("euclidean_distance").reset_index(drop=True)
        sum_capa = 0; sum_capa_tier3 = 0; cand_idx = []; 
        for i, row in df_sorted.iterrows():
            cand_idx.append(i)
            sum_capa += int(row["eff"])
            if row["is_tier3"] == 1: sum_capa_tier3 += int(row["eff"]); 
            if (sum_capa >= N * buffer_ratio): break
        if not cand_idx:
            cand_idx = list(range(len(df_sorted)))
        df_cand = df_sorted.loc[cand_idx].copy()
        
        df_selected = df_cand.copy()


        # ================================================================= #
        # 위에서 선택된 목록에 최소 조건을 만족하는지 확인하고 부족할 시 추가
        # 규칙 1: 상급종합병원(Tier 3) 최소 2개 보장
        final_tier3 = df_selected[df_selected["is_tier3"] == 1]
        num_to_ensure_tier3 = 2 - len(final_tier3)
        if num_to_ensure_tier3 > 0:
            print(f"  INFO: 최종 목록의 상급병원이 {len(final_tier3)}개. 최소 2개를 위해 '추가'합니다.")
            # 전체 병원 목록에서 아직 선택되지 않은 가장 가까운 상급병원을 찾아서 최소 2개가 될때까지 추가
            candidates = df_sorted[(df_sorted["is_tier3"] == 1) & (~df_sorted.index.isin(df_selected.index))]
            if not candidates.empty:
                hospitals_to_add = candidates.head(num_to_ensure_tier3)
                df_selected = pd.concat([df_selected, hospitals_to_add])

        # 규칙 2: 상급종합병원이 환자 40% 수용 용량 보장 (Tier 3 기준, 환자수가 많을때 최소 red환자 10% 이상 + 확률분포 고려한 비율)
        target_capa = N * 0.4
        current_capa = df_selected[df_selected["is_tier3"] == 1]["eff"].sum()
        while current_capa < target_capa:
            print(f"  INFO: 상급병원 용량이 {current_capa}/{target_capa}. 용량을 위해 '추가'합니다.")
            candidates = df_sorted[(df_sorted["is_tier3"] == 1) & (~df_sorted.index.isin(df_selected.index))]
            if candidates.empty: print("  WARNING: 추가할 상급병원이 더 이상 없습니다."); break
            hospital_to_add = candidates.head(1)
            df_selected = pd.concat([df_selected, hospital_to_add])
            current_capa = df_selected[df_selected["is_tier3"] == 1]["eff"].sum()

        # 규칙 3: 그 외 병원(Tier 2 등) 최소 1개 보장 (우연히 가장 가까이 있는 병원이 상급종합병원뿐일때 64개의 룰 중 실패하는 룰이 존재하므로)
        if len(df_selected[df_selected["is_tier3"] == 0]) == 0:
            print("  INFO: 최종 목록에 Tier 2 병원이 없음. 시뮬레이션 오류 방지를 위해 '추가'합니다.")
            candidates = df_sorted[(df_sorted["is_tier3"] == 0) & (~df_sorted.index.isin(df_selected.index))]
            if not candidates.empty:
                df_selected = pd.concat([df_selected, candidates.head(1)])

        # ================================================================= #
        # 규칙 4: 헬기장 병원 최소 보장 (UAV 대수 이상)
        if "헬기장 여부" in df_selected.columns:
            # UAV 대수 확인 (파라미터에서)
            uav_n = int(max(0, uav_count))

            if uav_n > 0:
                helipad_hospitals = df_selected[df_selected["헬기장 여부"] == 1]
                num_helipad = len(helipad_hospitals)

                # UAV 대수만큼 헬기장 병원이 없으면 추가
                num_to_ensure_helipad = uav_n - num_helipad

                if num_to_ensure_helipad > 0:
                    print(f"  INFO: 헬기장 병원이 {num_helipad}개인데 UAV는 {uav_n}대. 최소 {uav_n}개 헬기장 병원 확보를 위해 '{num_to_ensure_helipad}개' 추가합니다.")

                    # 전체 병원 목록에서 헬기장 있는 병원 중 아직 선택되지 않은 것 찾기
                    candidates_helipad = df_sorted[
                        (df_sorted["헬기장 여부"] == 1) &
                        (~df_sorted.index.isin(df_selected.index))
                    ]

                    if not candidates_helipad.empty:
                        # 필요한 만큼 헬기장 병원 추가
                        hospitals_to_add = candidates_helipad.head(num_to_ensure_helipad)
                        df_selected = pd.concat([df_selected, hospitals_to_add])
                        added_names = ", ".join(hospitals_to_add['요양기관명'].values)
                        print(f"    → 추가된 헬기장 병원: {added_names}")
                    else:
                        print(f"  ⚠️ 경고: 전체 데이터에 헬기장 병원이 {num_helipad}개밖에 없습니다. UAV {uav_n}대 운용이 불가능합니다.")
                else:
                    print(f"  ✓ 헬기장 병원 {num_helipad}개 (UAV {uav_n}대 운용 가능)")
            else:
                print("  INFO: UAV 대수가 0이므로 헬기장 병원 보장 로직을 건너뜁니다.")
        else:
            print("  ⚠️ '헬기장 여부' 컬럼이 원본 데이터에 없습니다. 헬기장 보장 로직을 건너뜁니다.")

        # ================================================================= #
        # 규칙 5: UAV 이송을 위한 교집합 병원 보장 (헬기장+Tier)
        if "헬기장 여부" in df_selected.columns:
            uav_n = int(max(0, uav_count))

            if uav_n > 0:
                # 5-1: Red UAV 이송용 헬기장+Tier3 병원 최소 1개 보장
                helipad_tier3_hospitals = df_selected[
                    (df_selected["헬기장 여부"] == 1) &
                    (df_selected["is_tier3"] == 1)
                ]

                if len(helipad_tier3_hospitals) == 0:
                    print("  INFO: Red UAV 이송용 헬기장+Tier3 병원이 없음. 추가 중...")
                    candidates = df_sorted[
                        (df_sorted["헬기장 여부"] == 1) &
                        (df_sorted["is_tier3"] == 1) &
                        (~df_sorted.index.isin(df_selected.index))
                    ]

                    if not candidates.empty:
                        hospital_to_add = candidates.head(1)
                        df_selected = pd.concat([df_selected, hospital_to_add])
                        added_name = hospital_to_add['요양기관명'].values[0]
                        print(f"    → 추가됨: {added_name}")
                    else:
                        print("  ⚠️ 경고: 전체 데이터에 헬기장+Tier3 병원 없음. Red UAV 이송 불가!")
                else:
                    print(f"  ✓ 헬기장+Tier3 병원 {len(helipad_tier3_hospitals)}개 (Red UAV 이송 가능)")

                # 5-2: Yellow UAV 이송용 헬기장+Tier2 병원 최소 1개 보장
                helipad_tier2_hospitals = df_selected[
                    (df_selected["헬기장 여부"] == 1) &
                    (df_selected["is_tier3"] == 0)
                ]

                if len(helipad_tier2_hospitals) == 0:
                    print("  INFO: Yellow UAV 이송용 헬기장+Tier2 병원이 없음. 추가 중...")
                    candidates = df_sorted[
                        (df_sorted["헬기장 여부"] == 1) &
                        (df_sorted["is_tier3"] == 0) &
                        (~df_sorted.index.isin(df_selected.index))
                    ]

                    if not candidates.empty:
                        hospital_to_add = candidates.head(1)
                        df_selected = pd.concat([df_selected, hospital_to_add])
                        added_name = hospital_to_add['요양기관명'].values[0]
                        print(f"    → 추가됨: {added_name}")
                    else:
                        print("  ⚠️ 경고: 전체 데이터에 헬기장+Tier2 병원 없음. Yellow UAV 이송 불가!")
                else:
                    print(f"  ✓ 헬기장+Tier2 병원 {len(helipad_tier2_hospitals)}개 (Yellow UAV 이송 가능)")
            else:
                print("  INFO: UAV 대수가 0이므로 헬기장+Tier 교집합 보장 로직을 건너뜁니다.")
        else:
            print("  ⚠️ '헬기장 여부' 컬럼이 원본 데이터에 없습니다. 헬기장+Tier 교집합 보장 로직을 건너뜁니다.")

        df_euc = df_selected.sort_values("euclidean_distance").reset_index(drop=True).copy()

        # ---------- (5.5) fixed_hos_num cap (RL obs 차원 일치) ----------
        fixed_n = getattr(self, "fixed_hos_num", None)
        if fixed_n is not None:
            target = int(fixed_n)
            if len(df_euc) > target:
                # 너무 많음 → 가까운 target 개로 cap. 보장 룰 깨지면 warning.
                capped = df_euc.head(target).copy()
                n_tier3 = int(capped["is_tier3"].sum())
                helipad_col = capped["헬기장 여부"] if "헬기장 여부" in capped.columns else None
                n_helipad = int((helipad_col == 1).sum()) if helipad_col is not None else 0
                n_helipad_t3 = int(((helipad_col == 1) & (capped["is_tier3"] == 1)).sum()) if helipad_col is not None else 0
                n_helipad_t2 = int(((helipad_col == 1) & (capped["is_tier3"] == 0)).sum()) if helipad_col is not None else 0
                breaks = []
                if n_tier3 < 2: breaks.append(f"Tier3<2({n_tier3})")
                if n_helipad < int(uav_count): breaks.append(f"helipad<uav({n_helipad}/{uav_count})")
                if n_helipad_t3 < 1: breaks.append("helipad+T3<1")
                if n_helipad_t2 < 1: breaks.append("helipad+T2<1")
                df_euc = capped.reset_index(drop=True)

                # ★ cap 후 헬기장 부족 보정: 가장 먼 일반 병원 제거 + 풀에서 헬기장 추가 (H 유지)
                if helipad_col is not None and n_helipad < int(uav_count):
                    deficit = int(uav_count) - n_helipad
                    already = set(df_euc["요양기관명"])
                    extra_helipad = df_sorted[
                        (df_sorted["헬기장 여부"] == 1) &
                        (~df_sorted["요양기관명"].isin(already))
                    ].head(deficit)
                    non_helipad = df_euc[df_euc["헬기장 여부"] == 0]
                    swap_n = min(len(extra_helipad), len(non_helipad))
                    if swap_n < deficit:
                        print(f"  ⚠️ 풀에 추가 가능한 헬기장이 {len(extra_helipad)}/{deficit}개만 있음 — 부분 보정")
                    if swap_n > 0:
                        to_remove = non_helipad.sort_values("euclidean_distance", ascending=False).head(swap_n)
                        df_euc = df_euc.drop(to_remove.index)
                        df_euc = pd.concat([df_euc, extra_helipad.head(swap_n)]).sort_values("euclidean_distance").reset_index(drop=True)
                        print(f"  ✓ cap 후 헬기장 부족 보정: 일반 {swap_n}곳 제거 → 헬기장 {swap_n}곳 추가 (헬기장 {n_helipad}→{n_helipad+swap_n}, H={len(df_euc)})")
                        n_helipad += swap_n
                        n_helipad_t3 = int(((df_euc["헬기장 여부"] == 1) & (df_euc["is_tier3"] == 1)).sum())
                        n_helipad_t2 = int(((df_euc["헬기장 여부"] == 1) & (df_euc["is_tier3"] == 0)).sum())
                        n_tier3 = int(df_euc["is_tier3"].sum())
                        breaks = []
                        if n_tier3 < 2: breaks.append(f"Tier3<2({n_tier3})")
                        if n_helipad < int(uav_count): breaks.append(f"helipad<uav({n_helipad}/{uav_count})")
                        if n_helipad_t3 < 1: breaks.append("helipad+T3<1")
                        if n_helipad_t2 < 1: breaks.append("helipad+T2<1")

                if breaks:
                    print(f"  ⚠️ fixed_hos_num={target} cap 후에도 보장 룰 깨짐: {', '.join(breaks)} — best-effort 진행")
                else:
                    print(f"  ✓ fixed_hos_num={target} cap 적용 (보장 룰 모두 유지)")
            elif len(df_euc) < target:
                deficit = target - len(df_euc)
                extra = df_sorted[~df_sorted.index.isin(df_euc.index)].head(deficit)
                if len(extra) < deficit:
                    print(f"  ⚠️ fixed_hos_num={target} 채우기 부족 (전체 풀 모자람): {len(df_euc)+len(extra)}곳에서 멈춤")
                df_euc = pd.concat([df_euc, extra]).sort_values("euclidean_distance").reset_index(drop=True)
                print(f"  📌 fixed_hos_num={target} 채우기: {deficit}개 추가")

        print(f" 최종 생성된 병원: {len(df_euc)}곳 (상급: {df_euc['is_tier3'].sum()}곳, 종합 등: {len(df_euc) - df_euc['is_tier3'].sum()}곳)")

        # ---------- (6) EUC 파일은 나중에 road 순서로 저장 (인덱스 일치 보장) ----------
        # ★ CRITICAL: distance_Hos2Site_euc.csv는 road 순서를 따라야 h_states와 인덱스가 일치
        # ★ 따라서 이 시점에서는 euc_info만 저장하고, distance는 road 재정렬 후 저장합니다.

        euc_info = df_euc[["operating_rooms", "capa", "종별코드", "요양기관명", "헬기장 여부"]].copy()
        euc_info.columns = ["수술실수", "병상수", "종별코드", "요양기관명", "헬기장 여부"]
        euc_info_path = os.path.join(save_folder, "hospital_info_euc.csv")
        euc_info.to_csv(euc_info_path, index=True, index_label="Index", encoding="utf-8-sig")
        
        routes_dir_hos = os.path.join(save_folder, "routes", "hos2site")
        ensure_dir(routes_dir_hos)
        # ---------- (7) ROAD 거리 & 시간 계산 & 저장 (선정 병원만) ----------
        road_distances = []
        road_durations = []

        for j, (_, row) in enumerate(df_euc.iterrows()):
            end = (row["y좌표"], row["x좌표"])
            road_km, duration_min = self.get_road_distance(
                start=(latitude, longitude), end=end,  # site → hospital
                save_json_dir=routes_dir_hos, route_type="hos2site",
                source_index=j, name=row.get("요양기관명", f"hospital_{j}"),
                start_label="site", goal_label="hospital"
            )
            road_distances.append(road_km)
            road_durations.append(duration_min)
            time.sleep(0.05)

        df_euc = df_euc.copy()
        df_euc["road_distance"] = road_distances
        df_euc["road_duration"] = road_durations
        df_road = df_euc.sort_values("road_duration").reset_index(drop=True).copy()

        # distance_Hos2Site_road.csv에 duration 컬럼 추가
        dist_road_df = pd.DataFrame({
            "distance": df_road["road_distance"],
            "duration": df_road["road_duration"]
        })
        dist_road_path = os.path.join(save_folder, "distance_Hos2Site_road.csv")
        dist_road_df.to_csv(dist_road_path, index=True, index_label="Index", encoding="utf-8-sig")

        # ★ CRITICAL FIX: distance_Hos2Site_euc.csv를 road 순서로 저장 (인덱스 일치 보장)
        # df_road는 road_duration 기준으로 정렬되어 있으므로, h_states와 동일한 인덱스 순서를 가집니다.
        # euclidean_distance 값은 유지하되, 순서만 road 기준으로 변경합니다.
        dist_euc_df = pd.DataFrame({"distance": df_road["euclidean_distance"]})
        dist_euc_path = os.path.join(save_folder, "distance_Hos2Site_euc.csv")
        dist_euc_df.to_csv(dist_euc_path, index=True, index_label="Index", encoding="utf-8-sig")

        road_info = df_road[["operating_rooms", "capa", "종별코드", "요양기관명", "헬기장 여부"]].copy()
        road_info.columns = ["수술실수", "병상수", "종별코드", "요양기관명", "헬기장 여부"]
        road_info_path = os.path.join(save_folder, "hospital_info_road.csv")
        road_info.to_csv(road_info_path, index=True, index_label="Index", encoding="utf-8-sig")

        print(f"  ✅ 병원 정보 생성 완료 (distance_Hos2Site_euc.csv는 road 순서로 저장됨)")


    
    def make_uav_info(self, latitude, longitude, incident_size, uav_count, save_folder):
        """UAV 정보 생성 - hospital_info_road.csv 기반 (★핵심 변경★)
        - hospital_info_road.csv에서 "헬기장 여부"=1인 병원만 필터링
        - 사고지점 기준 거리 계산 후 가장 가까운 N개 헬기장 병원 선정
        - 각 병원당 최대 1개 UAV 배정
        - ★ uav_info 병원 = hospital_info의 부분집합 보장 (인덱스 일치)
        - CSV 구조: Index, init_distance, 수술실수, 병상수, 종별코드, 요양기관명
        """
        print(f"  🚁 UAV 정보 생성 중 (hospital_info_road.csv 기반)...")

        import os
        import pandas as pd
        from haversine import haversine

        # 0) 파라미터 정리
        try:
            uav_n = int(max(0, int(uav_count)))
        except Exception:
            uav_n = 0
        if uav_n <= 0:
            print("⚠️ UAV 대수가 0입니다. UAV 정보 생성 생략.")
            # 생략안하고 UAV=0대일 때 실험을 위한 빈 CSV 파일 생성 (헤더만 포함)
            empty_df = pd.DataFrame(columns=["Index", "init_distance", "수술실수", "병상수", "종별코드", "요양기관명"])
            save_path = os.path.join(save_folder, "uav_info.csv")
            empty_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"  빈 UAV 정보 파일 생성 완료: {save_path}")
            return

        # 1) ★ hospital_info_road.csv 로드 (기존 엑셀 대신!)
        hospital_info_path = os.path.join(save_folder, "hospital_info_road.csv")
        if not os.path.exists(hospital_info_path):
            print(f"❌ {hospital_info_path} 파일이 없습니다.")
            print("   make_hospital_info()를 먼저 실행해주세요.")
            raise FileNotFoundError(f"❌ {hospital_info_path} 파일이 없습니다.")

        try:
            df_hospital_pool = pd.read_csv(hospital_info_path, encoding="utf-8-sig")
        except Exception as e:
            print(f"❌ hospital_info_road.csv 로드 실패: {e}")
            return

        # 2) "헬기장 여부" 컬럼 확인 (필수)
        if "헬기장 여부" not in df_hospital_pool.columns:
            print("❌ '헬기장 여부' 컬럼이 hospital_info_road.csv에 없습니다.")
            print("   make_hospital_info()에서 헬기장 컬럼 추가 로직을 확인해주세요.")
            raise KeyError("❌ hospital_info_road.csv에 '헬기장 여부' 컬럼이 없습니다.")

        # 3) hospital_info 내에서 헬기장 병원만 필터링
        df_helipad_in_pool = df_hospital_pool[df_hospital_pool["헬기장 여부"] == 1].copy()

        if df_helipad_in_pool.empty:
            print("❌ hospital_info_road.csv에 헬기장이 있는 병원이 없습니다.")
            print("   make_hospital_info()의 헬기장 보장 로직을 확인해주세요.")
            raise ValueError("❌ hospital_info에 헬기장이 있는 병원이 없습니다.")

        # 4) 헬기장 병원 개수 검증 (UAV 대수와 비교)
        if len(df_helipad_in_pool) < uav_n:
            print(f"❌ hospital_info에 헬기장 병원이 {len(df_helipad_in_pool)}개밖에 없어 UAV {uav_n}대를 배치할 수 없습니다.")
            print(f"   make_hospital_info()의 헬기장 보장 로직을 확인하거나 UAV 대수를 줄여주세요.")
            raise ValueError(
                f"❌ hospital_info에 헬기장 병원이 {len(df_helipad_in_pool)}개밖에 없어 "
                f"UAV {uav_n}대를 배치할 수 없습니다."
            )

        # 5) 사고지점-병원 유클리드 거리 계산 (hospital_info에는 좌표가 없으므로 원본 Excel에서 가져와야 함)
        # ★ hospital_info_road.csv에 이미 거리 정보가 있을 수 있지만, 안전하게 원본에서 좌표를 가져옴
        try:
            df_full_excel = pd.read_excel(self.hospital_data_path, engine="openpyxl")
        except Exception as e:
            print(f"❌ 원본 Excel 데이터 로드 실패: {e}")
            return

        # 병원명 기준으로 좌표 매칭
        df_helipad_in_pool = df_helipad_in_pool.merge(
            df_full_excel[["요양기관명", "x좌표", "y좌표"]],
            on="요양기관명",
            how="left"
        )

        # 좌표가 없는 병원 체크
        if df_helipad_in_pool[["x좌표", "y좌표"]].isnull().any().any():
            missing_hospitals = df_helipad_in_pool[df_helipad_in_pool[["x좌표", "y좌표"]].isnull().any(axis=1)]["요양기관명"].tolist()
            print(f"⚠️ 경고: 다음 병원의 좌표 정보가 없습니다: {missing_hospitals}")
            df_helipad_in_pool = df_helipad_in_pool.dropna(subset=["x좌표", "y좌표"])

        df_helipad_in_pool["distance"] = df_helipad_in_pool.apply(
            lambda row: haversine((row["y좌표"], row["x좌표"]), (latitude, longitude)),
            axis=1
        )

        # 6) 거리순 정렬 (가까운 헬기장 병원부터)
        df_helipad_in_pool = df_helipad_in_pool.sort_values("distance").reset_index(drop=True)

        # 7) 상위 N개 선정 (각 병원 최대 1개 UAV)
        df_selected = df_helipad_in_pool.head(uav_n).copy()

        # 8) CSV 저장 (hospital_info와 동일한 병원 사용, 인덱스 일치 보장)
        result_df = pd.DataFrame({
            "uav_id": range(len(df_selected)),                 # UAV 번호 (0..)
            "hospital_idx": df_selected["Index"].astype(int),
            "init_distance": df_selected["distance"].round(3),
            "수술실수": df_selected["수술실수"],
            "병상수": df_selected["병상수"],
            "종별코드": df_selected["종별코드"],
            "요양기관명": df_selected["요양기관명"]
        })

        save_path = os.path.join(save_folder, "uav_info.csv")
        result_df.to_csv(save_path, index=False, encoding="utf-8-sig")

        print(f"  ✅ UAV 정보 생성 완료: {len(result_df)}개 UAV")
        print(f"     헬기장 병원: {', '.join(df_selected['요양기관명'].head(3).tolist())}{'...' if len(df_selected) > 3 else ''}")
        print(f"     ★ hospital_info의 부분집합으로 생성됨 (인덱스 일치 보장)")



    def make_patient_info(self, save_folder):
        """환자 정보 생성 (하드코딩된 값 사용)"""
        print(f"  👥 환자 정보 생성 중...")
        types = self.patient_config["ratio"].keys()
        rows = []
        for t in types:
            α, β = self.patient_config["rescue_param"][t]
            rows.append({
                "type": t,
                "ratio": self.patient_config["ratio"][t],
                "rescue_param_alpha": α,
                "rescue_param_beta": β,
                "treat_tier3": self.patient_config["treat_tier3"][t],
                "treat_tier2": self.patient_config["treat_tier2"][t],
                "treat_tier3_mean": self.patient_config["treat_tier3_mean"][t],
                "treat_tier2_mean": self.patient_config["treat_tier2_mean"][t]
            })
        df = pd.DataFrame(rows)
        save_path = os.path.join(save_folder, "patient_info.csv")
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"  ✅ 환자 정보 생성 완료")

    def make_distance_Hos2Hos(self, save_folder):
        """병원 간 거리 행렬 생성"""
        print(f"  📐 병원간 거리 행렬 생성 중...")
        try:
            df_full = pd.read_excel(self.hospital_data_path, engine="openpyxl")
        except Exception as e:
            print(f"❌ 병원 데이터 로드 실패: {e}")
            return

        # Euclidean (★ CRITICAL FIX: road 순서 기준으로 생성)
        try:
            # ★ hospital_info_euc.csv 대신 hospital_info_road.csv 사용 (인덱스 일치 보장)
            file_road = os.path.join(save_folder, "hospital_info_road.csv")
            df_road_hos = pd.read_csv(file_road, encoding="utf-8-sig")
            names_road = df_road_hos["요양기관명"].tolist()
            coords_road = []
            for name in names_road:
                row = df_full[df_full["요양기관명"] == name]
                if not row.empty:
                    coords_road.append((row.iloc[0]["y좌표"], row.iloc[0]["x좌표"]))
                else:
                    coords_road.append((0, 0))
            N = len(coords_road)
            matrix = np.zeros((N, N))
            for i in range(N):
                for j in range(i, N):
                    if i == j:
                        dist = 0
                    else:
                        dist = haversine(coords_road[i], coords_road[j])
                    matrix[i][j] = dist
                    matrix[j][i] = dist
            save_path_euc = os.path.join(save_folder, "distance_Hos2Hos_euc.csv")
            pd.DataFrame(matrix).to_csv(save_path_euc, index=True, encoding="utf-8-sig")
            print(f"  ✅ 병원간 유클리드 거리 행렬 생성 완료 (road 순서 기준)")
        except Exception as e:
            print(f"❌ 유클리드 거리 계산 실패: {e}")

        # Road (엑셀 파일 사용 - 기존 계산 데이터)
        try:
            file_road = os.path.join(save_folder, "hospital_info_road.csv")
            df_road = pd.read_csv(file_road, encoding="utf-8-sig")
            names_road = df_road["요양기관명"].tolist()

            # Load pre-calculated distance matrix from Excel
            excel_path = os.path.join(self.base_path, "scenarios", "DISTANCE_MATRIX_FINAL.xlsx")
            print(f"  📂 엑셀 거리 행렬 로드 중: {excel_path}")
            df_matrix = pd.read_excel(excel_path, sheet_name="Distance_Matrix", engine="openpyxl")

            # Use first column as index (hospital names)
            df_matrix_indexed = df_matrix.set_index(df_matrix.columns[0])  # Use first column as index

            # Build distance matrix by looking up values
            N = len(names_road)
            matrix = np.zeros((N, N))
            missing_hospitals = []

            for i in range(N):
                for j in range(N):
                    if i == j:
                        matrix[i][j] = 0
                    else:
                        hospital_i = names_road[i]
                        hospital_j = names_road[j]

                        # Look up distance from Excel matrix
                        if hospital_i in df_matrix_indexed.index and hospital_j in df_matrix_indexed.columns:
                            dist = df_matrix_indexed.loc[hospital_i, hospital_j]
                            matrix[i][j] = float(dist) if pd.notna(dist) else 0
                        else:
                            matrix[i][j] = 0
                            if hospital_i not in missing_hospitals:
                                missing_hospitals.append(hospital_i)
                            if hospital_j not in missing_hospitals:
                                missing_hospitals.append(hospital_j)

            if missing_hospitals:
                print(f"  ⚠️ 엑셀에서 찾지 못한 병원 ({len(missing_hospitals)}개): {missing_hospitals[:5]}...")

            save_path_road = os.path.join(save_folder, "distance_Hos2Hos_road.csv")
            pd.DataFrame(matrix).to_csv(save_path_road, index=True, encoding="utf-8-sig")
            print(f"  ✅ 병원간 도로 거리 행렬 생성 완료 (엑셀 데이터 사용)")
        except Exception as e:
            print(f"❌ 도로 거리 계산 실패: {e}")
        print(f"  ✅ 병원간 거리 행렬 생성 완료")

    def _sanitize_coeff_text(self, text: str) -> str:
        """'1.1,1' 또는 '[1.1, 1]' → '1.1, 1' 로 정리"""
        if not text:
            return "1,1"
        t = text.strip()
        if t.startswith("[") and t.endswith("]"):
            t = t[1:-1]
        parts = [p.strip() for p in t.split(",") if p.strip() != ""]
        if len(parts) != 2:
            return "1,1"
        # 숫자 검증 (실패 시 기본)
        try:
            a = float(parts[0]); b = float(parts[1])
        except Exception:
            return "1,1"
        return f"{a},{b}".replace(",", ", ")
    
    def make_config_yaml(self, latitude, longitude, incident_size, amb_velocity,
                         uav_velocity, total_samples, random_seed, save_folder, is_use_time=True,
                         amb_handover_time=0, uav_handover_time=0, duration_coeff=1.0):
        """Config YAML 파일 생성"""
        print(f"  ⚙️ Config YAML 생성 중...")
        folder_name = f"({latitude},{longitude})"
        config_filename = f"config_{folder_name}.yaml"
        config_path = os.path.join(save_folder, config_filename)
        relative_folder = f"./scenarios/{self.experiment_id}/{folder_name}"

        # departure_time 정보
        departure_time_field = ""
        if self.departure_time:
            departure_time_field = f'  departure_time: "{self.departure_time}" # API 조회 시각 (YYYYMMDDHHMM)\n'

        yaml_content = f"""#incident_info:
#  incident_size: {incident_size} # 사고 규모 (총 환자 수)
#  latitude: {latitude} # 위도
#  longitude: {longitude} # 경도
#  incident_type: null # 사고 타입 설정 가능하게 추후 확장

entity_info:
{departure_time_field}  patient:
    incident_size: {incident_size} # 사고 규모 (총 환자 수)
    latitude: {latitude} # 위도
    longitude: {longitude} # 경도
    incident_type: null # 사고 타입 설정 가능하게 추후 확장
    info_path: "{relative_folder}/patient_info.csv"
  hospital:
    load_data: True
    info_path: "{relative_folder}/hospital_info_road.csv"
    dist_Hos2Hos_euc_info: "{relative_folder}/distance_Hos2Hos_euc.csv"
    dist_Hos2Hos_road_info: "{relative_folder}/distance_Hos2Hos_road.csv"
    dist_Hos2Site_euc_info: "{relative_folder}/distance_Hos2Site_euc.csv"
    dist_Hos2Site_road_info: "{relative_folder}/distance_Hos2Site_road.csv"
    max_send_coeff: [{self._sanitize_coeff_text(self.max_send_coeff_text)}]
  ambulance:
    load_data: True
    dispatch_distance_info: "{relative_folder}/amb_info_road.csv"
    velocity: {amb_velocity} # unit: km/h
    handover_time: {amb_handover_time} # unit: minutes
    is_use_time: {('True' if is_use_time else 'False')} # True: API duration 사용, False: 거리/속도 기반 계산
    duration_coeff: {duration_coeff} # API duration 시간가중치 (기본값: 1.0, 환경적 요인 반영시 조정)
    road_provider: {self.road_provider or ('kakao' if is_use_time else 'osrm')} # 도로 데이터 공급자 (kakao | osrm) - 시나리오 생성 시 기록
  uav:
    load_data: True
    dispatch_distance_info: "{relative_folder}/uav_info.csv"
    velocity: {uav_velocity} # unit: km/h
    handover_time: {uav_handover_time} # unit: minutes
    is_use_time: False # UAV는 항상 유클리드 거리 기반

event_info_path: "./src/sim_src/event_info.json"

rule_info:
  isFullFactorial: True
  priority_rule: ["START", "ReSTART"]
  hos_select_rule: ["RedOnly", "YellowNearest"] # hos_select_rule: ["RedOnly", "YellowHalf"]
  red_mode_rule: ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]
  yellow_mode_rule: ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]

run_setting:
  totalSamples: {total_samples} # number of samples
  random_seed: {random_seed} # null, if do not want to fix
  rule_test: True
  eval_mode: True
  output_path: "./results/{self.experiment_id}"
  exp_indicator: "{folder_name}"
  save_info: True # NotImplemented"""
        with open(config_path, 'w', encoding='utf-8') as file:
            file.write(yaml_content)
        print(f"  ✅ Config YAML 생성 완료")
        absolute_config_path = os.path.abspath(config_path)
        print(f"CONFIG_PATH:{absolute_config_path}")
        return absolute_config_path

    def generate_scenario(self, latitude, longitude, incident_size, amb_count,
                          uav_count, amb_velocity, uav_velocity,
                          total_samples, random_seed, is_use_time=True,
                          amb_handover_time=0, uav_handover_time=0, duration_coeff=1.0):
        """
        완전한 시나리오 생성 (모든 CSV + YAML)
        Args:
            is_use_time: True면 API duration 사용, False면 거리/속도 기반 계산
            amb_handover_time: 구급차 환자 인계시간 (분)
            uav_handover_time: UAV 환자 인계시간 (분)
        Returns: 생성된 config 파일 경로
        """
        # 방어적 bool 변환 (혹시 호출 측에서 문자열을 넘겨도 정상 동작)
        is_use_time = bool(is_use_time) if not isinstance(is_use_time, str) else str2bool(is_use_time)

        # road_provider는 __init__에서 is_use_time을 보고 이미 결정되어 있다.
        # 다만 호출 시점 is_use_time과 __init__ 때 가정이 다르면 갱신하고 경고한다.
        expected_provider = "kakao" if is_use_time else "osrm"
        if expected_provider != self.road_provider:
            print(f"  ⚠️ road_provider mismatch: __init__={self.road_provider}, "
                  f"generate_scenario(is_use_time={is_use_time}) → {expected_provider}로 갱신")
            self.road_provider = expected_provider

        if self.road_provider == "kakao" and not self.kakao_api_key:
            raise RuntimeError(
                "is_use_time=True 모드는 --kakao_api_key가 필요합니다. "
                "키가 없다면 is_use_time=False로 OSRM 백엔드를 사용하세요."
            )
        print(f"  🛣️ 도로 데이터 공급자: {self.road_provider}"
              f"{' (' + self.osrm_url + ')' if self.road_provider == 'osrm' else ''}")

        print(f"""\n📍 좌표 ({latitude},{longitude}) 시나리오 생성 시작...""")
        start_time = time.time()
        folder_name = f"({latitude},{longitude})"
        save_folder = os.path.join(self.base_path, "scenarios", self.experiment_id, folder_name)
        os.makedirs(save_folder, exist_ok=True)

        # 이전 실행의 routes/ 폴더 정리 (재시도 시 JSON 누적 방지)
        import shutil as _shutil
        routes_cleanup = os.path.join(save_folder, "routes")
        if os.path.exists(routes_cleanup):
            _shutil.rmtree(routes_cleanup)

        # 역지오코딩은 orchestrator.py에서 수행하므로 간단한 정보만 출력
        coordinate_info = {
            "latitude": latitude,
            "longitude": longitude,
            "full_address": "",
            "road_address": "",
            "area1": "",
            "area2": "",
            "area3": "",
            "area4": "",
            "is_valid": False
        }
        print(f"COORDINATE_INFO:{json.dumps(coordinate_info, ensure_ascii=False)}")
        print(f"  📍 좌표: ({latitude}, {longitude}) - 역지오코딩은 orchestrator에서 수행")

        # 생성 파이프라인
        self.make_amb_info(latitude, longitude, incident_size, amb_count, save_folder)
        self.make_hospital_info(latitude, longitude, incident_size, save_folder, uav_count)
        self.make_uav_info(latitude, longitude, incident_size, uav_count, save_folder)
        self.make_patient_info(save_folder)
        self.make_distance_Hos2Hos(save_folder)
        config_path = self.make_config_yaml(
            latitude, longitude, incident_size,
            amb_velocity, uav_velocity, total_samples,
            random_seed, save_folder, is_use_time,
            amb_handover_time, uav_handover_time, duration_coeff
        )
        
        elapsed = round(time.time() - start_time, 2)
        print(f"  ⏱️ 시나리오 생성 완료 ({elapsed}초)")
        print(f"API_CALL_COUNT:{self.api_call_count}")
        print(f"CONFIG_PATH:{config_path}")
        return config_path

# CLI 실행용
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCI 시나리오 동적 생성 (크로스 환경 호환)")
    parser.add_argument("--base_path", required=True, help="프로젝트 루트 경로")
    parser.add_argument("--latitude", type=float, required=False, help="위도")
    parser.add_argument("--longitude", type=float, required=False, help="경도")
    parser.add_argument("--incident_size", type=int, default=100, help="환자 수")
    parser.add_argument("--amb_count", type=int, default=30, help="구급차 수")
    parser.add_argument("--uav_count", type=int, default=25, help="UAV 수")
    parser.add_argument("--amb_velocity", type=int, default=40, help="구급차 속도")
    parser.add_argument("--uav_velocity", type=int, default=80, help="UAV 속도")
    parser.add_argument("--total_samples", type=int, default=1000, help="시뮬레이션 반복 수")
    parser.add_argument("--random_seed", type=int, default=0, help="랜덤 시드")
    parser.add_argument("--experiment_id", type=str, default=None, help="실험 ID")
    # 고급 옵션(ENV 또는 CLI 둘 다 허용)
    # parser.add_argument("--queue_policy", type=str, help='예: "0", "capa/2", "0.5"')
    parser.add_argument("--buffer_ratio", type=float, help="후보군 버퍼 배수 (기본 1.5)")
    parser.add_argument("--util_by_tier", type=str, help='예: "1:0.90,11:0.75,etc:0.60"')
    parser.add_argument("--hospital_max_send_coeff", type=str, default=None, help="전송계수 'a,b' 형식 (예: 1.1,1.0). 미입력시 ENV(MCI_MAX_SEND_COEFF) 또는 기본 1,1")

    # 카카오 API 관련 파라미터
    parser.add_argument("--kakao_api_key", type=str, default=None, help="카카오 모빌리티 REST API 키 (is_use_time=true 모드 필수)")
    parser.add_argument("--departure_time", type=str, default=None, help="출발시간 (YYYYMMDDHHMM 형식, 예: 202512241800)")
    parser.add_argument("--is_use_time", type=str2bool, default=False,
                        help="True: 카카오 API duration 기반 / False: OSRM 정적 거리 기반(distance/velocity). 시뮬 재실행 시에도 OSRM duration을 활용 가능")
    # OSRM 백엔드 (is_use_time=false일 때 사용)
    parser.add_argument("--osrm_url", type=str, default=None,
                        help="OSRM HTTP API base URL (기본: env MCI_OSRM_URL 또는 https://router.project-osrm.org)")
    parser.add_argument("--amb_handover_time", type=float, default=10.0, help="구급차 환자 인계시간 (분)")
    parser.add_argument("--uav_handover_time", type=float, default=15.0, help="UAV 환자 인계시간 (분)")
    parser.add_argument("--duration_coeff", type=float, default=1.0, help="API duration 시간가중치 (기본값: 1.0)")
    parser.add_argument("--fixed_hos_num", type=int, default=None,
                        help="모든 좌표에서 hos_num 고정 (RL 학습/평가 obs 차원 일치). 보장 룰 후 가까운 N개로 cap. 미지정 시 buffer_ratio 동적 결정")

    args = parser.parse_args()
    try:
        # UTF-8 출력 설정
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

    try:
        # args.is_use_time는 이미 str2bool로 파싱되어 bool
        generator = ScenarioGenerator(
            args.base_path,
            args.experiment_id,
            kakao_api_key=args.kakao_api_key,
            departure_time=args.departure_time,
            osrm_url=args.osrm_url,
            is_use_time=args.is_use_time,
            fixed_hos_num=args.fixed_hos_num,
        )

        # CLI가 주어지면 ENV 기본값을 덮어씀
        if args.hospital_max_send_coeff:
            generator.max_send_coeff_text = args.hospital_max_send_coeff
        # if args.queue_policy is not None:
        #     generator.queue_policy = args.queue_policy
        if args.buffer_ratio is not None:
            generator.buffer_ratio = float(args.buffer_ratio)
        if args.util_by_tier:
            m = parse_util_map(args.util_by_tier)
            if m:
                generator.util_by_tier = m

        # 현재 적용값 재출력
        print(f"buffer_ratio={generator.buffer_ratio}")

        # 현재 프로젝트 흐름에서는 사고 좌표를 외부에서 직접 전달한다.
        if args.latitude is None or args.longitude is None:
            print("❌ --latitude, --longitude 인자가 필요합니다.")
            sys.exit(1)
        latitude, longitude = args.latitude, args.longitude
        
        # 시나리오 생성
        config_path = generator.generate_scenario(
            latitude, longitude,
            args.incident_size, args.amb_count, args.uav_count,
            args.amb_velocity, args.uav_velocity,
            args.total_samples, args.random_seed,
            is_use_time=args.is_use_time,
            amb_handover_time=args.amb_handover_time,
            uav_handover_time=args.uav_handover_time,
            duration_coeff=args.duration_coeff
        )
        
        if config_path:
            print(f"\n✅ 시나리오 생성 성공!")
            print(f"📄 Config 파일: {config_path}")
        else:
            print("❌ 시나리오 생성 실패")
            sys.exit(1)
            
    except Exception as e:
        print(f"💥 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
