# -*- coding: utf-8 -*-
"""UAV 전용 시나리오 생성기.

설계 단순화 (v2):
- 원본 엑셀에서 '헬기장 여부 == 1' 인 모든 병원을 master pool 로 사용
- 모든 거리 = haversine 직선거리, road CSV 는 헤더만 (시뮬에서 euc 폴백)
- uav_info = master pool 중 가장 가까운 uav_count 개

CLI 예:
    python src/sce_src/make_uav_scenario.py \
        --base_path . --experiment_id helipad_center \
        --latitude 36.245107096 --longitude 127.462534992 \
        --incident_size 30 --uav_count 3 --rule_test
"""
import argparse
import math
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from haversine import haversine


PATIENT_CONFIG = {
    "ratio":            {"Red": 0.1, "Yellow": 0.3, "Green": 0.5, "Black": 0.1},
    "rescue_param":     {"Red": (6, 5), "Yellow": (2, 13), "Green": (1, 22), "Black": (0, 0)},
    "treat_tier3":      {"Red": True,  "Yellow": True, "Green": True, "Black": True},
    "treat_tier2":      {"Red": False, "Yellow": True, "Green": True, "Black": True},
    "treat_tier3_mean": {"Red": 40,    "Yellow": 20,   "Green": 10,   "Black": 0},
    "treat_tier2_mean": {"Red": float("inf"), "Yellow": 30, "Green": 15, "Black": 0},
}

UTIL_BY_TIER = {1: 0.656, 11: 0.461, "etc": 0.461}
# max_send = a*수술실수 + b*병상수.  현재: 수술실수는 그대로 + 병상수 절반
MAX_SEND_COEFF_TEXT = "1, 0.5"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _get_util(code) -> float:
    try:
        return UTIL_BY_TIER.get(int(code), UTIL_BY_TIER["etc"])
    except Exception:
        return UTIL_BY_TIER["etc"]


class UAVScenarioGenerator:
    def __init__(self, base_path: str, experiment_id: str, hospital_data_path: str):
        self.base_path = os.path.abspath(base_path)
        self.hospital_data_path = os.path.abspath(hospital_data_path)
        if not os.path.exists(self.hospital_data_path):
            raise FileNotFoundError(f"병원 데이터 파일이 없습니다: {self.hospital_data_path}")

        base_id = (experiment_id or datetime.now().strftime("%Y%m%d%H%M")).strip()
        if base_id.startswith("exp_"):
            base_id = base_id[4:]
        base_id = re.sub(r"\s+", "_", base_id).strip("_") or datetime.now().strftime("%Y%m%d%H%M")
        self.experiment_id = f"exp_{base_id}_uav"

        print(f"📁 base path: {self.base_path}")
        print(f"📁 hospital data: {self.hospital_data_path}")
        print(f"🆔 experiment_id: {self.experiment_id}")

    # ---------- AMB 비활성: 빈 CSV (0바이트) ----------
    def make_amb_info_empty(self, save_folder: str):
        path = os.path.join(save_folder, "amb_info_road.csv")
        open(path, "w", encoding="utf-8-sig").close()
        print(f"  🚑 빈 amb_info_road.csv 생성 (0바이트)")

    # ---------- 병원 풀: 헬기장만 ----------
    def make_hospital_info(self, latitude: float, longitude: float, save_folder: str) -> pd.DataFrame:
        print("  🏥 헬기장 병원 풀 추출 중 (haversine 직선거리)...")
        df_full = pd.read_excel(self.hospital_data_path, engine="openpyxl")

        cols_needed = ["요양기관명", "종별코드", "응급실병상수", "x좌표", "y좌표", "헬기장 여부"]
        for c in cols_needed:
            if c not in df_full.columns:
                raise KeyError(f"필수 컬럼 누락: {c}")

        # 헬기장만 추출
        df = df_full[df_full["헬기장 여부"] == 1].copy().reset_index(drop=True)
        if df.empty:
            raise ValueError("엑셀에 헬기장 병원이 없습니다.")
        df["헬기장 여부"] = df["헬기장 여부"].astype(int)
        print(f"  헬기장 병원 풀: {len(df)}개")

        # 사고지점→병원 직선거리
        coords = list(zip(df["y좌표"], df["x좌표"]))  # (lat, lon)
        df["euclidean_distance"] = [haversine(c, (latitude, longitude)) for c in coords]

        # 용량/등급 (UTIL 가동률 반영, MCI_ADV 와 동일 공식)
        df["util"] = df["종별코드"].apply(_get_util)
        df["capa"] = (df["응급실병상수"] * (1 - df["util"])).apply(
            lambda x: int(max(0, math.floor(x)))
        )
        conditions = [df["종별코드"] == 1, df["종별코드"] == 11]
        df["operating_rooms"] = np.select(conditions, [3, 2], default=1)
        df["eff"] = df["operating_rooms"] + df["capa"]
        df["is_tier3"] = (df["종별코드"].astype(int) == 1).astype(int)

        # 사고지점 기준 직선거리순 정렬 (인덱스 = 거리 순서)
        df = df.sort_values("euclidean_distance").reset_index(drop=True)

        n_t3 = int(df["is_tier3"].sum())
        n_t2 = len(df) - n_t3
        print(f"  최종 병원 {len(df)}개 (Tier3: {n_t3}, Tier2: {n_t2})")

        # ----- 저장 (sim_src 가 요구하는 파일들) -----
        # 명명 규칙: _road = 0바이트(UAV 전용이라 미사용), _euc = 실제 데이터
        info_df = df[["operating_rooms", "capa", "종별코드", "요양기관명", "헬기장 여부"]].copy()
        info_df.columns = ["수술실수", "병상수", "종별코드", "요양기관명", "헬기장 여부"]
        info_df.to_csv(os.path.join(save_folder, "hospital_info_euc.csv"),
                       index=True, index_label="Index", encoding="utf-8-sig")

        # 사고지점-병원 직선거리 (euc)
        pd.DataFrame({"distance": df["euclidean_distance"]}).to_csv(
            os.path.join(save_folder, "distance_Hos2Site_euc.csv"),
            index=True, index_label="Index", encoding="utf-8-sig")

        # 병원-병원 직선거리 행렬 (euc)
        H = len(df)
        coords_arr = list(zip(df["y좌표"], df["x좌표"]))
        mat = np.zeros((H, H), dtype=np.float32)
        for i in range(H):
            for j in range(i + 1, H):
                d = haversine(coords_arr[i], coords_arr[j])
                mat[i, j] = d
                mat[j, i] = d
        pd.DataFrame(mat).to_csv(os.path.join(save_folder, "distance_Hos2Hos_euc.csv"),
                                 index=True, encoding="utf-8-sig")

        # ----- road CSV: 0바이트 placeholder (UAV 전용이라 미사용; 시뮬에서 euc 폴백) -----
        for fname in ("distance_Hos2Site_road.csv",
                      "distance_Hos2Hos_road.csv",
                      "hospital_info_road.csv"):
            open(os.path.join(save_folder, fname), "w", encoding="utf-8-sig").close()

        # uav_info 생성에 좌표가 필요해서 임시 저장
        coord_path = os.path.join(save_folder, "_hospital_coords.csv")
        df[["요양기관명", "x좌표", "y좌표"]].to_csv(coord_path, index=False, encoding="utf-8-sig")

        print("  ✅ 병원/거리 파일 생성 완료 (_road 시리즈는 0바이트, 실 데이터는 _euc)")
        return df

    # ---------- UAV 정보 ----------
    def make_uav_info(self, latitude: float, longitude: float,
                      uav_count: int, save_folder: str):
        print(f"  🚁 UAV 정보 생성 중 (UAV {uav_count}대 / 헬기장 풀에서 가까운 순)...")
        path_uav = os.path.join(save_folder, "uav_info.csv")

        if uav_count <= 0:
            pd.DataFrame(columns=["uav_id", "hospital_idx", "init_distance",
                                  "수술실수", "병상수", "종별코드", "요양기관명"]).to_csv(
                path_uav, index=False, encoding="utf-8-sig")
            print("  빈 uav_info.csv 생성")
            return

        df_hos = pd.read_csv(os.path.join(save_folder, "hospital_info_euc.csv"), encoding="utf-8-sig")
        if len(df_hos) < uav_count:
            raise ValueError(f"헬기장 병원 {len(df_hos)}개 < uav_count={uav_count}")

        coords = pd.read_csv(os.path.join(save_folder, "_hospital_coords.csv"), encoding="utf-8-sig")
        df_hos = df_hos.merge(coords, on="요양기관명", how="left")
        df_hos["distance"] = df_hos.apply(
            lambda r: haversine((r["y좌표"], r["x좌표"]), (latitude, longitude)), axis=1)
        # hospital_info_road 가 이미 거리순으로 정렬되어 있어 distance 도 단조증가지만 안전하게 재정렬
        df_hos = df_hos.sort_values("distance").reset_index(drop=True)

        sel = df_hos.head(uav_count).copy()
        out = pd.DataFrame({
            "uav_id": range(len(sel)),
            "hospital_idx": sel["Index"].astype(int),
            "init_distance": sel["distance"].round(3),
            "수술실수": sel["수술실수"],
            "병상수": sel["병상수"],
            "종별코드": sel["종별코드"],
            "요양기관명": sel["요양기관명"],
        })
        out.to_csv(path_uav, index=False, encoding="utf-8-sig")
        os.remove(os.path.join(save_folder, "_hospital_coords.csv"))
        names = ", ".join(sel["요양기관명"].head(min(3, len(sel))).tolist())
        print(f"  ✅ UAV {len(out)}대 → {names}{'...' if len(sel) > 3 else ''}")

    # ---------- 환자 ----------
    def make_patient_info(self, save_folder: str):
        rows = []
        for t in PATIENT_CONFIG["ratio"].keys():
            a, b = PATIENT_CONFIG["rescue_param"][t]
            rows.append({
                "type": t,
                "ratio": PATIENT_CONFIG["ratio"][t],
                "rescue_param_alpha": a,
                "rescue_param_beta": b,
                "treat_tier3": PATIENT_CONFIG["treat_tier3"][t],
                "treat_tier2": PATIENT_CONFIG["treat_tier2"][t],
                "treat_tier3_mean": PATIENT_CONFIG["treat_tier3_mean"][t],
                "treat_tier2_mean": PATIENT_CONFIG["treat_tier2_mean"][t],
            })
        pd.DataFrame(rows).to_csv(os.path.join(save_folder, "patient_info.csv"),
                                  index=False, encoding="utf-8-sig")
        print("  ✅ patient_info.csv 생성")

    # ---------- YAML ----------
    def make_config_yaml(self, latitude, longitude, incident_size,
                         uav_count, uav_velocity, uav_handover_time,
                         total_samples, random_seed, save_folder,
                         rule_test: bool = False) -> str:
        folder_name = f"({latitude},{longitude})"
        relative_folder = f"./scenarios/{self.experiment_id}/{folder_name}"
        config_path = os.path.join(save_folder, f"config_{folder_name}.yaml")

        if rule_test:
            rule_block = (
                "rule_info:\n"
                "  isFullFactorial: False\n"
                '  priority_rule: ["START", "ReSTART"]\n'
                '  hos_select_rule: ["RedOnly", "YellowNearest"]\n'
                '  red_mode_rule: ["OnlyUAV"]\n'
                '  yellow_mode_rule: ["OnlyUAV"]\n'
            )
        else:
            rule_block = (
                "rule_info:\n"
                "  isFullFactorial: False\n"
                '  priority_rule: ["START"]\n'
                '  hos_select_rule: ["RedOnly"]\n'
                '  red_mode_rule: ["OnlyUAV"]\n'
                '  yellow_mode_rule: ["OnlyUAV"]\n'
            )

        yaml_content = f"""# UAV-only scenario (auto-generated, helipad-pool only)
entity_info:
  patient:
    incident_size: {incident_size}
    latitude: {latitude}
    longitude: {longitude}
    incident_type: null
    info_path: "{relative_folder}/patient_info.csv"
  hospital:
    load_data: True
    info_path: "{relative_folder}/hospital_info_euc.csv"
    dist_Hos2Hos_euc_info: "{relative_folder}/distance_Hos2Hos_euc.csv"
    dist_Hos2Hos_road_info: "{relative_folder}/distance_Hos2Hos_road.csv"
    dist_Hos2Site_euc_info: "{relative_folder}/distance_Hos2Site_euc.csv"
    dist_Hos2Site_road_info: "{relative_folder}/distance_Hos2Site_road.csv"
    max_send_coeff: [{MAX_SEND_COEFF_TEXT}]   # [수술실수 가중치, 병상수 가중치] -> hos_max_send = a*수술실수 + b*병상수
  ambulance:
    load_data: True
    dispatch_distance_info: "{relative_folder}/amb_info_road.csv"
    velocity: 40
    handover_time: 10.0
    is_use_time: False
    duration_coeff: 1.0
    road_provider: none
  uav:
    load_data: True
    dispatch_distance_info: "{relative_folder}/uav_info.csv"
    velocity: {uav_velocity}
    handover_time: {uav_handover_time}
    is_use_time: False

event_info_path: "./src/sim_src/event_info.json"

{rule_block}
run_setting:
  totalSamples: {total_samples}
  random_seed: {random_seed}
  rule_test: {rule_test}
  eval_mode: True
  output_path: "./results/{self.experiment_id}"
  exp_indicator: "{folder_name}"
  save_info: True
"""
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        abspath = os.path.abspath(config_path)
        print(f"  ✅ config YAML: {abspath}")
        return abspath

    # ---------- 통합 ----------
    def generate(self, latitude: float, longitude: float,
                 incident_size: int, uav_count: int,
                 uav_velocity: int, uav_handover_time: float,
                 total_samples: int, random_seed: int,
                 rule_test: bool = False) -> str:
        t0 = time.time()
        folder_name = f"({latitude},{longitude})"
        save_folder = os.path.join(self.base_path, "scenarios", self.experiment_id, folder_name)
        ensure_dir(save_folder)

        self.make_amb_info_empty(save_folder)
        self.make_hospital_info(latitude, longitude, save_folder)
        self.make_uav_info(latitude, longitude, uav_count, save_folder)
        self.make_patient_info(save_folder)
        cfg = self.make_config_yaml(
            latitude, longitude, incident_size,
            uav_count, uav_velocity, uav_handover_time,
            total_samples, random_seed, save_folder, rule_test=rule_test)

        print(f"\n⏱️ 생성 완료 ({round(time.time() - t0, 2)}초)")
        print(f"CONFIG_PATH:{cfg}")
        return cfg


def parse_args():
    p = argparse.ArgumentParser(description="UAV 전용 시나리오 생성기 (헬기장 풀, haversine only)")
    p.add_argument("--base_path", default=".", help="MCI_UAV 레포 루트")
    p.add_argument("--experiment_id", default=None, help="실험 ID (없으면 타임스탬프)")
    p.add_argument("--hospital_data", default=None,
                   help="병원 엑셀 경로. 기본: <base_path>/scenarios/엑셀 결합 데이터.xlsx")
    p.add_argument("--latitude", type=float, required=True)
    p.add_argument("--longitude", type=float, required=True)
    p.add_argument("--incident_size", type=int, default=30)
    p.add_argument("--uav_count", type=int, default=3)
    p.add_argument("--uav_velocity", type=int, default=80, help="km/h")
    p.add_argument("--uav_handover_time", type=float, default=15.0, help="minutes")
    p.add_argument("--total_samples", type=int, default=1000)
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--rule_test", action="store_true",
                   help="설정 시 START/ReSTART × RedOnly/YellowNearest 4개 룰 활성화")
    return p.parse_args()


class _Tee:
    """stdout 을 콘솔과 파일에 동시 기록."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except UnicodeEncodeError:
                st.write(s.encode('ascii', errors='replace').decode('ascii'))
    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    base = os.path.abspath(args.base_path)
    hospital_data = args.hospital_data or os.path.join(base, "scenarios", "엑셀 결합 데이터.xlsx")

    # ----- 시나리오 생성 로그 파일 준비 -----
    log_handle = None
    log_dir = os.path.join(base, "experiment_logs")
    os.makedirs(log_dir, exist_ok=True)
    coord_str = f"({args.latitude},{args.longitude})"
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_id_for_filename = (args.experiment_id or "auto").strip()
    log_path = os.path.join(log_dir, f"scenario_{exp_id_for_filename}_{coord_str}_{ts}.log")
    try:
        log_handle = open(log_path, 'w', encoding='utf-8')
        sys.stdout = _Tee(sys.__stdout__, log_handle)
        print(f"[LOG] scenario log -> {log_path}")
    except Exception as e:
        print(f"[LOG] 로그 파일 설정 실패 (콘솔만 출력): {e}")
        log_handle = None

    try:
        gen = UAVScenarioGenerator(base, args.experiment_id, hospital_data)
        gen.generate(
            latitude=args.latitude, longitude=args.longitude,
            incident_size=args.incident_size, uav_count=args.uav_count,
            uav_velocity=args.uav_velocity, uav_handover_time=args.uav_handover_time,
            total_samples=args.total_samples, random_seed=args.random_seed,
            rule_test=args.rule_test,
        )
    finally:
        if log_handle is not None:
            sys.stdout = sys.__stdout__
            log_handle.close()


if __name__ == "__main__":
    main()
