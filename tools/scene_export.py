# -*- coding: utf-8 -*-
"""시나리오 폴더 → Unity 디지털트윈용 컴팩트 scene.json.

카카오 route JSON(중첩 구조)과 CSV를 평탄화해서, Unity C# 로더가
좌표/경로/엔티티만 쉽게 읽도록 단일 scene.json 으로 내보낸다.

사용:
  python scene_export.py --scenario "Y:/scenarios/exp_plan1_대전_dep_202605261530"
  # 좌표 서브폴더 자동 탐색. 여러 개면 --coord "(36.3505,127.3845)" 로 지정.
  # 출력: <coord 폴더>/scene.json  (또는 --out 지정)

필요 env: 표준 라이브러리만 (yaml 있으면 사용, 없으면 정규식 폴백)
"""
from __future__ import annotations
import argparse, csv, glob, json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_yaml_min(path):
    """config yaml 에서 site lat/lon, incident_size, departure_time 추출 (yaml 없어도 동작)."""
    txt = open(path, encoding="utf-8").read()
    def grab(key):
        m = re.search(rf"{key}\s*:\s*([-\d.]+)", txt)
        return float(m.group(1)) if m else None
    dep = re.search(r'departure_time\s*:\s*"?(\d+)"?', txt)
    return {
        "latitude": grab("latitude"),
        "longitude": grab("longitude"),
        "incident_size": int(grab("incident_size")) if grab("incident_size") else None,
        "departure_time": dep.group(1) if dep else None,
    }


def read_csv_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pick(coord_dir, *names):
    """후보 파일명 중 먼저 존재하는 경로 반환(신규 포맷 우선, 구 포맷 폴백)."""
    for n in names:
        p = os.path.join(coord_dir, n)
        if os.path.exists(p):
            return p
    return os.path.join(coord_dir, names[0])


def col(row, *names):
    """여러 후보 컬럼명 중 존재하는 값 반환."""
    for n in names:
        if n in row and row[n] != "":
            return row[n]
    return None


def flatten_route(path):
    """카카오 route JSON → {idx, name, origin[lat,lon], dist_km, dur_min, path[[lat,lon],...], states[...]}.

    states: 각 정점이 속한 도로 구간의 traffic_state(0~5, 혼잡도) — path와 동일 길이. Unity NPC 교통량/속도 변조용.
    """
    d = json.load(open(path, encoding="utf-8"))
    meta = d.get("meta", {})
    origin = meta.get("center") or meta.get("hospital")  # [lon,lat]
    pts = []
    states = []
    try:
        routes = d["payload"]["kakao_response"]["routes"]
        for r in routes:
            for sec in r.get("sections", []):
                for road in sec.get("roads", []):
                    v = road.get("vertexes", [])
                    ts = road.get("traffic_state", 0)
                    try:
                        ts = int(ts)
                    except (TypeError, ValueError):
                        ts = 0
                    for i in range(0, len(v) - 1, 2):
                        lon, lat = v[i], v[i + 1]
                        pts.append([round(lat, 7), round(lon, 7)])
                        states.append(ts)
    except (KeyError, IndexError, TypeError):
        pass
    return {
        "idx": meta.get("source_index"),
        "name": meta.get("name"),
        "origin": [origin[1], origin[0]] if origin else None,  # [lat,lon]
        "dist_km": meta.get("distance_km"),
        "dur_min": meta.get("duration_min"),
        "path": pts,
        "states": states,
    }


def _flat_route(r):
    """{idx, path[[lat,lon]], states} → {idx, pts:[lat,lon,...], states:[...]} (JsonUtility 친화)."""
    pts = []
    for lat, lon in r["path"]:
        pts.append(lat); pts.append(lon)
    return {"idx": r["idx"], "pts": pts, "states": r.get("states", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, help="시나리오 폴더 (exp_*) 또는 좌표 폴더")
    ap.add_argument("--coord", default=None, help="좌표 서브폴더명 예:(36.3505,127.3845)")
    ap.add_argument("--out", default=None, help="출력 scene.json 경로 (기본=좌표폴더/scene.json)")
    args = ap.parse_args()

    root = args.scenario
    # 좌표 폴더 결정
    if os.path.exists(os.path.join(root, "config_" + (args.coord or "") + ".yaml")) or \
       glob.glob(os.path.join(root, "config_*.yaml")):
        coord_dir = root
    else:
        subs = [d for d in glob.glob(os.path.join(root, "(*)")) if os.path.isdir(d)]
        if args.coord:
            coord_dir = os.path.join(root, args.coord)
        elif len(subs) == 1:
            coord_dir = subs[0]
        else:
            print(f"[ERROR] 좌표 폴더 여러개 → --coord 지정 필요: {[os.path.basename(s) for s in subs]}")
            sys.exit(2)
    coord_name = os.path.basename(coord_dir.rstrip("/\\"))
    exp_id = os.path.basename(os.path.dirname(coord_dir.rstrip("/\\")))

    cfgs = glob.glob(os.path.join(coord_dir, "config_*.yaml"))
    cfg = read_yaml_min(cfgs[0]) if cfgs else {}
    site = {"lat": cfg.get("latitude"), "lon": cfg.get("longitude")}
    inc = cfg.get("incident_size")

    # 환자 비율 → 수 (JsonUtility 친화: 리스트)
    patients = []
    pinfo = os.path.join(coord_dir, "patient_info.csv")
    if os.path.exists(pinfo) and inc:
        for row in read_csv_rows(pinfo):
            t = col(row, "type"); ratio = col(row, "ratio")
            if t and ratio is not None:
                patients.append({"type": t, "count": int(round(float(ratio) * inc))})

    # 경로 로드 (좌표 출처)
    def load_routes(sub):
        out = []
        for fp in sorted(glob.glob(os.path.join(coord_dir, "routes", sub, "*.json"))):
            out.append(flatten_route(fp))
        return out
    center_routes = load_routes("center2site")
    hos_routes = load_routes("hos2site")
    hos_origin = {r["idx"]: r["origin"] for r in hos_routes if r["idx"] is not None}
    amb_origin = {r["idx"]: r["origin"] for r in center_routes if r["idx"] is not None}

    # 병원 (Phase 1: 통합 hospital_info.csv, 구 이름/구 포맷 폴백)
    hospitals = []
    hpath = pick(coord_dir, "hospital_info.csv", "hospitals.csv", "hospital_info_road.csv")
    if os.path.exists(hpath):
        for row in read_csv_rows(hpath):
            idx = int(col(row, "Index", "index") or len(hospitals))
            o = hos_origin.get(idx)
            hospitals.append({
                "idx": idx,
                "name": col(row, "요양기관명", "name"),
                "lat": o[0] if o else None, "lon": o[1] if o else None,
                "beds": int(float(col(row, "병상수") or 0)),
                "ors": int(float(col(row, "수술실수") or 0)),
                "helipad": str(col(row, "헬기장 여부", "헬기장여부") or "0").strip() in ("1", "True", "true"),
            })

    # AMB 기지 (Phase 1: amb_station_info.csv 고유 센터당 1행, 구 이름/구 포맷 폴백)
    amb_bases = []
    apath = pick(coord_dir, "amb_station_info.csv", "amb_bases.csv", "amb_info_road.csv")
    if os.path.exists(apath):
        for row in read_csv_rows(apath):
            idx = int(col(row, "Index", "index") or len(amb_bases))
            o = amb_origin.get(idx)
            amb_bases.append({
                "idx": idx,
                "name": col(row, "안전센터/소방서이름", "name"),
                "lat": o[0] if o else None, "lon": o[1] if o else None,
                "count": int(float(col(row, "보유대수") or 1)),
                "duration_min": float(col(row, "duration") or 0),
            })

    # UAV (Phase 1: uav_info.csv superset)
    uavs = []
    upath = pick(coord_dir, "uav_info.csv", "uav.csv")
    if os.path.exists(upath):
        for row in read_csv_rows(upath):
            uavs.append({
                "uav_id": int(float(col(row, "uav_id") or len(uavs))),
                "hospital_idx": int(float(col(row, "hospital_idx") or 0)),
            })

    scene = {
        "exp_id": exp_id, "coord": coord_name,
        "site": site, "incident_size": inc, "departure_time": cfg.get("departure_time"),
        "patients": patients,
        "hospitals": hospitals,
        "amb_bases": amb_bases,
        "uavs": uavs,
        "routes": {
            "center2site": [_flat_route(r) for r in center_routes],
            "hos2site": [_flat_route(r) for r in hos_routes],
        },
    }

    out = args.out or os.path.join(coord_dir, "scene.json")
    json.dump(scene, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[OK] {out}")
    print(f"  site={site} incident={inc} patients={patients}")
    print(f"  hospitals={len(hospitals)} (좌표있음 {sum(1 for h in hospitals if h['lat'])}, 헬기장 {sum(1 for h in hospitals if h['helipad'])})")
    print(f"  amb_bases={len(amb_bases)} uavs={len(uavs)}")
    print(f"  routes: center2site={len(center_routes)} hos2site={len(hos_routes)}  "
          f"(평균 점수 {sum(len(r['path']) for r in center_routes)//max(1,len(center_routes))}/"
          f"{sum(len(r['path']) for r in hos_routes)//max(1,len(hos_routes))})")


if __name__ == "__main__":
    main()
