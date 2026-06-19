"""
Seoul MCI 파일럿 — Unity scene_data.json 빌더.

입력:
  tools/seoul_pilot/_work/scenario/(<lat>,<lon>)/
    hospitals.csv, amb_bases.csv, uav.csv, patient_info.csv,  # (구 *_road.csv/uav_info.csv 폴백)
    routes/hos2site/<idx>_<hospital_name>.json   (direction: site->hospital)
    routes/center2site/<idx>_<center_name>.json  (direction: center->site)
  tools/seoul_pilot/_work/trace/trace_*.json

출력:
  external/ml-agents/UAV_test/Assets/StreamingAssets/SeoulPilot/
    scene_data.json
    trace.json   (그대로 복사)

좌표:
  EPSG:4326(lon,lat) → EPSG:5186(E,N) → Seoul_Ortho_Tiles 로컬 (x, z) = (E - aX, N - aY)
  Seoul_Ortho_Tiles 자체의 worldPos = (41684.9766, 0, 0). 빌더는 그 안의 로컬 좌표만 출력.
"""
from __future__ import annotations

import csv
import json
import math
import re
import shutil
from pathlib import Path

from pyproj import Transformer


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "_work"
SCENARIO_DIR = WORK / "scenario" / "(37.5666,126.9784)"
TRACE_DIR = WORK / "trace"

ASSETS = ROOT.parent.parent / "external" / "ml-agents" / "UAV_test" / "Assets"
OUT_DIR = ASSETS / "StreamingAssets" / "SeoulPilot"
ANCHOR_JSON = ASSETS / "GIS_Seoul" / "anchor.json"

SEOUL_ROOT_WORLD_X = 41684.9766  # 문서화 용도. Unity 측은 parent transform 으로 처리.
CRUISE_ALT = 150.0
SIM_TIME_SCALE = 30.0


def load_anchor() -> tuple[float, float]:
    j = json.loads(ANCHOR_JSON.read_text(encoding="utf-8"))
    return float(j["epsgAnchorX"]), float(j["epsgAnchorY"])


def make_transformer():
    return Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def norm_name(s: str) -> str:
    s = s.replace(" ", "").replace("_", "").replace("-", "")
    return s.casefold()


def find_route(route_dir: Path, target_name: str) -> Path | None:
    target = norm_name(target_name)
    candidates = []
    for p in sorted(route_dir.glob("*.json")):
        stem = p.stem
        m = re.match(r"\d+_(.+)$", stem)
        body = m.group(1) if m else stem
        if norm_name(body) == target:
            return p
        candidates.append((stem, body))
    # token fallback — 첫 토큰(소방서명) 또는 마지막 토큰(안전센터명) 부분일치
    for p in sorted(route_dir.glob("*.json")):
        stem = p.stem
        m = re.match(r"\d+_(.+)$", stem)
        body = norm_name(m.group(1) if m else stem)
        if body in target or target in body:
            return p
    return None


def route_coords(path: Path) -> list[list[float]]:
    """OSRM GeoJSON LineString → [[lon, lat], ...]"""
    d = json.loads(path.read_text(encoding="utf-8"))
    geo = d["payload"]["osrm_response"]["routes"][0]["geometry"]
    return list(geo["coordinates"])


def to_local(lon: float, lat: float, tfm, ax: float, ay: float) -> tuple[float, float]:
    e, n = tfm.transform(lon, lat)
    return (e - ax, n - ay)


def main() -> int:
    if not SCENARIO_DIR.exists():
        raise SystemExit(f"scenario not found: {SCENARIO_DIR}")
    if not ANCHOR_JSON.exists():
        raise SystemExit(f"anchor.json missing: {ANCHOR_JSON}")

    ax, ay = load_anchor()
    tfm = make_transformer()
    print(f"[anchor] EPSG:5186 = ({ax:.3f}, {ay:.3f})")

    # Phase 1: 통합/카운트 포맷 우선, 구 포맷 폴백
    def _pick(*names):
        for n in names:
            if (SCENARIO_DIR / n).exists():
                return SCENARIO_DIR / n
        return SCENARIO_DIR / names[0]
    hospitals = read_csv(_pick("hospitals.csv", "hospital_info_road.csv"))
    ambs = read_csv(_pick("amb_bases.csv", "amb_info_road.csv"))
    uavs = read_csv(_pick("uav.csv", "uav_info.csv"))

    h2s = SCENARIO_DIR / "routes" / "hos2site"
    c2s = SCENARIO_DIR / "routes" / "center2site"
    if not h2s.exists() or not c2s.exists():
        raise SystemExit(f"routes folders missing under {SCENARIO_DIR/'routes'}")

    # ---- 사고현장 좌표: routes 의 site endpoint 평균
    site_endpoints: list[tuple[float, float]] = []
    for p in sorted(h2s.glob("*.json")):
        # hos2site: 첫 점=site, 마지막 점=hospital
        coords = route_coords(p)
        site_endpoints.append((coords[0][0], coords[0][1]))
    for p in sorted(c2s.glob("*.json")):
        # center2site: 마지막 점=site
        coords = route_coords(p)
        site_endpoints.append((coords[-1][0], coords[-1][1]))

    if not site_endpoints:
        raise SystemExit("no route endpoints to derive site coordinate")
    s_lon = sum(p[0] for p in site_endpoints) / len(site_endpoints)
    s_lat = sum(p[1] for p in site_endpoints) / len(site_endpoints)
    var_lon = sum((p[0] - s_lon) ** 2 for p in site_endpoints) / len(site_endpoints)
    var_lat = sum((p[1] - s_lat) ** 2 for p in site_endpoints) / len(site_endpoints)
    site_x, site_z = to_local(s_lon, s_lat, tfm, ax, ay)
    print(
        f"[site] mean lon/lat=({s_lon:.6f},{s_lat:.6f}) "
        f"std~=({math.sqrt(var_lon):.2e},{math.sqrt(var_lat):.2e}) "
        f"local=({site_x:.2f},{site_z:.2f})  n={len(site_endpoints)}"
    )

    # ---- 병원: hos2site 의 마지막 점이 병원 lon/lat
    hosp_out = []
    for i, row in enumerate(hospitals):
        name = (row.get("요양기관명") or row.get("name") or "").strip()
        has_heli = str(row.get("헬기장 여부", "0")).strip() in ("1", "True", "true")
        route_path = find_route(h2s, name)
        if route_path is None:
            # 폴백: prefix 인덱스 매칭
            for p in h2s.glob(f"{i:03d}_*.json"):
                route_path = p
                break
        if route_path is None:
            print(f"[hosp {i}] route NOT FOUND for '{name}', skipping (placed at site)")
            x, z = site_x, site_z
        else:
            coords = route_coords(route_path)
            x, z = to_local(coords[-1][0], coords[-1][1], tfm, ax, ay)
        hosp_out.append({
            "id": i,
            "name": name,
            "x": round(x, 3),
            "z": round(z, 3),
            "has_helipad": has_heli,
            "roof_y": 25.0,
        })

    # ---- AMB 센터: center2site 의 첫 점이 119안전센터 lon/lat. route 폴리라인도 보관.
    amb_out = []
    used_routes: set[str] = set()
    for i, row in enumerate(ambs):
        name = (row.get("안전센터/소방서이름") or row.get("name") or "").strip()
        route_path = find_route(c2s, name)
        if route_path is not None:
            used_routes.add(route_path.name)
        else:
            print(f"[amb {i}] route NOT FOUND for '{name}'")
        if route_path is None:
            x, z = site_x, site_z
            route_to_site: list[list[float]] = []
        else:
            coords = route_coords(route_path)
            cx, cz = to_local(coords[0][0], coords[0][1], tfm, ax, ay)
            x, z = cx, cz
            route_to_site = []
            for lon, lat in coords:
                lx, lz = to_local(lon, lat, tfm, ax, ay)
                route_to_site.append([round(lx, 3), round(lz, 3)])
        amb_out.append({
            "id": i,
            "center_idx": i,
            "name": name,
            "x": round(x, 3),
            "z": round(z, 3),
            "route_to_site": route_to_site,
        })

    # ---- UAV: hospital_idx 로 출발지 결정 (옥상)
    uav_out = []
    for i, row in enumerate(uavs):
        hidx = int(row.get("hospital_idx", row.get("hospital", 0)))
        uav_out.append({"id": i, "hospital_idx": hidx})

    # ---- patient count
    cfg_path = next(SCENARIO_DIR.glob("config_*.yaml"), None)
    patient_count = 20
    if cfg_path is not None:
        for line in cfg_path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*incident_size:\s*(\d+)", line)
            if m:
                patient_count = int(m.group(1))
                break

    scene = {
        "region": "seoul",
        "scene_offset": {"x": SEOUL_ROOT_WORLD_X, "y": 0.0, "z": 0.0},
        "sim_time_scale": SIM_TIME_SCALE,
        "cruise_alt": CRUISE_ALT,
        "incident": {
            "x": round(site_x, 3),
            "z": round(site_z, 3),
            "patient_count": patient_count,
            "lon": s_lon,
            "lat": s_lat,
        },
        "hospitals": hosp_out,
        "amb_centers": amb_out,
        "uavs": uav_out,
        "ambs": [
            {
                "id": a["id"],
                "center_idx": a["center_idx"],
                "route_to_site": a["route_to_site"],
            }
            for a in amb_out
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scene_path = OUT_DIR / "scene_data.json"
    scene_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {scene_path}")

    # ---- trace 복사
    trace_src = next(TRACE_DIR.glob("trace_*.json"), None)
    if trace_src is None:
        raise SystemExit(f"no trace_*.json in {TRACE_DIR}")
    trace_dst = OUT_DIR / "trace.json"
    shutil.copyfile(trace_src, trace_dst)
    print(f"[copy] {trace_src.name} -> {trace_dst}")

    # ---- 검증 요약
    print(
        f"[summary] hospitals={len(hosp_out)} amb_centers={len(amb_out)} "
        f"uavs={len(uav_out)} patient_count={patient_count}"
    )
    print(f"[helipad hosp ids]={[h['id'] for h in hosp_out if h['has_helipad']]}")

    # transport_start / hospital_arrival 쌍 수 확인
    trace_j = json.loads(trace_dst.read_text(encoding="utf-8"))
    n_ts = n_ha = 0
    for run_events in trace_j.values():
        for e in run_events:
            if e.get("event") == "transport_start":
                n_ts += 1
            elif e.get("event") == "hospital_arrival":
                n_ha += 1
    print(f"[trace] transport_start={n_ts} hospital_arrival={n_ha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
