# -*- coding: utf-8 -*-
"""OSM 태그 → 도로변 장애물(가로등·가로수·소화전·볼라드·가드레일·버스쉘터…) 사이드카.

자율주행 시뮬에서 **도로변 정지 장애물**은 (a)LiDAR/레이더 클러터, (b)시야 차폐(occlusion),
(c)이탈 시 충돌 대상, (d)차선 폭 인지 단서로 전부 필요하다. 정밀도로지도(LGV2)에는 차도 기하만
있고 노변 시설물이 없으므로 OSM 태그에서 뽑아 별도 사이드카로 만든다.

수집 태그(도로변에 실제로 서 있는 것만 — 건물/POI 는 제외):
  점(node)
    streetlamp   highway=street_lamp        가로등 지주(가장 흔한 노변 장애물)
    tree         natural=tree               가로수
    hydrant      emergency=fire_hydrant     소화전
    bollard      barrier=bollard            볼라드(보도 진입 방지)
    pole         power=pole | man_made=utility_pole   전신주
    bench        amenity=bench              벤치
    bin          amenity=waste_basket       휴지통
    shelter      amenity=shelter | shelter_type=public_transport  버스쉘터
    busstop      highway=bus_stop           정류장 표지
    trafficsign  traffic_sign=* | highway=give_way|stop            도로표지
    postbox      amenity=post_box           우체통
    parkingmeter amenity=vending_machine[vending=parking_tickets]
  선(way, 폴리라인)
    guardrail    barrier=guard_rail         가드레일(차도 경계 — 충돌 시 치명적)
    fence        barrier=fence
    wall         barrier=wall | barrier=retaining_wall
    hedge        barrier=hedge
    kerb         barrier=kerb               연석(차선 이탈 판정용)

출력: tools/nationwide_v2/streetfurniture/<region>.txt (EPSG:5186 절대좌표, 소수 2자리)
  점  : "<kind> <E> <N>"
  선  : "<kind> <E1> <N1> <E2> <N2> ..."
  ※ Unity `CAR_test/Assets/Scripts/V2/StreetFurnitureV2.cs` 가 이 파일을 읽어
     프로시저럴 메시 + 콜라이더로 배치한다(originE/N 을 빼서 씬 로컬 좌표로).

사용:
  python tools/v2_street_furniture.py --region seoul_gangnamgu \
      --bbox5186 202000 543000 205000 546000
  (무인자 기본 = 강남 파일럿 9타일)

엔드포인트는 osm_overpass_endpoints.py 규약(로컬 Overpass 우선, 없으면 공개 미러 로테이션)을 따른다.
"""
import argparse
import os
import sys
import time

import requests
from pyproj import Transformer

from osm_overpass_endpoints import overpass_endpoints

TOOLS = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(TOOLS, "nationwide_v2", "streetfurniture")
ENDPOINTS = overpass_endpoints()
HEADERS = {"User-Agent": "MCI-UAV-research/1.0 (autonomous driving sim)"}

# kind -> Overpass node 필터 목록
NODE_FILTERS = {
    "streetlamp": ['node["highway"="street_lamp"]'],
    "tree": ['node["natural"="tree"]'],
    "hydrant": ['node["emergency"="fire_hydrant"]'],
    "bollard": ['node["barrier"="bollard"]'],
    "pole": ['node["power"="pole"]', 'node["man_made"="utility_pole"]'],
    "bench": ['node["amenity"="bench"]'],
    "bin": ['node["amenity"="waste_basket"]'],
    "shelter": ['node["amenity"="shelter"]', 'node["shelter_type"="public_transport"]'],
    "busstop": ['node["highway"="bus_stop"]'],
    "trafficsign": ['node["traffic_sign"]', 'node["highway"="give_way"]', 'node["highway"="stop"]'],
    "postbox": ['node["amenity"="post_box"]'],
}
# kind -> Overpass way 필터 목록(폴리라인)
WAY_FILTERS = {
    "guardrail": ['way["barrier"="guard_rail"]'],
    "fence": ['way["barrier"="fence"]'],
    "wall": ['way["barrier"="wall"]', 'way["barrier"="retaining_wall"]'],
    "hedge": ['way["barrier"="hedge"]'],
    "kerb": ['way["barrier"="kerb"]'],
}

TO_WGS84 = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
TO_5186 = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)


def bbox_wgs84(min_e, min_n, max_e, max_n):
    """EPSG:5186 사각형 → WGS84 bbox(south, west, north, east). 네 모서리를 다 변환해 감싼다."""
    corners = [
        TO_WGS84.transform(min_e, min_n),
        TO_WGS84.transform(max_e, min_n),
        TO_WGS84.transform(min_e, max_n),
        TO_WGS84.transform(max_e, max_n),
    ]
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    return min(lats), min(lons), max(lats), max(lons)


def overpass(query, rounds=4):
    backoff = 5
    for _ in range(rounds):
        for endpoint in ENDPOINTS:
            try:
                response = requests.post(
                    endpoint, data={"data": query}, headers=HEADERS, timeout=180)
                if response.status_code == 200:
                    return response.json().get("elements", [])
                if response.status_code in (429, 503, 504):
                    continue
                response.raise_for_status()
            except Exception:
                continue
        time.sleep(backoff)
        backoff = min(backoff * 2, 90)
    return []


def fetch_nodes(bbox, kinds):
    parts = []
    for kind in kinds:
        for flt in NODE_FILTERS[kind]:
            parts.append(f"{flt}({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});")
    # ⚠`out tags` 는 node 의 **lat/lon 을 주지 않는다**(태그만) → 전부 좌표 없음으로 버려진다.
    #   node 는 `out body`(기본 out) 여야 좌표가 온다. way 의 `out tags geom` 함정과 같은 계열.
    query = f"[out:json][timeout:180];({''.join(parts)});out body;"
    return overpass(query)


def fetch_ways(bbox, kinds):
    parts = []
    for kind in kinds:
        for flt in WAY_FILTERS[kind]:
            parts.append(f"{flt}({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});")
    # geom 이 있어야 좌표를 얻는다(members 아님 — way 는 out geom 으로 충분)
    query = f"[out:json][timeout:180];({''.join(parts)});out geom;"
    return overpass(query)


def classify_node(tags):
    """OSM 태그 → 우리 kind. 우선순위가 높은 것부터."""
    if tags.get("highway") == "street_lamp":
        return "streetlamp"
    if tags.get("natural") == "tree":
        return "tree"
    if tags.get("emergency") == "fire_hydrant":
        return "hydrant"
    if tags.get("barrier") == "bollard":
        return "bollard"
    if tags.get("power") == "pole" or tags.get("man_made") == "utility_pole":
        return "pole"
    if tags.get("amenity") == "shelter" or tags.get("shelter_type") == "public_transport":
        return "shelter"
    if tags.get("highway") == "bus_stop":
        return "busstop"
    if tags.get("amenity") == "bench":
        return "bench"
    if tags.get("amenity") == "waste_basket":
        return "bin"
    if "traffic_sign" in tags or tags.get("highway") in ("give_way", "stop"):
        return "trafficsign"
    if tags.get("amenity") == "post_box":
        return "postbox"
    return None


def classify_way(tags):
    barrier = tags.get("barrier")
    if barrier == "guard_rail":
        return "guardrail"
    if barrier == "fence":
        return "fence"
    if barrier in ("wall", "retaining_wall"):
        return "wall"
    if barrier == "hedge":
        return "hedge"
    if barrier == "kerb":
        return "kerb"
    return None


def inside(e, n, box):
    return box[0] <= e < box[2] and box[1] <= n < box[3]


REGION_INDEX = os.path.join(TOOLS, "nationwide_v2", "region_index.json")


def load_region_index():
    """v2_region_index.py 산출 — 지역별 EPSG:5186 bbox. 하드코딩 bbox 를 대체한다."""
    if not os.path.exists(REGION_INDEX):
        return {}
    import json
    with open(REGION_INDEX, encoding="utf-8") as f:
        data = json.load(f)
    return {r["name"]: r for r in data.get("regions", [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="seoul_gangnamgu")
    parser.add_argument(
        "--bbox5186", nargs=4, type=float, default=None,
        help="minE minN maxE maxN (EPSG:5186). 생략 시 region_index.json 의 지역 bbox 사용")
    parser.add_argument("--all", action="store_true",
                        help="차선그래프가 있는 전 지역 순회(재개 가능 — 기존 출력은 건너뜀)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    catalog = load_region_index()
    if args.all:
        os.makedirs(OUTDIR, exist_ok=True)
        names = [n for n, r in catalog.items() if r.get("hasLaneGraph")]
        names.sort()
        print(f"[all] 대상 {len(names)} 지역")
        done = 0
        for name in names:
            out_path = os.path.join(OUTDIR, f"{name}.txt")
            if os.path.exists(out_path) and not args.force:
                continue
            region = catalog[name]
            box = (region["minE"], region["minN"], region["maxE"], region["maxN"])
            try:
                collect(name, box, out_path)
                done += 1
            except Exception as exc:                     # 한 지역 실패가 전체를 멈추지 않게
                print(f"[fail] {name}: {exc}")
        print(f"[all] 신규 수집 {done}")
        return 0

    os.makedirs(OUTDIR, exist_ok=True)
    out_path = os.path.join(OUTDIR, f"{args.region}.txt")
    if os.path.exists(out_path) and not args.force:
        print(f"[skip] 이미 존재: {out_path} (--force 로 재수집)")
        return 0

    if args.bbox5186:
        box = tuple(args.bbox5186)
    elif args.region in catalog:
        r = catalog[args.region]
        box = (r["minE"], r["minN"], r["maxE"], r["maxN"])
    else:
        box = (202000.0, 543000.0, 205000.0, 546000.0)
    return collect(args.region, box, out_path)


def collect(region_name, box, out_path):
    wgs = bbox_wgs84(*box)
    print(f"[{region_name}] 5186={box} → wgs84 s{wgs[0]:.5f} w{wgs[1]:.5f} n{wgs[2]:.5f} e{wgs[3]:.5f}")

    lines = []
    counts = {}

    nodes = fetch_nodes(wgs, list(NODE_FILTERS.keys()))
    print(f"[node] 수신 {len(nodes)}")
    for element in nodes:
        tags = element.get("tags") or {}
        kind = classify_node(tags)
        if kind is None:
            continue
        lat, lon = element.get("lat"), element.get("lon")
        if lat is None or lon is None:
            continue
        e, n = TO_5186.transform(lon, lat)
        if not inside(e, n, box):
            continue
        lines.append(f"{kind} {e:.2f} {n:.2f}")
        counts[kind] = counts.get(kind, 0) + 1

    ways = fetch_ways(wgs, list(WAY_FILTERS.keys()))
    print(f"[way] 수신 {len(ways)}")
    for element in ways:
        tags = element.get("tags") or {}
        kind = classify_way(tags)
        if kind is None:
            continue
        geometry = element.get("geometry") or []
        # ⚠큰 bbox 는 geometry 에 null 포인트가 섞여 온다 — 반드시 걸러낸다(전 tools 공통 함정)
        points = [p for p in geometry if p and "lat" in p and "lon" in p]
        if len(points) < 2:
            continue
        coords = []
        for p in points:
            e, n = TO_5186.transform(p["lon"], p["lat"])
            coords.append((e, n))
        # 구역 안에 한 점이라도 걸치면 채택(경계 걸침 보존)
        if not any(inside(e, n, box) for (e, n) in coords):
            continue
        flat = " ".join(f"{e:.2f} {n:.2f}" for (e, n) in coords)
        lines.append(f"{kind} {flat}")
        counts[kind] = counts.get(kind, 0) + 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")

    total = sum(counts.values())
    print(f"[out] {out_path} — {total}건")
    for kind in sorted(counts, key=lambda k: -counts[k]):
        print(f"   {kind:12s} {counts[kind]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
