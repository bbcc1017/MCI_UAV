# -*- coding: utf-8 -*-
"""한 시군구 bbox에서 OSM이 제공하는 '현실 요소' 종류·개수 탐색(구현 우선순위 판단용)."""
import json, os, sys, requests

TOOLS = os.path.dirname(os.path.abspath(__file__))
SGG = json.load(open(os.path.join(TOOLS, "nationwide", "sgg.json"), encoding="utf-8"))
name = sys.argv[1] if len(sys.argv) > 1 else "gyeonggi_gwangmyeongsi"
d = next(s for s in SGG if s["name"] == name)
b = d["bbox"]  # minLat,minLon,maxLat,maxLon
bb = f"{b[0]},{b[1]},{b[2]},{b[3]}"
EP = "https://overpass-api.de/api/interpreter"
H = {"User-Agent": "MCI-UAV-research/1.0"}

# (라벨, overpass 조각)
Q = {
    "신호등 traffic_signals":      f'node["highway"="traffic_signals"]({bb});',
    "횡단보도 crossing(node)":      f'node["highway"="crossing"]({bb});',
    "버스정류장 bus_stop":          f'node["highway"="bus_stop"]({bb});',
    "정지선 stop":                  f'node["highway"="stop"]({bb});',
    "가로수 tree(node)":            f'node["natural"="tree"]({bb});',
    "공원 park":                    f'way["leisure"="park"]({bb});',
    "녹지 grass/forest":            f'(way["landuse"~"grass|forest|meadow|recreation_ground"]({bb}););',
    "수계 water(폴리곤)":           f'way["natural"="water"]({bb});',
    "하천 waterway":                f'way["waterway"~"river|stream|canal"]({bb});',
    "철도 railway":                 f'way["railway"~"rail|subway|light_rail"]({bb});',
    "병원 amenity=hospital":        f'(node["amenity"="hospital"]({bb});way["amenity"="hospital"]({bb}););',
    "학교 amenity=school":          f'(node["amenity"="school"]({bb});way["amenity"="school"]({bb}););',
    "소방서 fire_station":          f'(node["amenity"="fire_station"]({bb});way["amenity"="fire_station"]({bb}););',
    "주유소 fuel":                  f'node["amenity"="fuel"]({bb});',
    "주차장 parking":               f'way["amenity"="parking"]({bb});',
    "교량 bridge(way)":             f'way["bridge"="yes"]["highway"]({bb});',
    "터널 tunnel(way)":             f'way["tunnel"="yes"]["highway"]({bb});',
    "속도제한 maxspeed보유 도로":   f'way["highway"]["maxspeed"]({bb});',
}

print(f"[탐색] {name}  bbox={bb}\n")
for label, frag in Q.items():
    q = f"[out:json][timeout:60];{frag}out count;"
    try:
        r = requests.post(EP, data={"data": q}, headers=H, timeout=90)
        if r.status_code != 200:
            print(f"  {label:28s}: HTTP {r.status_code}")
            continue
        els = r.json().get("elements", [])
        cnt = 0
        for e in els:
            if e.get("type") == "count":
                tags = e.get("tags", {})
                cnt = int(tags.get("total") or tags.get("nodes") or tags.get("ways") or 0)
        print(f"  {label:28s}: {cnt}")
    except Exception as e:
        print(f"  {label:28s}: 실패 {str(e)[:50]}")
