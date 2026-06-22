# -*- coding: utf-8 -*-
"""OSM(Overpass API) 도로망 → 시군구별 roads.txt (Unity 리본 메시용).

vworld 건물/정사영상과 같은 좌표계(WGS84→RegionRegistry)로, 시군구 폴리곤 안의
highway way를 받아 등급별 폭과 폴리라인을 라인 포맷으로 저장한다.

  roads.txt 라인: "<width_m> <lat1> <lon1> <lat2> <lon2> ..."   (정점 위경도)

사용:
  python tools/osm_roads.py fetch --only gyeonggi_gwangmyeongsi,seoul_geumcheongu
  python tools/osm_roads.py fetch                 # sgg.json 전체(오래 걸림)

산출물: tools/nationwide/sgg/vw_<name>/roads.txt  (해당 region 씬에 임포트)
"""
import argparse
import json
import math
import os
import sys
import time

import requests

from osm_overpass_endpoints import overpass_endpoints

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(TOOLS, "nationwide")
SGG_JSON = os.path.join(ROOT, "sgg.json")
SGGDIR = os.path.join(ROOT, "sgg")
ENDPOINTS = overpass_endpoints()
HEADERS = {"User-Agent": "MCI-UAV-research/1.0 (academic disaster sim)"}

# highway 등급 → 도로 폭(m). lanes 태그 있으면 lanes*3.5 우선.
WIDTH = {
    "motorway": 24, "motorway_link": 12, "trunk": 20, "trunk_link": 10,
    "primary": 15, "primary_link": 8, "secondary": 11, "secondary_link": 7,
    "tertiary": 8, "tertiary_link": 6, "unclassified": 6, "residential": 6,
    "living_street": 5, "service": 4,
}
GRADES = "|".join(WIDTH.keys())


def overpass(bbox, retries=4):
    """bbox=(minLat,minLon,maxLat,maxLon) 안의 highway way를 geometry째로."""
    q = (f"[out:json][timeout:120];"
         f'way["highway"~"^({GRADES})$"]'
         f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});out geom;")
    backoff = 5
    for attempt in range(retries):
        last_error = None
        for ep in ENDPOINTS:
            try:
                r = requests.post(ep, data={"data": q}, headers=HEADERS, timeout=180)
                if r.status_code == 200:
                    return r.json().get("elements", [])
                if r.status_code in (429, 503, 504):
                    last_error = RuntimeError(f"overpass {r.status_code} from {ep}")
                    continue
                r.raise_for_status()
            except Exception as e:  # noqa: BLE001
                last_error = e
                continue
        if attempt == retries - 1:
            if last_error:
                raise last_error
            return []
        print(f"  overpass retry {attempt + 1}/{retries}; waiting {backoff}s")
        time.sleep(backoff)
        backoff = min(backoff * 2, 120)
    return []


def point_in_rings(lat, lon, rings):
    """ray casting — rings=[[ [lon,lat],... ], ...]."""
    inside = False
    for ring in rings:
        n = len(ring)
        j = n - 1
        for i in range(n):
            xi, yi = ring[i][0], ring[i][1]
            xj, yj = ring[j][0], ring[j][1]
            if ((yi > lat) != (yj > lat)) and \
               (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        if inside:
            return True
    return False


def width_of(tags):
    lanes = tags.get("lanes")
    if lanes:
        try:
            return max(3.5, float(str(lanes).split(";")[0]) * 3.5)
        except Exception:  # noqa: BLE001
            pass
    return WIDTH.get(tags.get("highway"), 6)


def fetch_one(sgg):
    """시군구 1개 도로 → roads.txt. 중심점이 폴리곤 안인 way만(이웃 누수 방지)."""
    name = sgg["name"]
    outdir = os.path.join(SGGDIR, f"vw_{name}")
    if not os.path.isdir(outdir):
        print(f"[osm] {name}: vw 폴더 없음 — 스킵")
        return 0
    bb = sgg["bbox"]   # minLat,minLon,maxLat,maxLon
    ways = overpass(bb)
    rings = sgg["rings"]
    lines, kept = [], 0
    for w in ways:
        geom = w.get("geometry") or []
        if len(geom) < 2:
            continue
        # 중심점 폴리곤 판정(시군구 귀속)
        clat = sum(p["lat"] for p in geom) / len(geom)
        clon = sum(p["lon"] for p in geom) / len(geom)
        if not point_in_rings(clat, clon, rings):
            continue
        wid = width_of(w.get("tags", {}))
        coords = " ".join(f"{p['lat']:.7f} {p['lon']:.7f}" for p in geom)
        lines.append(f"{wid:.1f} {coords}")
        kept += 1
    out = os.path.join(outdir, "roads.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[osm] {name}: way {len(ways)} → 귀속 {kept} → roads.txt ({os.path.getsize(out) / 1024:.0f}KB)")
    time.sleep(1.0)   # overpass 예의
    return kept


def cmd_fetch(args):
    with open(SGG_JSON, encoding="utf-8") as f:
        sggs = json.load(f)
    if args.only:
        names = set(args.only.split(","))
        sggs = [s for s in sggs if s["name"] in names]
    print(f"[osm] 대상 시군구 {len(sggs)}개")
    total = 0
    for i, s in enumerate(sggs):
        out = os.path.join(SGGDIR, f"vw_{s['name']}", "roads.txt")
        if os.path.exists(out) and not args.force:
            continue
        try:
            total += fetch_one(s)
        except Exception as e:  # noqa: BLE001
            print(f"[osm] {s['name']} 실패: {str(e)[:100]}")
        if (i + 1) % 10 == 0:
            print(f"[osm] 진행 {i + 1}/{len(sggs)}")
    print(f"[osm] 완료 — 총 도로 {total}개")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--only", default=None)
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=cmd_fetch)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
