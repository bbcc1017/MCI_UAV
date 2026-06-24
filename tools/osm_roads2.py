# -*- coding: utf-8 -*-
"""OSM(Overpass) 도로망 → 시군구별 enriched roads2/<name>.txt.

osm_roads.py(폭+좌표만)의 상위호환. 차선수·일방통행·도로등급을 함께 저장해
Unity에서 중앙선/차선점선/일방통행을 표현하고 NPC 교통 방향을 정한다.

  roads2/<name>.txt 라인:
    "<class> <lanes> <oneway> <struct> <lat1> <lon1> <lat2> <lon2> ..."
      class  : highway 등급 토큰(motorway/primary/secondary/...). 폭/마킹 스타일 결정.
      lanes  : 총 차로수(int, 0=미상 → class 기본 사용).
      oneway : 0=양방향, 1=정방향 일방, -1=역방향 일방(geometry 역순 주행).
      struct : 0=지상, 1=교량(bridge — 물/계곡 위 고가로 시각화), 2=터널(tunnel).
               하위호환: struct 없는 옛 파일은 4번째가 좌표(소수점)라 임포터가 구분해 0 처리.

사용:
  conda run -n UAV python tools/osm_roads2.py fetch --only seoul_jongnogu
  conda run -n UAV python tools/osm_roads2.py fetch          # sgg.json 전체(오래 걸림)
  conda run -n UAV python tools/osm_roads2.py fetch --force   # 기존 파일 덮어씀
"""
import argparse
import json
import os
import time

import requests

from osm_overpass_endpoints import overpass_endpoints

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(TOOLS, "nationwide")
SGG_JSON = os.path.join(ROOT, "sgg.json")
OUTDIR = os.path.join(ROOT, "roads2")
# 여러 미러 로테이션 — 한 곳이 429/504면 다음 미러를 즉시 시도(백오프 대기 최소화).
ENDPOINTS = overpass_endpoints()
HEADERS = {"User-Agent": "MCI-UAV-research/1.0 (academic disaster sim)"}

CLASSES = [
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential",
    "living_street", "service",
]
GRADES = "|".join(CLASSES)

# 보행로(인도/광장/자전거도로/계단·산책로) — 별도 레이어(footways/<name>.txt). 차도와 분리.
FW_CLASSES = ["footway", "pedestrian", "path", "cycleway", "steps"]
FW_GRADES = "|".join(FW_CLASSES)
OUTDIR_FW = os.path.join(ROOT, "footways")


def overpass(bbox, grades=GRADES, rounds=4):
    """bbox=(minLat,minLon,maxLat,maxLon) 안의 highway way를 geometry+tags.
    grades=정규식 등급(기본=차도 GRADES, 보행로는 FW_GRADES 전달).
    미러를 순회하며 한 곳이 429/504면 즉시 다음 미러로 — 한 라운드 전부 실패 시에만 백오프."""
    q = (f"[out:json][timeout:120];"
         f'way["highway"~"^({grades})$"]'
         f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});out tags geom;")
    backoff = 5
    for rnd in range(rounds):
        for ep in ENDPOINTS:
            try:
                r = requests.post(ep, data={"data": q}, headers=HEADERS, timeout=180)
                if r.status_code == 200:
                    return r.json().get("elements", [])
                if r.status_code in (429, 503, 504):
                    continue   # 다음 미러 즉시
                r.raise_for_status()
            except Exception as e:  # noqa: BLE001
                continue   # 다음 미러
        print(f"  전 미러 실패(라운드 {rnd + 1}) — {backoff}s 대기", flush=True)
        time.sleep(backoff)
        backoff = min(backoff * 2, 90)
    return []


def point_in_rings(lat, lon, rings):
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


def parse_lanes(tags):
    v = tags.get("lanes")
    if not v:
        return 0
    try:
        return max(0, int(float(str(v).split(";")[0])))
    except Exception:  # noqa: BLE001
        return 0


def parse_oneway(tags):
    v = str(tags.get("oneway", "")).strip().lower()
    if v in ("yes", "true", "1"):
        return 1
    if v in ("-1", "reverse"):
        return -1
    # 고속도로/링크는 사실상 일방
    hw = tags.get("highway", "")
    if hw in ("motorway", "motorway_link", "trunk_link", "primary_link",
              "secondary_link", "tertiary_link"):
        return 1
    return 0


def parse_struct(tags):
    """0=지상, 1=교량(bridge), 2=터널(tunnel). bridge가 tunnel보다 우선."""
    b = str(tags.get("bridge", "")).strip().lower()
    if b and b not in ("no", "false", "0"):
        return 1
    if str(tags.get("man_made", "")).strip().lower() in ("bridge",):
        return 1
    t = str(tags.get("tunnel", "")).strip().lower()
    if t and t not in ("no", "false", "0"):
        return 2
    if str(tags.get("covered", "")).strip().lower() in ("yes", "tunnel"):
        return 2
    return 0


def fetch_one(sgg, grades=GRADES, outdir=OUTDIR, tag="osm2"):
    name = sgg["name"]
    bb = sgg["bbox"]
    rings = sgg["rings"]
    ways = overpass(bb, grades)
    lines, kept = [], 0
    for w in ways:
        geom = w.get("geometry") or []
        # 대형 area에서 bbox 경계를 넘는 way는 geometry에 null/불완전 점이 섞임 → 제거(안 하면 'NoneType' 오류).
        geom = [p for p in geom if p and "lat" in p and "lon" in p]
        if len(geom) < 2:
            continue
        clat = sum(p["lat"] for p in geom) / len(geom)
        clon = sum(p["lon"] for p in geom) / len(geom)
        if not point_in_rings(clat, clon, rings):
            continue
        tags = w.get("tags", {})
        cls = tags.get("highway", "residential")
        lanes = parse_lanes(tags)
        oneway = parse_oneway(tags)
        struct = parse_struct(tags)
        coords = " ".join(f"{p['lat']:.7f} {p['lon']:.7f}" for p in geom)
        lines.append(f"{cls} {lanes} {oneway} {struct} {coords}")
        kept += 1
    out = os.path.join(outdir, f"{name}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[{tag}] {name}: way {len(ways)} → 귀속 {kept} ({os.path.getsize(out) / 1024:.0f}KB)", flush=True)
    time.sleep(1.0)
    return kept


def _fetch_all(args, grades, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    with open(SGG_JSON, encoding="utf-8") as f:
        sggs = json.load(f)
    if args.only:
        names = set(args.only.split(","))
        sggs = [s for s in sggs if s["name"] in names]
    print(f"[{tag}] 대상 시군구 {len(sggs)}개 → {outdir}", flush=True)
    total, done = 0, 0
    for i, s in enumerate(sggs):
        out = os.path.join(outdir, f"{s['name']}.txt")
        if os.path.exists(out) and not args.force:
            done += 1
            continue
        try:
            total += fetch_one(s, grades, outdir, tag)
            done += 1
        except Exception as e:  # noqa: BLE001
            print(f"[{tag}] {s['name']} 실패: {str(e)[:100]}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"[{tag}] 진행 {i + 1}/{len(sggs)} (완료 {done})", flush=True)
    print(f"[{tag}] 완료 — 처리 {done}/{len(sggs)}, 총 way {total}개", flush=True)


def cmd_fetch(args):
    _fetch_all(args, GRADES, OUTDIR, "osm2")


def cmd_footway(args):
    _fetch_all(args, FW_GRADES, OUTDIR_FW, "fw")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--only", default=None)
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=cmd_fetch)
    fw = sub.add_parser("footway", help="보행로(인도/광장/자전거도로) → footways/<name>.txt")
    fw.add_argument("--only", default=None)
    fw.add_argument("--force", action="store_true")
    fw.set_defaults(func=cmd_footway)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
