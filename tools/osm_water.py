# -*- coding: utf-8 -*-
"""OSM 수계(강·호수·저수지) 폴리곤 → 시군구별 area/<name>.txt 의 water 라인 교체.

⚠️왜 별도 스크립트인가 (2026-07-13 발각):
  1) `osm_areas.py` 의 Overpass 쿼리는 **`way[...]` 만** 조회한다. 그런데 **한강·낙동강·금강·
     소양호 등 국내 주요 수계는 전부 OSM multipolygon `relation`** 이라 통째로 누락됐다.
     실측: 강남구 area/*.txt 의 water 폴리곤 8개·총 0.10 km²(연못뿐) — **한강이 없다**.
     로컬 Overpass 확인 결과 한강 = relation 152336 / 3769500, 탄천·청계천·중랑천·석촌호수도 전부 relation.
  2) 대형 수계는 여러 시군구에 걸쳐 있는데 osm_areas 의 **center-point 귀속**은 폴리곤을 한 구에만
     붙인다 → 여기서는 **구 경계로 클립(shapely intersection)** 한다.

출력 포맷은 기존 area/<name>.txt 와 동일( "water lat lon lat lon ..." = 폐합 링 1개 )
→ **Unity 임포터(BuildAreas) 수정 불필요**. 다만 그 포맷엔 구멍 개념이 없으므로,
   내곽링(예: 한강의 여의도·밤섬)은 **키홀 절개**로 외곽링에 병합한다
   (외곽↔내곽 최근접 점쌍을 왕복으로 이어 붙여 링 하나로 만드는 표준 기법).

사용:
  python tools/osm_water.py fetch --only seoul_gangnamgu
  python tools/osm_water.py fetch                 # 전국 255(기존 완료분 skip)
  python tools/osm_water.py fetch --force
"""
import argparse
import json
import os
import time

import requests
from shapely.geometry import Polygon
from shapely.ops import unary_union

from osm_overpass_endpoints import overpass_endpoints

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(TOOLS, "nationwide")
SGG_JSON = os.path.join(ROOT, "sgg.json")
AREADIR = os.path.join(ROOT, "area")
STAMP = os.path.join(ROOT, ".water2")
ENDPOINTS = overpass_endpoints()
H = {"User-Agent": "MCI-UAV-research/1.0 (academic disaster sim)"}

MIN_AREA_DEG2 = 2e-8   # ≈ 200 m² 미만 물웅덩이는 버림
SIMPLIFY_DEG = 1e-5    # ≈ 1.1 m — 강 리본엔 충분


def overpass(bbox, rounds=4):
    """way + **relation** 둘 다. relation 은 out geom 이 members[].geometry 를 준다."""
    b = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    q = ("[out:json][timeout:180];("
         f'way["natural"="water"]({b});'
         f'way["waterway"="riverbank"]({b});'
         f'way["landuse"="reservoir"]({b});'
         f'relation["natural"="water"]({b});'
         f'relation["waterway"="riverbank"]({b});'
         f'relation["landuse"="reservoir"]({b});'
         # ⚠️`out tags geom` 은 안 된다 — Overpass 의 `tags` 모드는 **멤버를 출력하지 않아**
         #   relation 이 members:[] 로 와서 링을 못 만든다(한강이 통째로 사라진 진범).
         #   `body` 모드여야 members 가 나온다.
         ");out body geom;")
    backoff = 5
    for _ in range(rounds):
        for ep in ENDPOINTS:
            try:
                r = requests.post(ep, data={"data": q}, headers=H, timeout=220)
                if r.status_code == 200:
                    return r.json().get("elements", [])
                if r.status_code in (429, 503, 504):
                    continue
                r.raise_for_status()
            except Exception:
                continue
        time.sleep(backoff)
        backoff = min(backoff * 2, 90)
    return []


def coords(geom):
    # ⚠️큰 bbox 응답은 geometry 에 null 원소가 섞인다(잘린 way) → 반드시 걸러야 KeyError 안 남
    return [(p["lon"], p["lat"]) for p in geom if p and "lon" in p and "lat" in p]


def stitch(segs, tol=1e-9):
    """멀티폴리곤 relation 의 멤버 way 조각들을 폐합 링으로 잇는다."""
    def near(a, b):
        return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol

    rings, pool = [], [list(s) for s in segs if len(s) >= 2]
    while pool:
        cur = pool.pop(0)
        grew = True
        while grew and not near(cur[0], cur[-1]):
            grew = False
            for i, s in enumerate(pool):
                if near(cur[-1], s[0]):
                    cur.extend(s[1:])
                elif near(cur[-1], s[-1]):
                    cur.extend(list(reversed(s))[1:])
                elif near(cur[0], s[-1]):
                    cur = s[:-1] + cur
                elif near(cur[0], s[0]):
                    cur = list(reversed(s))[:-1] + cur
                else:
                    continue
                pool.pop(i)
                grew = True
                break
        if len(cur) >= 4 and near(cur[0], cur[-1]):
            rings.append(cur)
    return rings


def polys_from_elements(els):
    """way/relation 요소들 → shapely Polygon 리스트(구멍 포함)."""
    out = []
    for e in els:
        try:
            if e["type"] == "way":
                c = coords(e.get("geometry") or [])
                if len(c) >= 4:
                    p = Polygon(c)
                    if p.is_valid or p.buffer(0).is_valid:
                        out.append(p if p.is_valid else p.buffer(0))
            elif e["type"] == "relation":
                outer_segs, inner_segs = [], []
                for m in e.get("members", []):
                    if m.get("type") != "way":
                        continue
                    c = coords(m.get("geometry") or [])
                    if len(c) < 2:
                        continue
                    (inner_segs if m.get("role") == "inner" else outer_segs).append(c)
                outers = [Polygon(r) for r in stitch(outer_segs) if len(r) >= 4]
                inners = [Polygon(r) for r in stitch(inner_segs) if len(r) >= 4]
                for o in outers:
                    if not o.is_valid:
                        o = o.buffer(0)
                    holes = [i for i in inners if i.is_valid and o.contains(i.representative_point())]
                    if holes:
                        o = o.difference(unary_union(holes))
                    out.append(o)
        except Exception:
            continue
    return [p for p in out if not p.is_empty and p.area > 0]


def keyhole(poly):
    """구멍 있는 Polygon → 링 1개. 외곽↔내곽 최근접 점쌍을 왕복 연결(키홀 절개)."""
    ext = list(poly.exterior.coords)
    for interior in poly.interiors:
        inner = list(interior.coords)[:-1]     # 닫힘 중복점 제거
        if len(inner) < 3:
            continue
        bi = bj = 0
        bd = float("inf")
        for i, p in enumerate(ext):
            for j, q in enumerate(inner):
                d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
                if d < bd:
                    bd, bi, bj = d, i, j
        loop = inner[bj:] + inner[:bj] + [inner[bj]]   # 내곽 한 바퀴 후 진입점 복귀
        ext = ext[:bi + 1] + loop + ext[bi:]           # 절개선을 왕복(면적 0 슬릿)
    return ext


def water_lines(district_poly, els):
    ws = polys_from_elements(els)
    if not ws:
        return []
    merged = unary_union(ws)
    clipped = merged.intersection(district_poly)   # ★대형 수계를 구 경계로 자른다
    if clipped.is_empty:
        return []
    geoms = list(getattr(clipped, "geoms", [clipped]))
    lines = []
    for g in geoms:
        if g.geom_type != "Polygon" or g.area < MIN_AREA_DEG2:
            continue
        g = g.simplify(SIMPLIFY_DEG, preserve_topology=True)
        if g.is_empty or g.geom_type != "Polygon":
            continue
        ring = keyhole(g)
        if len(ring) < 3:
            continue
        toks = []
        for lon, lat in ring:
            toks.append(f"{lat:.7f}")
            toks.append(f"{lon:.7f}")
        lines.append("water " + " ".join(toks))
    return lines


def rewrite_area(name, lines):
    """area/<name>.txt 에서 water 라인만 교체(park/green 보존)."""
    path = os.path.join(AREADIR, name + ".txt")
    keep = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            keep = [ln.rstrip("\n") for ln in f if not ln.startswith("water ")]
    os.makedirs(AREADIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ln in keep:
            if ln.strip():
                f.write(ln + "\n")
        for ln in lines:
            f.write(ln + "\n")
    return len(keep), len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch"])
    ap.add_argument("--only", default=None, help="시군구 name (쉼표 구분)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    with open(SGG_JSON, "r", encoding="utf-8") as f:
        sgg = json.load(f)
    os.makedirs(STAMP, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None

    for d in sgg:
        name = d["name"]
        if only and name not in only:
            continue
        stamp = os.path.join(STAMP, name + ".ok")
        if os.path.exists(stamp) and not args.force:
            continue
        rings = d.get("rings") or []
        if not rings:
            continue
        # 구 경계(가장 큰 링) — 섬이 여러 개면 합집합
        dp = unary_union([Polygon(r) for r in rings if len(r) >= 4]).buffer(0)
        bb = d["bbox"]   # [minlat, minlon, maxlat, maxlon]
        els = overpass(bb)
        lines = water_lines(dp, els)
        kept, nw = rewrite_area(name, lines)
        km2 = 0.0
        try:
            wpolys = polys_from_elements(els)
            if wpolys:
                km2 = unary_union(wpolys).intersection(dp).area * (111.0 * 88.0)  # deg² → 대략 km²(위도 37°)
        except Exception:
            pass
        print(f"{name}: 요소 {len(els)} → water 링 {nw}개 (~{km2:.2f} km²), park/green {kept}행 유지", flush=True)
        with open(stamp, "w", encoding="utf-8") as f:
            f.write("ok\n")


if __name__ == "__main__":
    main()
