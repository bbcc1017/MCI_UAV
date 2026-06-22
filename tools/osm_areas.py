# -*- coding: utf-8 -*-
"""OSM 영역요소(공원/녹지/수계) 폴리곤 → 시군구별 area/<name>.txt.

각 라인: "<type> <lat1> <lon1> <lat2> <lon2> ..."  (폐합 링 정점)
  type: park(공원) | green(녹지) | water(수계)

사용: python tools/osm_areas.py fetch --only gyeonggi_gwangmyeongsi
"""
import argparse, json, os, time, requests

from osm_overpass_endpoints import overpass_endpoints

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(TOOLS, "nationwide")
SGG_JSON = os.path.join(ROOT, "sgg.json")
OUTDIR = os.path.join(ROOT, "area")
ENDPOINTS = overpass_endpoints()
H = {"User-Agent": "MCI-UAV-research/1.0 (academic disaster sim)"}


def classify(tags):
    if tags.get("natural") == "water" or tags.get("waterway") in ("riverbank",) or tags.get("water"):
        return "water"
    if tags.get("leisure") == "park":
        return "park"
    if tags.get("landuse") in ("grass", "forest", "meadow", "recreation_ground", "village_green"):
        return "green"
    if tags.get("leisure") in ("garden", "pitch", "playground", "golf_course"):
        return "green"
    return None


def overpass(bbox, rounds=4):
    b = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    q = ("[out:json][timeout:120];("
         f'way["leisure"="park"]({b});'
         f'way["landuse"~"grass|forest|meadow|recreation_ground|village_green"]({b});'
         f'way["leisure"~"garden|pitch|playground|golf_course"]({b});'
         f'way["natural"="water"]({b});'
         f'way["waterway"="riverbank"]({b});'
         ");out tags geom;")
    backoff = 5
    for _ in range(rounds):
        for ep in ENDPOINTS:
            try:
                r = requests.post(ep, data={"data": q}, headers=H, timeout=160)
                if r.status_code == 200:
                    return r.json().get("elements", [])
                if r.status_code in (429, 503, 504):
                    continue
                r.raise_for_status()
            except Exception:
                continue
        time.sleep(backoff); backoff = min(backoff * 2, 90)
    return []


def point_in_rings(lat, lon, rings):
    inside = False
    for ring in rings:
        n = len(ring); j = n - 1
        for i in range(n):
            xi, yi = ring[i][0], ring[i][1]
            xj, yj = ring[j][0], ring[j][1]
            if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        if inside:
            return True
    return False


def fetch_one(sgg):
    name = sgg["name"]
    els = overpass(sgg["bbox"])
    rings = sgg["rings"]
    lines, kept = [], 0
    for e in els:
        geom = e.get("geometry") or []
        if len(geom) < 3:
            continue
        t = classify(e.get("tags", {}))
        if not t:
            continue
        clat = sum(p["lat"] for p in geom) / len(geom)
        clon = sum(p["lon"] for p in geom) / len(geom)
        if not point_in_rings(clat, clon, rings):
            continue
        coords = " ".join(f"{p['lat']:.7f} {p['lon']:.7f}" for p in geom)
        lines.append(f"{t} {coords}")
        kept += 1
    out = os.path.join(OUTDIR, f"{name}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[area] {name}: 폴리곤 {len(els)} → 귀속 {kept} ({os.path.getsize(out)/1024:.0f}KB)", flush=True)
    time.sleep(1.0)
    return kept


def cmd_fetch(args):
    os.makedirs(OUTDIR, exist_ok=True)
    sggs = json.load(open(SGG_JSON, encoding="utf-8"))
    if args.only:
        names = set(args.only.split(","))
        sggs = [s for s in sggs if s["name"] in names]
    print(f"[area] 대상 {len(sggs)}개 → {OUTDIR}", flush=True)
    done = 0
    for i, s in enumerate(sggs):
        out = os.path.join(OUTDIR, f"{s['name']}.txt")
        if os.path.exists(out) and not args.force:
            done += 1; continue
        try:
            fetch_one(s); done += 1
        except Exception as e:
            print(f"[area] {s['name']} 실패: {str(e)[:80]}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"[area] 진행 {i+1}/{len(sggs)}", flush=True)
    print(f"[area] 완료 — 처리 {done}/{len(sggs)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--only", default=None)
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=cmd_fetch)
    a = ap.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
