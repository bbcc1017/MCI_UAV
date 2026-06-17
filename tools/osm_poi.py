# -*- coding: utf-8 -*-
"""OSM 주요 POI(병원/학교/소방서/경찰서/주유소) → 시군구별 poi/<name>.txt.

각 라인: "<type> <lat> <lon> <name>"   (name은 공백→_ 치환, 없으면 '-')
  type: hospital | school | fire_station | police | fuel

사용: python tools/osm_poi.py fetch --only gyeonggi_gwangmyeongsi
"""
import argparse, json, os, time, requests

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(TOOLS, "nationwide")
SGG_JSON = os.path.join(ROOT, "sgg.json")
OUTDIR = os.path.join(ROOT, "poi")
ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]
H = {"User-Agent": "MCI-UAV-research/1.0 (academic disaster sim)"}
AMENITIES = ["hospital", "school", "fire_station", "police", "fuel"]


def overpass(bbox, rounds=4):
    b = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    parts = "".join(
        f'node["amenity"="{a}"]({b});way["amenity"="{a}"]({b});' for a in AMENITIES)
    q = f"[out:json][timeout:90];({parts});out tags center;"
    backoff = 5
    for _ in range(rounds):
        for ep in ENDPOINTS:
            try:
                r = requests.post(ep, data={"data": q}, headers=H, timeout=120)
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
    seen = set()
    lines, kept = [], 0
    for e in els:
        tags = e.get("tags", {})
        a = tags.get("amenity")
        if a not in AMENITIES:
            continue
        if e.get("type") == "node":
            lat, lon = e.get("lat"), e.get("lon")
        else:
            c = e.get("center") or {}
            lat, lon = c.get("lat"), c.get("lon")
        if lat is None or not point_in_rings(lat, lon, rings):
            continue
        key = (a, round(lat, 5), round(lon, 5))
        if key in seen:
            continue
        seen.add(key)
        nm = (tags.get("name") or "-").replace(" ", "_")
        lines.append(f"{a} {lat:.7f} {lon:.7f} {nm}")
        kept += 1
    out = os.path.join(OUTDIR, f"{name}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[poi] {name}: 요소 {len(els)} → 귀속 {kept} ({os.path.getsize(out)/1024:.0f}KB)", flush=True)
    time.sleep(1.0)
    return kept


def cmd_fetch(args):
    os.makedirs(OUTDIR, exist_ok=True)
    sggs = json.load(open(SGG_JSON, encoding="utf-8"))
    if args.only:
        names = set(args.only.split(","))
        sggs = [s for s in sggs if s["name"] in names]
    print(f"[poi] 대상 {len(sggs)}개 → {OUTDIR}", flush=True)
    done = 0
    for i, s in enumerate(sggs):
        out = os.path.join(OUTDIR, f"{s['name']}.txt")
        if os.path.exists(out) and not args.force:
            done += 1; continue
        try:
            fetch_one(s); done += 1
        except Exception as e:
            print(f"[poi] {s['name']} 실패: {str(e)[:80]}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"[poi] 진행 {i+1}/{len(sggs)}", flush=True)
    print(f"[poi] 완료 — 처리 {done}/{len(sggs)}", flush=True)


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
