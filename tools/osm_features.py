# -*- coding: utf-8 -*-
"""OSM 포인트 현실요소(신호등/횡단보도/버스정류장) → 시군구별 feat/<name>.txt.

각 라인: "<type> <lat> <lon>"
  type: signal(신호등) | crossing(횡단보도) | busstop(버스정류장)

사용:
  conda run/직접: python tools/osm_features.py fetch --only gyeonggi_gwangmyeongsi
  전체: python tools/osm_features.py fetch
"""
import argparse, json, os, time, requests

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(TOOLS, "nationwide")
SGG_JSON = os.path.join(ROOT, "sgg.json")
OUTDIR = os.path.join(ROOT, "feat")
ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]
H = {"User-Agent": "MCI-UAV-research/1.0 (academic disaster sim)"}

TYPES = {  # type 라벨 → highway 태그값
    "signal": "traffic_signals",
    "crossing": "crossing",
    "busstop": "bus_stop",
}


def overpass(bbox, rounds=4):
    parts = "".join(f'node["highway"="{v}"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});' for v in TYPES.values())
    q = f"[out:json][timeout:90];({parts});out;"
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
        time.sleep(backoff)
        backoff = min(backoff * 2, 90)
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
    rev = {v: k for k, v in TYPES.items()}
    lines, kept = [], 0
    for e in els:
        if e.get("type") != "node":
            continue
        lat, lon = e.get("lat"), e.get("lon")
        if lat is None or not point_in_rings(lat, lon, rings):
            continue
        hw = e.get("tags", {}).get("highway")
        t = rev.get(hw)
        if not t:
            continue
        lines.append(f"{t} {lat:.7f} {lon:.7f}")
        kept += 1
    out = os.path.join(OUTDIR, f"{name}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[feat] {name}: 노드 {len(els)} → 귀속 {kept} ({os.path.getsize(out)/1024:.0f}KB)", flush=True)
    time.sleep(1.0)
    return kept


def cmd_fetch(args):
    os.makedirs(OUTDIR, exist_ok=True)
    sggs = json.load(open(SGG_JSON, encoding="utf-8"))
    if args.only:
        names = set(args.only.split(","))
        sggs = [s for s in sggs if s["name"] in names]
    print(f"[feat] 대상 {len(sggs)}개 → {OUTDIR}", flush=True)
    done = 0
    for i, s in enumerate(sggs):
        out = os.path.join(OUTDIR, f"{s['name']}.txt")
        if os.path.exists(out) and not args.force:
            done += 1; continue
        try:
            fetch_one(s); done += 1
        except Exception as e:
            print(f"[feat] {s['name']} 실패: {str(e)[:80]}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"[feat] 진행 {i+1}/{len(sggs)}", flush=True)
    print(f"[feat] 완료 — 처리 {done}/{len(sggs)}", flush=True)


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
