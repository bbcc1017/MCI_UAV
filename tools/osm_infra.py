# -*- coding: utf-8 -*-
"""OSM 인프라(UAV 비행장애물·철도·헬기장) → 시군구별 infra/<name>.txt.

각 라인:
  점형: "<type> <lat> <lon>"                     type: tower | substation | station | helipad
  선형: "<type> <lat> <lon> <lat> <lon> ..."       type: powerline | rail

UAV 트윈 의미: 송전선/철탑/고탑=저고도 충돌위험, 철도/지하철=구조물, 헬기장=실제 LZ.
osm_poi.py 패턴(미러 로테이션·중심점 귀속·재개) + ways는 out geom으로 형상 취득.

사용: PYTHONIOENCODING=utf-8 .../python.exe tools/osm_infra.py fetch --only seoul_gangnamgu
"""
import argparse, json, os, time, requests

from osm_overpass_endpoints import overpass_endpoints

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(TOOLS, "nationwide")
SGG_JSON = os.path.join(ROOT, "sgg.json")
OUTDIR = os.path.join(ROOT, "infra")
ENDPOINTS = overpass_endpoints()
H = {"User-Agent": "MCI-UAV-research/1.0 (academic disaster sim)"}

# Overpass 셀렉터 — 점/선 인프라
SELECTORS = """
  way["power"="line"]({b});
  way["power"="minor_line"]({b});
  node["power"="tower"]({b});
  way["power"="substation"]({b});
  node["power"="substation"]({b});
  node["man_made"="tower"]({b});
  way["man_made"="tower"]({b});
  node["man_made"="mast"]({b});
  way["railway"="rail"]({b});
  way["railway"="subway"]({b});
  way["railway"="light_rail"]({b});
  way["railway"="tram"]({b});
  way["railway"="narrow_gauge"]({b});
  node["railway"="station"]({b});
  node["aeroway"="helipad"]({b});
  way["aeroway"="helipad"]({b});
"""


def overpass(bbox, rounds=4):
    b = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    q = f"[out:json][timeout:120];({SELECTORS.format(b=b)});out geom tags;"
    backoff = 5
    for _ in range(rounds):
        for ep in ENDPOINTS:
            try:
                r = requests.post(ep, data={"data": q}, headers=H, timeout=180)
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


def classify(tags):
    p = tags.get("power"); m = tags.get("man_made"); rw = tags.get("railway"); aw = tags.get("aeroway")
    if p in ("line", "minor_line"): return "powerline", "line"
    if p == "tower": return "tower", "point"
    if p == "substation": return "substation", "point"
    if m in ("tower", "mast"): return "tower", "point"
    if rw in ("rail", "subway", "light_rail", "tram", "narrow_gauge"): return "rail", "line"
    if rw == "station": return "station", "point"
    if aw == "helipad": return "helipad", "point"
    return None, None


def geom_latlon(e):
    """node=lat/lon, way=geometry 배열. 점 귀속용 대표좌표(선=중점) 반환."""
    if e.get("type") == "node":
        return e.get("lat"), e.get("lon"), None
    g = e.get("geometry")
    if not g:
        return None, None, None
    pts = [(p["lat"], p["lon"]) for p in g if p.get("lat") is not None]
    if not pts:
        return None, None, None
    mid = pts[len(pts) // 2]
    return mid[0], mid[1], pts


def fetch_one(sgg):
    name = sgg["name"]
    els = overpass(sgg["bbox"])
    rings = sgg["rings"]
    seen = set()
    lines, kept = [], 0
    for e in els:
        ty, kind = classify(e.get("tags", {}))
        if ty is None:
            continue
        rlat, rlon, pts = geom_latlon(e)
        if rlat is None or not point_in_rings(rlat, rlon, rings):
            continue
        if kind == "point":
            key = (ty, round(rlat, 5), round(rlon, 5))
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"{ty} {rlat:.7f} {rlon:.7f}")
            kept += 1
        else:  # line — 형상 좌표열(점 과다 시 ~10m 솎음은 임포터가 처리)
            if not pts or len(pts) < 2:
                continue
            coords = " ".join(f"{la:.7f} {lo:.7f}" for la, lo in pts)
            lines.append(f"{ty} {coords}")
            kept += 1
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"{name}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[infra] {name}: 요소 {len(els)} → 귀속 {kept} ({os.path.getsize(out)/1024:.0f}KB)", flush=True)
    time.sleep(1.0)
    return kept


def cmd_fetch(args):
    os.makedirs(OUTDIR, exist_ok=True)
    sggs = json.load(open(SGG_JSON, encoding="utf-8"))
    if args.only:
        names = set(args.only.split(","))
        sggs = [s for s in sggs if s["name"] in names]
    print(f"[infra] 대상 {len(sggs)}개 → {OUTDIR}", flush=True)
    done = 0
    for i, s in enumerate(sggs):
        out = os.path.join(OUTDIR, f"{s['name']}.txt")
        if os.path.exists(out) and not args.force:
            done += 1; continue
        try:
            fetch_one(s); done += 1
        except Exception as e:
            print(f"[infra] {s['name']} 실패: {str(e)[:80]}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"[infra] 진행 {i+1}/{len(sggs)}", flush=True)
    print(f"[infra] 완료 — 처리 {done}/{len(sggs)}", flush=True)


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
