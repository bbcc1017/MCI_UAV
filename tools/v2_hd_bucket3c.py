#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v2 HD맵 스테이지3c 버킷터 — 신호등(점)/과속방지턱(폴리곤)/가드레일(선) → 1km 타일별.

 · 신호등(C1_TRAFFICLIGHT, Point 7,115) → 타일 버킷 + 1.5m 그리드 디듀프 → `hd_signals/tile.txt`(`lat lon`)
 · 과속방지턱(C4_SPEEDBUMP, MultiPolygon 360) → centroid 버킷 → `hd_speedbumps/tile.txt`(`lat lon…` 외곽링)
 · 가드레일(C3_VEHICLEPROTECTION, MultiLineString 10,294) → 타일 클리핑 → `hd_guardrails/tile.txt`(`lat lon…`)
좌표=EPSG:4326 lon lat, z무시(DEM 드레이프). 실행: UAV env python tools/v2_hd_bucket3c.py --region seoul_gangnamgu
"""
import os, json, argparse
from pyproj import Transformer

P5186 = ("+proj=tmerc +lat_0=38 +lon_0=127 +k=1 +x_0=200000 +y_0=600000 "
         "+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
TILE_M = 1000.0
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HD = os.path.join(REPO, "tools", "nationwide", "hdmap")
_fwd = Transformer.from_crs("EPSG:4326", P5186, always_xy=True)
_inv = Transformer.from_crs(P5186, "EPSG:4326", always_xy=True)


def liang_barsky(x0, y0, x1, y1, xmin, ymin, xmax, ymax):
    dx, dy = x1 - x0, y1 - y0
    p = (-dx, dx, -dy, dy); q = (x0 - xmin, xmax - x0, y0 - ymin, ymax - y0)
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0: return None
        else:
            t = qi / pi
            if pi < 0: u0 = max(u0, t)
            else: u1 = min(u1, t)
    return None if u0 > u1 else (x0 + u0 * dx, y0 + u0 * dy, x0 + u1 * dx, y0 + u1 * dy)


def clip_polyline(en, rect):
    xmin, ymin, xmax, ymax = rect
    pieces, cur = [], []
    for i in range(len(en) - 1):
        seg = liang_barsky(en[i][0], en[i][1], en[i + 1][0], en[i + 1][1], xmin, ymin, xmax, ymax)
        if seg is None:
            if len(cur) >= 2: pieces.append(cur)
            cur = []; continue
        a = (seg[0], seg[1]); b = (seg[2], seg[3])
        if not cur: cur = [a, b]
        elif abs(a[0] - cur[-1][0]) < 1e-6 and abs(a[1] - cur[-1][1]) < 1e-6: cur.append(b)
        else:
            if len(cur) >= 2: pieces.append(cur)
            cur = [a, b]
    if len(cur) >= 2: pieces.append(cur)
    return pieces


def load(layer, region):
    return json.load(open(os.path.join(HD, layer, region + ".geojson"), encoding="utf-8"))["features"]


def emit(buckets, out):
    os.makedirs(out, exist_ok=True)
    for (tx, tz), lines in sorted(buckets.items()):
        with open(os.path.join(out, f"tile_{tx}_{tz}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="seoul_gangnamgu")
    args = ap.parse_args()
    reg = args.region
    ob = os.path.join(REPO, "tools", "nationwide_v2")

    # ── 신호등(점 + 1.5m 디듀프) ──
    sig = {}; seen = set(); ns = 0
    for f in load("lt_p_c1trafficlight", reg):
        g = f.get("geometry") or {}
        if g.get("type") != "Point":
            continue
        c = g["coordinates"]
        if len(c) < 2:
            continue
        lon, lat = c[0], c[1]
        E, N = _fwd.transform(lon, lat)
        key = (round(E / 1.5), round(N / 1.5))       # 1.5m 그리드 디듀프
        if key in seen:
            continue
        seen.add(key)
        sig.setdefault((int(E // 1000), int(N // 1000)), []).append(f"{lat:.7f} {lon:.7f}")
        ns += 1
    emit(sig, os.path.join(ob, "hd_signals", reg))
    print(f"[3c] 신호등 {ns}개(디듀프 후) / {len(sig)}타일", flush=True)

    # ── 과속방지턱(centroid 버킷) ──
    sb = {}; nb = 0
    for f in load("lt_c_c4speedbump", reg):
        g = f.get("geometry") or {}
        if g.get("type") != "MultiPolygon":
            continue
        for poly in g["coordinates"]:
            if not poly:
                continue
            ring = [(c[0], c[1]) for c in poly[0] if len(c) >= 2]
            if len(ring) < 3:
                continue
            clon = sum(p[0] for p in ring) / len(ring); clat = sum(p[1] for p in ring) / len(ring)
            E, N = _fwd.transform(clon, clat)
            coords = " ".join(f"{lat:.7f} {lon:.7f}" for lon, lat in ring)
            sb.setdefault((int(E // 1000), int(N // 1000)), []).append(coords)
            nb += 1
    emit(sb, os.path.join(ob, "hd_speedbumps", reg))
    print(f"[3c] 방지턱 {nb}개 / {len(sb)}타일", flush=True)

    # ── 가드레일(클리핑) ──
    gr = {}; ng = np = 0
    for f in load("lt_l_c3vehicleprotectionsafety", reg):
        g = f.get("geometry") or {}
        if g.get("type") != "MultiLineString":
            continue
        for ls in g["coordinates"]:
            ll = [(c[0], c[1]) for c in ls if len(c) >= 2]
            if len(ll) < 2:
                continue
            en = [_fwd.transform(lon, lat) for lon, lat in ll]
            ng += 1
            Es = [e for e, n in en]; Ns = [n for e, n in en]
            for tx in range(int(min(Es) // 1000), int(max(Es) // 1000) + 1):
                for tz in range(int(min(Ns) // 1000), int(max(Ns) // 1000) + 1):
                    rect = (tx * TILE_M, tz * TILE_M, (tx + 1) * TILE_M, (tz + 1) * TILE_M)
                    for pc in clip_polyline(en, rect):
                        lls = [_inv.transform(e, n) for e, n in pc]
                        coords = " ".join(f"{lat:.7f} {lon:.7f}" for lon, lat in lls)
                        gr.setdefault((tx, tz), []).append(coords)
                        np += 1
    emit(gr, os.path.join(ob, "hd_guardrails", reg))
    print(f"[3c] 가드레일 {ng}선 → 조각 {np} / {len(gr)}타일", flush=True)


if __name__ == "__main__":
    main()
