#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v2 HD맵 타일 버킷터 — 정밀도로지도 노면선표시/보도구간 → 1km EPSG:5186 타일별.

디지털트윈 v2 Phase 2 스테이지3b: HD geojson(EPSG:4326, z 무시=DEM 드레이프)을
 · 노면선(B2_SURFACELINEMARK, MultiLineString) → 타일 Liang-Barsky 클리핑 → `hd_lanes/<region>/tile.txt`
   (`<color> <dash> lat lon …`, color=W/Y/B, dash=S/D)
 · 보도(A4_SUBSIDIARYSECTION, MultiPolygon) → centroid 버킷 → `hd_sidewalks/<region>/tile.txt`
   (`<subtype> lat lon …` 외곽링)
C# TileBakerV2 가 노면선=얇은 리본(도로 위), 보도=드레이프 폴리곤(콘크리트)으로 굽는다.

노면선 색상: type 첫자리 3=청(버스), kind∈{501중앙·502유턴·530·531}=황, 그 외 백. 점선=type 둘째자리 2.
실행: PYTHONIOENCODING=utf-8 /c/Users/User/anaconda3/envs/UAV/python.exe tools/v2_hd_bucket.py --region seoul_gangnamgu
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
YELLOW_KIND = {"501", "502", "530", "531"}


def liang_barsky(x0, y0, x1, y1, xmin, ymin, xmax, ymax):
    dx, dy = x1 - x0, y1 - y0
    p = (-dx, dx, -dy, dy); q = (x0 - xmin, xmax - x0, y0 - ymin, ymax - y0)
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
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


def lane_style(props):
    ty = props.get("type") or ""
    kind = props.get("kind") or ""
    color = "B" if ty[:1] == "3" else ("Y" if kind in YELLOW_KIND else "W")
    dash = "D" if (len(ty) >= 2 and ty[1] == "2") else "S"
    return color, dash


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
    outbase = os.path.join(REPO, "tools", "nationwide_v2")

    # ── 노면선(클리핑) ──
    lanes = {}
    nl = np = 0
    for f in load("lt_l_b2surfacelinemark", reg):
        g = f.get("geometry") or {}
        if g.get("type") != "MultiLineString":
            continue
        color, dash = lane_style(f["properties"])
        head = f"{color} {dash}"
        for ls in g["coordinates"]:
            ll = [(c[0], c[1]) for c in ls if len(c) >= 2]
            if len(ll) < 2:
                continue
            en = [_fwd.transform(lon, lat) for lon, lat in ll]
            nl += 1
            Es = [e for e, n in en]; Ns = [n for e, n in en]
            for tx in range(int(min(Es) // 1000), int(max(Es) // 1000) + 1):
                for tz in range(int(min(Ns) // 1000), int(max(Ns) // 1000) + 1):
                    rect = (tx * TILE_M, tz * TILE_M, (tx + 1) * TILE_M, (tz + 1) * TILE_M)
                    for pc in clip_polyline(en, rect):
                        lls = [_inv.transform(e, n) for e, n in pc]
                        coords = " ".join(f"{lat:.7f} {lon:.7f}" for lon, lat in lls)
                        lanes.setdefault((tx, tz), []).append(f"{head} {coords}")
                        np += 1
    emit(lanes, os.path.join(outbase, "hd_lanes", reg))
    print(f"[hd] 노면선 {nl}선 → 조각 {np} / {len(lanes)}타일", flush=True)

    # ── 보도(centroid 버킷) ──
    sw = {}
    ns = 0
    for f in load("lt_c_a4subsidiarysection", reg):
        g = f.get("geometry") or {}
        if g.get("type") != "MultiPolygon":
            continue
        st = f["properties"].get("subtype") or "0"
        for poly in g["coordinates"]:
            if not poly:
                continue
            ring = [(c[0], c[1]) for c in poly[0] if len(c) >= 2]
            if len(ring) < 3:
                continue
            clon = sum(p[0] for p in ring) / len(ring); clat = sum(p[1] for p in ring) / len(ring)
            E, N = _fwd.transform(clon, clat)
            key = (int(E // 1000), int(N // 1000))
            coords = " ".join(f"{lat:.7f} {lon:.7f}" for lon, lat in ring)
            sw.setdefault(key, []).append(f"{st} {coords}")
            ns += 1
    emit(sw, os.path.join(outbase, "hd_sidewalks", reg))
    print(f"[hd] 보도 {ns}폴리곤 / {len(sw)}타일 → 완료", flush=True)


if __name__ == "__main__":
    main()
