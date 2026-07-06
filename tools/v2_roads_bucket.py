#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v2 도로 타일 클리퍼 — roads2/&lt;region&gt;.txt 를 1km EPSG:5186 타일 bbox 로 클리핑.

디지털트윈 v2 Phase 2 스테이지3(도로면): 도로는 타일 경계를 가로지르는 폴리라인이라 건물(점)과 달리
**Liang-Barsky 세그먼트 클리핑**으로 타일별 조각으로 잘라야 한다(경계서 갭·과overhang 없음). 각 타일이
그 안의 도로 조각만 받는다. C# TileBakerV2.BuildRoads 가 조각별 리본을 타일로컬로 굽는다.

입력 roads2 라인: `class lanes oneway struct lat lon lat lon …`  (⚠️건물과 달리 lat lon 순서)
출력: tools/nationwide_v2/roads_tiles/&lt;region&gt;/tile_x_z.txt (동일 포맷, 타일별 클리핑 조각)
실행: PYTHONIOENCODING=utf-8 /c/Users/User/anaconda3/envs/UAV/python.exe tools/v2_roads_bucket.py --region seoul_gangnamgu
"""
import os, argparse
from pyproj import Transformer

P5186 = ("+proj=tmerc +lat_0=38 +lon_0=127 +k=1 +x_0=200000 +y_0=600000 "
         "+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
TILE_M = 1000.0
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_fwd = Transformer.from_crs("EPSG:4326", P5186, always_xy=True)   # (lon,lat)->(E,N)
_inv = Transformer.from_crs(P5186, "EPSG:4326", always_xy=True)   # (E,N)->(lon,lat)


def liang_barsky(x0, y0, x1, y1, xmin, ymin, xmax, ymax):
    """세그먼트 (x0,y0)-(x1,y1) 를 rect 로 클리핑 → (ax,ay,bx,by) or None."""
    dx, dy = x1 - x0, y1 - y0
    p = (-dx, dx, -dy, dy); q = (x0 - xmin, xmax - x0, y0 - ymin, ymax - y0)
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                u0 = max(u0, t)
            else:
                u1 = min(u1, t)
    if u0 > u1:
        return None
    return (x0 + u0 * dx, y0 + u0 * dy, x0 + u1 * dx, y0 + u1 * dy)


def clip_polyline(en, rect):
    """폴리라인(E,N 리스트) → rect 내부 연속 조각들의 리스트."""
    xmin, ymin, xmax, ymax = rect
    pieces, cur = [], []
    for i in range(len(en) - 1):
        seg = liang_barsky(en[i][0], en[i][1], en[i + 1][0], en[i + 1][1], xmin, ymin, xmax, ymax)
        if seg is None:
            if len(cur) >= 2:
                pieces.append(cur)
            cur = []
            continue
        a = (seg[0], seg[1]); b = (seg[2], seg[3])
        if not cur:
            cur = [a, b]
        elif abs(a[0] - cur[-1][0]) < 1e-6 and abs(a[1] - cur[-1][1]) < 1e-6:
            cur.append(b)
        else:
            if len(cur) >= 2:
                pieces.append(cur)
            cur = [a, b]
    if len(cur) >= 2:
        pieces.append(cur)
    return pieces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="seoul_gangnamgu")
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = args.src or os.path.join(REPO, "tools", "nationwide", "roads2", args.region + ".txt")
    out = args.out or os.path.join(REPO, "tools", "nationwide_v2", "roads_tiles", args.region)
    os.makedirs(out, exist_ok=True)

    buckets = {}
    nroad = npiece = 0
    with open(src, encoding="utf-8") as fp:
        for line in fp:
            tok = line.split()
            if len(tok) < 8:              # head4 + 최소 2점
                continue
            head = tok[:4]                # class lanes oneway struct
            nums = tok[4:]
            ll = []
            for i in range(0, len(nums) - 1, 2):
                try:
                    ll.append((float(nums[i + 1]), float(nums[i])))   # (lon,lat) for transform
                except ValueError:
                    pass
            if len(ll) < 2:
                continue
            en = [_fwd.transform(lon, lat) for (lon, lat) in ll]
            nroad += 1
            Es = [e for e, n in en]; Ns = [n for e, n in en]
            tx0, tx1 = int(min(Es) // 1000), int(max(Es) // 1000)
            tz0, tz1 = int(min(Ns) // 1000), int(max(Ns) // 1000)
            for tx in range(tx0, tx1 + 1):
                for tz in range(tz0, tz1 + 1):
                    rect = (tx * TILE_M, tz * TILE_M, (tx + 1) * TILE_M, (tz + 1) * TILE_M)
                    for pc in clip_polyline(en, rect):
                        lls = [_inv.transform(e, n) for (e, n) in pc]   # (lon,lat)
                        coords = " ".join(f"{lat:.7f} {lon:.7f}" for (lon, lat) in lls)
                        buckets.setdefault((tx, tz), []).append(f"{' '.join(head)} {coords}")
                        npiece += 1

    for (tx, tz), lines in sorted(buckets.items()):
        with open(os.path.join(out, f"tile_{tx}_{tz}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    print(f"[roads] {args.region}: 도로 {nroad} → 조각 {npiece} / {len(buckets)}타일 → {out}", flush=True)


if __name__ == "__main__":
    main()
