#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v2 건물 타일 버킷터 — buildings.geojson(76k동) → 1km EPSG:5186 타일별 compact txt.

디지털트윈 v2 Phase 2 스테이지2: C# TileBakerV2 가 77MB geojson 을 직접 파싱하지 않도록,
Python(UAV env: json+pyproj)이 각 동을 centroid→5186→타일(floor(E/1000))로 버킷 분류하고
타일별로 `높이 유형 lon lat lon lat …`(외곽링) 을 쓴다. C# 는 이 소형 파일을
읽어 타일로컬 압출 + DEM 드레이프(바닥고도는 C# TerrainHeight 로 산출).

유형(2026-07-11 파사드 v2, usability 용도코드+층수 폴백 → 셰이더 버텍스컬러):
  0=일반  1=아파트(02000 공동주택·12층+)  2=업무(14000·6층+)  3=근생상가(03000/04000)  4=단독주택(01000·1~2층)

⚠️ 홀(내부링)은 이번 버전에서 미방출(외곽링만) — 대부분 단순 폴리곤. 홀 지원은 후속(geojson 원본 보존).
⚠️ MultiPolygon 은 파트별로 별도 동으로 분리.

산출: tools/nationwide_v2/buildings_tiles/&lt;region&gt;/tile_x_z.txt + buildings_manifest.json
실행: PYTHONIOENCODING=utf-8 /c/Users/User/anaconda3/envs/UAV/python.exe tools/v2_buildings_bucket.py --region seoul_gangnamgu
"""
import os, json, argparse
from pyproj import Transformer

P5186 = ("+proj=tmerc +lat_0=38 +lon_0=127 +k=1 +x_0=200000 +y_0=600000 "
         "+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
TILE_M = 1000.0
FLOOR_M = 3.5
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_tr = Transformer.from_crs("EPSG:4326", P5186, always_xy=True)   # (lon,lat)->(E,N)


def parts(geom):
    """geometry → [(outer_ring, n_inner_rings), ...] 파트별."""
    t = geom.get("type"); c = geom.get("coordinates")
    if t == "Polygon":
        return [(c[0], len(c) - 1)] if c else []
    if t == "MultiPolygon":
        return [(poly[0], len(poly) - 1) for poly in c if poly]
    return []


def btype(props):
    """건물 유형 — usability(GIS건물통합 용도코드) 우선, 결측 시 층수/명칭 폴백."""
    u = str(props.get("usability") or "").strip()
    fl = int(props.get("grnd_flr") or 0)
    nm = props.get("bld_nm") or ""
    if u == "02000" or "아파트" in nm:
        return 1
    if u == "14000":
        return 2
    if u in ("03000", "04000"):
        return 3
    if u == "01000":
        return 4
    if fl >= 12:
        return 1
    if fl >= 6:
        return 2
    if 0 < fl <= 2:
        return 4
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="seoul_gangnamgu")
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = args.src or os.path.join(REPO, "tools", "nationwide_v2", "buildings", args.region, "buildings.geojson")
    out = args.out or os.path.join(REPO, "tools", "nationwide_v2", "buildings_tiles", args.region)
    os.makedirs(out, exist_ok=True)

    print(f"[bldg] {args.region}: {src} 로드", flush=True)
    gj = json.load(open(src, encoding="utf-8"))
    feats = gj["features"]
    print(f"[bldg] {len(feats)} feature → 타일 버킷", flush=True)

    buckets = {}   # (x,z) -> list of "h lon lat ..." lines
    total = holes = multi = 0
    for f in feats:
        geom = f.get("geometry"); props = f.get("properties") or {}
        if not geom:
            continue
        pl = parts(geom)
        if len(pl) > 1:
            multi += 1
        h = props.get("height") or 0
        if not h or h < 1:
            h = max(1.0, float(props.get("grnd_flr") or 0)) * FLOOR_M
        h = min(max(float(h), 3.0), 600.0)
        t = btype(props)
        for ring, n_inner in pl:
            if not ring or len(ring) < 3:
                continue
            if n_inner:
                holes += 1
            lons = [p[0] for p in ring]; lats = [p[1] for p in ring]
            clon = sum(lons) / len(lons); clat = sum(lats) / len(lats)
            E, N = _tr.transform(clon, clat)
            key = (int(E // 1000), int(N // 1000))
            coords = " ".join(f"{lon:.7f} {lat:.7f}" for lon, lat in ring)
            buckets.setdefault(key, []).append(f"{h:.1f} {t} {coords}")
            total += 1

    manifest = {"region": args.region, "crs": "EPSG:4326_rings", "tile_m": TILE_M,
                "building_count": total, "tiles": {}}
    for (tx, tz), lines in sorted(buckets.items()):
        with open(os.path.join(out, f"tile_{tx}_{tz}.txt"), "w", encoding="utf-8") as fp:
            fp.write("\n".join(lines))
        manifest["tiles"][f"{tx}_{tz}"] = len(lines)
    json.dump(manifest, open(os.path.join(out, "buildings_manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[bldg] 완료 — {total}동 → {len(buckets)}타일 (홀 {holes} 미방출, MultiPolygon {multi}) → {out}", flush=True)


if __name__ == "__main__":
    main()
