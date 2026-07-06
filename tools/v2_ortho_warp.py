#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v2 정사 프리워프 — z19 모자이크(EPSG:3857) → 1km EPSG:5186 타일 4096² JPG.

디지털트윈 v2 Phase 2: 소스 정사(vworld_fetch tiles)는 웹메르카토르(3857)라 5186 정렬
타일과 ~1-2° 수렴각+1.26배 스케일 차이가 난다. GDAL 로 각 1km 5186 타일 창(-te)에
정확히 워프해 두면, C# TileBakerV2 는 5186 정렬 드레이프 메시 + 자명한 UV(local/1000+0.5)
만 담당하면 된다(3857 재투영을 C#에서 안 함).

⚠️ GDAL3 축순서 함정 회피: 타깃 5186 을 EPSG 코드가 아니라 PROJ 문자열로 지정 →
   항상 (easting, northing) 순서. KoreaGeo(=EPSG:5186)와 동일 파라미터.
⚠️ qgis_batch env(GDAL) 전용 — UAV env 엔 raster 라이브러리 없음.

산출: <out>/tile_<x>_<z>.jpg (기본 Assets/TilesV2/<region>/ortho) + tiles_index.json
      (C# 베이커가 읽어 타일 순회; 타일 x,z 는 floor(E/1000)=TileIndex 와 동일 정의).

실행:
  PYTHONIOENCODING=utf-8 /c/Users/User/anaconda3/envs/qgis_batch/python.exe \
    tools/v2_ortho_warp.py --region seoul_gangnamgu [--limit 4] [--force]
"""
import os, sys, json, math, argparse, glob
from osgeo import gdal, osr
from pyproj import Transformer

gdal.UseExceptions()

# EPSG:5186 (Korea 2000 / Central Belt 2010) — PROJ 문자열(E/N 순서 강제). KoreaGeo 와 동일.
P5186 = ("+proj=tmerc +lat_0=38 +lon_0=127 +k=1 +x_0=200000 +y_0=600000 "
         "+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
TILE_M = 1000.0

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_mosaic_vrt(src_dir, manifest):
    """모자이크 JPG 들에 3857 georef 를 붙여 하나의 가상 모자이크(.vrt)로 결합."""
    part_vrts = []
    for mo in manifest["mosaics"]:
        jpg = os.path.join(src_dir, mo["file"])
        if not os.path.exists(jpg):
            continue
        minx, miny, maxx, maxy = mo["epsg3857_bounds"]
        vrt = f"/vsimem/{os.path.splitext(mo['file'])[0]}.vrt"
        # -a_ullr ulx uly lrx lry = minx,maxy,maxx,miny (north-up)
        gdal.Translate(vrt, jpg, format="VRT", outputSRS="EPSG:3857",
                       outputBounds=[minx, maxy, maxx, miny])
        part_vrts.append(vrt)
    if not part_vrts:
        raise SystemExit("모자이크 JPG 를 못 찾음")
    mosaic = "/vsimem/_mosaic.vrt"
    gdal.BuildVRT(mosaic, part_vrts)
    return mosaic, part_vrts


def coverage_bounds_5186(manifest):
    """모자이크 union 3857 bounds → 5186 AABB (경계 조밀 샘플로 회전 반영)."""
    xs = [b for mo in manifest["mosaics"] for b in (mo["epsg3857_bounds"][0], mo["epsg3857_bounds"][2])]
    ys = [b for mo in manifest["mosaics"] for b in (mo["epsg3857_bounds"][1], mo["epsg3857_bounds"][3])]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    tr = Transformer.from_crs("EPSG:3857", P5186, always_xy=True)
    # 경계선 따라 샘플(회전으로 코너만으론 부족)
    n = 40
    E, N = [], []
    for i in range(n + 1):
        t = i / n
        for (x, y) in ((minx + (maxx - minx) * t, miny), (minx + (maxx - minx) * t, maxy),
                       (minx, miny + (maxy - miny) * t), (maxx, miny + (maxy - miny) * t)):
            e, nn = tr.transform(x, y)
            E.append(e); N.append(nn)
    return min(E), min(N), max(E), max(N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="seoul_gangnamgu")
    ap.add_argument("--src", default=None, help="ortho19/<region> (모자이크+manifest)")
    ap.add_argument("--out", default=None, help="타일 JPG 출력 (기본 Assets/TilesV2/<region>/ortho)")
    ap.add_argument("--index_out", default=None)
    ap.add_argument("--px", type=int, default=4096)
    ap.add_argument("--resample", default="bilinear")
    ap.add_argument("--jpeg_quality", type=int, default=88)
    ap.add_argument("--limit", type=int, default=0, help="디버그: 타일 수 상한(0=전체)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src_dir = args.src or os.path.join(REPO, "tools", "nationwide_v2", "ortho19", args.region)
    out_dir = args.out or os.path.join(REPO, "external", "ml-agents", "UAV_test",
                                       "Assets", "TilesV2", args.region, "ortho")
    idx_out = args.index_out or os.path.join(REPO, "tools", "nationwide_v2",
                                             "ortho19_tiles", args.region, "tiles_index.json")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(idx_out), exist_ok=True)

    manifest = json.load(open(os.path.join(src_dir, "tiles_manifest.json"), encoding="utf-8"))
    print(f"[warp] {args.region}: 모자이크 {len(manifest['mosaics'])}장 → VRT 결합", flush=True)
    mosaic_vrt, _ = build_mosaic_vrt(src_dir, manifest)

    minE, minN, maxE, maxN = coverage_bounds_5186(manifest)
    x0, x1 = math.floor(minE / TILE_M), math.floor(maxE / TILE_M)
    z0, z1 = math.floor(minN / TILE_M), math.floor(maxN / TILE_M)
    print(f"[warp] 5186 커버리지 E[{minE:.0f},{maxE:.0f}] N[{minN:.0f},{maxN:.0f}] "
          f"→ 타일 x[{x0}..{x1}] z[{z0}..{z1}] ({(x1-x0+1)*(z1-z0+1)}칸 후보)", flush=True)

    warp_srs = osr.SpatialReference(); warp_srs.ImportFromProj4(P5186)
    tiles, done, empty = [], 0, 0
    cand = [(x, z) for z in range(z0, z1 + 1) for x in range(x0, x1 + 1)]
    for (x, z) in cand:
        if args.limit and done >= args.limit:
            break
        te = [x * TILE_M, z * TILE_M, (x + 1) * TILE_M, (z + 1) * TILE_M]  # minE,minN,maxE,maxN
        jpg = os.path.join(out_dir, f"tile_{x}_{z}.jpg")
        rel = os.path.relpath(jpg, os.path.join(REPO, "external", "ml-agents", "UAV_test", "Assets"))
        if os.path.exists(jpg) and not args.force:
            tiles.append({"x": x, "z": z, "bounds5186": te, "asset": "Assets/" + rel.replace("\\", "/")})
            done += 1
            continue
        mem = gdal.Warp("", mosaic_vrt, format="MEM", dstSRS=P5186,
                        outputBounds=te, width=args.px, height=args.px,
                        resampleAlg=args.resample, dstNodata=0, outputType=gdal.GDT_Byte,
                        multithread=True)
        arr = mem.ReadAsArray()
        if arr is None or int(arr.max()) == 0:
            empty += 1
            mem = None
            continue
        gdal.Translate(jpg, mem, format="JPEG", creationOptions=[f"QUALITY={args.jpeg_quality}"])
        mem = None
        # JPEG 사이드카(.aux.xml 등) 정리
        for junk in glob.glob(jpg + ".aux.xml"):
            os.remove(junk)
        tiles.append({"x": x, "z": z, "bounds5186": te, "asset": "Assets/" + rel.replace("\\", "/")})
        done += 1
        if done % 10 == 0:
            print(f"[warp] {done} 타일 완료(빈칸 {empty} 스킵)", flush=True)

    index = {
        "region": args.region, "frame": manifest.get("frame") or "Sudogwon",
        "crs": "EPSG:5186", "tile_m": TILE_M, "px": args.px,
        "tile_count": len(tiles), "tiles": tiles,
    }
    json.dump(index, open(idx_out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[warp] 완료 — 타일 {len(tiles)}개(빈칸 {empty} 스킵) → {out_dir}", flush=True)
    print(f"[warp] 인덱스 → {idx_out}", flush=True)


if __name__ == "__main__":
    main()
