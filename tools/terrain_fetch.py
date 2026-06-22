# -*- coding: utf-8 -*-
"""Copernicus GLO-30(30m) DEM → 시군구별 heightmap(float32 격자 + 메타).

지형(고도)을 디지털트윈에 입히기 위한 1단계. vWorld DEM API는 2019 폐쇄(보안규정)라
무료·무제약인 **Copernicus GLO-30**(AWS 공개버킷)을 GDAL /vsicurl/ 로 직접 윈도우 읽기 한다
(전체 다운로드·API키 불필요). 한국은 전부 N/E 1°타일.

출력(tools/nationwide/terrain/<region>):
  <region>.bin   float32 고도(m), row-major, 북→남 행·서→동 열
  <region>.json  {region,frame,origin_lat,origin_lon,dlat,dlon,nrows,ncols,min_h,max_h,nodata}
Unity 임포터가 node(i,j): lat=origin_lat+i*dlat, lon=origin_lon+j*dlon → RegionRegistry 프레임
변환 + y=고도 로 지형 메시를 굽는다.

실행(GDAL 있는 env):
  /c/Users/User/anaconda3/envs/qgis_batch/python.exe tools/terrain_fetch.py --region daejeon_seogu --unity-copy
"""
import argparse
import json
import os
import sys

import numpy as np
from osgeo import gdal

gdal.UseExceptions()
gdal.SetConfigOption("GDAL_HTTP_UNSAFESSL", "YES")
gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

TOOLS = os.path.dirname(os.path.abspath(__file__))
SGG_JSON = os.path.join(TOOLS, "nationwide", "sgg.json")
OUTDIR = os.path.join(TOOLS, "nationwide", "terrain")
UNITY_DST = os.path.join(TOOLS, os.pardir, "external", "ml-agents", "UAV_test",
                         "Assets", "Scenes", "Regions", "terrain")

S3 = "https://copernicus-dem-30m.s3.amazonaws.com"


def tile_url(lat_deg, lon_deg):
    ns = f"N{lat_deg:02d}" if lat_deg >= 0 else f"S{abs(lat_deg):02d}"
    ew = f"E{lon_deg:03d}" if lon_deg >= 0 else f"W{abs(lon_deg):03d}"
    name = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
    return f"/vsicurl/{S3}/{name}/{name}.tif"


def covering_tiles(min_lat, min_lon, max_lat, max_lon):
    tiles = []
    for la in range(int(np.floor(min_lat)), int(np.floor(max_lat)) + 1):
        for lo in range(int(np.floor(min_lon)), int(np.floor(max_lon)) + 1):
            tiles.append(tile_url(la, lo))
    return tiles


def load_region(name):
    sggs = json.load(open(SGG_JSON, encoding="utf-8"))
    for s in sggs:
        if s["name"] == name:
            return s
    sys.exit(f"시군구 '{name}' 없음")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True, help="시군구 슬러그 (예: daejeon_seogu)")
    ap.add_argument("--pad", type=float, default=0.01, help="bbox 여유(도) — 가장자리 잘림 방지")
    ap.add_argument("--unity-copy", action="store_true")
    args = ap.parse_args()

    s = load_region(args.region)
    frame = s.get("frame")
    minLat, minLon, maxLat, maxLon = s["bbox"]
    minLat -= args.pad; minLon -= args.pad; maxLat += args.pad; maxLon += args.pad

    tiles = covering_tiles(minLat, minLon, maxLat, maxLon)
    print(f"[terrain] {args.region} frame={frame} bbox=({minLat:.4f},{minLon:.4f},{maxLat:.4f},{maxLon:.4f}) "
          f"타일 {len(tiles)}개", flush=True)
    for t in tiles:
        print("   ", t)

    # 여러 타일이면 VRT로 합치고, 하나면 그대로 연다
    if len(tiles) == 1:
        ds = gdal.Open(tiles[0])
    else:
        vrt = gdal.BuildVRT("", tiles)
        ds = vrt
    if ds is None:
        sys.exit("[terrain] DEM 열기 실패 — 네트워크/타일명 확인")

    gt = ds.GetGeoTransform()   # (ox, pw, 0, oy, 0, ph)  ox=좌상단 lon, oy=좌상단 lat, ph<0
    ox, pw, _, oy, _, ph = gt
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()

    # bbox → 픽셀 윈도우
    xoff = max(0, int((minLon - ox) / pw))
    yoff = max(0, int((maxLat - oy) / ph))            # ph<0 → 위쪽(maxLat)이 작은 행
    xend = min(ds.RasterXSize, int(np.ceil((maxLon - ox) / pw)))
    yend = min(ds.RasterYSize, int(np.ceil((minLat - oy) / ph)))
    xsize, ysize = xend - xoff, yend - yoff
    if xsize <= 1 or ysize <= 1:
        sys.exit(f"[terrain] 윈도우 비정상 xsize={xsize} ysize={ysize}")

    arr = band.ReadAsArray(xoff, yoff, xsize, ysize).astype(np.float32)
    # 노드 중심 좌표(행0=북, 열0=서)
    origin_lat = oy + (yoff + 0.5) * ph
    origin_lon = ox + (xoff + 0.5) * pw

    valid = arr if nodata is None else arr[arr != nodata]
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
        # nodata(해상 등)는 0으로 — 내륙 시군구엔 거의 없음
        fill = float(np.nanmin(valid)) if valid.size else 0.0
        arr = np.where(np.isnan(arr), fill, arr).astype(np.float32)
    min_h, max_h = float(arr.min()), float(arr.max())

    os.makedirs(OUTDIR, exist_ok=True)
    bin_path = os.path.join(OUTDIR, f"{args.region}.bin")
    arr.tofile(bin_path)   # float32 row-major (북→남, 서→동)
    meta = {
        "region": args.region, "frame": frame,
        "origin_lat": origin_lat, "origin_lon": origin_lon,
        "dlat": ph, "dlon": pw,           # dlat<0 (행 증가=남쪽)
        "nrows": int(ysize), "ncols": int(xsize),
        "min_h": min_h, "max_h": max_h,
        "source": "Copernicus GLO-30",
    }
    json.dump(meta, open(os.path.join(OUTDIR, f"{args.region}.json"), "w", encoding="utf-8"), indent=1)
    print(f"[terrain] {ysize}x{xsize} 격자 → {bin_path} ({os.path.getsize(bin_path)/1048576:.1f}MB) "
          f"고도 {min_h:.0f}~{max_h:.0f}m", flush=True)

    if args.unity_copy:
        os.makedirs(os.path.abspath(UNITY_DST), exist_ok=True)
        import shutil
        for ext in (".bin", ".json"):
            shutil.copy(os.path.join(OUTDIR, args.region + ext),
                        os.path.join(os.path.abspath(UNITY_DST), args.region + ext))
        print(f"[terrain] Unity 복사 → {os.path.abspath(UNITY_DST)}", flush=True)


if __name__ == "__main__":
    main()
