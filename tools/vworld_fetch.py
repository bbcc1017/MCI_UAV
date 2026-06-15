# -*- coding: utf-8 -*-
"""
vworld Open API 기반 지역 자료 자동 수집 (수동 다운로드 대체 파일럿)

  1) 건물: WFS lt_c_bldginfo(GIS건물통합정보) — footprint + 층수/높이/용도.
     호출당 1000피처 제한 → 꽉 차면 bbox 4분할 재귀.
  2) 정사영상: WMTS Satellite z18/z19 타일 → 16x16(4096px) 모자이크 + 매니페스트(EPSG:3857/4326 bounds).

사용:
  python vworld_fetch.py buildings --bbox 36.34 127.37 36.36 127.40 --out out_daejeon
  python vworld_fetch.py tiles     --bbox 36.34 127.37 36.36 127.40 --zoom 18 --out out_daejeon

키: 환경변수 VWORLD_API_KEY.
주의: 오픈API 운영정책(일일 트래픽/활용 범위)을 따르세요. 연구 내부 활용 기준으로 설계.
"""
import argparse
import json
import math
import os
import sys
import time

import requests
from PIL import Image

KEY = os.environ.get("VWORLD_API_KEY")
WFS_URL = "https://api.vworld.kr/req/wfs"
WMTS_URL = "https://api.vworld.kr/req/wmts/1.0.0/{key}/Satellite/{z}/{y}/{x}.jpeg"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "MCI-UAV-research/1.0"


# ---------------- 건물 (WFS) ----------------

def wfs_fetch(bbox, max_features=1000):
    """bbox=(minLat,minLon,maxLat,maxLon). WFS 1.1.0 + EPSG:4326은 lat,lon 축 순서."""
    params = {
        "SERVICE": "WFS", "VERSION": "1.1.0", "REQUEST": "GetFeature",
        "TYPENAME": "lt_c_bldginfo",
        "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},EPSG:4326",
        "SRSNAME": "EPSG:4326", "OUTPUT": "application/json",
        "MAXFEATURES": str(max_features), "KEY": KEY, "DOMAIN": "localhost",
    }
    for attempt in range(4):
        try:
            r = SESSION.get(WFS_URL, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001 — 재시도 후 포기
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
            print(f"  retry wfs {bbox}: {e}")


def harvest_buildings(bbox, depth=0):
    """재귀 분할 수집: 1000피처 가득 차면 4분할."""
    js = wfs_fetch(bbox)
    n = js.get("numberReturned", len(js.get("features", [])))
    if n >= 1000 and depth < 10:
        mlat = (bbox[0] + bbox[2]) / 2
        mlon = (bbox[1] + bbox[3]) / 2
        feats = []
        for sub in [(bbox[0], bbox[1], mlat, mlon), (bbox[0], mlon, mlat, bbox[3]),
                    (mlat, bbox[1], bbox[2], mlon), (mlat, mlon, bbox[2], bbox[3])]:
            feats.extend(harvest_buildings(sub, depth + 1))
        return feats
    print(f"  bbox={tuple(round(v, 4) for v in bbox)} -> {n} features")
    time.sleep(0.15)
    return js.get("features", [])


def cmd_buildings(args):
    bbox = tuple(args.bbox)
    feats = harvest_buildings(bbox)
    # 중복 제거(분할 경계에서 겹침) — feature id 기준
    seen, uniq = set(), []
    for f in feats:
        fid = f.get("id")
        if fid in seen:
            continue
        seen.add(fid)
        uniq.append(f)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "buildings.geojson")
    with open(path, "w", encoding="utf-8") as fp:
        json.dump({"type": "FeatureCollection", "features": uniq}, fp, ensure_ascii=False)
    hs = [f["properties"].get("height") or 0 for f in uniq]
    fl = [f["properties"].get("grnd_flr") or 0 for f in uniq]
    print(f"[buildings] {len(uniq)}동 저장 -> {path}")
    print(f"  height>0: {sum(1 for h in hs if h > 0)}동, 층수>0: {sum(1 for v in fl if v > 0)}동 "
          f"(높이 없으면 grnd_flr*3.5 압출 - 기존 SHP 파이프라인과 동일 규칙)")


# ---------------- 정사영상 (WMTS) ----------------

def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def tile_bounds_3857(x, y, z):
    """타일의 EPSG:3857 bounds (Unity 배치용 georef)."""
    n = 2 ** z
    world = 20037508.342789244
    minx = -world + (x / n) * 2 * world
    maxx = -world + ((x + 1) / n) * 2 * world
    maxy = world - (y / n) * 2 * world
    miny = world - ((y + 1) / n) * 2 * world
    return minx, miny, maxx, maxy


def cmd_tiles(args):
    bbox = tuple(args.bbox)  # minLat,minLon,maxLat,maxLon
    z = args.zoom
    x0, y0 = lonlat_to_tile(bbox[1], bbox[2], z)   # 좌상(maxLat)
    x1, y1 = lonlat_to_tile(bbox[3], bbox[0], z)   # 우하(minLat)
    # 16타일(모자이크) 블록 경계로 스냅 — 모자이크에 빈(검정) 영역이 안 생기게
    x0, y0 = (x0 // 16) * 16, (y0 // 16) * 16
    x1, y1 = (x1 // 16) * 16 + 15, (y1 // 16) * 16 + 15
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    print(f"[tiles] z={z} 타일 {nx}x{ny} = {nx * ny}장 (모자이크 {math.ceil(nx / 16)}x{math.ceil(ny / 16)})")
    os.makedirs(args.out, exist_ok=True)
    manifest = {"zoom": z, "tile_origin": [x0, y0], "mosaics": []}
    from concurrent.futures import ThreadPoolExecutor
    from io import BytesIO

    def fetch_one(job):
        tx, ty, gx, gy = job
        url = WMTS_URL.format(key=KEY, z=z, y=gy, x=gx)
        for attempt in range(3):
            try:
                r = SESSION.get(url, timeout=20)
                if r.status_code == 200:
                    return tx, ty, r.content
                return tx, ty, None
            except Exception:  # noqa: BLE001
                time.sleep(1 + attempt)
        return tx, ty, None

    for my in range(math.ceil(ny / 16)):
        for mx in range(math.ceil(nx / 16)):
            mosaic = Image.new("RGB", (4096, 4096))
            jobs = []
            for ty in range(16):
                for tx in range(16):
                    gx, gy = x0 + mx * 16 + tx, y0 + my * 16 + ty
                    if gx > x1 or gy > y1:
                        continue
                    jobs.append((tx, ty, gx, gy))
            with ThreadPoolExecutor(max_workers=8) as ex:
                for tx, ty, data in ex.map(fetch_one, jobs):
                    if data:
                        mosaic.paste(Image.open(BytesIO(data)), (tx * 256, ty * 256))
            time.sleep(args.delay)
            name = f"mosaic_{mx:03d}_{my:03d}.jpg"
            mosaic.save(os.path.join(args.out, name), quality=92)
            tl = tile_bounds_3857(x0 + mx * 16, y0 + my * 16, z)
            br = tile_bounds_3857(x0 + mx * 16 + 15, y0 + my * 16 + 15, z)
            manifest["mosaics"].append({
                "file": name,
                "epsg3857_bounds": [tl[0], br[1], br[2], tl[3]],  # minx,miny,maxx,maxy
            })
            print(f"  {name} 저장")
    with open(os.path.join(args.out, "tiles_manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=1)
    print(f"[tiles] 완료 -> {args.out} (manifest의 3857 bounds를 GDAL로 EPSG:5186 변환해 기존 배치 파이프라인에 연결)")


# ---------------- 패킹 (대형 지역용 경량 포맷) ----------------
# C#(MiniJsonReader)으로 수백MB GeoJSON을 파싱하면 에디터가 수 분 멈춤 →
# 라인 기반 buildings.txt(h lon1 lat1 lon2 lat2 ...)로 변환. 여러 폴더를 한 번에 처리하며
# feature id를 공유해 밴드 경계의 중복 건물 제거.

def cmd_pack(args):
    seen = set()
    for d in args.dirs:
        src = os.path.join(d, "buildings.geojson")
        if not os.path.exists(src):
            print(f"[pack] {src} 없음 - 스킵")
            continue
        with open(src, encoding="utf-8") as fp:
            js = json.load(fp)
        lines, dup = [], 0
        for f in js["features"]:
            fid = f.get("id")
            if fid in seen:
                dup += 1
                continue
            seen.add(fid)
            p = f.get("properties", {})
            h = p.get("height") or 0
            if not h or h < 1:
                h = max(1, p.get("grnd_flr") or 1) * 3.5
            geom = f.get("geometry") or {}
            polys = geom.get("coordinates") or []
            if geom.get("type") == "Polygon":
                polys = [polys]
            # 비정상 높이 보정 — 실존 초고층 존중(롯데타워 555m, 부산 엘시티 411.6m):
            #  · h>600(이론상 국내 최대 초과) 또는 h가 층수×8을 크게 넘는 경우만 단위오류/쓰레기로 판정
            fl = p.get("grnd_flr") or 0
            if h > 600 or (h > 200 and fl > 0 and h > fl * 8):
                h = fl * 3.5 if fl > 0 else min(h, 600)
            for rings in polys:
                if not rings:
                    continue
                ring = rings[0]
                if len(ring) < 3:
                    continue
                # 비건물 쓰레기 피처 방어: footprint 대각이 1.5km 넘으면 건물이 아님(행정경계 등)
                lons = [c[0] for c in ring]; lats = [c[1] for c in ring]
                dx = (max(lons) - min(lons)) * 89000.0
                dz = (max(lats) - min(lats)) * 111000.0
                if dx * dx + dz * dz > 1500.0 ** 2:
                    continue
                coords = " ".join(f"{c[0]:.7f} {c[1]:.7f}" for c in ring)
                lines.append(f"{h:.1f} {coords}")
        out = os.path.join(d, "buildings.txt")
        with open(out, "w", encoding="utf-8") as fp:
            fp.write("\n".join(lines))
        print(f"[pack] {d}: {len(lines)}링 (중복 {dup} 제거) -> buildings.txt "
              f"{os.path.getsize(out) / 1048576:.1f}MB")


# ---------------- 바다 컬링 + 밴드 중복 제거 ----------------
# 직사각 bbox 수집의 순수 바다 모자이크(휘도 std가 매우 낮음)를 제거해 해안선 모양을 따르게 하고,
# 16타일 블록 스냅으로 인접 밴드가 공유하게 된 동일 모자이크(중복)도 제거.

def cmd_cullsea(args):
    import numpy as np
    seen_bounds = set()
    for d in args.dirs:
        mani_path = os.path.join(d, "tiles_manifest.json")
        with open(mani_path, encoding="utf-8") as fp:
            mani = json.load(fp)
        keep, sea, dup = [], 0, 0
        for m in mani["mosaics"]:
            f = os.path.join(d, m["file"])
            if not os.path.exists(f):
                continue
            key = tuple(round(v, 1) for v in m["epsg3857_bounds"])
            if key in seen_bounds:
                os.remove(f); dup += 1; continue
            im = np.asarray(Image.open(f).resize((128, 128))).astype(float)
            if im.mean(axis=2).std() < args.std:
                os.remove(f); sea += 1; continue
            seen_bounds.add(key)
            keep.append(m)
        mani["mosaics"] = keep
        with open(mani_path, "w", encoding="utf-8") as fp:
            json.dump(mani, fp, indent=1)
        print(f"[cullsea] {os.path.basename(d)}: 유지 {len(keep)}, 바다 제거 {sea}, 중복 제거 {dup}")


def main():
    if not KEY:
        sys.exit("VWORLD_API_KEY 환경변수가 없습니다.")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("buildings")
    b.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("minLat", "minLon", "maxLat", "maxLon"))
    b.add_argument("--out", default="vworld_out")
    b.set_defaults(func=cmd_buildings)
    t = sub.add_parser("tiles")
    t.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("minLat", "minLon", "maxLat", "maxLon"))
    t.add_argument("--zoom", type=int, default=18)
    t.add_argument("--delay", type=float, default=0.05)
    t.add_argument("--out", default="vworld_out")
    t.set_defaults(func=cmd_tiles)
    pk = sub.add_parser("pack")
    pk.add_argument("--dirs", nargs="+", required=True)
    pk.set_defaults(func=cmd_pack)
    cs = sub.add_parser("cullsea")
    cs.add_argument("--dirs", nargs="+", required=True)
    cs.add_argument("--std", type=float, default=5.0)
    cs.set_defaults(func=cmd_cullsea)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
