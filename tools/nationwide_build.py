# -*- coding: utf-8 -*-
"""남한 전체(시군구 단위) vworld 빌드 오케스트레이터.

분할 설계:
  · 단위 = z16 16x16 타일 블록(4096px 모자이크, 약 7.8km).
  · 시군구 폴리곤(WFS lt_c_adsigg)으로 블록을 프레임(RegionRegistry 좌표섬)별로
    정확히 1회씩 시군구에 할당 → 모자이크 중복/빈땅 없음. 바다 블록 자동 제외.
  · 프레임 경계 블록은 양쪽 프레임에 모두 등장(섬이 달라 unity 충돌 없음).
  · 건물은 블록당 1회 수집(재귀 1000피처 분할), 어셈블 때 중심점 폴리곤 판정으로
    시군구에 귀속 → 중복 없음.
  · 씬 = 시군구당 1개. tiles_manifest.json에 "frame" 필드 추가 →
    VworldRegionImporter가 모든 정점에 해당 프레임을 강제(가장자리 늘어남 버그 방지).

스테이지(전부 재실행 안전·재개 가능):
  python tools/nationwide_build.py sgg        # 시군구 폴리곤 수집 → sgg.json
  python tools/nationwide_build.py alloc      # 블록 할당 → alloc.json
  python tools/nationwide_build.py seed       # 기존 vw_*(zoom16) 모자이크 재활용
  python tools/nationwide_build.py fetch [--only daejeon_donggu,...] [--tiles-only|--buildings-only]
  python tools/nationwide_build.py assemble [--only ...]   # 시군구별 vw 디렉터리 + pack
  python tools/nationwide_build.py sggindex   # Unity 런타임용 sgg_index.json
  python tools/nationwide_build.py status

키: 환경변수 VWORLD_API_KEY. python = anaconda3/envs/MCI.
"""
import argparse
import json
import math
import os
import re
import sys
import time

import requests

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(TOOLS, "nationwide")
BLOCKS = os.path.join(ROOT, "blocks")
SGGDIR = os.path.join(ROOT, "sgg")
SGG_JSON = os.path.join(ROOT, "sgg.json")
ALLOC_JSON = os.path.join(ROOT, "alloc.json")
STATE_JSON = os.path.join(ROOT, "blocks_state.json")
UNITY_REGIONS = os.path.join(os.path.dirname(TOOLS), "external", "ml-agents",
                             "UAV_test", "Assets", "Scenes", "Regions")

KEY = os.environ.get("VWORLD_API_KEY")
WFS_URL = "https://api.vworld.kr/req/wfs"
WMTS_URL = "https://api.vworld.kr/req/wmts/1.0.0/{key}/Satellite/{z}/{y}/{x}.jpeg"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "MCI-UAV-research/1.0"
Z = 16
WORLD = 20037508.342789244

# 시도코드 → (slug, RegionRegistry 프레임). 신구 행정코드 병기(강원 42/51, 전북 45/52).
SIDO = {
    "11": ("seoul", "Sudogwon"), "26": ("busan", "Busan"), "27": ("daegu", "Gyeongbuk"),
    "28": ("incheon", "Sudogwon"), "29": ("gwangju", "Jeonnam"), "30": ("daejeon", "Daejeon"),
    "31": ("ulsan", "Busan"), "36": ("sejong", "Daejeon"), "41": ("gyeonggi", "Sudogwon"),
    "42": ("gangwon", "Gangwon"), "51": ("gangwon", "Gangwon"),
    "43": ("chungbuk", "Chungbuk"), "44": ("chungnam", "Chungnam"),
    "45": ("jeonbuk", "Jeonbuk"), "52": ("jeonbuk", "Jeonbuk"),
    "46": ("jeonnam", "Jeonnam"), "47": ("gyeongbuk", "Gyeongbuk"),
    "48": ("gyeongnam", "Gyeongnam"), "50": ("jeju", "Jeju"),
}


# ---------------- 공통 ----------------

def tile_to_lonlat(x, y, z=Z):
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def lonlat_to_tile(lon, lat, z=Z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def block_bounds_3857(bx, by):
    """블록(bx,by) = 타일 x[bx*16, bx*16+15], y[by*16, by*16+15]."""
    n = 2 ** Z
    minx = -WORLD + (bx * 16 / n) * 2 * WORLD
    maxx = -WORLD + ((bx + 1) * 16 / n) * 2 * WORLD
    maxy = WORLD - (by * 16 / n) * 2 * WORLD
    miny = WORLD - ((by + 1) * 16 / n) * 2 * WORLD
    return minx, miny, maxx, maxy


def block_bounds_lonlat(bx, by):
    minx, miny, maxx, maxy = block_bounds_3857(bx, by)
    lon0 = minx / WORLD * 180.0
    lon1 = maxx / WORLD * 180.0
    lat0 = math.degrees(2 * math.atan(math.exp(miny / WORLD * math.pi)) - math.pi / 2)
    lat1 = math.degrees(2 * math.atan(math.exp(maxy / WORLD * math.pi)) - math.pi / 2)
    return lat0, lon0, lat1, lon1   # minLat,minLon,maxLat,maxLon


def simplify_ring(ring, tol=0.001):
    """Douglas-Peucker(반복 스택판). ring=[[lon,lat],...]."""
    if len(ring) <= 4:
        return ring
    keep = [False] * len(ring)
    keep[0] = keep[-1] = True
    stack = [(0, len(ring) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        ax, ay = ring[i0]
        bx, by = ring[i1]
        dx, dy = bx - ax, by - ay
        norm = math.hypot(dx, dy)
        dmax, imax = 0.0, -1
        for i in range(i0 + 1, i1):
            if norm < 1e-12:  # 닫힌 링: 기준 선분이 0길이 → 점-점 거리
                d = math.hypot(ring[i][0] - ax, ring[i][1] - ay)
            else:
                d = abs(dx * (ay - ring[i][1]) - dy * (ax - ring[i][0])) / norm
            if d > dmax:
                dmax, imax = d, i
        if dmax > tol:
            keep[imax] = True
            stack.append((i0, imax))
            stack.append((imax, i1))
    return [p for p, k in zip(ring, keep) if k]


def load_sgg():
    with open(SGG_JSON, encoding="utf-8") as fp:
        return json.load(fp)


def sgg_paths(sgg):
    """시군구의 outer ring들을 matplotlib Path 목록으로."""
    from matplotlib.path import Path
    return [Path(r) for r in sgg["rings"]]


# ---------------- 1) 시군구 폴리곤 ----------------

def cmd_sgg(args):
    os.makedirs(ROOT, exist_ok=True)
    found = {}
    cells = []
    lat = 33.0
    while lat < 38.8:
        lon = 124.5
        while lon < 132.0:
            cells.append((lat, lon, lat + 1.0, lon + 1.0))
            lon += 1.0
        lat += 1.0
    for i, bb in enumerate(cells):
        params = {
            "SERVICE": "WFS", "VERSION": "1.1.0", "REQUEST": "GetFeature",
            "TYPENAME": "lt_c_adsigg",
            "BBOX": f"{bb[0]},{bb[1]},{bb[2]},{bb[3]},EPSG:4326",
            "SRSNAME": "EPSG:4326", "OUTPUT": "application/json",
            "MAXFEATURES": "1000", "KEY": KEY, "DOMAIN": "localhost",
        }
        for attempt in range(4):
            try:
                r = SESSION.get(WFS_URL, params=params, timeout=60)
                r.raise_for_status()
                js = r.json()
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
                print(f"  retry cell {bb}: {e}")
        feats = js.get("features", [])
        for f in feats:
            p = f["properties"]
            cd = p["sig_cd"]
            geom = f["geometry"]
            polys = geom["coordinates"]
            if geom["type"] == "Polygon":
                polys = [polys]
            rings = []
            for rs in polys:
                if not rs:
                    continue
                ring = simplify_ring([list(c) for c in rs[0]], tol=0.0008)
                if len(ring) >= 4:
                    rings.append(ring)
            if not rings:
                continue
            sido = SIDO.get(cd[:2])
            if sido is None:
                print(f"  !! 미지정 시도코드 {cd} ({p.get('full_nm')}) — 스킵")
                continue
            if cd in found:
                # 같은 시군구의 다른 파트(섬 등) — 링 병합(중복 파트는 시그니처로 스킵)
                s = found[cd]
                exist = {(len(r), round(r[0][0], 6), round(r[0][1], 6)) for r in s["rings"]}
                for ring in rings:
                    sig = (len(ring), round(ring[0][0], 6), round(ring[0][1], 6))
                    if sig in exist:
                        continue
                    s["rings"].append(ring)
                lons = [c[0] for ring in s["rings"] for c in ring]
                lats = [c[1] for ring in s["rings"] for c in ring]
                s["bbox"] = [min(lats), min(lons), max(lats), max(lons)]
                continue
            eng = re.sub(r"[^a-z]", "", (p.get("sig_eng_nm") or "").split(",")[0].lower())
            slug = f"{sido[0]}_{eng or cd}"
            lons = [c[0] for ring in rings for c in ring]
            lats = [c[1] for ring in rings for c in ring]
            found[cd] = {
                "code": cd, "name": slug, "kor": p.get("full_nm", ""),
                "frame": sido[1],
                "bbox": [min(lats), min(lons), max(lats), max(lons)],
                "rings": rings,
            }
        print(f"[sgg] cell {i + 1}/{len(cells)} {bb[:2]} -> 누적 {len(found)}")
        time.sleep(0.2)
    # slug 충돌 처리(코드 접미)
    by_slug = {}
    for s in found.values():
        by_slug.setdefault(s["name"], []).append(s)
    for slug, lst in by_slug.items():
        if len(lst) > 1:
            for s in lst:
                s["name"] = f"{slug}{s['code']}"
    out = sorted(found.values(), key=lambda s: s["code"])
    with open(SGG_JSON, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False)
    print(f"[sgg] {len(out)}개 시군구 -> {SGG_JSON} "
          f"({os.path.getsize(SGG_JSON) / 1048576:.1f}MB)")


# ---------------- 2) 블록 할당 ----------------

def cmd_alloc(args):
    import numpy as np
    sggs = load_sgg()
    frames = {}
    for s in sggs:
        frames.setdefault(s["frame"], []).append(s)
    alloc = {"zoom": Z, "frames": {}}
    uniq = set()
    for fname, lst in sorted(frames.items()):
        paths = {s["name"]: sgg_paths(s) for s in lst}
        verts = {s["name"]: np.array([c for r in s["rings"] for c in r]) for s in lst}
        # 프레임 전체 블록 범위
        minLat = min(s["bbox"][0] for s in lst)
        minLon = min(s["bbox"][1] for s in lst)
        maxLat = max(s["bbox"][2] for s in lst)
        maxLon = max(s["bbox"][3] for s in lst)
        x0, y0 = lonlat_to_tile(minLon, maxLat)
        x1, y1 = lonlat_to_tile(maxLon, minLat)
        bx0, by0, bx1, by1 = x0 // 16, y0 // 16, x1 // 16, y1 // 16
        entries = []
        N = 7  # NxN 내부 샘플
        for by in range(by0, by1 + 1):
            for bx in range(bx0, bx1 + 1):
                lat0, lon0, lat1, lon1 = block_bounds_lonlat(bx, by)
                pts = np.array([[lon0 + (lon1 - lon0) * (i + 0.5) / N,
                                 lat0 + (lat1 - lat0) * (j + 0.5) / N]
                                for j in range(N) for i in range(N)])
                best, bestN = None, 0
                for s in lst:
                    bb = s["bbox"]
                    if lat1 < bb[0] or lat0 > bb[2] or lon1 < bb[1] or lon0 > bb[3]:
                        continue
                    inside = np.zeros(len(pts), dtype=bool)
                    for p in paths[s["name"]]:
                        inside |= p.contains_points(pts)
                    n = int(inside.sum())
                    # 경계 정점 포함 테스트 — 샘플 격자가 놓치는 작은 섬/슬리버 구제
                    if n == 0:
                        v = verts[s["name"]]
                        nv = int(((v[:, 0] >= lon0) & (v[:, 0] <= lon1)
                                  & (v[:, 1] >= lat0) & (v[:, 1] <= lat1)).sum())
                        if nv > 0:
                            n = 1  # 땅 존재 증거(소유 경쟁에선 최저 가중)
                    if n > bestN:
                        best, bestN = s["name"], n
                if best:
                    entries.append([bx, by, best])
                    uniq.add((bx, by))
        alloc["frames"][fname] = entries
        print(f"[alloc] {fname}: 블록 {len(entries)}")
    alloc["unique_blocks"] = sorted([list(b) for b in uniq])
    with open(ALLOC_JSON, "w", encoding="utf-8") as fp:
        json.dump(alloc, fp)
    total = sum(len(v) for v in alloc["frames"].values())
    print(f"[alloc] 프레임 합 {total}, 고유 블록 {len(uniq)} "
          f"(경계 중복 {total - len(uniq)}) -> {ALLOC_JSON}")


# ---------------- 3) 기존 모자이크 시드 ----------------

def cmd_seed(args):
    os.makedirs(BLOCKS, exist_ok=True)
    n_copy = n_skip = 0
    import shutil
    for d in sorted(os.listdir(TOOLS)):
        if not d.startswith("vw_"):
            continue
        mf = os.path.join(TOOLS, d, "tiles_manifest.json")
        if not os.path.isfile(mf):
            continue
        with open(mf, encoding="utf-8") as fp:
            man = json.load(fp)
        if man.get("zoom") != Z:
            print(f"[seed] {d}: zoom={man.get('zoom')} != {Z} — 스킵")
            continue
        for m in man["mosaics"]:
            minx, miny, maxx, maxy = m["epsg3857_bounds"]
            n = 2 ** Z
            tx = (minx + WORLD) / (2 * WORLD) * n
            ty = (WORLD - maxy) / (2 * WORLD) * n
            bx, by = round(tx) // 16, round(ty) // 16
            # 정렬 검증(블록 경계와 1m 이내)
            eb = block_bounds_3857(bx, by)
            if abs(eb[0] - minx) > 1 or abs(eb[3] - maxy) > 1:
                continue
            dst = os.path.join(BLOCKS, f"m_{bx}_{by}.jpg")
            src = os.path.join(TOOLS, d, m["file"])
            if os.path.exists(dst) or not os.path.exists(src):
                n_skip += 1
                continue
            shutil.copy2(src, dst)
            n_copy += 1
    print(f"[seed] 복사 {n_copy}, 스킵(중복/누락) {n_skip}")


# ---------------- 4) 페치 ----------------

def load_state():
    if os.path.exists(STATE_JSON):
        with open(STATE_JSON, encoding="utf-8") as fp:
            return json.load(fp)
    return {}


def save_state(st):
    tmp = STATE_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(st, fp)
    os.replace(tmp, STATE_JSON)


def harvest_buildings(bbox, depth=0):
    params = {
        "SERVICE": "WFS", "VERSION": "1.1.0", "REQUEST": "GetFeature",
        "TYPENAME": "lt_c_bldginfo",
        "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},EPSG:4326",
        "SRSNAME": "EPSG:4326", "OUTPUT": "application/json",
        "MAXFEATURES": "1000", "KEY": KEY, "DOMAIN": "localhost",
    }
    js = None
    for attempt in range(5):
        try:
            r = SESSION.get(WFS_URL, params=params, timeout=40)
            r.raise_for_status()
            js = r.json()
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    n = js.get("numberReturned", len(js.get("features", [])))
    if n >= 1000 and depth < 12:
        mlat = (bbox[0] + bbox[2]) / 2
        mlon = (bbox[1] + bbox[3]) / 2
        feats = []
        for sub in [(bbox[0], bbox[1], mlat, mlon), (bbox[0], mlon, mlat, bbox[3]),
                    (mlat, bbox[1], bbox[2], mlon), (mlat, mlon, bbox[2], bbox[3])]:
            feats.extend(harvest_buildings(sub, depth + 1))
        return feats
    time.sleep(0.1)
    return js.get("features", [])


def fetch_block_tiles(bx, by, workers=6):
    from concurrent.futures import ThreadPoolExecutor
    from io import BytesIO
    from PIL import Image
    x0, y0 = bx * 16, by * 16
    jobs = [(tx, ty) for ty in range(16) for tx in range(16)]
    fails = 0

    def one(j):
        tx, ty = j
        url = WMTS_URL.format(key=KEY, z=Z, y=y0 + ty, x=x0 + tx)
        for attempt in range(3):
            try:
                r = SESSION.get(url, timeout=20)
                if r.status_code == 200:
                    # JPEG 매직 검사 — 커버리지 밖/에러문서가 200으로 오는 경우 방어
                    if r.content[:2] == b"\xff\xd8":
                        return tx, ty, r.content
                    return tx, ty, ("BAD", r.content[:120])
                if r.status_code in (403, 429):
                    return tx, ty, "BLOCKED"
                return tx, ty, None
            except Exception:  # noqa: BLE001
                time.sleep(1 + attempt)
        return tx, ty, None

    mosaic = Image.new("RGB", (4096, 4096))
    blocked = False
    bad_sample = None
    n_ok = n_bad = 0   # bad=비이미지 200(커버리지 밖, 영구) / fails=네트워크 등(일시)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for tx, ty, data in ex.map(one, jobs):
            if data == "BLOCKED":
                blocked = True
            elif isinstance(data, tuple):
                n_bad += 1
                if bad_sample is None:
                    bad_sample = data[1]
            elif data:
                try:
                    mosaic.paste(Image.open(BytesIO(data)), (tx * 256, ty * 256))
                    n_ok += 1
                except Exception:  # noqa: BLE001 — 손상 이미지
                    fails += 1
            else:
                fails += 1
    if blocked:
        return None, "blocked"
    if n_ok == 0:
        # 전 타일 결손 — 쿼터 메시지/할당 오류 진단용으로 내용 노출
        return None, f"no-tiles bad={n_bad} sample={bad_sample!r}"
    if fails > 64:  # 일시 실패가 1/4 이상이면 불완전 — 저장 안 하고 다음 런에 재시도
        return None, f"fails={fails}"
    return mosaic, None   # 영구 결손(n_bad)은 검정으로 남기고 저장


def cmd_fetch(args):
    os.makedirs(BLOCKS, exist_ok=True)
    with open(ALLOC_JSON, encoding="utf-8") as fp:
        alloc = json.load(fp)
    blocks = [tuple(b) for b in alloc["unique_blocks"]]
    if args.only:
        names = set(args.only.split(","))
        sel = set()
        for entries in alloc["frames"].values():
            for bx, by, sgg in entries:
                if sgg in names:
                    sel.add((bx, by))
        blocks = [b for b in blocks if b in sel]
    st = load_state()
    todo_t = [b for b in blocks
              if not os.path.exists(os.path.join(BLOCKS, f"m_{b[0]}_{b[1]}.jpg"))]
    todo_b = [b for b in blocks
              if not os.path.exists(os.path.join(BLOCKS, f"b_{b[0]}_{b[1]}.json"))]
    print(f"[fetch] 대상 블록 {len(blocks)} | 타일 잔여 {len(todo_t)} | 건물 잔여 {len(todo_b)}")
    backoff = 60
    n_done = 0
    t0 = time.time()
    for i, (bx, by) in enumerate(blocks):
        mpath = os.path.join(BLOCKS, f"m_{bx}_{by}.jpg")
        bpath = os.path.join(BLOCKS, f"b_{bx}_{by}.json")
        unav = mpath + ".unavailable"   # 영구 비제공(DMZ/서해5도 등 보안지역) 마커
        if not args.buildings_only and not os.path.exists(mpath) and not os.path.exists(unav):
            while True:
                mosaic, err = fetch_block_tiles(bx, by)
                if err == "blocked":
                    print(f"[fetch] 쿼터/차단 감지 — {backoff}s 대기")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 3600)
                    continue
                backoff = 60
                if mosaic is not None:
                    mosaic.save(mpath, quality=92)
                elif err and err.startswith("no-tiles"):
                    with open(unav, "w") as fp:
                        fp.write(err)
                    print(f"[fetch] 블록 {bx},{by} 영구 비제공(보안지역) — 마커 기록")
                else:
                    print(f"[fetch] 블록 {bx},{by} 타일 불완전({err}) — 다음에 재시도")
                break
        if not args.tiles_only and not os.path.exists(bpath):
            bb = block_bounds_lonlat(bx, by)
            try:
                feats = harvest_buildings(bb)
            except Exception as e:  # noqa: BLE001
                print(f"[fetch] 블록 {bx},{by} 건물 실패: {e} — {backoff}s 대기 후 계속")
                time.sleep(backoff)
                backoff = min(backoff * 2, 3600)
                continue
            seen, uniq = set(), []
            for f in feats:
                fid = f.get("id")
                if fid in seen:
                    continue
                seen.add(fid)
                uniq.append(f)
            tmp = bpath + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fp:
                json.dump({"features": uniq}, fp, ensure_ascii=False)
            os.replace(tmp, bpath)
        n_done += 1
        if n_done % 10 == 0:
            el = time.time() - t0
            print(f"[fetch] {n_done}/{len(blocks)} ({el / 60:.0f}분 경과, "
                  f"{n_done / max(el, 1) * 3600:.0f}블록/시간)")
            st["last"] = [bx, by]
            save_state(st)
    print("[fetch] 완료")


# ---------------- 5) 어셈블(시군구별 vw 디렉터리 + pack) ----------------

def cmd_assemble(args):
    import shutil
    import numpy as np
    sys.path.insert(0, TOOLS)
    import vworld_fetch as vf
    sggs = {s["name"]: s for s in load_sgg()}
    with open(ALLOC_JSON, encoding="utf-8") as fp:
        alloc = json.load(fp)
    only = set(args.only.split(",")) if args.only else None
    os.makedirs(SGGDIR, exist_ok=True)

    # 프레임별: 블록 → 소유 시군구. 건물 귀속은 블록 단위로 frame 내 모든 sgg 중심점 판정.
    for fname, entries in sorted(alloc["frames"].items()):
        by_sgg = {}
        for bx, by, sgg in entries:
            by_sgg.setdefault(sgg, []).append((bx, by))
        # 블록 미보유 시군구(7.8km 블록을 이웃이 전부 소유한 작은 구 — 서울 구 등)도 포함:
        # 영상은 이웃 씬이 커버하므로 '건물 전용' 씬으로 어셈블.
        frame_sggs = sorted(set(by_sgg.keys())
                            | {s["name"] for s in sggs.values() if s["frame"] == fname})
        if only and not (set(frame_sggs) & only):
            continue
        paths_cache = {n: sgg_paths(sggs[n]) for n in frame_sggs}
        bbox_cache = {n: sggs[n]["bbox"] for n in frame_sggs}

        for sgg_name in frame_sggs:
            if only and sgg_name not in only:
                continue
            blocks = by_sgg.get(sgg_name, [])
            outdir = os.path.join(SGGDIR, f"vw_{sgg_name}")
            marker = os.path.join(outdir, ".packed")
            if os.path.exists(marker) and not args.force:
                continue
            # 모든 블록 자료가 준비됐는지 (영구 비제공 모자이크 마커는 통과 — 건물만 필수)
            def mosaic_ok(b):
                return (os.path.exists(os.path.join(BLOCKS, f"m_{b[0]}_{b[1]}.jpg"))
                        or os.path.exists(os.path.join(BLOCKS, f"m_{b[0]}_{b[1]}.jpg.unavailable")))
            missing = [b for b in blocks
                       if not mosaic_ok(b)
                       or not os.path.exists(os.path.join(BLOCKS, f"b_{b[0]}_{b[1]}.json"))]
            if missing:
                print(f"[assemble] {sgg_name}: 블록 {len(missing)}/{len(blocks)} 미수집 — 보류")
                continue
            os.makedirs(outdir, exist_ok=True)
            mani = {"zoom": Z, "frame": fname, "mosaics": []}
            for bx, by in blocks:
                fn = f"m_{bx}_{by}.jpg"
                if not os.path.exists(os.path.join(BLOCKS, fn)):
                    print(f"[assemble] {sgg_name}: {fn} 영구 비제공 — 모자이크 생략(건물만)")
                    continue
                dst = os.path.join(outdir, fn)
                if not os.path.exists(dst):
                    try:
                        os.link(os.path.join(BLOCKS, fn), dst)
                    except OSError:
                        shutil.copy2(os.path.join(BLOCKS, fn), dst)
                minx, miny, maxx, maxy = block_bounds_3857(bx, by)
                mani["mosaics"].append(
                    {"file": fn, "epsg3857_bounds": [minx, miny, maxx, maxy]})
            with open(os.path.join(outdir, "tiles_manifest.json"), "w", encoding="utf-8") as fp:
                json.dump(mani, fp, indent=1)

            # 건물: 이 sgg의 블록 + 이웃 sgg 소유 블록에 걸친 내 건물(중심점 판정)
            # → frame 내 전체 블록을 훑되 sgg bbox와 겹치는 블록만.
            sbb = bbox_cache[sgg_name]
            feats, seen = [], set()
            for bx, by, _owner in entries:
                lat0, lon0, lat1, lon1 = block_bounds_lonlat(bx, by)
                if lat1 < sbb[0] or lat0 > sbb[2] or lon1 < sbb[1] or lon0 > sbb[3]:
                    continue
                bpath = os.path.join(BLOCKS, f"b_{bx}_{by}.json")
                if not os.path.exists(bpath):
                    continue
                with open(bpath, encoding="utf-8") as fp:
                    bf = json.load(fp)["features"]
                if not bf:
                    continue
                cents = []
                for f in bf:
                    geom = f.get("geometry") or {}
                    polys = geom.get("coordinates") or []
                    if geom.get("type") == "Polygon":
                        polys = [polys]
                    try:
                        ring = polys[0][0]
                        cx = sum(c[0] for c in ring) / len(ring)
                        cy = sum(c[1] for c in ring) / len(ring)
                    except Exception:  # noqa: BLE001
                        cx = cy = float("nan")
                    cents.append([cx, cy])
                cents = np.array(cents)
                inside = np.zeros(len(bf), dtype=bool)
                for p in paths_cache[sgg_name]:
                    inside |= p.contains_points(cents)
                for f, ok in zip(bf, inside):
                    if not ok:
                        continue
                    fid = f.get("id")
                    if fid in seen:
                        continue
                    seen.add(fid)
                    feats.append(f)
            with open(os.path.join(outdir, "buildings.geojson"), "w", encoding="utf-8") as fp:
                json.dump({"type": "FeatureCollection", "features": feats}, fp,
                          ensure_ascii=False)
            # 기존 pack 규칙 재사용(높이 보정 등)
            ns = argparse.Namespace(dirs=[outdir])
            vf.cmd_pack(ns)
            with open(marker, "w") as fp:
                fp.write("ok")
            print(f"[assemble] {sgg_name}({fname}): 모자이크 {len(blocks)}, 건물 {len(feats)}")


# ---------------- 6) Unity 런타임 sgg_index ----------------

def cmd_sggindex(args):
    sggs = load_sgg()
    out = []
    for s in sggs:
        rings = [simplify_ring(r, tol=0.001) for r in s["rings"]]
        out.append({
            "name": s["name"], "kor": s["kor"], "frame": s["frame"],
            "bbox": s["bbox"],
            "rings": [[round(v, 5) for c in r for v in c] for r in rings],
        })
    path = os.path.join(UNITY_REGIONS, "sgg_index.json")
    with open(path, "w", encoding="utf-8") as fp:
        json.dump({"sggs": out}, fp, ensure_ascii=False)
    print(f"[sggindex] {len(out)}개 -> {path} ({os.path.getsize(path) / 1048576:.1f}MB)")


# ---------------- 7) 상태 ----------------

def cmd_status(args):
    with open(ALLOC_JSON, encoding="utf-8") as fp:
        alloc = json.load(fp)
    blocks = [tuple(b) for b in alloc["unique_blocks"]]
    nt = sum(1 for b in blocks
             if os.path.exists(os.path.join(BLOCKS, f"m_{b[0]}_{b[1]}.jpg"))
             or os.path.exists(os.path.join(BLOCKS, f"m_{b[0]}_{b[1]}.jpg.unavailable")))
    nb = sum(1 for b in blocks
             if os.path.exists(os.path.join(BLOCKS, f"b_{b[0]}_{b[1]}.json")))
    packed = []
    if os.path.isdir(SGGDIR):
        packed = [d for d in os.listdir(SGGDIR)
                  if os.path.exists(os.path.join(SGGDIR, d, ".packed"))]
    print(f"고유 블록 {len(blocks)} | 모자이크 {nt} ({nt / len(blocks) * 100:.1f}%) | "
          f"건물 {nb} ({nb / len(blocks) * 100:.1f}%) | packed 시군구 {len(packed)}")


def main():
    if not KEY:
        sys.exit("VWORLD_API_KEY 환경변수가 없습니다.")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sgg").set_defaults(func=cmd_sgg)
    sub.add_parser("alloc").set_defaults(func=cmd_alloc)
    sub.add_parser("seed").set_defaults(func=cmd_seed)
    f = sub.add_parser("fetch")
    f.add_argument("--only", default=None)
    f.add_argument("--tiles-only", action="store_true")
    f.add_argument("--buildings-only", action="store_true")
    f.set_defaults(func=cmd_fetch)
    a = sub.add_parser("assemble")
    a.add_argument("--only", default=None)
    a.add_argument("--force", action="store_true")
    a.set_defaults(func=cmd_assemble)
    sub.add_parser("sggindex").set_defaults(func=cmd_sggindex)
    sub.add_parser("status").set_defaults(func=cmd_status)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
