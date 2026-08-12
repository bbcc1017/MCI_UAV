"""vWorld GIS건물통합정보 → 지역별 buildings.geojson 일괄 수집(재개 가능).

왜 필요한가: 건물 높이 인덱스(`bldg_height_index.py` → `tools/nationwide/bldgindex/`)의 원천이
강남구 1곳뿐이라, Unity 런타임의 실측 건물 라벨·항공장애표시등·옥상 구조물이 강남 밖에서는
실측 데이터 없이 돌아간다(메시 프로브 폴백으로 동작은 하지만 이름·층수가 없다).

이 스크립트는 `vworld_fetch.py` 의 buildings 수집을 시군구 bbox 로 돌려
`tools/nationwide_v2/buildings/<region>/buildings.geojson` 을 채우고, 이어서
`bldg_height_index.py` 로 인덱스까지 만든다.

⚠️vWorld WFS 는 bbox 당 1000건 상한이라 재귀 4분할로 훑는다 — 도심 구 하나가 수백 요청이다.
   API 키 일일 쿼터를 고려해 --limit 로 끊어 돌리고, 이미 받은 지역은 자동으로 건너뛴다.
⚠️sgg.json 의 bbox 는 **[minLat, minLon, maxLat, maxLon]**(위도 먼저) — vworld_fetch 인자 순서와 같다.

사용:
  python tools/vworld_buildings_batch.py --regions busan_junggu
  python tools/vworld_buildings_batch.py --metro --limit 10
  python tools/vworld_buildings_batch.py --sido busan,daegu --index-only
"""

import argparse
import io
import json
import os
import subprocess
import sys
import time

SGG_JSON = os.path.join("tools", "nationwide", "sgg.json")
OUT_ROOT = os.path.join("tools", "nationwide_v2", "buildings")
INDEX_DIR = os.path.join("tools", "nationwide", "bldgindex")
METRO_PREFIXES = ("seoul", "busan", "daegu", "incheon",
                  "gwangju", "daejeon", "ulsan", "sejong")


def load_sgg():
    with io.open(SGG_JSON, encoding="utf-8") as fh:
        return {s["name"]: s for s in json.load(fh)}


def select(sgg, args):
    names = sorted(sgg)
    if args.regions:
        want = [r for r in args.regions.split(",") if r]
        missing = [r for r in want if r not in sgg]
        if missing:
            sys.exit(f"sgg.json 에 없는 지역: {','.join(missing)}")
        return want
    if args.all_regions:
        return names
    prefixes = METRO_PREFIXES if args.metro else tuple(
        s for s in args.sido.split(",") if s)
    if not prefixes:
        sys.exit("--regions/--metro/--sido/--all 중 하나 필요")
    return [n for n in names if n.startswith(tuple(p + "_" for p in prefixes))]


def fetch_one(region, bbox, python_exe):
    """vworld_fetch.py buildings 를 하위 프로세스로 — 재귀 분할·재시도 로직을 그대로 쓴다."""
    out_dir = os.path.join(OUT_ROOT, region)
    cmd = [python_exe, os.path.join("tools", "vworld_fetch.py"), "buildings",
           "--bbox", *[str(v) for v in bbox], "--out", out_dir]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run(cmd, env=env).returncode == 0


def index_one(region, python_exe):
    cmd = [python_exe, os.path.join("tools", "bldg_height_index.py"),
           "--regions", region]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run(cmd, env=env).returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", default="")
    parser.add_argument("--metro", action="store_true")
    parser.add_argument("--sido", default="")
    parser.add_argument("--all", action="store_true", dest="all_regions")
    parser.add_argument("--limit", type=int, default=0,
                        help="이번 실행에서 새로 수집할 지역 수 상한(0=제한 없음)")
    parser.add_argument("--force", action="store_true", help="이미 있는 geojson 도 다시 수집")
    parser.add_argument("--index-only", action="store_true", dest="index_only",
                        help="수집은 건너뛰고 이미 받은 geojson 으로 인덱스만 생성")
    parser.add_argument("--python", default=sys.executable, dest="python_exe")
    args = parser.parse_args()

    sgg = load_sgg()
    regions = select(sgg, args)
    os.makedirs(INDEX_DIR, exist_ok=True)

    fetched = indexed = skipped = 0
    for region in regions:
        geojson = os.path.join(OUT_ROOT, region, "buildings.geojson")
        have = os.path.exists(geojson)
        if not args.index_only and (args.force or not have):
            if args.limit and fetched >= args.limit:
                print(f"[limit] {args.limit}개 수집 후 중단 — 나머지는 재실행하면 이어받는다")
                break
            started = time.time()
            print(f"[fetch] {region} …")
            if not fetch_one(region, sgg[region]["bbox"], args.python_exe):
                print(f"[fail] {region}: vworld 수집 실패 — 건너뜀")
                continue
            fetched += 1
            have = os.path.exists(geojson)
            print(f"[fetch] {region} 완료 ({time.time() - started:.0f}s)")
        elif have:
            skipped += 1
        if have and index_one(region, args.python_exe):
            indexed += 1

    print(f"[done] 수집 {fetched} · 기수집 {skipped} · 인덱스 {indexed} / 대상 {len(regions)}")


if __name__ == "__main__":
    main()
