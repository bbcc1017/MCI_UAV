# -*- coding: utf-8 -*-
"""전국 255 시군구 → CAR_test 자율주행 씬용 지역 인덱스(EPSG:5186 bbox + 산출물 보유 현황).

CAR_test 는 지금까지 `seoul_gangnamgu` 와 bbox(202000~205000, 543000~546000) 가 코드 곳곳에
하드코딩돼 있어 다른 지역으로 못 옮겼다. 이 인덱스가 그 하드코딩을 대체한다.

입력  tools/nationwide/sgg.json (name/kor/bbox(WGS84 minlon,minlat,maxlon,maxlat)/rings)
출력  tools/nationwide_v2/region_index.json
      [{name, kor, minE,minN,maxE,maxN, centerE,centerN, hasLaneGraph, hasWalk, hasStdLink, hasFurniture}]

Unity `V2/RegionCatalogV2.cs` 가 이 파일을 읽어 지역 전환 시 원점/경계/데이터 경로를 정한다.
정밀도로지도 미수록 6구는 hasLaneGraph=false 로 남아 주행 대상에서 자동 제외된다.

실행: PYTHONIOENCODING=utf-8 <UAV env python> tools/v2_region_index.py
"""
import io
import json
import os

from pyproj import Transformer

TOOLS = os.path.dirname(os.path.abspath(__file__))
SGG = os.path.join(TOOLS, "nationwide", "sgg.json")
LANEGRAPH = os.path.join(TOOLS, "nationwide_v2", "lanegraph")
FURNITURE = os.path.join(TOOLS, "nationwide_v2", "streetfurniture")
OUT = os.path.join(TOOLS, "nationwide_v2", "region_index.json")

TO_5186 = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)


def main():
    with io.open(SGG, encoding="utf-8", errors="replace") as f:
        regions = json.load(f)

    index = []
    for entry in regions:
        name = entry.get("name")
        if not name:
            continue
        bbox = entry.get("bbox") or []
        if len(bbox) != 4:
            continue
        # ⚠sgg.json bbox 는 **[minlat, minlon, maxlat, maxlon]** — 위도가 먼저다.
        #   lon/lat 순으로 읽으면 좌표변환이 inf 를 뱉는다(실측 확인).
        min_lat, min_lon, max_lat, max_lon = bbox
        corners = [
            TO_5186.transform(min_lon, min_lat),
            TO_5186.transform(max_lon, min_lat),
            TO_5186.transform(min_lon, max_lat),
            TO_5186.transform(max_lon, max_lat),
        ]
        es = [c[0] for c in corners]
        ns = [c[1] for c in corners]
        min_e, max_e = min(es), max(es)
        min_n, max_n = min(ns), max(ns)

        index.append({
            "name": name,
            "kor": entry.get("kor", ""),
            "minE": round(min_e, 1), "minN": round(min_n, 1),
            "maxE": round(max_e, 1), "maxN": round(max_n, 1),
            "centerE": round((min_e + max_e) * 0.5, 1),
            "centerN": round((min_n + max_n) * 0.5, 1),
            "hasLaneGraph": os.path.exists(os.path.join(LANEGRAPH, name + ".bin")),
            "hasWalk": os.path.exists(os.path.join(LANEGRAPH, name + ".walk.bin")),
            "hasStdLink": os.path.exists(os.path.join(LANEGRAPH, name + ".stdlink.bin")),
            "hasFurniture": os.path.exists(os.path.join(FURNITURE, name + ".txt")),
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump({"regions": index}, f, ensure_ascii=False, indent=1)

    lane = sum(1 for r in index if r["hasLaneGraph"])
    walk = sum(1 for r in index if r["hasWalk"])
    furniture = sum(1 for r in index if r["hasFurniture"])
    print(f"[out] {OUT} — {len(index)} 지역")
    print(f"   차선그래프 {lane}/{len(index)} · 보행(walk.bin) {walk} · 노변장애물 {furniture}")
    missing = [r["name"] for r in index if not r["hasLaneGraph"]]
    if missing:
        print("   차선그래프 없음(정밀도로지도 미수록):", ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
