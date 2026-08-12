"""건물 높이 인덱스 — vWorld GIS건물통합정보(SHP 유래) 속성에서 실제 높이를 뽑아
Unity 런타임이 바로 읽는 소형 텍스트로 만든다.

왜 필요한가: HUD 의 건물 높이 라벨을 하향 레이캐스트로 재는 것은 **메시 상단 높이**일 뿐
실제 건축물 대장 높이가 아니다(옥탑·정사 오차 포함). 원천 속성(height, 결측 시 지상층수×3.5m,
v2 파사드 버킷터와 같은 규칙)을 쓰면 라벨이 실측치가 된다.

입력: tools/nationwide_v2/buildings/<region>/buildings.geojson  (vworld_fetch.py buildings 산출)
출력: tools/nationwide/bldgindex/<region>.txt
      한 줄 = "lon lat height floors 건물명"  (건물명은 줄 끝까지, 없으면 생략)
      centroid 를 --merge-m 격자로 뭉쳐 같은 칸은 **최고 높이 동**만 남긴다(아파트 단지 라벨 폭주 방지).

사용:
  python tools/bldg_height_index.py --regions seoul_gangnamgu
  python tools/bldg_height_index.py --all
"""

import argparse
import json
import math
import os

FLOOR_M = 3.5           # v2_buildings_bucket.py 와 동일 규칙
# 발자국 대비 최대 세장비(높이 / √바닥면적). 세계 최슬림 초고층이 ~24:1, 국내 아파트는 5~8:1 이라
# 15 는 실재 건물을 절대 자르지 않으면서 오기입만 걸러낸다.
# (실측 인천 남동구: 정상 51층 타워 면적 1536㎡ 통과 / 층수 337·128 오기입 276·559㎡ 탈락)
MAX_SLENDERNESS = 15.0
SRC_DIR = os.path.join("tools", "nationwide_v2", "buildings")
OUT_DIR = os.path.join("tools", "nationwide", "bldgindex")


def ring_points(geometry):
    """MultiPolygon/Polygon 의 외곽 링들만 (좌표 리스트) 로 뽑는다."""
    kind = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if kind == "Polygon":
        return [coords[0]] if coords else []
    if kind == "MultiPolygon":
        return [part[0] for part in coords if part]
    return []


def height_of(props):
    height = props.get("height") or 0
    try:
        height = float(height)
    except (TypeError, ValueError):
        height = 0.0
    floors = props.get("grnd_flr") or 0
    try:
        floors = int(floors)
    except (TypeError, ValueError):
        floors = 0
    if height < 1.0:
        height = max(1.0, float(floors)) * FLOOR_M
    # 원천 속성 노이즈 방어 — 층수 대비 비현실적으로 큰 height 는 오기입이다
    # (실측: "정원빌딩" 10층 351m). 층당 6m 를 넘으면 층수 기반값으로 대체한다.
    if floors >= 1 and height > floors * 6.0:
        height = floors * FLOOR_M
    return min(max(height, 3.0), 600.0), floors


def ring_area_m2(ring, lat):
    """외곽 링의 평면 면적(㎡) — 신발끈 공식, 위도로 경도 스케일 보정."""
    scale_y = 111320.0
    scale_x = scale_y * math.cos(math.radians(lat))
    total = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i][0] * scale_x, ring[i][1] * scale_y
        x2, y2 = ring[(i + 1) % n][0] * scale_x, ring[(i + 1) % n][1] * scale_y
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def plausible(height, area_m2):
    """발자국이 지탱할 수 없는 높이 = 원천 오기입(층수 결측·오기입이 ×3.5 로 부풀려진 경우)."""
    if area_m2 <= 1.0:
        return True
    return height <= MAX_SLENDERNESS * math.sqrt(area_m2)


def clean_name(props):
    name = str(props.get("bld_nm") or "").strip()
    if not name or name.upper() == "NN":
        return ""
    return " ".join(name.split())[:24]


def process(region, args):
    path = os.path.join(args.src_dir, region, "buildings.geojson")
    if not os.path.exists(path):
        print(f"[skip] {region}: buildings.geojson 없음 ({path})")
        return False
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    features = data.get("features") or []

    # (격자키) -> (height, lon, lat, floors, name) — 같은 칸은 최고 높이만
    merged = {}
    bogus = 0
    step = args.merge_m / 111320.0        # 위도 도 단위(경도는 cos 보정 생략 — 병합 격자라 무해)
    for feature in features:
        geometry = feature.get("geometry")
        props = feature.get("properties") or {}
        if not geometry:
            continue
        height, floors = height_of(props)
        name = clean_name(props)
        for ring in ring_points(geometry):
            if not ring or len(ring) < 3:
                continue
            lons = [p[0] for p in ring]
            lats = [p[1] for p in ring]
            lon = sum(lons) / len(lons)
            lat = sum(lats) / len(lats)
            if not plausible(height, ring_area_m2(ring, lat)):
                bogus += 1
                continue
            key = (int(lat / step), int(lon / step))
            previous = merged.get(key)
            if previous is None or height > previous[0]:
                merged[key] = (height, lon, lat, floors, name)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, region + ".txt")
    rows = sorted(merged.values(), key=lambda row: -row[0])
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        for height, lon, lat, floors, name in rows:
            fh.write(f"{lon:.6f} {lat:.6f} {height:.1f} {floors}"
                     + (f" {name}" if name else "") + "\n")
    tallest = rows[0] if rows else (0, 0, 0, 0, "")
    print(f"[ok] {region}: {len(features)}동 → {len(rows)}행"
          f" | 최고 {tallest[0]:.0f}m {tallest[4]}"
          + (f" | 세장비 탈락 {bogus}" if bogus else "")
          + f" | {os.path.getsize(out_path) / 1024:.0f}KB")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--merge-m", type=float, default=25.0, dest="merge_m")
    parser.add_argument("--src-dir", default=SRC_DIR, dest="src_dir")
    parser.add_argument("--out-dir", default=OUT_DIR, dest="out_dir")
    args = parser.parse_args()

    regions = [r for r in args.regions.split(",") if r]
    if args.all and os.path.isdir(args.src_dir):
        regions = sorted(name for name in os.listdir(args.src_dir)
                         if os.path.isdir(os.path.join(args.src_dir, name)))
    if not regions:
        parser.error("--regions 또는 --all 필요")
    done = sum(1 for region in regions if process(region, args))
    print(f"[done] {done}/{len(regions)}")


if __name__ == "__main__":
    main()
