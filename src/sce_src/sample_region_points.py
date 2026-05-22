"""17개 광역시도 폴리곤 내부에서 랜덤 평가점을 샘플링한다 (피드백 3 — 일반화 평가).

배경: plan1nat_f3 평가는 학습에 쓴 17개 대표 좌표(시청/도청)에서 그대로 봐서
hold-out 일반화 시험이 아니다. 학습한 전국 단일 정책을 "학습 좌표와 분리된"
지역 내부의 임의 지점에서 평가하려고, scenarios/ctprvn.shp(통계청 시도 경계,
EPSG:5179) 폴리곤 안에서 균일 랜덤 점을 뽑는다.

- 좌표계: shp 는 EPSG:5179(미터) → pyproj 로 WGS84(EPSG:4326) 위경도 변환.
- 폴리곤 내부 판정: 모든 ring(외곽+섬+구멍) 에 대한 even-odd ray casting.
  → 섬은 포함, 구멍(호수 등)은 제외. shp 의 ring 방향과 무관하게 정확.
- bbox 거부 샘플링(rejection sampling). 본토 면적이 압도적이라 섬보다 본토
  위주로 뽑힌다 — 도로망 위 점일 확률이 높아 Kakao 경로탐색 실패가 적다.

출력: scenarios/plan1nat_eval_points.json  {region_short: [[lat,lon], ...]}

예:
  python src/sce_src/sample_region_points.py --n 5 --seed 20260522
"""
import argparse
import json
import os

import numpy as np
import shapefile  # pyshp
from pyproj import Transformer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))

# 통계청 ctprvn CTP_KOR_NM(full) → 프로젝트 short name (cross_location_eval.LOCATIONS 와 일치)
KOR_TO_SHORT = {
    "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구",
    "인천광역시": "인천", "광주광역시": "광주", "대전광역시": "대전",
    "울산광역시": "울산", "세종특별자치시": "세종", "경기도": "경기",
    "강원특별자치도": "강원", "충청북도": "충북", "충청남도": "충남",
    "전북특별자치도": "전북", "전라북도": "전북", "전라남도": "전남",
    "경상북도": "경북", "경상남도": "경남", "제주특별자치도": "제주",
}


def _ring_contains(pt, ring):
    """even-odd ray casting — 점 pt 가 단일 ring(닫힌 좌표열) 안인가."""
    x, y = pt
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi):
            inside = not inside
        j = i
    return inside


def _polygon_contains(pt, rings):
    """모든 ring 에 대한 even-odd 누적 — 섬 포함 / 구멍 제외를 자동 처리."""
    inside = False
    for ring in rings:
        if _ring_contains(pt, ring):
            inside = not inside
    return inside


def _shape_rings(shape):
    """pyshp polygon shape → ring 좌표열 리스트로 분해."""
    parts = list(shape.parts) + [len(shape.points)]
    return [shape.points[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]


def sample_region(shape, n, rng, transformer):
    """폴리곤(5179) 내부 n개 점을 균일 랜덤 추출 → WGS84 (lat,lon) 반환."""
    rings = _shape_rings(shape)
    xmin, ymin, xmax, ymax = shape.bbox
    pts, tries, max_tries = [], 0, n * 100000
    while len(pts) < n and tries < max_tries:
        tries += 1
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        if _polygon_contains((x, y), rings):
            lon, lat = transformer.transform(x, y)
            pts.append([round(lat, 6), round(lon, 6)])
    if len(pts) < n:
        raise RuntimeError(f"샘플 부족: {len(pts)}/{n} (tries={tries})")
    return pts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shp", default=os.path.join(REPO, "scenarios", "ctprvn.shp"))
    p.add_argument("--n", type=int, default=5, help="지역당 샘플 점 수")
    p.add_argument("--seed", type=int, default=20260522)
    p.add_argument("--out", default=os.path.join(REPO, "scenarios",
                                                 "plan1nat_eval_points.json"))
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    transformer = Transformer.from_crs(5179, 4326, always_xy=True)

    sf = shapefile.Reader(args.shp, encoding="cp949")
    fields = [f[0] for f in sf.fields[1:]]
    kor_idx = fields.index("CTP_KOR_NM")

    out = {}
    for sr in sf.shapeRecords():
        kor = sr.record[kor_idx].strip()
        short = KOR_TO_SHORT.get(kor)
        if short is None:
            print(f"⚠️ 매핑 없는 지역명: {kor!r} — 건너뜀")
            continue
        pts = sample_region(sr.shape, args.n, rng, transformer)
        out[short] = pts
        print(f"  {short:4s} ({kor}): " +
              ", ".join(f"({la},{lo})" for la, lo in pts))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[sample] {len(out)}개 지역 × {args.n}점 → {args.out}")


if __name__ == "__main__":
    main()
