# -*- coding: utf-8 -*-
"""OSM(Overpass) 도로망 → 시군구별 enriched roads2/<name>.txt.

osm_roads.py(폭+좌표만)의 상위호환. 차선수·일방통행·도로등급을 함께 저장해
Unity에서 중앙선/차선점선/일방통행을 표현하고 NPC 교통 방향을 정한다.

  roads2/<name>.txt 라인:
    "<class> <lanes> <oneway> <struct> <lat1> <lon1> <lat2> <lon2> ..."
      class  : highway 등급 토큰(motorway/primary/secondary/...). 폭/마킹 스타일 결정.
      lanes  : 총 차로수(int, 0=미상 → class 기본 사용).
      oneway : 0=양방향, 1=정방향 일방, -1=역방향 일방(geometry 역순 주행).
      struct : 0=지상, 1=교량(bridge — 물/계곡 위 고가로 시각화), 2=터널(tunnel).
               하위호환: struct 없는 옛 파일은 4번째가 좌표(소수점)라 임포터가 구분해 0 처리.

귀속 방식(2026-07-02 개정 — '씬 한복판 도로 끊김' 수정):
  기존 = bbox 질의 + null 좌표 제거 + way 중심점 폴리곤 귀속. 세 가지 결함이 겹쳐
  간선이 씬 한복판에서 끊겼다(로드 34개 구 병합 기준 간선 데드엔드 696곳 실측):
    ① bbox 밖 좌표는 null로 와서 제거만 하면 빠진 구간을 직선으로 이어붙인 가짜 도로가 됨
    ② 잘린 geometry의 중심점으로 귀속 → 경계 way가 어느 구에도 귀속 안 되거나(구멍)
    ③ 귀속돼도 자기 bbox에서 잘린 채 끝남(이웃 구는 그 이어짐을 안 가져감)
  개정 = 질의 bbox를 MARGIN_DEG 확장(경계 way의 온전한 geometry 확보) + null 지점에서 way 분할
  + **최근접 폴리곤 통짜 귀속**: 각 run의 중간점이 포함되는(없으면 가장 가까운) 시군구에
  통째로 귀속(전국 STRtree). 자르지 않으므로 이웃 구로 넘어가는 오버행은 그대로 이어진다.
  ⚠️ 폴리곤 '정확 절단'은 기각 — 통계청 시군구 경계는 단순화돼 인접 폴리곤 사이에 무주공산
  슬리버가 있고, 고속도로 중앙선을 따라가는 경계에서 way가 잘게 썰려 새 구멍이 생겼다(실측).

사용:
  conda run -n UAV python tools/osm_roads2.py fetch --only seoul_jongnogu
  conda run -n UAV python tools/osm_roads2.py fetch          # sgg.json 전체(오래 걸림)
  conda run -n UAV python tools/osm_roads2.py fetch --force   # 기존 파일 덮어씀
"""
import argparse
import json
import os
import time

import requests

from osm_overpass_endpoints import overpass_endpoints

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(TOOLS, "nationwide")
SGG_JSON = os.path.join(ROOT, "sgg.json")
OUTDIR = os.path.join(ROOT, "roads2")
# 여러 미러 로테이션 — 한 곳이 429/504면 다음 미러를 즉시 시도(백오프 대기 최소화).
ENDPOINTS = overpass_endpoints()
HEADERS = {"User-Agent": "MCI-UAV-research/1.0 (academic disaster sim)"}

CLASSES = [
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential",
    "living_street", "service",
]
GRADES = "|".join(CLASSES)

# 보행로(인도/광장/자전거도로/계단·산책로) — 별도 레이어(footways/<name>.txt). 차도와 분리.
FW_CLASSES = ["footway", "pedestrian", "path", "cycleway", "steps"]
FW_GRADES = "|".join(FW_CLASSES)
OUTDIR_FW = os.path.join(ROOT, "footways")

MARGIN_DEG = 0.035    # 질의 bbox 확장(도) ≈ 3.5km — 경계 way의 geometry를 온전히 받아옴


def overpass(bbox, grades=GRADES, rounds=4):
    """bbox=(minLat,minLon,maxLat,maxLon) 안의 highway way를 geometry+tags.
    grades=정규식 등급(기본=차도 GRADES, 보행로는 FW_GRADES 전달).
    미러를 순회하며 한 곳이 429/504면 즉시 다음 미러로 — 한 라운드 전부 실패 시에만 백오프."""
    q = (f"[out:json][timeout:120];"
         f'way["highway"~"^({grades})$"]'
         f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});out tags geom;")
    backoff = 5
    for rnd in range(rounds):
        for ep in ENDPOINTS:
            try:
                r = requests.post(ep, data={"data": q}, headers=HEADERS, timeout=180)
                if r.status_code == 200:
                    return r.json().get("elements", [])
                if r.status_code in (429, 503, 504):
                    continue   # 다음 미러 즉시
                r.raise_for_status()
            except Exception as e:  # noqa: BLE001
                continue   # 다음 미러
        print(f"  전 미러 실패(라운드 {rnd + 1}) — {backoff}s 대기", flush=True)
        time.sleep(backoff)
        backoff = min(backoff * 2, 90)
    return []


def point_in_rings(lat, lon, rings):
    inside = False
    for ring in rings:
        n = len(ring)
        j = n - 1
        for i in range(n):
            xi, yi = ring[i][0], ring[i][1]
            xj, yj = ring[j][0], ring[j][1]
            if ((yi > lat) != (yj > lat)) and \
               (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        if inside:
            return True
    return False


def parse_lanes(tags):
    v = tags.get("lanes")
    if not v:
        return 0
    try:
        return max(0, int(float(str(v).split(";")[0])))
    except Exception:  # noqa: BLE001
        return 0


def parse_oneway(tags):
    v = str(tags.get("oneway", "")).strip().lower()
    if v in ("yes", "true", "1"):
        return 1
    if v in ("-1", "reverse"):
        return -1
    # 고속도로/링크는 사실상 일방
    hw = tags.get("highway", "")
    if hw in ("motorway", "motorway_link", "trunk_link", "primary_link",
              "secondary_link", "tertiary_link"):
        return 1
    return 0


def parse_struct(tags):
    """0=지상, 1=교량(bridge), 2=터널(tunnel). bridge가 tunnel보다 우선."""
    b = str(tags.get("bridge", "")).strip().lower()
    if b and b not in ("no", "false", "0"):
        return 1
    if str(tags.get("man_made", "")).strip().lower() in ("bridge",):
        return 1
    t = str(tags.get("tunnel", "")).strip().lower()
    if t and t not in ("no", "false", "0"):
        return 2
    if str(tags.get("covered", "")).strip().lower() in ("yes", "tunnel"):
        return 2
    return 0


def _split_at_nulls(geom):
    """bbox 밖 좌표(null/불완전)에서 way를 분할 — 제거 후 이어붙이면 빠진 구간을
    직선으로 건너뛰는 가짜 도로가 되므로, 그 지점에서 폴리라인을 끊는다."""
    runs, cur = [], []
    for p in geom:
        if p and "lat" in p and "lon" in p:
            cur.append((p["lat"], p["lon"]))
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
    if len(cur) >= 2:
        runs.append(cur)
    return runs


def _district_poly(rings):
    """sgg rings([lon,lat] 목록들) → shapely (Multi)Polygon. 섬/다중부 지원, invalid는 buffer(0) 교정."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    polys = []
    for ring in rings:
        if len(ring) < 4:
            continue
        try:
            p = Polygon(ring)
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty:
                polys.append(p)
        except Exception:  # noqa: BLE001
            continue
    return unary_union(polys) if polys else None


class OwnerIndex:
    """전국 시군구 폴리곤 소유권 인덱스 — 점을 '포함하는(없으면 최근접)' 구로 유일 귀속.
    폴리곤 절단 대신 통짜 귀속에 사용: 경계 슬리버(무주공산)도 최근접 구가 가져가 구멍이 없다."""

    def __init__(self, sggs):
        from shapely.strtree import STRtree
        self.names, self.geoms = [], []
        for s in sggs:
            poly = _district_poly(s["rings"])
            if poly is None:
                continue
            self.names.append(s["name"])
            self.geoms.append(poly)
        self.tree = STRtree(self.geoms)

    def owner_of(self, lat, lon):
        from shapely.geometry import Point
        pt = Point(lon, lat)
        for i in self.tree.query(pt):          # bbox 후보만 — 명시 contains로 방향 혼동 회피
            if self.geoms[i].contains(pt):
                return self.names[i]
        return self.names[self.tree.nearest(pt)]


def fetch_one(sgg, grades=GRADES, outdir=OUTDIR, tag="osm2", owner=None):
    name = sgg["name"]
    bb = sgg["bbox"]
    if owner is None:
        owner = OwnerIndex([sgg])
    qbb = (bb[0] - MARGIN_DEG, bb[1] - MARGIN_DEG, bb[2] + MARGIN_DEG, bb[3] + MARGIN_DEG)
    ways = overpass(qbb, grades)
    lines, kept = [], 0
    for w in ways:
        tags = w.get("tags", {})
        cls = tags.get("highway", "residential")
        lanes = parse_lanes(tags)
        oneway = parse_oneway(tags)
        struct = parse_struct(tags)
        for run in _split_at_nulls(w.get("geometry") or []):
            mid = run[len(run) // 2]           # 중간 '정점'(선 위의 점 — 평균은 L자에서 밖으로 샘)
            if owner.owner_of(mid[0], mid[1]) != name:
                continue
            coords = " ".join(f"{la:.7f} {lo:.7f}" for la, lo in run)
            lines.append(f"{cls} {lanes} {oneway} {struct} {coords}")
            kept += 1
    out = os.path.join(outdir, f"{name}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[{tag}] {name}: way {len(ways)} → 통짜귀속 {kept} ({os.path.getsize(out) / 1024:.0f}KB)", flush=True)
    time.sleep(1.0)
    return kept


def _fetch_all(args, grades, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    with open(SGG_JSON, encoding="utf-8") as f:
        sggs_all = json.load(f)
    owner = OwnerIndex(sggs_all)   # 귀속은 항상 전국 인덱스 기준(--only여도)
    sggs = sggs_all
    if args.only:
        names = set(args.only.split(","))
        sggs = [s for s in sggs_all if s["name"] in names]
    print(f"[{tag}] 대상 시군구 {len(sggs)}개 → {outdir}", flush=True)
    total, done = 0, 0
    for i, s in enumerate(sggs):
        out = os.path.join(outdir, f"{s['name']}.txt")
        if os.path.exists(out) and not args.force:
            done += 1
            continue
        try:
            total += fetch_one(s, grades, outdir, tag, owner)
            done += 1
        except Exception as e:  # noqa: BLE001
            print(f"[{tag}] {s['name']} 실패: {str(e)[:100]}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"[{tag}] 진행 {i + 1}/{len(sggs)} (완료 {done})", flush=True)
    print(f"[{tag}] 완료 — 처리 {done}/{len(sggs)}, 총 way {total}개", flush=True)


def cmd_fetch(args):
    _fetch_all(args, GRADES, args.outdir or OUTDIR, "osm2")


def cmd_footway(args):
    _fetch_all(args, FW_GRADES, args.outdir or OUTDIR_FW, "fw")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--only", default=None)
    f.add_argument("--force", action="store_true")
    f.add_argument("--outdir", default=None, help="출력 디렉터리 재지정(스테이징 검증용)")
    f.set_defaults(func=cmd_fetch)
    fw = sub.add_parser("footway", help="보행로(인도/광장/자전거도로) → footways/<name>.txt")
    fw.add_argument("--only", default=None)
    fw.add_argument("--force", action="store_true")
    fw.add_argument("--outdir", default=None, help="출력 디렉터리 재지정(스테이징 검증용)")
    fw.set_defaults(func=cmd_footway)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
