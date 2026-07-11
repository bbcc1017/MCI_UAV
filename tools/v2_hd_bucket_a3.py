#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v2 HD맵 도로 3D 버킷터 — 정밀도로지도 A3 차도면/A2 링크/B3 노면표시/B1 표지/C5 장벽/C6 지주/A5 주차장 → 1km 타일별.

도로 3D 전환(2026-07-11): 전 레이어 **z(고도) 보존** — 고가/교량/IC·JC 표면을 DEM 드레이프 아닌 실높이로 베이크.
 · 차도구간면(A3_DRIVEWAYSECTION, MultiPolygon 3D) → Sutherland-Hodgman 사각 클리핑(z 선형보간)
   → `hd_roadsurf/<region>/tile.txt` (`<kind> <roadtype> lat lon z …` 외곽링; 홀=미방출·카운트만)
 · 주행경로링크(A2_LINK, MultiLineString 3D) → Liang-Barsky 3D 클리핑
   → `hd_links/tile.txt` (`<laneno> <linktype> <roadrank> <from> <to> <rlink> <llink> lat lon z …`, 결측='-')
 · 노면표시면(B3_SURFACEMARK) → centroid 버킷 → `hd_marks/tile.txt` (`<kind> lat lon z …` 외곽링)
 · 안전표지(B1, 점) → `hd_signs/tile.txt` (`<kind> lat lon z`)
 · 높이장벽(C5, 선) → 클리핑 → `hd_heightbarriers/tile.txt` (`lat lon z …`)
 · 지주(C6, 점) → `hd_posts/tile.txt` (`<kind> lat lon z`)
 · 주차장면(A5) → centroid 버킷 → `hd_parkinglots/tile.txt` (`<kind> lat lon z …`)
C# TileBakerV2 가 A3=z 직접 삼각분할(도로 표면), A2=IC/JC 고어 식별, 나머지=프롭/오버레이로 굽는다.
좌표=EPSG:4326 lon lat z. 실행: UAV env python tools/v2_hd_bucket_a3.py --region seoul_gangnamgu
"""
import os, json, argparse
from pyproj import Transformer

P5186 = ("+proj=tmerc +lat_0=38 +lon_0=127 +k=1 +x_0=200000 +y_0=600000 "
         "+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
TILE_M = 1000.0
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HD = os.path.join(REPO, "tools", "nationwide", "hdmap")
_fwd = Transformer.from_crs("EPSG:4326", P5186, always_xy=True)
_inv = Transformer.from_crs(P5186, "EPSG:4326", always_xy=True)


def liang_barsky3(p0, p1, xmin, ymin, xmax, ymax):
    """3D 세그먼트를 사각에 클립 — z 는 u 선형보간."""
    (x0, y0, z0), (x1, y1, z1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    p = (-dx, dx, -dy, dy); q = (x0 - xmin, xmax - x0, y0 - ymin, ymax - y0)
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0: return None
        else:
            t = qi / pi
            if pi < 0: u0 = max(u0, t)
            else: u1 = min(u1, t)
    if u0 > u1:
        return None
    return ((x0 + u0 * dx, y0 + u0 * dy, z0 + u0 * (z1 - z0)),
            (x0 + u1 * dx, y0 + u1 * dy, z0 + u1 * (z1 - z0)))


def clip_polyline3(enz, rect):
    xmin, ymin, xmax, ymax = rect
    pieces, cur = [], []
    for i in range(len(enz) - 1):
        seg = liang_barsky3(enz[i], enz[i + 1], xmin, ymin, xmax, ymax)
        if seg is None:
            if len(cur) >= 2: pieces.append(cur)
            cur = []; continue
        a, b = seg
        if not cur: cur = [a, b]
        elif abs(a[0] - cur[-1][0]) < 1e-6 and abs(a[1] - cur[-1][1]) < 1e-6: cur.append(b)
        else:
            if len(cur) >= 2: pieces.append(cur)
            cur = [a, b]
    if len(cur) >= 2: pieces.append(cur)
    return pieces


def clip_ring3(ring, rect):
    """Sutherland-Hodgman 사각 클리핑(볼록 클립 영역) — 교차점 z 선형보간, 연속 중복 제거."""
    xmin, ymin, xmax, ymax = rect
    edges = (  # (내부판정, 교차 t 계산용 축/경계)
        (lambda p: p[0] >= xmin, 0, xmin),
        (lambda p: p[0] <= xmax, 0, xmax),
        (lambda p: p[1] >= ymin, 1, ymin),
        (lambda p: p[1] <= ymax, 1, ymax),
    )
    out = list(ring)
    for inside, ax, bound in edges:
        if not out:
            return []
        cur, prev = [], out[-1]
        for pt in out:
            pin, cin = inside(prev), inside(pt)
            if pin != cin:  # 경계 교차 — z 포함 보간
                d = pt[ax] - prev[ax]
                t = 0.0 if d == 0 else (bound - prev[ax]) / d
                cur.append(tuple(prev[k] + t * (pt[k] - prev[k]) for k in range(3)))
            if cin:
                cur.append(pt)
            prev = pt
        out = cur
    dedup = []
    for pt in out:
        if not dedup or abs(pt[0] - dedup[-1][0]) > 1e-6 or abs(pt[1] - dedup[-1][1]) > 1e-6:
            dedup.append(pt)
    if len(dedup) >= 2 and abs(dedup[0][0] - dedup[-1][0]) < 1e-6 and abs(dedup[0][1] - dedup[-1][1]) < 1e-6:
        dedup.pop()
    return dedup if len(dedup) >= 3 else []


def ring_area(ring):
    s = 0.0
    for i in range(len(ring)):
        x0, y0 = ring[i][0], ring[i][1]
        x1, y1 = ring[(i + 1) % len(ring)][0], ring[(i + 1) % len(ring)][1]
        s += x0 * y1 - x1 * y0
    return abs(s) * 0.5


def load(layer, region):
    return json.load(open(os.path.join(HD, layer, region + ".geojson"), encoding="utf-8"))["features"]


def emit(buckets, out):
    os.makedirs(out, exist_ok=True)
    for (tx, tz), lines in sorted(buckets.items()):
        with open(os.path.join(out, f"tile_{tx}_{tz}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def to_enz(coords):
    """[lon,lat,z] 목록 → [(E,N,z)]; z 결측=0."""
    enz = []
    for c in coords:
        if len(c) < 2:
            continue
        E, N = _fwd.transform(c[0], c[1])
        enz.append((E, N, float(c[2]) if len(c) >= 3 else 0.0))
    return enz


def fmt_pts(enz):
    out = []
    for E, N, z in enz:
        lon, lat = _inv.transform(E, N)
        out.append(f"{lat:.7f} {lon:.7f} {z:.2f}")
    return " ".join(out)


def prop(props, key):
    v = props.get(key)
    s = str(v).strip() if v is not None else ""
    return s.replace(" ", "_") if s else "-"


def tile_range(enz):
    Es = [p[0] for p in enz]; Ns = [p[1] for p in enz]
    return (int(min(Es) // 1000), int(max(Es) // 1000),
            int(min(Ns) // 1000), int(max(Ns) // 1000))


def bucket_poly_clip(feats, head_fn, min_area=0.5):
    """면 레이어 → 타일 클리핑(z 보존) — shapely 교차로 멀티파트 분리 방출.

    Sutherland-Hodgman 은 타일을 드나드는 오목 폴리곤에서 경계 브리지(零폭 통로)를 만들어
    500m+ 삼각형 경고를 유발 → shapely intersection 으로 파트별 정상 링 방출.
    z 는 최근접 원본 정점에서 복원(도로 경계 정점 간격 2~3m → 오차 미미).
    """
    from shapely.geometry import Polygon, box
    buckets, npoly, npiece, nhole = {}, 0, 0, 0
    for f in feats:
        g = f.get("geometry") or {}
        if g.get("type") != "MultiPolygon":
            continue
        head = head_fn(f.get("properties") or {})
        for poly in g["coordinates"]:
            if not poly:
                continue
            nhole += max(0, len(poly) - 1)
            ring = to_enz(poly[0])
            if len(ring) < 3:
                continue
            npoly += 1
            try:
                shp = Polygon([(p[0], p[1]) for p in ring])
                if not shp.is_valid:
                    shp = shp.buffer(0)          # 자가교차 정리
            except Exception:  # noqa: BLE001
                continue
            x0, x1, z0, z1 = tile_range(ring)
            for tx in range(x0, x1 + 1):
                for tz in range(z0, z1 + 1):
                    cell = box(tx * TILE_M, tz * TILE_M, (tx + 1) * TILE_M, (tz + 1) * TILE_M)
                    inter = shp.intersection(cell)
                    if inter.is_empty:
                        continue
                    geoms = getattr(inter, "geoms", [inter])
                    for part in geoms:
                        if part.geom_type != "Polygon" or part.area < min_area:
                            continue
                        nhole += len(part.interiors)
                        cl = [_restore_z(x, y, ring) for x, y in part.exterior.coords[:-1]]
                        if len(cl) < 3:
                            continue
                        buckets.setdefault((tx, tz), []).append(f"{head} {fmt_pts(cl)}")
                        npiece += 1
    return buckets, npoly, npiece, nhole


def _restore_z(x, y, ring):
    """클립 정점의 z = 최근접 원본 정점 z."""
    best, bz = float("inf"), 0.0
    for e, n, z in ring:
        d = (e - x) * (e - x) + (n - y) * (n - y)
        if d < best:
            best, bz = d, z
    return (x, y, bz)


def bucket_poly_centroid(feats, head_fn):
    buckets, n = {}, 0
    for f in feats:
        g = f.get("geometry") or {}
        if g.get("type") != "MultiPolygon":
            continue
        head = head_fn(f.get("properties") or {})
        for poly in g["coordinates"]:
            if not poly:
                continue
            ring = to_enz(poly[0])
            if len(ring) < 3:
                continue
            cE = sum(p[0] for p in ring) / len(ring); cN = sum(p[1] for p in ring) / len(ring)
            buckets.setdefault((int(cE // 1000), int(cN // 1000)), []).append(f"{head} {fmt_pts(ring)}")
            n += 1
    return buckets, n


def bucket_line_clip(feats, head_fn):
    buckets, nline, npiece = {}, 0, 0
    for f in feats:
        g = f.get("geometry") or {}
        if g.get("type") != "MultiLineString":
            continue
        head = head_fn(f.get("properties") or {})
        for ls in g["coordinates"]:
            enz = to_enz(ls)
            if len(enz) < 2:
                continue
            nline += 1
            x0, x1, z0, z1 = tile_range(enz)
            for tx in range(x0, x1 + 1):
                for tz in range(z0, z1 + 1):
                    rect = (tx * TILE_M, tz * TILE_M, (tx + 1) * TILE_M, (tz + 1) * TILE_M)
                    for pc in clip_polyline3(enz, rect):
                        line = f"{head} {fmt_pts(pc)}" if head else fmt_pts(pc)
                        buckets.setdefault((tx, tz), []).append(line)
                        npiece += 1
    return buckets, nline, npiece


def bucket_point(feats, head_fn):
    buckets, n = {}, 0
    for f in feats:
        g = f.get("geometry") or {}
        if g.get("type") != "Point":
            continue
        c = g["coordinates"]
        if len(c) < 2:
            continue
        E, N = _fwd.transform(c[0], c[1])
        z = float(c[2]) if len(c) >= 3 else 0.0
        head = head_fn(f.get("properties") or {})
        lon, lat = _inv.transform(E, N)
        buckets.setdefault((int(E // 1000), int(N // 1000)), []).append(f"{head} {lat:.7f} {lon:.7f} {z:.2f}")
        n += 1
    return buckets, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="seoul_gangnamgu")
    args = ap.parse_args()
    reg = args.region
    ob = os.path.join(REPO, "tools", "nationwide_v2")

    # ── A3 차도구간면(사각 클리핑, z 보존) ──
    b, npoly, npiece, nhole = bucket_poly_clip(
        load("lt_c_a3drivewaysection", reg),
        lambda pr: f"{prop(pr, 'kind')} {prop(pr, 'roadtype')}")
    emit(b, os.path.join(ob, "hd_roadsurf", reg))
    print(f"[a3] 차도면 {npoly}폴리곤 → 조각 {npiece} / {len(b)}타일 (홀 {nhole} 미방출)", flush=True)

    # ── A2 주행경로링크(클리핑, 위상 속성 유지) ──
    b, nline, npiece = bucket_line_clip(
        load("lt_l_a2link", reg),
        lambda pr: (f"{prop(pr, 'laneno')} {prop(pr, 'linktype')} {prop(pr, 'roadrank')} "
                    f"{prop(pr, 'fromnodeid')} {prop(pr, 'tonodeid')} {prop(pr, 'r_linkid')} {prop(pr, 'l_linkid')}"))
    emit(b, os.path.join(ob, "hd_links", reg))
    print(f"[a2] 링크 {nline}선 → 조각 {npiece} / {len(b)}타일", flush=True)

    # ── B3 노면표시면(centroid — 화살표/횡단보도/도류대 등 소형) ──
    b, n = bucket_poly_centroid(load("lt_c_b3surfacemark", reg), lambda pr: prop(pr, "kind"))
    emit(b, os.path.join(ob, "hd_marks", reg))
    print(f"[b3] 노면표시 {n}면 / {len(b)}타일", flush=True)

    # ── B1 안전표지(점) ──
    b, n = bucket_point(load("lt_p_b1safetysign", reg), lambda pr: prop(pr, "kind"))
    emit(b, os.path.join(ob, "hd_signs", reg))
    print(f"[b1] 표지 {n}점 / {len(b)}타일", flush=True)

    # ── C5 높이장벽(선) ──
    b, nline, npiece = bucket_line_clip(load("lt_l_c5heightbarrier", reg), lambda pr: "")
    emit(b, os.path.join(ob, "hd_heightbarriers", reg))
    print(f"[c5] 장벽 {nline}선 → 조각 {npiece} / {len(b)}타일", flush=True)

    # ── C6 지주(점) ──
    b, n = bucket_point(load("lt_p_c6postpoint", reg), lambda pr: prop(pr, "kind"))
    emit(b, os.path.join(ob, "hd_posts", reg))
    print(f"[c6] 지주 {n}점 / {len(b)}타일", flush=True)

    # ── A5 주차장면(centroid) ──
    b, n = bucket_poly_centroid(load("lt_c_a5parkinglot", reg), lambda pr: prop(pr, "kind"))
    emit(b, os.path.join(ob, "hd_parkinglots", reg))
    print(f"[a5] 주차장 {n}면 / {len(b)}타일 → 완료", flush=True)


if __name__ == "__main__":
    main()
