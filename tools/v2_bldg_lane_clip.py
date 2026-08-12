#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v2 건물 ↔ 차선 회랑 충돌 진단/제거 — 자율주행 차도 한복판에 선 건물을 걸러낸다.

배경: `v2_buildings_bucket.py` 는 vWorld GIS건물통합정보 footprint 를 그대로 타일에 버킷하고
TileBakerV2 가 압출한다. 그 원천에는 **도로 위에 얹힌 동**이 섞여 있다(공사 중 가설물·철거 후
잔존 레코드·좌표 오차·MultiPolygon 슬리버 등). 그 결과 CAR_test 자율주행 씬에서 교차로 한복판에
건물 콜라이더가 서고 NPC/에고가 들이받는다.

판정: 정밀도로지도 **A2 주행링크 중심선 ±(lane_half+margin)** 회랑 + **A3 차도면 폴리곤** 과의
겹침 면적. 차도 밖(보도·인도·건물부지) 건물은 당연히 남긴다.

산출:
  --report : 겹치는 동 목록(겹침면적 내림차순) + 요약. 파일 안 건드림.
  --apply  : ① 타일 txt 에서 해당 라인 제거(원본은 `<tile>.txt.orig` 백업, 재실행 시 백업 기준 = 멱등)
             ② `buildings_tiles/<region>/laneclip.json` 에 제거된 동의 **타일로컬 XZ 폴리곤** 기록
                (이미 베이크된 Unity 메시를 재베이크 없이 트리밍하기 위한 입력)

실행(Windows):
  PYTHONIOENCODING=utf-8 /c/Users/User/anaconda3/envs/UAV/python.exe tools/v2_bldg_lane_clip.py --region seoul_gangnamgu --report
  PYTHONIOENCODING=utf-8 /c/Users/User/anaconda3/envs/UAV/python.exe tools/v2_bldg_lane_clip.py --region seoul_gangnamgu --apply
"""
import os, json, glob, argparse
from pyproj import Transformer
from shapely.geometry import Polygon, LineString, Point, shape
from shapely.strtree import STRtree
from shapely.ops import unary_union

P5186 = ("+proj=tmerc +lat_0=38 +lon_0=127 +k=1 +x_0=200000 +y_0=600000 "
         "+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
TILE_M = 1000.0
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_tr = Transformer.from_crs("EPSG:4326", P5186, always_xy=True)
_inv = Transformer.from_crs(P5186, "EPSG:4326", always_xy=True)


def hd_path(layer, region):
    return os.path.join(REPO, "tools", "nationwide", "hdmap", layer, f"{region}.geojson")


def load_lanes(region):
    """A2 주행링크 → EPSG:5186 LineString 목록."""
    p = hd_path("lt_l_a2link", region)
    if not os.path.exists(p):
        raise SystemExit(f"[clip] A2 링크 없음: {p}")
    gj = json.load(open(p, encoding="utf-8"))
    out = []
    for f in gj["features"]:
        g = f.get("geometry") or {}
        t = g.get("type"); c = g.get("coordinates")
        parts = [c] if t == "LineString" else (c if t == "MultiLineString" else [])
        for line in parts:
            pts = [(x, y) for x, y, *_ in line]
            if len(pts) < 2:
                continue
            E, N = _tr.transform([p[0] for p in pts], [p[1] for p in pts])
            out.append(LineString(list(zip(E, N))))
    return out


def load_driveway(region):
    """A3 차도면 폴리곤 → EPSG:5186 (없으면 빈 리스트)."""
    p = hd_path("lt_c_a3drivewaysection", region)
    if not os.path.exists(p):
        return []
    gj = json.load(open(p, encoding="utf-8"))
    out = []
    for f in gj["features"]:
        g = f.get("geometry") or {}
        t = g.get("type"); c = g.get("coordinates")
        polys = [c] if t == "Polygon" else (c if t == "MultiPolygon" else [])
        for poly in polys:
            ring = poly[0] if poly else None
            if not ring or len(ring) < 4:
                continue
            E, N = _tr.transform([q[0] for q in ring], [q[1] for q in ring])
            pg = Polygon(list(zip(E, N)))
            if not pg.is_valid:
                pg = pg.buffer(0)
            if not pg.is_empty:
                out.append(pg)
    return out


def load_nodes(region, radius):
    """A1 교차로 노드 → 반경 radius 원판(EPSG:5186).

    ⚠A3 차도면은 커버리지가 희박해(강남 364폴리곤) **교차로 내부가 마스크에서 빠진다**.
    지하철역·지하상가 같은 지하구조물이 교차로 한복판에 얹혀도 안 걸리는 이유가 이것이다.
    교차로 안에는 실제 건물이 있을 수 없으므로 노드 원판을 통째로 마스크에 넣는다.
    """
    p = hd_path("lt_p_a1node", region)
    if not os.path.exists(p):
        return []
    gj = json.load(open(p, encoding="utf-8"))
    out = []
    for f in gj["features"]:
        g = f.get("geometry") or {}
        if g.get("type") != "Point":
            continue
        c = g.get("coordinates")
        if not c or len(c) < 2:
            continue
        E, N = _tr.transform(c[0], c[1])
        out.append(Point(E, N).buffer(radius, resolution=8))
    return out


def parse_tile(path):
    """타일 txt → [(원본라인, 높이, 유형, [(lon,lat)…])]"""
    recs = []
    for line in open(path, encoding="utf-8").read().splitlines():
        tok = line.split()
        if len(tok) < 8:
            continue
        try:
            h = float(tok[0]); bt = int(tok[1])
        except ValueError:
            continue
        ring = []
        for i in range(2, len(tok) - 1, 2):
            try:
                ring.append((float(tok[i]), float(tok[i + 1])))
            except ValueError:
                pass
        if len(ring) >= 3:
            recs.append((line, h, bt, ring))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="seoul_gangnamgu")
    ap.add_argument("--lane-half", type=float, default=1.70, help="차선 반폭(m)")
    ap.add_argument("--margin", type=float, default=0.60, help="차체 여유(m) — 회랑 확장분")
    ap.add_argument("--min-overlap", type=float, default=0.5, help="이 면적(m²) 이상 겹쳐야 후보")
    ap.add_argument("--node-radius", type=float, default=14.0,
                    help="A1 교차로 노드 원판 반경(m) — 교차로 내부 건물 제거용")
    ap.add_argument("--drop-h", type=float, default=4.0,
                    help="이 높이 이하면 통째 제거 — height/grnd_flr 결측 폴백(3.5m) 유령 레코드 대역")
    ap.add_argument("--drop-ratio", type=float, default=0.20,
                    help="발자국 대비 침범비율이 이 값 이상이면 통째 제거")
    ap.add_argument("--drop-dist", type=float, default=1.20,
                    help="차선 중심선까지 거리가 이 값(m) 미만이면 통째 제거(주행선 직격)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()
    if not (args.report or args.apply):
        args.report = True

    region = args.region
    tdir = os.path.join(REPO, "tools", "nationwide_v2", "buildings_tiles", region)
    tiles = sorted(glob.glob(os.path.join(tdir, "tile_*.txt")))
    if not tiles:
        raise SystemExit(f"[clip] 타일 txt 없음: {tdir}")

    lanes = load_lanes(region)
    drive = load_driveway(region)
    nodes = load_nodes(region, args.node_radius)
    drive = drive + nodes
    print(f"[clip] {region}: A2 {len(lanes)}선 · A3+교차로원판 {len(drive)}면(노드 {len(nodes)}) · 타일 {len(tiles)}개", flush=True)

    lane_tree = STRtree(lanes)
    drive_tree = STRtree(drive) if drive else None
    half = args.lane_half + args.margin

    offenders = []          # (overlap, tile, h, bt, lon, lat, dist, on_a3, line, ring)
    total = 0
    for tp in tiles:
        orig = tp + ".orig"
        src = orig if os.path.exists(orig) else tp
        base = os.path.basename(tp)[len("tile_"):-len(".txt")]
        tx, tz = (int(v) for v in base.split("_"))
        for line, h, bt, ring in parse_tile(src):
            total += 1
            E, N = _tr.transform([p[0] for p in ring], [p[1] for p in ring])
            pg = Polygon(list(zip(E, N)))
            if not pg.is_valid:
                pg = pg.buffer(0)
            if pg.is_empty:
                continue
            probe = pg.buffer(half)
            hit = [lanes[i] for i in lane_tree.query(probe)]
            ov = 0.0; dist = 1e9
            if hit:
                corr = unary_union([ln.buffer(half, cap_style=2) for ln in hit])
                inter = pg.intersection(corr)
                ov = inter.area
                dist = min(pg.distance(ln) for ln in hit)
            on_a3 = False
            if drive_tree is not None:
                for i in drive_tree.query(pg):
                    if drive[i].intersects(pg):
                        a3 = drive[i].intersection(pg).area
                        if a3 > args.min_overlap:
                            on_a3 = True
                            ov = max(ov, a3)
                        break
            if ov >= args.min_overlap:
                c = pg.centroid
                lon, lat = _inv.transform(c.x, c.y)
                offenders.append(dict(overlap=ov, tile=(tx, tz), h=h, bt=bt, lon=lon, lat=lat,
                                      dist=dist, on_a3=on_a3, line=line, area=pg.area,
                                      cE=c.x, cN=c.y, ring_en=list(zip(E, N))))

    offenders.sort(key=lambda d: -d["overlap"])
    print(f"[clip] 건물 {total}동 중 차도 침범 {len(offenders)}동 "
          f"(회랑 반폭 {half:.2f}m, 최소겹침 {args.min_overlap}m²)", flush=True)
    # 요약 — 침범 비율(겹침/발자국)·높이 분포. 비율↑ = 통째로 차도 위(유령 레코드), 비율↓ = 실제 건물 모서리.
    if offenders:
        rat = sorted(d["overlap"] / max(d["area"], 1e-6) for d in offenders)
        for lo, hi in ((0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.01)):
            n = sum(1 for r in rat if lo <= r < hi)
            print(f"    침범비율 {lo:.0%}~{hi:.0%}: {n}동", flush=True)
        for lo, hi in ((0, 4), (4, 10), (10, 30), (30, 1000)):
            n = sum(1 for d in offenders if lo <= d["h"] < hi)
            print(f"    높이 {lo}~{hi}m: {n}동", flush=True)
    for d in offenders[:args.top]:
        print(f"  tile_{d['tile'][0]}_{d['tile'][1]}  겹침 {d['overlap']:7.1f}m²/{d['area']:8.1f}m²  "
              f"h={d['h']:5.1f} t={d['bt']}  중심선거리 {d['dist']:5.2f}m  A3={'Y' if d['on_a3'] else 'n'}  "
              f"({d['lat']:.6f},{d['lon']:.6f})", flush=True)

    # ── 제거 대상 판정 ───────────────────────────────────────────
    #   유령(높이 결측 폴백)·대부분 차도 위·주행선 직격만 통째 제거. 실제 건물이 모서리만
    #   스치는 경우(비율↓·높이↑)는 남긴다 — 중심선에서 여전히 2m 넘게 떨어져 차폭에 안 걸린다.
    def drops(d):
        return (d["h"] <= args.drop_h or
                d["overlap"] / max(d["area"], 1e-6) >= args.drop_ratio or
                d["dist"] < args.drop_dist)
    kept = [d for d in offenders if not drops(d)]
    offenders = [d for d in offenders if drops(d)]
    print(f"[clip] 제거 {len(offenders)}동 / 잔류(모서리 스침) {len(kept)}동", flush=True)
    for d in kept[:10]:
        print(f"    잔류 tile_{d['tile'][0]}_{d['tile'][1]} h={d['h']:.1f} "
              f"겹침 {d['overlap']:.1f}m² 비율 {d['overlap']/max(d['area'],1e-6):.1%} "
              f"거리 {d['dist']:.2f}m ({d['lat']:.6f},{d['lon']:.6f})", flush=True)

    if not args.apply:
        return

    # ── 제거 적용 ────────────────────────────────────────────────
    by_tile = {}
    for d in offenders:
        by_tile.setdefault(d["tile"], []).append(d)
    clip = {"region": region, "tile_m": TILE_M, "lane_half": args.lane_half,
            "margin": args.margin, "removed": len(offenders), "tiles": {}}
    for tp in tiles:
        base = os.path.basename(tp)[len("tile_"):-len(".txt")]
        tx, tz = (int(v) for v in base.split("_"))
        orig = tp + ".orig"
        if not os.path.exists(orig):
            os.replace(tp, orig)
        drop = {d["line"] for d in by_tile.get((tx, tz), [])}
        lines = open(orig, encoding="utf-8").read().splitlines()
        keep = [l for l in lines if l not in drop]
        with open(tp, "w", encoding="utf-8") as fp:
            fp.write("\n".join(keep))
        if drop:
            cE, cN = tx * TILE_M + TILE_M * 0.5, tz * TILE_M + TILE_M * 0.5
            clip["tiles"][f"{tx}_{tz}"] = [
                {"h": d["h"], "poly": [[round(e - cE, 3), round(n - cN, 3)] for e, n in d["ring_en"]]}
                for d in by_tile[(tx, tz)]]
    outp = os.path.join(tdir, "laneclip.json")
    json.dump(clip, open(outp, "w", encoding="utf-8"), ensure_ascii=False)
    # C# 파서용 평문(사전 구조 없이 한 줄 = 한 동): `tileX tileZ nPts x z x z …` (타일로컬 m)
    txtp = os.path.join(tdir, "laneclip.txt")
    with open(txtp, "w", encoding="utf-8") as fp:
        for key, polys in clip["tiles"].items():
            tx, tz = key.split("_")
            for d in polys:
                pts = " ".join(f"{x:.3f} {z:.3f}" for x, z in d["poly"])
                fp.write(f"{tx} {tz} {len(d['poly'])} {pts}\n")
    print(f"[clip] 적용 — {len(offenders)}동 제거, 타일 txt 갱신(.orig 백업) → {outp}", flush=True)


if __name__ == "__main__":
    main()
