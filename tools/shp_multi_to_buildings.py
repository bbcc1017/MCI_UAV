# -*- coding: utf-8 -*-
"""3분할(여러 개) AL_D010 SHP → 하나의 Unity buildings.bin.

skill `korean-gis-to-unity-nodxf-v2`의 shp_to_buildings.py와 동일한 포맷/높이공식을
쓰되, 여러 --shp 를 순서대로 한 .bin 에 합쳐 쓴다 (경기도처럼 건물정보가 3분할인 경우).

높이 공식: A16(0<x<200) → A26*3.5(≤120) → 6m
.bin 포맷(LE): [i32 count] 반복{ [i32 ring_count][f32 h][f32 cx][f32 cy]
  반복{ [i32 vc][i32 is_hole][f32×2×vc xy] } }

사용:
  python shp_multi_to_buildings.py --out out.bin --shp a.shp b.shp c.shp
필요 env: qgis_batch (osgeo/GDAL)
"""
from __future__ import annotations
import argparse, os, struct, sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from osgeo import ogr


def compute_height(a16, a26):
    if a16 is not None and 0.0 < a16 < 200.0:
        return float(a16)
    if a26 is not None and a26 > 0:
        return float(min(a26 * 3.5, 120.0))
    return 6.0


def ring_to_points(ring):
    n = ring.GetPointCount()
    pts = []
    for i in range(n):
        x, y, _ = ring.GetPoint(i)
        pts.append((x, y))
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts.pop()
    return pts


def process_shp(path, fo, hstat):
    drv = ogr.GetDriverByName("ESRI Shapefile")
    ds = drv.Open(path, 0)
    if ds is None:
        print(f"[ERROR] SHP open fail: {path}", file=sys.stderr)
        return 0, 0, 0
    lyr = ds.GetLayer(0)
    ldef = lyr.GetLayerDefn()
    fields = [ldef.GetFieldDefn(i).GetName() for i in range(ldef.GetFieldCount())]
    has_a16 = "A16" in fields
    has_a26 = "A26" in fields
    total = lyr.GetFeatureCount()
    print(f"[INFO] {os.path.basename(path)}: features={total}, A16={has_a16}, A26={has_a26}")
    written = no_geom = empty = 0
    for fi in range(total):
        feat = lyr.GetFeature(fi)
        if feat is None:
            continue
        geom = feat.GetGeometryRef()
        if geom is None:
            no_geom += 1
            continue
        a16 = feat.GetField("A16") if has_a16 else None
        a26 = feat.GetField("A26") if has_a26 else None
        h = compute_height(a16, a26)
        if a16 is not None and 0.0 < a16 < 200.0:
            hstat["a16"] += 1
        elif a26 is not None and a26 > 0:
            hstat["a26"] += 1
        else:
            hstat["fallback6"] += 1

        gt = geom.GetGeometryType()
        polys = []
        if gt in (ogr.wkbPolygon, ogr.wkbPolygon25D, ogr.wkbPolygonM, ogr.wkbPolygonZM):
            polys.append(geom)
        elif gt in (ogr.wkbMultiPolygon, ogr.wkbMultiPolygon25D):
            for i in range(geom.GetGeometryCount()):
                polys.append(geom.GetGeometryRef(i))
        else:
            continue

        for poly in polys:
            rc = poly.GetGeometryCount()
            if rc == 0:
                empty += 1
                continue
            rings = []
            for r in range(rc):
                pts = ring_to_points(poly.GetGeometryRef(r))
                if len(pts) < 3:
                    continue
                rings.append((r > 0, pts))
            if not rings:
                empty += 1
                continue
            env = poly.GetEnvelope()
            cx = (env[0] + env[1]) / 2
            cy = (env[2] + env[3]) / 2
            fo.write(struct.pack("<i", len(rings)))
            fo.write(struct.pack("<f", h))
            fo.write(struct.pack("<ff", cx, cy))
            for is_hole, pts in rings:
                fo.write(struct.pack("<i", len(pts)))
                fo.write(struct.pack("<i", 1 if is_hole else 0))
                for x, y in pts:
                    fo.write(struct.pack("<ff", x, y))
            written += 1
        if (fi + 1) % 50000 == 0:
            print(f"  [{fi+1}/{total}] written={written}")
    return written, no_geom, empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shp", required=True, nargs="+", help="입력 SHP 여러 개")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    hstat = {"a16": 0, "a26": 0, "fallback6": 0}
    total_written = total_nogeom = total_empty = 0
    with open(args.out, "wb") as fo:
        fo.write(struct.pack("<i", 0))  # placeholder count
        for shp in args.shp:
            w, ng, em = process_shp(shp, fo, hstat)
            total_written += w
            total_nogeom += ng
            total_empty += em
            print(f"  -> 누적 written={total_written}")
        fo.seek(0)
        fo.write(struct.pack("<i", total_written))

    print(f"[DONE] written={total_written}, no_geom={total_nogeom}, empty={total_empty}")
    print(f"[HEIGHT] A16={hstat['a16']}, A26={hstat['a26']}, fallback6={hstat['fallback6']}")
    print(f"[OUT] {args.out}  ({os.path.getsize(args.out):,} bytes)")


if __name__ == "__main__":
    main()
