"""v2 보행 인프라 추출 — 정밀도로지도 b3(노면표시)/a4(보도)/c1(신호) → WLK2 사이드카 바이너리.

자율주행 RL 씬(CAR_test)에 '횡단보도 건너는 보행자'를 넣기 위한 데이터. LGV2 차선 bin(seoul_gangnamgu.bin)은
건드리지 않고 별도 <region>.walk.bin 을 생성(기존 LaneRoadBuilderV2/씬 무영향). Unity PedestrianManagerV2·
TrafficSignalDirectorV2 가 BinaryReader 로 직독.

입력(tools/nationwide/hdmap/<layer>/<region>.geojson, 좌표 [lon,lat,z]):
 · lt_c_b3surfacemark  노면표시(면) — kind {5321,534,533}=횡단보도/자전거횡단도(보행신호 근접 실측 확정) → 건너는 구역
 · lt_p_c1trafficlight 신호(점) — type 11=보행등 → 횡단보도 walk 페이즈 위치
 · lt_c_a4subsidiarysection 보도(면, name=보도) → 보행자 스폰/배회 영역(경량: 중심+반경)

출력:
 · <region>.walk.bin   WLK2 리틀엔디언(앵커는 LGV2 bin 과 동일 = Unity 월드변환 일치)
 · <region>.walk.report.json  검증 통계

WLK2 포맷(리틀엔디언, f32 좌표는 anchorE/N 상대):
  'WLK2' u32ver | u32×3 nCross nPedSig nSide | f64 anchorE anchorN
  crossA f32×3[nCross] | crossB f32×3[nCross] | crossW f32[nCross] | crossKind u16[nCross] | crossPed i32[nCross]
  pedPos f32×3[nPedSig]
  sideCenter f32×2[nSide] | sideRadius f32[nSide]
  (crossA/B = 횡단보도 장축 양끝(양쪽 연석), crossW = 횡단 폭(깊이), crossPed = 최근접 보행신호 인덱스)

실행: PYTHONIOENCODING=utf-8 <UAV env python> tools/v2_walk_extract.py --region seoul_gangnamgu
"""
import argparse
import json
import os
import struct

import numpy as np
from pyproj import Transformer

P5186 = ("+proj=tmerc +lat_0=38 +lon_0=127 +k=1 +x_0=200000 +y_0=600000 "
         "+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HD = os.path.join(REPO, "tools", "nationwide", "hdmap")
OUT = os.path.join(REPO, "tools", "nationwide_v2", "lanegraph")
_fwd = Transformer.from_crs("EPSG:4326", P5186, always_xy=True)

CROSS_KINDS = {5321, 534, 533}     # 횡단보도/자전거횡단도(보행신호 근접 실측 확정)
PED_SIGNAL_TYPE = 11               # 보행등


def load(layer, region):
    return json.load(open(os.path.join(HD, layer, region + ".geojson"), encoding="utf-8"))["features"]


def norm_int(v, default=0):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def poly_points(geom):
    """MultiPolygon → 전 외곽링 [lon,lat,z] 평탄 리스트."""
    if geom.get("type") != "MultiPolygon":
        return []
    out = []
    for poly in geom.get("coordinates", []):
        if poly:
            out.extend(poly[0])   # 외곽 링만
    return out


def read_lane_anchor(region):
    """LGV2 bin 헤더에서 anchorE/N 읽기(offset 32, f64×2) — 좌표 앵커 공유."""
    p = os.path.join(OUT, region + ".bin")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        head = f.read(48)
    if head[:4] != b"LGV2":
        return None
    return struct.unpack_from("<2d", head, 32)   # anchorE, anchorN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="seoul_gangnamgu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    reg = args.region

    # ── 보행신호(type 11) ──
    ped_en = []
    for f in load("lt_p_c1trafficlight", reg):
        p = f.get("properties") or {}
        if norm_int(p.get("type")) != PED_SIGNAL_TYPE:
            continue
        g = f.get("geometry") or {}
        if g.get("type") != "Point":
            continue
        c = g["coordinates"]
        E, N = _fwd.transform(c[0], c[1])
        ped_en.append((E, N, c[2] if len(c) > 2 else 0.0))
    ped_arr = np.asarray(ped_en, dtype=np.float64) if ped_en else np.zeros((0, 3))

    # 보행신호 40m 그리드(횡단보도 최근접 검색)
    grid = {}
    for i, (E, N, _) in enumerate(ped_en):
        grid.setdefault((int(E // 40), int(N // 40)), []).append(i)

    def nearest_ped(E, N):
        gx, gy = int(E // 40), int(N // 40)
        best, bi = 1e18, -1
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for i in grid.get((gx + a, gy + b), []):
                    d = (ped_en[i][0] - E) ** 2 + (ped_en[i][1] - N) ** 2
                    if d < best:
                        best, bi = d, i
        return bi, best ** 0.5

    # ── 횡단보도(b3 CROSS_KINDS) → PCA 장축 양끝 ──
    cA, cB, cW, cKind, cPed = [], [], [], [], []
    with_ped = 0
    for f in load("lt_c_b3surfacemark", reg):
        p = f.get("properties") or {}
        kind = norm_int(p.get("kind"))
        if kind not in CROSS_KINDS:
            continue
        pts = poly_points(f.get("geometry") or {})
        if len(pts) < 3:
            continue
        arr = np.asarray([(c[0], c[1], c[2] if len(c) > 2 else 0.0) for c in pts], dtype=np.float64)
        E, N = _fwd.transform(arr[:, 0], arr[:, 1])
        xy = np.column_stack([E, N])
        z = float(np.median(arr[:, 2]))
        ctr = xy.mean(axis=0)
        d = xy - ctr
        # PCA: 공분산 고유벡터 → 장축(횡단 방향)/단축(깊이)
        cov = np.cov(d.T)
        w, v = np.linalg.eigh(cov)
        major = v[:, np.argmax(w)]
        minor = v[:, np.argmin(w)]
        tmaj = d @ major
        tmin = d @ minor
        A = ctr + major * tmaj.min()
        B = ctr + major * tmaj.max()
        width = float(tmin.max() - tmin.min())
        pi, pdist = nearest_ped(ctr[0], ctr[1])
        if pdist < 20.0:
            with_ped += 1
        cA.append((A[0], A[1], z))
        cB.append((B[0], B[1], z))
        cW.append(width)
        cKind.append(kind)
        cPed.append(pi if pdist < 25.0 else -1)

    # ── 보도(a4 name=보도) → 중심+반경 ──
    sC, sR = [], []
    for f in load("lt_c_a4subsidiarysection", reg):
        p = f.get("properties") or {}
        if str(p.get("name")).strip() != "보도":
            continue
        pts = poly_points(f.get("geometry") or {})
        if len(pts) < 3:
            continue
        arr = np.asarray([(c[0], c[1]) for c in pts], dtype=np.float64)
        E, N = _fwd.transform(arr[:, 0], arr[:, 1])
        xy = np.column_stack([E, N])
        ctr = xy.mean(axis=0)
        r = float(np.sqrt(((xy - ctr) ** 2).sum(axis=1)).max())
        sC.append((ctr[0], ctr[1]))
        sR.append(min(r, 60.0))   # 과대 폴리곤 클램프

    # ── 앵커 ──
    anchor = read_lane_anchor(reg)
    if anchor is None:
        allE = [a[0] for a in cA] + [q[0] for q in ped_en] + [c[0] for c in sC]
        allN = [a[1] for a in cA] + [q[1] for q in ped_en] + [c[1] for c in sC]
        anchor = (float(np.mean(allE)) if allE else 0.0, float(np.mean(allN)) if allN else 0.0)
        print(f"[walk] ⚠ LGV2 bin 없음 → 자체 앵커 {anchor}")
    aE, aN = anchor

    nCross, nPed, nSide = len(cA), len(ped_en), len(sC)

    def rel3(lst):
        a = np.asarray(lst, dtype=np.float64).reshape(-1, 3)
        a[:, 0] -= aE
        a[:, 1] -= aN
        return a.astype("<f4")

    def rel2(lst):
        a = np.asarray(lst, dtype=np.float64).reshape(-1, 2)
        a[:, 0] -= aE
        a[:, 1] -= aN
        return a.astype("<f4")

    out_path = args.out or os.path.join(OUT, reg + ".walk.bin")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as fo:
        fo.write(b"WLK2")
        fo.write(struct.pack("<I", 1))
        fo.write(struct.pack("<3I", nCross, nPed, nSide))
        fo.write(struct.pack("<2d", aE, aN))
        if nCross:
            fo.write(rel3(cA).tobytes())
            fo.write(rel3(cB).tobytes())
            fo.write(np.asarray(cW, dtype="<f4").tobytes())
            fo.write(np.asarray(cKind, dtype="<u2").tobytes())
            fo.write(np.asarray(cPed, dtype="<i4").tobytes())
        if nPed:
            fo.write(rel3(ped_en).tobytes())
        if nSide:
            fo.write(rel2(sC).tobytes())
            fo.write(np.asarray(sR, dtype="<f4").tobytes())

    widths = np.asarray(cW) if cW else np.zeros(1)
    lengths = np.asarray([np.hypot(cB[i][0] - cA[i][0], cB[i][1] - cA[i][1]) for i in range(nCross)]) if nCross else np.zeros(1)
    report = {
        "region": reg,
        "anchorE": aE, "anchorN": aN,
        "n_crosswalk": nCross,
        "n_ped_signal": nPed,
        "n_sidewalk": nSide,
        "crosswalk_with_pedsignal_pct": round(100.0 * with_ped / max(1, nCross), 1),
        "crosswalk_len_m_median": round(float(np.median(lengths)), 1),
        "crosswalk_width_m_median": round(float(np.median(widths)), 1),
        "kinds": {str(int(k)): int((np.asarray(cKind) == k).sum()) for k in sorted(set(cKind))} if cKind else {},
        "bin_bytes": os.path.getsize(out_path),
    }
    rp = os.path.join(OUT, reg + ".walk.report.json")
    json.dump(report, open(rp, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=int)
    print(f"[walk] {reg}: 횡단보도 {nCross} · 보행신호 {nPed} · 보도 {nSide} "
          f"· 신호근접 {report['crosswalk_with_pedsignal_pct']}% · {out_path} ({report['bin_bytes']}B)")


if __name__ == "__main__":
    main()
