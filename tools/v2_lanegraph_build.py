#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v2 차선그래프 빌더 — 정밀도로지도 A2/A1/B2/C1 → LGV2 바이너리(자율주행 시험용 차선 단위 주행 그래프).

입력(tools/nationwide/hdmap/<layer>/<region>.geojson, 좌표 [lon,lat,z]):
 · lt_l_a2link          주행경로링크 — fromnodeid/tonodeid(방향 위상)·l_linkid/r_linkid(차선변경 인접)·laneno·roadrank
 · lt_p_a1node          주행경로노드 — nodetype(제로패딩 "07"→7 정규화)
 · lt_l_b2surfacelinemark 노면선표시 — kind 530=정지선(l/r_linkid 대부분 null → 기하 매핑), 그 외=차선 경계(폭 복원)
 · lt_p_c1trafficlight  신호등 — linkid 로 규제 대상 링크 직결(type 1/2=차량등만 로직 대상)

산출(tools/nationwide_v2/lanegraph/):
 · <region>.bin          LGV2 리틀엔디언 바이너리(Unity LaneGraphV2 가 BinaryReader 로 직독)
 · <region>.report.json  검증 통계 + 합격기준 판정(§합격기준)
 · <region>.debug.geojson (--debug-geojson) 정지선 s점·교차로 그룹·신호-링크 연결선 — QGIS 육안 검수용

그래프 규약:
 · successor(L) = { M | M.fromnodeid == L.tonodeid } — 링크=차선 1:1, 교차로 내부 회전링크 포함 위상 완결
   (succ 목록은 직진성(진행방위 일치) 오름차순 정렬 — [0]=최직진)
 · from/to 가 A1 에 없으면 말단 좌표로 가상 노드(nodetype=200) — 같은 id 는 링크 간 공유
 · 신호접근 확장: C1 직결 링크에서 l/r 인접 체인(말단 30m 내·방위 정합)을 따라 동일 접근로 전 차선에 전파
 · 정지선 stopS: ①B2 530 속성 연계 ②말단 50m × 530 세그먼트 2D 교차 ③말단 6m 내 수선투영 ④length−3m
 · 교차로 그룹 = 접근로 말단 + 차량등 위치의 union-find 클러스터(반경 --radius) → 진행방위로 현시축 병합(|dot|>0.5)
 · 제한속도: A2 결측 → roadrank 기본표(안전속도 5030) + --speed-overrides(byRoadno/byRank kmh)
 · 차선폭: B2(530 제외)의 l/r_linkid 역참조 경계선과의 수선거리 중앙값, 클램프 [2.6,4.5], 실패=3.25

LGV2 포맷(전부 리틀엔디언, f32 좌표는 anchorE/N 상대):
  'LGV2' u32ver | u32×6 nLinks nNodes nPts nSucc nSignals nGroups | f64 anchorE anchorN
  ptStart i32[nLinks+1] | pts f32[nPts×3](E N z) | fromNode/toNode/leftIdx/rightIdx i32[nLinks]
  laneNo/linkType/roadRank/flags u8[nLinks] (flags: 1=경계절단 2=특수차로(laneno≥91) 4=신호접근)
  length/speedMs/laneWidth/stopS f32[nLinks] (stopS<0=없음) | groupIdx i32[nLinks] | axisIdx u8[nLinks]
  succStart i32[nLinks+1] | succ i32[nSucc]
  nodePos f32[nNodes×3] | nodeType u8[nNodes]
  groupPos f32[nGroups×3] | groupAxes u8 | groupFlags u8(1=경계·비활성) | groupOffset f32[0,1)
  sigPos f32[nSignals×3] | sigType u8 | sigLink i32(-1=미부착)

합격기준(report.json 의 criteria — 미달 시 원인 규명 후 진행):
 · 최대 약연결 성분 ≥95% 링크 · 비경계·고립(succ/l/r 모두 무) dead-end <2%(테이퍼=l/r 만 있는 정상 위상은 제외)
 · C1 차량등 링크 부착 ≥99%
 · 신호접근 stopS(교차/투영) ≥85% · 차선폭 복원 ≥90% · 그룹 축수 1~2 지배(>4축은 목록 검수)

실행: PYTHONIOENCODING=utf-8 <UAV env python> tools/v2_lanegraph_build.py --region seoul_gangnamgu --debug-geojson
"""
import os, json, math, argparse, struct, hashlib
import numpy as np
from pyproj import Transformer

P5186 = ("+proj=tmerc +lat_0=38 +lon_0=127 +k=1 +x_0=200000 +y_0=600000 "
         "+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HD = os.path.join(REPO, "tools", "nationwide", "hdmap")
_fwd = Transformer.from_crs("EPSG:4326", P5186, always_xy=True)
_inv = Transformer.from_crs(P5186, "EPSG:4326", always_xy=True)

F_BOUNDARY, F_SPECIAL, F_SIGNAL = 1, 2, 4
SPEED_BY_RANK = {1: 80, 2: 60, 3: 50, 4: 50, 5: 40, 6: 40, 7: 30, 8: 30, 9: 30}


def load(layer, region):
    return json.load(open(os.path.join(HD, layer, region + ".geojson"), encoding="utf-8"))["features"]


def norm_int(v, default=0):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def sid(v):
    """속성 id 문자열 정규화 — 빈값/None → ''."""
    s = str(v).strip() if v is not None else ""
    return "" if s in ("", "None", "null", "-") else s


def to_en_batch(coords):
    """[[lon,lat,z],…] → np.array (n,3) E,N,z — pyproj 벡터화."""
    a = np.asarray([(c[0], c[1], c[2] if len(c) > 2 else 0.0) for c in coords], dtype=np.float64)
    E, N = _fwd.transform(a[:, 0], a[:, 1])
    return np.column_stack([E, N, a[:, 2]])


def seg_dirs(pts):
    """말단 진행방위(마지막 유효 세그먼트 단위벡터, 2D)와 시점 방위."""
    def unit(a, b):
        d = b[:2] - a[:2]; n = math.hypot(d[0], d[1])
        return (d / n) if n > 1e-6 else np.array([1.0, 0.0])
    i = len(pts) - 2
    while i > 0 and math.hypot(*(pts[i + 1][:2] - pts[i][:2])) < 0.05:
        i -= 1
    j = 0
    while j < len(pts) - 2 and math.hypot(*(pts[j + 1][:2] - pts[j][:2])) < 0.05:
        j += 1
    return unit(pts[j], pts[j + 1]), unit(pts[i], pts[i + 1])


class DSU:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def point_seg_s(p, a, b):
    """점 p 를 세그먼트 ab 에 투영 — (수선거리, ab 상 비율 t)."""
    d = b - a; L2 = d[0] * d[0] + d[1] * d[1]
    if L2 < 1e-12:
        return math.hypot(*(p - a)), 0.0
    t = max(0.0, min(1.0, ((p - a) @ d) / L2))
    q = a + t * d
    return math.hypot(*(p - q)), t


def seg_intersect(a, b, c, d):
    """2D 세그먼트 ab×cd 교차 — 교차 시 ab 상 비율 t, 아니면 None."""
    r = b - a; s = d - c
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-12:
        return None
    t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / den
    u = ((c[0] - a[0]) * r[1] - (c[1] - a[1]) * r[0]) / den
    return t if (0.0 <= t <= 1.0 and 0.0 <= u <= 1.0) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="seoul_gangnamgu")
    ap.add_argument("--radius", type=float, default=35.0, help="교차로 클러스터 반경 m")
    ap.add_argument("--boundary-margin", type=float, default=200.0, help="구경계 폴리곤 경계절단 플래그 여유 m")
    ap.add_argument("--group-margin", type=float, default=250.0, help="경계 그룹 비활성 여유 m")
    ap.add_argument("--speed-overrides", default=None, help="JSON {byRoadno:{도로번호:kmh}, byRank:{rank:kmh}}")
    ap.add_argument("--debug-geojson", action="store_true")
    args = ap.parse_args()
    reg = args.region
    out_dir = os.path.join(REPO, "tools", "nationwide_v2", "lanegraph")
    os.makedirs(out_dir, exist_ok=True)
    rep = {"region": reg}

    def emit_nodata(reason):
        # 정밀도로지도 미수록 지역(vWorld WFS 가 HTTP200 + 0피처로 응답 — 실측 확인). A2 자체가 0이거나,
        # 구내 링크 0(이웃 구 bbox spill 만 잡힘)인 경우 무데이터로 통일 — bin 미생성(기존 spill bin 정리).
        rep["criteria"] = {"no_data": "SKIP"}
        rep["criteria_note"] = reason
        binp = os.path.join(out_dir, reg + ".bin")
        if os.path.exists(binp):
            os.remove(binp)
        with open(os.path.join(out_dir, reg + ".report.json"), "w", encoding="utf-8") as fo:
            json.dump(rep, fo, ensure_ascii=False, indent=1, default=int)
        print(f"[skip] {reg}: {reason} — bin 미생성", flush=True)

    ov_roadno, ov_rank = {}, {}
    if args.speed_overrides:
        ov = json.load(open(args.speed_overrides, encoding="utf-8"))
        ov_roadno = {str(k): float(v) for k, v in (ov.get("byRoadno") or {}).items()}
        ov_rank = {norm_int(k): float(v) for k, v in (ov.get("byRank") or {}).items()}

    # ── A2 링크 로드 ─────────────────────────────────────────────
    lid, lfrom, lto, l_l, l_r = [], [], [], [], []
    laneno, ltype, lrank, lroadno, lpts = [], [], [], [], []
    dup, multipart = 0, 0
    idmap = {}
    for f in load("lt_l_a2link", reg):
        g = f.get("geometry") or {}
        if g.get("type") != "MultiLineString":
            continue
        parts = [p for p in g["coordinates"] if len(p) >= 2]
        if not parts:
            continue
        if len(parts) > 1:
            multipart += 1
        coords = [c for p in parts for c in p]
        p = f.get("properties") or {}
        gid = sid(p.get("id"))
        if not gid or gid in idmap:
            dup += 1
            continue
        idmap[gid] = len(lid)
        lid.append(gid)
        lfrom.append(sid(p.get("fromnodeid"))); lto.append(sid(p.get("tonodeid")))
        l_l.append(sid(p.get("l_linkid"))); l_r.append(sid(p.get("r_linkid")))
        laneno.append(norm_int(p.get("laneno"))); ltype.append(norm_int(p.get("linktype")))
        lrank.append(norm_int(p.get("roadrank"))); lroadno.append(sid(p.get("roadno")))
        lpts.append(to_en_batch(coords))
    n = len(lid)
    print(f"[a2] 링크 {n} (중복/무id {dup}, 멀티파트 {multipart})", flush=True)
    rep["links"] = {"count": n, "dup_or_noid": dup, "multipart": multipart}
    if n == 0:  # A2 자체 0(예: 부산 남구/수영구, 인천 강화군, 경북 울릉군) — WFS HTTP200+0피처 실측 확인
        emit_nodata("A2 미수록(정밀도로지도 커버리지 공백)")
        return

    # 호장·방위
    cum = []
    for pts in lpts:
        d = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        cum.append(np.concatenate([[0.0], np.cumsum(d)]))
    length = np.array([c[-1] for c in cum], dtype=np.float64)
    tan0 = np.zeros((n, 2)); tan1 = np.zeros((n, 2))
    for i, pts in enumerate(lpts):
        tan0[i], tan1[i] = seg_dirs(pts)

    def sample(i, s):
        c = cum[i]; s = min(max(s, 0.0), c[-1])
        k = int(np.searchsorted(c, s, side="right")) - 1
        k = min(max(k, 0), len(c) - 2)
        t = 0.0 if c[k + 1] <= c[k] else (s - c[k]) / (c[k + 1] - c[k])
        return lpts[i][k] + t * (lpts[i][k + 1] - lpts[i][k])

    # ── A1 노드 + 가상 노드 ─────────────────────────────────────
    node_pos, node_type, nidx = [], [], {}
    for f in load("lt_p_a1node", reg):
        g = f.get("geometry") or {}
        if g.get("type") != "Point":
            continue
        p = f.get("properties") or {}
        gid = sid(p.get("id"))
        if not gid or gid in nidx:
            continue
        c = g["coordinates"]
        E, N = _fwd.transform(c[0], c[1])
        nidx[gid] = len(node_pos)
        node_pos.append((E, N, c[2] if len(c) > 2 else 0.0))
        node_type.append(min(255, norm_int(p.get("nodetype"))))
    n_a1 = len(node_pos)
    virt = 0
    fromN = np.full(n, -1, dtype=np.int64); toN = np.full(n, -1, dtype=np.int64)

    def resolve(node_id, at):
        nonlocal virt
        if node_id and node_id in nidx:
            return nidx[node_id]
        virt += 1
        key = node_id if node_id else f"__anon{virt}"
        if key in nidx:
            virt -= 1
            return nidx[key]
        nidx[key] = len(node_pos)
        node_pos.append(tuple(at)); node_type.append(200)
        return nidx[key]

    for i in range(n):
        fromN[i] = resolve(lfrom[i], lpts[i][0])
        toN[i] = resolve(lto[i], lpts[i][-1])
    print(f"[a1] 노드 {n_a1} + 가상 {len(node_pos) - n_a1} (미해결 id 포함 {virt})", flush=True)
    rep["nodes"] = {"a1": n_a1, "virtual": len(node_pos) - n_a1}

    # ── successor(직진성 정렬 CSR)·l/r 인접·경계 플래그 ────────
    by_from = {}
    for i in range(n):
        by_from.setdefault(lfrom[i], []).append(i)
    succ_lists, pred_cnt = [], np.zeros(n, dtype=np.int64)
    for i in range(n):
        cand = [j for j in by_from.get(lto[i], []) if j != i]
        cand.sort(key=lambda j: -float(tan1[i] @ tan0[j]))  # [0]=최직진
        succ_lists.append(cand)
        for j in cand:
            pred_cnt[j] += 1
    # 구간(section) 이음새 봉합 — tonodeid 는 있는데 대응 fromnodeid 링크가 없는 세대차 절단.
    # 같은 차선이 0.6m 내에서 방위 일치로 이어지면 합성 successor 추가(합성 없으면 간선 한복판 dead-end).
    start_grid = {}
    for i in range(n):
        s0 = lpts[i][0]
        start_grid.setdefault((int(s0[0] // 2), int(s0[1] // 2)), []).append(i)
    stitched = 0
    for i in range(n):
        if succ_lists[i]:
            continue
        e = lpts[i][-1]
        best, bestd = -1, 0.6
        cx, cy = int(e[0] // 2), int(e[1] // 2)
        for gx in range(cx - 1, cx + 2):
            for gy in range(cy - 1, cy + 2):
                for j in start_grid.get((gx, gy), []):
                    if j == i:
                        continue
                    d = math.hypot(e[0] - lpts[j][0][0], e[1] - lpts[j][0][1])
                    if d <= bestd and float(tan1[i] @ tan0[j]) > 0.85:
                        best, bestd = j, d
        if best >= 0:
            succ_lists[i].append(best); pred_cnt[best] += 1; stitched += 1
    print(f"[stitch] 이음새 봉합 {stitched}", flush=True)
    leftI = np.full(n, -1, dtype=np.int64); rightI = np.full(n, -1, dtype=np.int64)
    bad_l = bad_r = 0
    for i in range(n):
        if l_l[i]:
            leftI[i] = idmap.get(l_l[i], -1)
            bad_l += int(leftI[i] < 0)
        if l_r[i]:
            rightI[i] = idmap.get(l_r[i], -1)
            bad_r += int(rightI[i] < 0)
    pred_lists = [[] for _ in range(n)]
    for i in range(n):
        for j in succ_lists[i]:
            pred_lists[j].append(i)

    allpts = np.vstack(lpts)
    minE, minN = allpts[:, 0].min(), allpts[:, 1].min()
    maxE, maxN = allpts[:, 0].max(), allpts[:, 1].max()

    # 구 경계 = sgg.json rings([lon,lat]) 실폴리곤 — bbox 모서리는 오목 접경(서초 등)을 놓친다
    segs = None
    try:
        sgg = json.load(open(os.path.join(REPO, "tools", "nationwide", "sgg.json"), encoding="utf-8"))
        ent = next(e for e in sgg if e.get("name") == reg)
        As, Bs = [], []
        for ring in ent.get("rings") or []:
            rr = np.asarray(ring, dtype=np.float64)
            E, N = _fwd.transform(rr[:, 0], rr[:, 1])
            a = np.column_stack([E, N])
            As.append(a); Bs.append(np.vstack([a[1:], a[:1]]))
        if As:
            A = np.vstack(As); B = np.vstack(Bs)
            D = B - A
            segs = (A, D, np.maximum((D * D).sum(1), 1e-12))
    except StopIteration:
        print(f"[warn] sgg.json 에 {reg} 없음 — bbox 경계 폴백", flush=True)

    def edge_dist(p):
        if segs is None:
            return min(p[0] - minE, maxE - p[0], p[1] - minN, maxN - p[1])
        A, D, L2 = segs
        t = np.clip(((p[0] - A[:, 0]) * D[:, 0] + (p[1] - A[:, 1]) * D[:, 1]) / L2, 0.0, 1.0)
        return float(np.min(np.hypot(p[0] - (A[:, 0] + t * D[:, 0]), p[1] - (A[:, 1] + t * D[:, 1]))))

    # WFS bbox 페치는 구 폴리곤 밖 spill 링크를 포함 — 폴리곤 밖 = 전부 경계권(그곳 절단은 페치 경계)
    # ⚠️다중 링(섬 포함 해안 시군구) 경계는 자가교차가 흔해 raw union 이 GEOS TopologyException 으로 죽는다
    # → 링별 buffer(0) 정리 후 unary_union, 그래도 실패하면 링별 prepared 리스트로 any 판정(폴백).
    _prep, _preps = None, None
    from shapely.geometry import Point as _Pt
    if segs is not None:
        from shapely.geometry import Polygon as _Poly
        from shapely.ops import unary_union as _uu
        from shapely.prepared import prep as _prep_fn
        polys = []
        for a2 in As:
            if len(a2) < 3:
                continue
            g = _Poly(a2)
            if not g.is_valid:
                g = g.buffer(0)          # 자가교차 정리(멀티폴리곤이 될 수 있음)
            if not g.is_empty:
                polys.append(g)
        try:
            _prep = _prep_fn(_uu(polys))
        except Exception as ex:          # noqa: BLE001 — GEOS 실패 시 링 단위 판정으로 폴백
            print(f"[warn] {reg}: 경계 union 실패({type(ex).__name__}) — 링별 판정 폴백", flush=True)
            _preps = [_prep_fn(g) for g in polys]

    def inpoly(p):
        if _prep is None and _preps is None:
            return True
        pt = _Pt(float(p[0]), float(p[1]))
        if _prep is not None:
            return _prep.contains(pt)
        return any(g.contains(pt) for g in _preps)

    def is_bnd(p):
        return (not inpoly(p)) or edge_dist(p) < args.boundary_margin

    # 구내 링크 0 = 이웃 구 bbox spill 만 존재(부산 영도구=중구 영도대교, 인천 옹진군). 부산 16구
    # 전수 스캔·WFS HTTP200+0피처 실측으로 해당 시군구 정밀도로지도 미수록 확정 → 무데이터로 통일.
    inside_mask = [inpoly(lpts[i][len(lpts[i]) // 2]) for i in range(n)]
    n_inside = sum(inside_mask)
    if n_inside == 0:
        emit_nodata("구내 링크 0 — 이웃 구 spill 만 존재(정밀도로지도 미수록)")
        return

    flags = np.zeros(n, dtype=np.uint8)
    dead_end = sum(1 for s in succ_lists if not s)
    for i in range(n):
        if laneno[i] >= 91:
            flags[i] |= F_SPECIAL
        if (not succ_lists[i] and is_bnd(lpts[i][-1])) or (pred_cnt[i] == 0 and is_bnd(lpts[i][0])):
            flags[i] |= F_BOUNDARY
    nb_ids = [i for i in range(n) if not succ_lists[i] and not (flags[i] & F_BOUNDARY)]
    nonb_dead = len(nb_ids)
    # 차선 테이퍼(감소)는 successor 없이 l/r 로만 이어지는 정상 위상 — 진짜 결함은 succ 도 l/r 도 없는 고립
    nb_no_lr = sum(1 for i in nb_ids if int(leftI[i]) < 0 and int(rightI[i]) < 0)
    nb_diag = {"special_91p": 0, "short_lt30m": 0, "near_start_15m": 0, "no_lr_isolated": nb_no_lr,
               "ltype_hist": {}}
    for i in nb_ids:  # 진단: 내부 dead-end 의 성격(자연 종단 vs 미연결)
        if laneno[i] >= 91:
            nb_diag["special_91p"] += 1
        if length[i] < 30:
            nb_diag["short_lt30m"] += 1
        nb_diag["ltype_hist"][str(ltype[i])] = nb_diag["ltype_hist"].get(str(ltype[i]), 0) + 1
        e = lpts[i][-1]
        cx, cy = int(e[0] // 2), int(e[1] // 2)
        found = False
        for gx in range(cx - 8, cx + 9):
            for gy in range(cy - 8, cy + 9):
                for j in start_grid.get((gx, gy), []):
                    if j != i and math.hypot(e[0] - lpts[j][0][0], e[1] - lpts[j][0][1]) <= 15.0:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        nb_diag["near_start_15m"] += int(found)
    samples = []
    for i in range(n):
        if succ_lists[i] or (flags[i] & F_BOUNDARY) or len(samples) >= 20:
            continue
        lon, lat = _inv.transform(lpts[i][-1][0], lpts[i][-1][1])
        samples.append({"id": lid[i], "lat": round(lat, 6), "lon": round(lon, 6),
                        "rank": lrank[i], "ltype": ltype[i], "laneno": laneno[i]})
    rep["topology"] = {"dead_end": dead_end, "dead_end_nonboundary": nonb_dead, "stitched": stitched,
                       "deadend_diag": nb_diag, "deadend_samples": samples,
                       "boundary_flagged": int((flags & F_BOUNDARY).astype(bool).sum()),
                       "invalid_left_ref": bad_l, "invalid_right_ref": bad_r,
                       "special_lane_91p": int((flags & F_SPECIAL).astype(bool).sum())}

    # 약연결 성분(무방향: succ + l/r) — 판정은 구내 링크 한정(강변/시계 구는 spill 조각이 성분수를 부풀림)
    dsu = DSU(n)
    for i in range(n):
        for j in succ_lists[i]:
            dsu.union(i, j)
        if leftI[i] >= 0: dsu.union(i, int(leftI[i]))
        if rightI[i] >= 0: dsu.union(i, int(rightI[i]))
    comp, comp_in = {}, {}
    for i in range(n):
        r0 = dsu.find(i)
        comp[r0] = comp.get(r0, 0) + 1
        if inside_mask[i]:
            comp_in[r0] = comp_in.get(r0, 0) + 1
    wcc = max(comp.values()) / n if n else 0.0
    wcc_in = (max(comp_in.values()) / n_inside) if n_inside else 0.0
    rep["links"]["inside"] = n_inside
    rep["topology"]["wcc_max_frac"] = round(wcc, 4)
    rep["topology"]["wcc_inside_frac"] = round(wcc_in, 4)
    rep["topology"]["wcc_count"] = len(comp)
    print(f"[graph] succ 합계 {sum(map(len, succ_lists))}, dead-end {dead_end}(비경계 {nonb_dead}), "
          f"약연결 전체 {wcc:.1%}/{len(comp)}성분 · 구내 {wcc_in:.1%}({n_inside}링크)", flush=True)

    # ── 제한속도 ────────────────────────────────────────────────
    speed = np.zeros(n, dtype=np.float64)
    rank_dist = {}
    for i in range(n):
        kmh = ov_roadno.get(lroadno[i]) or ov_rank.get(lrank[i]) or SPEED_BY_RANK.get(lrank[i], 50)
        speed[i] = kmh / 3.6
        rank_dist[lrank[i]] = rank_dist.get(lrank[i], 0) + 1
    rep["speed"] = {"rank_dist": {str(k): v for k, v in sorted(rank_dist.items())},
                    "overrides": {"byRoadno": len(ov_roadno), "byRank": len(ov_rank)}}

    # ── C1 신호등 → 링크 부착(직결 97% + 기하 폴백) ────────────
    end_grid = {}
    for i in range(n):
        e = lpts[i][-1]
        end_grid.setdefault((int(e[0] // 40), int(e[1] // 40)), []).append(i)
    sig_pos, sig_type, sig_link, sig_method = [], [], [], []
    ped_or_etc = attach_id = attach_geo = unattached = 0
    for f in load("lt_p_c1trafficlight", reg):
        g = f.get("geometry") or {}
        p = f.get("properties") or {}
        if g.get("type") != "Point":
            continue
        ty = norm_int(p.get("type"))
        if ty not in (1, 2):  # 차량등만 로직 대상(11=보행등 등 제외)
            ped_or_etc += 1
            continue
        c = g["coordinates"]
        E, N = _fwd.transform(c[0], c[1])
        z = c[2] if len(c) > 2 else 0.0
        li = idmap.get(sid(p.get("linkid")), -1)
        meth = "id"
        if li < 0:  # 폴백: 말단 40m 내·전방 원뿔 65°·연장 중심선 최근접
            best, bestd = -1, 1e9
            cx, cy = int(E // 40), int(N // 40)
            for gx in range(cx - 1, cx + 2):
                for gy in range(cy - 1, cy + 2):
                    for j in end_grid.get((gx, gy), []):
                        e = lpts[j][-1]
                        v = np.array([E - e[0], N - e[1]])
                        d = math.hypot(v[0], v[1])
                        if d > 40:
                            continue
                        if d > 1e-6 and (v / d) @ tan1[j] < 0.42:
                            continue
                        lat = abs(v[0] * tan1[j][1] - v[1] * tan1[j][0])  # 연장선 수선거리
                        if lat < bestd:
                            best, bestd = j, lat
            li, meth = best, "geo"
        if li >= 0:
            attach_id += meth == "id"; attach_geo += meth == "geo"
        else:
            unattached += 1; meth = "none"
        sig_pos.append((E, N, z)); sig_type.append(ty); sig_link.append(li); sig_method.append(meth)
    n_sig = len(sig_pos)
    att = attach_id + attach_geo
    rep["signals"] = {"vehicle": n_sig, "ped_or_etc_skipped": ped_or_etc, "attached_by_id": attach_id,
                      "attached_by_geo": attach_geo, "unattached": unattached,
                      "attach_frac": round(att / n_sig, 4) if n_sig else 1.0}
    print(f"[c1] 차량등 {n_sig}: 직결 {attach_id} + 기하 {attach_geo} + 미부착 {unattached} "
          f"(보행등 등 제외 {ped_or_etc})", flush=True)

    # ── 신호접근 확장(l/r 체인 → 동일 접근로 전 차선) ──────────
    direct = sorted({li for li in sig_link if li >= 0})
    approach = set(direct)
    for L in direct:
        for step in (leftI, rightI):
            cur, visited = L, {L}
            while True:
                m = int(step[cur])
                if m < 0 or m in visited:
                    break
                visited.add(m)
                if math.hypot(*(lpts[m][-1][:2] - lpts[L][-1][:2])) > 30 or tan1[m] @ tan1[L] < 0.7:
                    break
                approach.add(m); cur = m
    for i in approach:
        flags[i] |= F_SIGNAL
    rep["signals"]["approach_direct"] = len(direct)
    rep["signals"]["approach_expanded"] = len(approach)
    print(f"[approach] 직결 {len(direct)} → 확장 {len(approach)} 차선", flush=True)

    # ── B2: 정지선(530) + 차선 경계(폭) ─────────────────────────
    stops, bnd_left, bnd_right = [], {}, {}
    kind_dist = {}
    for f in load("lt_l_b2surfacelinemark", reg):
        g = f.get("geometry") or {}
        if g.get("type") != "MultiLineString":
            continue
        p = f.get("properties") or {}
        kind = norm_int(p.get("kind"))
        kind_dist[kind] = kind_dist.get(kind, 0) + 1
        for part in g["coordinates"]:
            if len(part) < 2:
                continue
            en = to_en_batch(part)
            if kind == 530:
                stops.append((en, sid(p.get("l_linkid")), sid(p.get("r_linkid"))))
            else:
                ll, rr = sid(p.get("l_linkid")), sid(p.get("r_linkid"))
                if rr in idmap:  # 이 선의 우측 링크 → 그 링크의 좌경계
                    bnd_left.setdefault(idmap[rr], []).append(en)
                if ll in idmap:
                    bnd_right.setdefault(idmap[ll], []).append(en)
    rep["b2"] = {"kind_dist": {str(k): v for k, v in sorted(kind_dist.items())}, "stoplines": len(stops)}

    # 정지선 stopS
    stopS = np.full(n, -1.0, dtype=np.float64)
    stop_grid = {}
    for si, (en, _, _) in enumerate(stops):
        for k in range(len(en) - 1):
            a, b = en[k], en[k + 1]
            for gx in range(int(min(a[0], b[0]) // 20), int(max(a[0], b[0]) // 20) + 1):
                for gy in range(int(min(a[1], b[1]) // 20), int(max(a[1], b[1]) // 20) + 1):
                    stop_grid.setdefault((gx, gy), []).append((si, k))
    meth_of = {}
    REAL_METH = ("attr", "x", "proj", "inh", "perp", "fan")
    m_attr = m_x = m_proj = m_inh = m_perp = m_fan = m_fb = 0
    for si, (en, sl, sr) in enumerate(stops):  # ①속성 연계(희귀 — 강남 4/2723)
        for key in (sl, sr):
            j = idmap.get(key, -1)
            if j >= 0 and stopS[j] < 0:
                mid = en[len(en) // 2]
                best = None
                for k in range(len(lpts[j]) - 1):
                    d, t = point_seg_s(mid[:2], lpts[j][k][:2], lpts[j][k + 1][:2])
                    if best is None or d < best[0]:
                        best = (d, cum[j][k] + t * (cum[j][k + 1] - cum[j][k]))
                if best and best[0] < 8:
                    stopS[j] = best[1]; meth_of[j] = "attr"; m_attr += 1

    def stop_cells(a, b):
        for gx in range(int(min(a[0], b[0]) // 20) - 1, int(max(a[0], b[0]) // 20) + 2):
            for gy in range(int(min(a[1], b[1]) // 20) - 1, int(max(a[1], b[1]) // 20) + 2):
                yield from stop_grid.get((gx, gy), [])

    pend = []
    for j in sorted(approach):
        if stopS[j] >= 0:
            continue
        s_tail = max(0.0, length[j] - 50.0)
        k0 = max(0, int(np.searchsorted(cum[j], s_tail, side="right")) - 1)
        hit = -1.0
        seen = set()
        for k in range(k0, len(lpts[j]) - 1):  # ②말단 50m × 정지선 교차
            a, b = lpts[j][k][:2], lpts[j][k + 1][:2]
            for si, sk in stop_cells(a, b):
                if (si, sk) in seen:
                    continue
                seen.add((si, sk))
                t = seg_intersect(a, b, stops[si][0][sk][:2], stops[si][0][sk + 1][:2])
                if t is not None:
                    hit = max(hit, cum[j][k] + t * (cum[j][k + 1] - cum[j][k]))
        if hit < 0:  # ②b 말단 연장 12m — 링크가 정지선 직전에 끝나는 규약 대응
            a = lpts[j][-1][:2]; b = a + tan1[j] * 12.0
            for si, sk in stop_cells(a, b):
                if seg_intersect(a, b, stops[si][0][sk][:2], stops[si][0][sk + 1][:2]) is not None:
                    hit = max(0.0, length[j] - 0.5)
                    break
        if hit >= 0:
            stopS[j] = hit; meth_of[j] = "x"; m_x += 1
            continue
        e = lpts[j][-1][:2]  # ③말단 6m 수선투영
        best = None
        for si, sk in stop_cells(e, e):
            d, _ = point_seg_s(e, stops[si][0][sk][:2], stops[si][0][sk + 1][:2])
            if best is None or d < best:
                best = d
        if best is not None and best <= 6.0:
            stopS[j] = max(0.0, length[j] - best); meth_of[j] = "proj"; m_proj += 1
        else:
            pend.append(j)
    for _ in range(3):  # ③b 인접 차선 정지점 상속(정지선이 일부 차선만 걸친 데이터 대응)
        rest = []
        for j in pend:
            got = False
            for nb in (int(leftI[j]), int(rightI[j])):
                if nb >= 0 and stopS[nb] >= 0 and meth_of.get(nb) in REAL_METH:
                    P = sample(nb, float(stopS[nb]))[:2]
                    k0 = max(0, int(np.searchsorted(cum[j], max(0.0, length[j] - 50.0), side="right")) - 1)
                    best = None
                    for k in range(k0, len(lpts[j]) - 1):
                        d, t = point_seg_s(P, lpts[j][k][:2], lpts[j][k + 1][:2])
                        if best is None or d < best[0]:
                            best = (d, cum[j][k] + t * (cum[j][k + 1] - cum[j][k]))
                    if best and best[0] <= 8.0:
                        stopS[j] = best[1]; meth_of[j] = "inh"; m_inh += 1; got = True
                        break
            if not got:
                rest.append(j)
        pend = rest
        if not pend:
            break
    rest = []  # ③d 수직 정지선 투영 — 정지선이 중심선을 안 가로지르는 포켓/짧은 도색(말단 25m·수직성 판별)
    for j in pend:
        e = lpts[j][-1][:2]
        best = None
        cx, cy = int(e[0] // 20), int(e[1] // 20)
        for gx in range(cx - 2, cx + 3):
            for gy in range(cy - 2, cy + 3):
                for si, sk in stop_grid.get((gx, gy), []):
                    sa, sb = stops[si][0][sk][:2], stops[si][0][sk + 1][:2]
                    sd = sb - sa
                    sn = math.hypot(sd[0], sd[1])
                    if sn < 1e-6 or abs(float((sd / sn) @ tan1[j])) > 0.5:
                        continue  # 차선과 평행(차선경계류) 배제 — 정지선은 차선에 수직
                    M = (sa + sb) * 0.5
                    if math.hypot(M[0] - e[0], M[1] - e[1]) > 25.0:
                        continue
                    k0 = max(0, int(np.searchsorted(cum[j], max(0.0, length[j] - 40.0), side="right")) - 1)
                    for k in range(k0, len(lpts[j]) - 1):
                        d, t = point_seg_s(M, lpts[j][k][:2], lpts[j][k + 1][:2])
                        if d <= 20.0 and (best is None or d < best[0]):
                            best = (d, cum[j][k] + t * (cum[j][k + 1] - cum[j][k]))
        if best:
            stopS[j] = best[1]; meth_of[j] = "perp"; m_perp += 1
        else:
            rest.append(j)
    pend = rest
    for _ in range(2):  # ③c 같은 접근 부채꼴 상속 — l/r 미연결 포켓/버스차로(말단 18m·평행 차선의 정지점 투영)
        rest = []
        for j in pend:
            e = lpts[j][-1]
            best = None
            cx, cy = int(e[0] // 40), int(e[1] // 40)
            for gx in range(cx - 1, cx + 2):
                for gy in range(cy - 1, cy + 2):
                    for k2 in end_grid.get((gx, gy), []):
                        if k2 == j or meth_of.get(k2) not in REAL_METH:
                            continue
                        if math.hypot(e[0] - lpts[k2][-1][0], e[1] - lpts[k2][-1][1]) > 18.0 \
                           or float(tan1[j] @ tan1[k2]) < 0.7:
                            continue
                        P = sample(k2, float(stopS[k2]))[:2]
                        k0 = max(0, int(np.searchsorted(cum[j], max(0.0, length[j] - 50.0), side="right")) - 1)
                        for k in range(k0, len(lpts[j]) - 1):
                            d, t = point_seg_s(P, lpts[j][k][:2], lpts[j][k + 1][:2])
                            if d <= 12.0 and (best is None or d < best[0]):
                                best = (d, cum[j][k] + t * (cum[j][k + 1] - cum[j][k]))
            if best:
                stopS[j] = best[1]; meth_of[j] = "fan"; m_fan += 1
            else:
                rest.append(j)
        pend = rest
        if not pend:
            break
    for j in pend:  # ④최종 폴백(A2 규약상 접근링크 말단≈정지선 — 구조적 근사)
        stopS[j] = max(length[j] * 0.5, length[j] - 3.0); meth_of[j] = "fb"; m_fb += 1
    hist530 = {"<=6": 0, "<=12": 0, "<=20": 0, "<=40": 0, ">40": 0}  # 진단: 폴백(구내)의 최근접 530 거리
    for j in pend:
        if not inpoly(lpts[j][-1]):
            continue
        e = lpts[j][-1][:2]
        best = None
        cx, cy = int(e[0] // 20), int(e[1] // 20)
        for gx in range(cx - 2, cx + 3):
            for gy in range(cy - 2, cy + 3):
                for si, sk in stop_grid.get((gx, gy), []):
                    d, _ = point_seg_s(e, stops[si][0][sk][:2], stops[si][0][sk + 1][:2])
                    if best is None or d < best:
                        best = d
        hist530[">40" if best is None or best > 40 else
                "<=6" if best <= 6 else "<=12" if best <= 12 else
                "<=20" if best <= 20 else "<=40"] += 1
    n_app = len(approach)
    real = m_attr + m_x + m_proj + m_inh + m_perp + m_fan
    n_app_in = sum(1 for j in approach if inpoly(lpts[j][-1]))
    real_in = sum(1 for j in approach if meth_of.get(j) in REAL_METH and inpoly(lpts[j][-1]))
    rep["stopline"] = {"by_attr": m_attr, "by_intersect": m_x, "by_project": m_proj, "by_inherit": m_inh,
                       "by_perp": m_perp, "by_fan": m_fan, "fallback": m_fb,
                       "real_frac_of_approach": round(real / n_app, 4) if n_app else 1.0,
                       "approach_inside": n_app_in,
                       "real_frac_inside": round(real_in / n_app_in, 4) if n_app_in else 1.0,
                       "fallback_inside_nearest530_hist": hist530}
    print(f"[stop] 속성 {m_attr} + 교차 {m_x} + 투영 {m_proj} + 상속 {m_inh} + 수직 {m_perp} + 부채꼴 {m_fan}"
          f" + 폴백 {m_fb} / 접근 {n_app} (구내 real {real_in}/{n_app_in})", flush=True)

    # 차선폭 — ①B2 경계 직접 복원 → ②l/r·succ/pred 이웃 전파(중앙값)
    width = np.full(n, 3.25, dtype=np.float64)
    known = np.zeros(n, dtype=bool)
    w_ok = 0
    for i in range(n):
        L = [np.asarray(x[:, :2]) for x in bnd_left.get(i, [])]
        R = [np.asarray(x[:, :2]) for x in bnd_right.get(i, [])]
        if not L and not R:
            continue
        ws = []
        for s in np.linspace(5.0, max(5.0, length[i] - 5.0), num=min(10, max(2, int(length[i] // 10)))):
            p = sample(i, float(s))[:2]
            dl = min((float(np.min(np.hypot(b[:, 0] - p[0], b[:, 1] - p[1]))) for b in L), default=None)
            dr = min((float(np.min(np.hypot(b[:, 0] - p[0], b[:, 1] - p[1]))) for b in R), default=None)
            if dl is not None and dl > 4.0: dl = None
            if dr is not None and dr > 4.0: dr = None
            if dl is not None and dr is not None: ws.append(dl + dr)
            elif dl is not None: ws.append(2 * dl)
            elif dr is not None: ws.append(2 * dr)
        if ws:
            width[i] = min(4.5, max(2.6, float(np.median(ws)))); known[i] = True; w_ok += 1
    w_prop = 0
    for _ in range(3):
        changed = 0
        for i in range(n):
            if known[i]:
                continue
            cand = [width[nb] for nb in (int(leftI[i]), int(rightI[i])) if nb >= 0 and known[nb]]
            if not cand:
                cand = [width[j2] for j2 in succ_lists[i] if known[j2]]
                cand += [width[j2] for j2 in pred_lists[i] if known[j2]]
            if cand:
                width[i] = float(np.median(cand)); known[i] = True; w_prop += 1; changed += 1
        if not changed:
            break
    w_all = w_ok + w_prop
    rep["width"] = {"direct": w_ok, "propagated": w_prop,
                    "frac": round(w_all / n, 4) if n else 1.0}
    print(f"[width] 직접 {w_ok} + 전파 {w_prop} = {w_all}/{n} ({w_all / n:.1%})", flush=True)

    # ── 교차로 그룹(union-find) + 현시축 ────────────────────────
    pts_cl = [(lpts[i][-1][0], lpts[i][-1][1], "L", i) for i in sorted(approach)]
    pts_cl += [(sig_pos[k][0], sig_pos[k][1], "S", k) for k in range(n_sig) if sig_link[k] >= 0]
    m = len(pts_cl)
    dsu2 = DSU(m)
    # 신호↔자기 링크 말단 강제 union — 속성 연계가 곧 정답. 광폭 교차로에서 근측 말단과 원측
    # 신호를 잇는 브리지가 되어 십자 교차로가 하나의 그룹으로 병합된다(축0/분열 그룹 해소).
    lpt_of = {pts_cl[a][3]: a for a in range(m) if pts_cl[a][2] == "L"}
    for a in range(m):
        if pts_cl[a][2] == "S":
            j = lpt_of.get(sig_link[pts_cl[a][3]])
            if j is not None:
                dsu2.union(a, j)
    cell = args.radius
    grid2 = {}
    for a, (x, y, _, _) in enumerate(pts_cl):
        grid2.setdefault((int(x // cell), int(y // cell)), []).append(a)
    r2 = args.radius * args.radius
    for a, (x, y, _, _) in enumerate(pts_cl):
        cx, cy = int(x // cell), int(y // cell)
        for gx in range(cx - 1, cx + 2):
            for gy in range(cy - 1, cy + 2):
                for b in grid2.get((gx, gy), []):
                    if b <= a:
                        continue
                    dx, dy = x - pts_cl[b][0], y - pts_cl[b][1]
                    if dx * dx + dy * dy <= r2:
                        dsu2.union(a, b)
    clusters = {}
    for a in range(m):
        clusters.setdefault(dsu2.find(a), []).append(a)
    groupIdx = np.full(n, -1, dtype=np.int64); axisIdx = np.zeros(n, dtype=np.uint8)
    g_pos, g_axes, g_flags, g_off = [], [], [], []
    axes_hist = {}
    big_groups = []
    for root, members in sorted(clusters.items()):
        gi = len(g_pos)
        xs = [pts_cl[a][0] for a in members]; ys = [pts_cl[a][1] for a in members]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        app_links = [pts_cl[a][3] for a in members if pts_cl[a][2] == "L"]
        axes = []  # 대표 방위(부호 무시 병합 |dot|>0.5)
        for i in app_links:
            b = tan1[i]
            hitax = -1
            for ax, v in enumerate(axes):
                if abs(float(b @ v)) > 0.5:
                    hitax = ax
                    break
            if hitax < 0:
                axes.append(b.copy()); hitax = len(axes) - 1
            groupIdx[i] = gi; axisIdx[i] = min(255, hitax)
        zmed = float(np.median([lpts[i][-1][2] for i in app_links])) if app_links else 0.0
        bnd = edge_dist((cx, cy)) < args.group_margin or any(flags[i] & F_BOUNDARY for i in app_links)
        g_pos.append((cx, cy, zmed)); g_axes.append(max(1, len(axes))); g_flags.append(1 if bnd else 0)
        h = int(hashlib.md5(f"{int(cx)}_{int(cy)}".encode()).hexdigest()[:8], 16)
        g_off.append((h % 1000) / 1000.0)
        axes_hist[len(axes)] = axes_hist.get(len(axes), 0) + 1
        if len(axes) > 4:
            lon, lat = _inv.transform(cx, cy)
            big_groups.append({"group": gi, "axes": len(axes), "n_approach": len(app_links),
                               "lat": round(lat, 6), "lon": round(lon, 6)})
    nG = len(g_pos)
    rep["groups"] = {"count": nG, "axes_hist": {str(k): v for k, v in sorted(axes_hist.items())},
                     "boundary_disabled": int(sum(g_flags)), "over4axes": big_groups[:50]}
    print(f"[group] 교차로 그룹 {nG} (축 분포 {dict(sorted(axes_hist.items()))}, 경계 비활성 {sum(g_flags)})",
          flush=True)

    # ── 합격기준 판정 ───────────────────────────────────────────
    # 구내 링크 0 = 그 시군구 자체엔 정밀도로지도가 없고 bbox spill 만 잡힌 경우(예: 옹진군 — 폴리곤은
    # 백령도까지 뻗지만 링크는 전부 인천 본토). 구내 기준 판정이 무의미하므로 N/A 로 표기한다.
    crit = {
        "wcc>=0.95": (wcc_in >= 0.95) if n_inside else None,
        "isolated_deadend<2%": (nb_no_lr / n if n else 0) < 0.02,
        "signal_attach>=99%": (att / n_sig if n_sig else 1) >= 0.99,
        "stopS_real>=85%": ((real_in / n_app_in) >= 0.85) if n_app_in else None,
        "width>=90%": (w_all / n if n else 1) >= 0.90,
    }
    def verdict(v):
        return "N/A" if v is None else ("PASS" if v else "FAIL")

    rep["criteria"] = {k: verdict(v) for k, v in crit.items()}
    rep["criteria_note"] = ("구내 링크 0 — bbox spill 만 존재(해당 시군구에 정밀도로지도 미수록)"
                            if not n_inside else "")
    print("[criteria] " + "  ".join(f"{k}:{verdict(v)}" for k, v in crit.items()), flush=True)

    # ── LGV2 바이너리 ───────────────────────────────────────────
    anchorE, anchorN = (minE + maxE) * 0.5, (minN + maxN) * 0.5
    ptStart = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        ptStart[i + 1] = ptStart[i] + len(lpts[i])
    nPts = int(ptStart[-1])
    pts_all = np.vstack(lpts) - np.array([anchorE, anchorN, 0.0])
    succStart = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        succStart[i + 1] = succStart[i] + len(succ_lists[i])
    succ_flat = np.array([j for s in succ_lists for j in s], dtype=np.int32)
    npos = np.asarray(node_pos, dtype=np.float64) - np.array([anchorE, anchorN, 0.0])
    gpos = (np.asarray(g_pos, dtype=np.float64) - np.array([anchorE, anchorN, 0.0])) if nG else np.zeros((0, 3))
    spos = (np.asarray(sig_pos, dtype=np.float64) - np.array([anchorE, anchorN, 0.0])) if n_sig else np.zeros((0, 3))

    bin_path = os.path.join(out_dir, reg + ".bin")
    with open(bin_path, "wb") as fo:
        fo.write(b"LGV2"); fo.write(struct.pack("<I", 1))
        fo.write(struct.pack("<6I", n, len(node_pos), nPts, len(succ_flat), n_sig, nG))
        fo.write(struct.pack("<2d", anchorE, anchorN))
        fo.write(ptStart.astype("<i4").tobytes())
        fo.write(pts_all.astype("<f4").tobytes())
        for arr in (fromN, toN, leftI, rightI):
            fo.write(arr.astype("<i4").tobytes())
        fo.write(np.clip(np.asarray(laneno), 0, 255).astype("<u1").tobytes())
        fo.write(np.clip(np.asarray(ltype), 0, 255).astype("<u1").tobytes())
        fo.write(np.clip(np.asarray(lrank), 0, 255).astype("<u1").tobytes())
        fo.write(flags.astype("<u1").tobytes())
        for arr in (length, speed, width, stopS):
            fo.write(np.asarray(arr).astype("<f4").tobytes())
        fo.write(groupIdx.astype("<i4").tobytes())
        fo.write(axisIdx.astype("<u1").tobytes())
        fo.write(succStart.astype("<i4").tobytes())
        fo.write(succ_flat.astype("<i4").tobytes())
        fo.write(npos.astype("<f4").tobytes())
        fo.write(np.asarray(node_type).astype("<u1").tobytes())
        fo.write(gpos.astype("<f4").tobytes())
        fo.write(np.asarray(g_axes, dtype=np.uint8).tobytes())
        fo.write(np.asarray(g_flags, dtype=np.uint8).tobytes())
        fo.write(np.asarray(g_off).astype("<f4").tobytes())
        fo.write(spos.astype("<f4").tobytes())
        fo.write(np.asarray(sig_type, dtype=np.uint8).tobytes())
        fo.write(np.asarray(sig_link, dtype=np.int64).astype("<i4").tobytes())
    rep["bin"] = {"path": os.path.relpath(bin_path, REPO).replace("\\", "/"),
                  "bytes": os.path.getsize(bin_path),
                  "anchorE": round(anchorE, 3), "anchorN": round(anchorN, 3), "nPts": nPts}
    print(f"[bin] {bin_path} ({os.path.getsize(bin_path) / 1e6:.1f} MB)", flush=True)

    with open(os.path.join(out_dir, reg + ".report.json"), "w", encoding="utf-8") as fo:
        json.dump(rep, fo, ensure_ascii=False, indent=1, default=int)  # Windows numpy int32 대응

    # ── 디버그 geojson(QGIS) ────────────────────────────────────
    if args.debug_geojson:
        feats = []
        for j in sorted(approach):
            if stopS[j] < 0:
                continue
            p = sample(j, float(stopS[j]))
            lon, lat = _inv.transform(p[0], p[1])
            feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                          "properties": {"what": "stopS", "link": lid[j], "s": round(float(stopS[j]), 1)}})
        for gi in range(nG):
            lon, lat = _inv.transform(g_pos[gi][0], g_pos[gi][1])
            feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                          "properties": {"what": "group", "axes": int(g_axes[gi]), "disabled": int(g_flags[gi])}})
        for k in range(n_sig):
            if sig_link[k] < 0 or sig_method[k] != "geo":
                continue
            e = lpts[sig_link[k]][-1]
            a = _inv.transform(sig_pos[k][0], sig_pos[k][1]); b = _inv.transform(e[0], e[1])
            feats.append({"type": "Feature",
                          "geometry": {"type": "LineString", "coordinates": [[a[0], a[1]], [b[0], b[1]]]},
                          "properties": {"what": "sig_geo_attach"}})
        with open(os.path.join(out_dir, reg + ".debug.geojson"), "w", encoding="utf-8") as fo:
            json.dump({"type": "FeatureCollection", "features": feats}, fo, ensure_ascii=False)
        print(f"[debug] {reg}.debug.geojson ({len(feats)} 피처)", flush=True)


if __name__ == "__main__":
    main()
