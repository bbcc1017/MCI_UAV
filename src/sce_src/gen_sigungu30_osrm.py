# -*- coding: utf-8 -*-
"""시군구 250개 × 30점 학습풀 — 도로 접근 가능 영역 균일 샘플링 (v18 E5).

기존 random4(시군구당 4점)로는 지역특화 모델을 학습하기에 표본이 너무 적다. 규칙(2파라미터)은
k≈3 에서 이미 포화하지만(v18 E4 학습곡선, 점근 46.8%·90% 도달 k=3.1) 924k 파라미터 정책은
전혀 다른 표본복잡도를 가진다. 여기서는 시군구당 30점을 만들어 **모델 자신의 학습곡선**을
{4, 8, 16, 30} 중첩 부분집합으로 측정할 수 있게 한다.

## 기존 생성기와 다른 점 — OSRM 스냅 게이트

``gen_eval_holdout_osrm.py`` 는 폴리곤 내부 균일점을 그대로 쓴다. 그런데 OSRM 은 좌표를
도로망에 **스냅**하고(실측 중위 176m·q90 1.2km, 산간·도서에서 큼) 그 사실이 route JSON 에만
남는다. 결과적으로 **AMB 는 스냅점 기준, UAV 는 원좌표 기준**으로 시간이 계산돼 UAV 이득이
과소추정된다. 이 생성기는 `/nearest` 로 스냅 거리를 미리 재고 **500m 초과 좌표를 기각**해
그 불일치를 원천에서 없앤다.

⚠️ **표기 주의**: 스냅 기각 때문에 표본은 더 이상 행정구역 면적에 균일하지 않다. 도로에서 먼
산간·도서 영역이 제외되므로 정확한 표현은
`uniform over road-accessible area (OSRM snap ≤ 500 m)` 다. 실측 수용률은 전체 73.5%,
도심 92~95%, 대면적 산간 50%(홍천군) 이며 거부된 영역은 구급차 접근이 비현실적인 지형이다.

## 기록하는 것

좌표마다 `snap_m`(스냅 거리)·`snap_lat/lon`·`snap_name`·`attempt`(그 점을 얻기까지의 시도 수),
시군구마다 `n_inside`(폴리곤 내부로 뽑힌 총 수)·`n_rej_snap`·`n_rej_near_eval`·`n_rej_osrm`·
`accept_rate`·스냅 거리 분위수·`excl_km`(적용된 평가셋 이격 반경)을 남긴다.

## 좌표 배제 정책

사용자 결정(2026-09-01): 새 모델을 새로 학습하므로 **기존 학습좌표와의 중복은 허용**한다.
다만 **평가셋과는 겹치면 안 된다** — 대표점250 · 외부250 · v15_blind250(미개봉). 기본 이격
1.0km, 소형 시군구에서 30점을 못 채우면 0.5→0.25→0.125→0 으로 완화하고 `excl_km` 에 기록한다.

## 2단 구성

    points   폴리곤 균일 샘플 + 스냅 게이트 → 좌표 원장 확정 (OSRM /nearest 만 사용, 빠름)
    build    확정된 좌표로 시나리오 생성 (gen_scenario_for_region, 재개 가능)

사용
----
    python src/sce_src/gen_sigungu30_osrm.py points --workers 32
    python src/sce_src/gen_sigungu30_osrm.py build  --workers 32
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
import urllib.request
from contextlib import redirect_stdout
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import shapefile
from pyproj import Transformer

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[1]
for _d in (str(THIS), str(REPO / "src/rl_src")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

OSRM_URL = os.environ.get("MCI_OSRM_URL", "http://127.0.0.1:5000")
EXP_PREFIX = "sigungu30/osrm"
SHP = REPO / "scenarios/sig.shp"
OUT_POINTS = REPO / "scenarios/manifests/sigungu30_points.json"
OUT_STATS = REPO / "scenarios/manifests/sigungu30_sampling_stats.json"
OUT_MANIFEST = REPO / "scenarios/manifests/sigungu30_manifest.json"
REF_POINTS = REPO / "scenarios/manifests/sigungu_osrm_train1000_random4_points.json"

# 학습 정본과 완전 동일한 시나리오 파라미터 (혼용 방지)
PARAMS = dict(incident_size=100, amb_count=30, uav_count=26, amb_velocity=50,
              uav_velocity=200, amb_handover_time=5.0, uav_handover_time=10.0,
              total_samples=1000, fixed_hos_num=47, uav_num=26)

SNAP_MAX_M = 500.0
EXCL_LADDER = (1.0, 0.5, 0.25, 0.125, 0.0)


# ------------------------------------------------------------------ 기하
def _rings(shape):
    parts = list(shape.parts) + [len(shape.points)]
    return [shape.points[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]


def _ring_contains(pt, ring):
    x, y = pt
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _contains(pt, rings):
    """짝수-홀수 규칙 — 섬은 포함, 구멍은 제외 (ring 방향 무관)."""
    inside = False
    for r in rings:
        if _ring_contains(pt, r):
            inside = not inside
    return inside


def hav_km(a, b):
    R = 6371.0088
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    return 2 * R * math.asin(math.sqrt(
        math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2))


def osrm_nearest(lat, lon, timeout=6.0):
    """(스냅거리 m, 스냅 lat, 스냅 lon, 도로명) 또는 None."""
    url = f"{OSRM_URL}/nearest/v1/driving/{lon},{lat}?number=1"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            d = json.load(r)
        if d.get("code") != "Ok" or not d.get("waypoints"):
            return None
        w = d["waypoints"][0]
        return (float(w["distance"]), float(w["location"][1]), float(w["location"][0]),
                w.get("name", "") or "")
    except Exception:
        return None


# ------------------------------------------------------- Phase 1: 좌표 확정
def _point_worker(task):
    (sigcd, name, sido, rings, bbox, seed, n_pts, excl, budget) = task
    tr = Transformer.from_crs(5179, 4326, always_xy=True)
    rng = np.random.default_rng(seed)
    xmin, ymin, xmax, ymax = bbox
    stat = dict(n_bbox=0, n_inside=0, n_rej_snap=0, n_rej_near_eval=0, n_rej_osrm=0)
    snaps, pts = [], []
    for excl_km in EXCL_LADDER:
        if len(pts) >= n_pts:
            break
        tried = 0
        while len(pts) < n_pts and tried < budget:
            tried += 1
            stat["n_bbox"] += 1
            x = rng.uniform(xmin, xmax)
            y = rng.uniform(ymin, ymax)
            if not _contains((x, y), rings):
                continue
            stat["n_inside"] += 1
            lon, lat = tr.transform(x, y)
            lat, lon = round(lat, 6), round(lon, 6)
            near = osrm_nearest(lat, lon)
            if near is None:
                stat["n_rej_osrm"] += 1
                continue
            sm, slat, slon, sname = near
            if sm > SNAP_MAX_M:
                stat["n_rej_snap"] += 1
                snaps.append(sm)
                continue
            if excl_km > 0 and any(hav_km((lat, lon), e) < excl_km for e in excl):
                stat["n_rej_near_eval"] += 1
                continue
            snaps.append(sm)
            pts.append(dict(sigcd=sigcd, name=name, sido=sido, pidx=len(pts),
                            lat=lat, lon=lon, snap_m=round(sm, 2),
                            snap_lat=round(slat, 6), snap_lon=round(slon, 6),
                            snap_name=sname, attempt=tried, excl_km=excl_km))
        stat["excl_km_final"] = excl_km
    a = np.asarray(snaps, float)
    acc = stat["n_inside"] and len(pts) / stat["n_inside"]
    stat.update(n_accepted=len(pts), accept_rate_of_inside=round(float(acc or 0), 4),
                snap_median_m=round(float(np.median(a)), 1) if a.size else None,
                snap_q90_m=round(float(np.percentile(a, 90)), 1) if a.size else None,
                snap_max_accepted_m=round(max((p["snap_m"] for p in pts), default=0), 1))
    return dict(sigcd=sigcd, name=name, points=pts, stat=stat,
                ok=len(pts) == n_pts)


def points_main(a) -> None:
    ref = json.load(open(REF_POINTS, encoding="utf-8"))
    meta = {}
    for k, v in ref.items():
        meta[str(v["sigcd"]) if "sigcd" in v else k.rsplit("_", 2)[-2]] = (v["name"], v["sido"])

    # 평가셋 좌표 (배제 대상) — 대표점250 · 외부250 · v15_blind250
    excl_all = []
    import re
    for mp in ("sigungu_osrm_eval250_representative_manifest.json",
               "distill_external_test250_osrm_manifest.json"):
        p = REPO / "scenarios/manifests" / mp
        if p.exists():
            for v in json.load(open(p, encoding="utf-8")).values():
                m = re.search(r"\((\-?\d+\.\d+),(\-?\d+\.\d+)\)", str(v))
                if m:
                    excl_all.append((float(m.group(1)), float(m.group(2))))
    bp = REPO / "scenarios/manifests/v15_blind250_points.json"
    if bp.exists():
        for v in json.load(open(bp, encoding="utf-8")).values():
            if isinstance(v, dict) and "lat" in v:
                excl_all.append((float(v["lat"]), float(v["lon"])))
    print(f"[points] 평가셋 배제 좌표 {len(excl_all)}개 · 스냅 상한 {SNAP_MAX_M:.0f}m · "
          f"시군구당 {a.n_points}점", flush=True)

    sf = shapefile.Reader(str(SHP), encoding="cp949")
    tasks = []
    for i, rec in enumerate(sf.records()):
        sigcd = str(rec[0])
        if sigcd not in meta:
            print(f"  ⚠ {sigcd} {rec[2]}: 기준 원장에 없음, 건너뜀")
            continue
        name, sido = meta[sigcd]
        sh = sf.shape(i)
        # 후보 배제 좌표를 bbox 근처로 미리 좁힌다(거리계산 절약)
        lo, la = Transformer.from_crs(5179, 4326, always_xy=True).transform(sh.bbox[0], sh.bbox[1])
        lo2, la2 = Transformer.from_crs(5179, 4326, always_xy=True).transform(sh.bbox[2], sh.bbox[3])
        ex = [e for e in excl_all if la - 0.05 <= e[0] <= la2 + 0.05 and lo - 0.05 <= e[1] <= lo2 + 0.05]
        tasks.append((sigcd, name, sido, _rings(sh), sh.bbox,
                      a.seed + i, a.n_points, ex, a.budget))
    if a.limit:
        tasks = tasks[:a.limit]
    print(f"[points] 시군구 {len(tasks)} · workers={a.workers}", flush=True)

    t0 = time.time()
    out, stats, bad = {}, {}, []
    with Pool(min(a.workers, len(tasks))) as pool:
        for n, r in enumerate(pool.imap_unordered(_point_worker, tasks), 1):
            stats[r["sigcd"]] = r["stat"]
            for p in r["points"]:
                out[f"{p['name']}_{p['sigcd']}_q{p['pidx']:02d}"] = p
            if not r["ok"]:
                bad.append((r["sigcd"], r["name"], r["stat"]["n_accepted"]))
            if n % 25 == 0 or n == len(tasks):
                print(f"  [{n}/{len(tasks)}] 누적 좌표 {len(out)} · {time.time()-t0:.0f}s", flush=True)

    OUT_POINTS.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    allsnap = np.array([p["snap_m"] for p in out.values()], float)
    rej = sum(s["n_rej_snap"] for s in stats.values())
    ins = sum(s["n_inside"] for s in stats.values())
    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "snap_max_m": SNAP_MAX_M, "points_per_sigungu": a.n_points,
        "n_sigungu": len(stats), "n_points": len(out),
        "sampling_note": ("행정구역 폴리곤 면적 균일 샘플 후 OSRM 스냅 500m 게이트. 따라서 표본은 "
                          "'도로 접근 가능 영역에서 균일'이며 면적 균일이 아니다."),
        "exclusion": {"targets": "대표점250 · 외부250 · v15_blind250",
                      "n_excluded_coords": len(excl_all),
                      "ladder_km": list(EXCL_LADDER),
                      "note": "기존 학습좌표(train1000·train_pool)는 배제하지 않는다 — 새 모델을 새로 학습하므로 무해."},
        "totals": {"n_inside_sampled": ins, "n_rejected_snap": rej,
                   "n_rejected_near_eval": sum(s["n_rej_near_eval"] for s in stats.values()),
                   "n_rejected_osrm_error": sum(s["n_rej_osrm"] for s in stats.values()),
                   "snap_reject_rate_of_inside": round(rej / max(ins, 1), 4)},
        "accepted_snap_m": {"median": round(float(np.median(allsnap)), 1),
                            "q90": round(float(np.percentile(allsnap, 90)), 1),
                            "max": round(float(allsnap.max()), 1)} if allsnap.size else None,
        "incomplete_regions": [{"sigcd": c, "name": n, "n": k} for c, n, k in bad],
        "per_sigungu": stats,
    }
    OUT_STATS.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[points] 좌표 {len(out)} / 목표 {len(tasks)*a.n_points} · {(time.time()-t0)/60:.1f}분")
    print(f"  폴리곤 내부 표본 {ins:,} → 스냅 기각 {rej:,} ({100*rej/max(ins,1):.1f}%)")
    if allsnap.size:
        print(f"  채택 스냅거리: 중위 {np.median(allsnap):.0f}m · q90 {np.percentile(allsnap,90):.0f}m · 최대 {allsnap.max():.0f}m")
    if bad:
        print(f"  ⚠ 30점 미달 {len(bad)}곳: {bad[:5]}")
    print(f"  → {OUT_POINTS}\n  → {OUT_STATS}")


# ------------------------------------------------- Phase 2: 시나리오 생성
def _build_worker(task):
    key, p = task
    from cross_location_eval import gen_scenario_for_region
    short = f"{p['name']}_{p['sigcd']}"
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            cfg = gen_scenario_for_region(
                short, p["lat"], p["lon"], base_path=str(REPO), exp_prefix=EXP_PREFIX,
                is_use_time=False, osrm_url=OSRM_URL, kakao_api_key=None,
                departure_time=None, **PARAMS)
        import yaml
        with open(cfg, encoding="utf-8") as f:
            c = yaml.safe_load(f)
        hp = c["entity_info"]["hospital"]["info_path"]
        if not os.path.isabs(hp):
            hp = os.path.join(REPO, hp)
        with open(hp, encoding="utf-8") as f:
            n = sum(1 for _ in f) - 1
        if n != PARAMS["fixed_hos_num"]:
            raise RuntimeError(f"hospital {n} != {PARAMS['fixed_hos_num']}")
        return dict(key=key, cfg=cfg, ok=True)
    except Exception as e:
        return dict(key=key, cfg=None, ok=False, err=str(e)[:200])


def build_main(a) -> None:
    pts = json.load(open(OUT_POINTS, encoding="utf-8"))
    tasks = [(k, v) for k, v in pts.items()]
    if a.limit:
        tasks = tasks[:a.limit]
    done = {}
    if OUT_MANIFEST.exists() and a.skip_done:
        done = json.load(open(OUT_MANIFEST, encoding="utf-8"))
        tasks = [(k, v) for k, v in tasks if k not in done or not os.path.exists(done[k])]
    print(f"[build] 좌표 {len(pts)} · 남은 {len(tasks)} · workers={a.workers}", flush=True)
    if not tasks:
        print("[build] 전부 완료 상태"); return
    t0, man, fail = time.time(), dict(done), []
    with Pool(min(a.workers, len(tasks))) as pool:
        for n, r in enumerate(pool.imap_unordered(_build_worker, tasks), 1):
            if r["ok"]:
                man[r["key"]] = r["cfg"]
            else:
                fail.append((r["key"], r.get("err", "")))
            if n % 100 == 0 or n == len(tasks):
                print(f"  [{n}/{len(tasks)}] ok={len(man)} fail={len(fail)} "
                      f"{time.time()-t0:.0f}s", flush=True)
                OUT_MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[build] 성공 {len(man)} · 실패 {len(fail)} · {(time.time()-t0)/60:.1f}분 → {OUT_MANIFEST}")
    if fail:
        print(f"  ⚠ 실패 예: {fail[:3]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("points", help="좌표 확정 (폴리곤 균일 + 스냅 500m 게이트)")
    p1.add_argument("--n_points", type=int, default=30)
    p1.add_argument("--workers", type=int, default=32)
    p1.add_argument("--seed", type=int, default=20260901)
    p1.add_argument("--budget", type=int, default=400000, help="시군구당 bbox 샘플 시도 상한")
    p1.add_argument("--limit", type=int, default=0)
    p2 = sub.add_parser("build", help="확정 좌표로 시나리오 생성")
    p2.add_argument("--workers", type=int, default=32)
    p2.add_argument("--limit", type=int, default=0)
    p2.add_argument("--skip_done", action="store_true", default=True)
    a = ap.parse_args()
    {"points": points_main, "build": build_main}[a.cmd](a)


if __name__ == "__main__":
    main()
