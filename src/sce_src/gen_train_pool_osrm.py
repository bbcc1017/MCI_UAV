# -*- coding: utf-8 -*-
"""D1 학습풀 확장(S1b): 시군구당 신규 3점 × 250 = 750점 OSRM 시나리오 생성 → train1000 매니페스트.

현 전국 학습풀(sigungu_osrm_manifest.json, 시군구 250 중심점)을 250→1000점으로 확장한다.
원형 gen_eval_holdout_osrm.py 의 폴리곤 샘플링·OSRM 연결·재시도 로직을 승계하되,
**오염 방지 제약**을 추가: 신규 좌표는
  (i) 기존 학습 중심점 250좌표(sigungu_osrm_manifest.json 경로에서 파싱)
  (ii) holdout 좌표 전체(eval_holdout_points.json 1000점 p0~p3 + plan1nat_eval_points.json 85점)
와 **최소 1km 이격**(haversine)한 시군구 폴리곤 내부 점만 허용.
⚠️ 초소형 시군구(부산 중구 2.8km² 등)는 1km 제약이 기하적으로 불가능할 수 있어
   반경을 1.0→0.5→0.25→0.125km 로 단계 완화(폴백)하고 points 파일에 radius_km 기록.

- 파라미터는 학습(시군구 OSRM)·holdout 과 **완전 일치**: incident100/amb30/uav_count26/
  uav_num26/fixed_hos_num47/vel50·200/handover5·10/total1000/seed0, OSRM(is_use_time=False).
- 좌표 확정: 최초 실행 시 250 시군구 × 3점 전량 샘플링 → train_pool_points.json 저장
  (이후 실행은 같은 좌표 재사용 → 스모크/재개 간 좌표 불변).
- 생성 실패(OSRM 경로 실패·병원수≠47·헬기장≠26) 시 같은 시군구 폴리곤서
  제약 만족 재샘플 재시도(원형 승계, 최대 10회) — 성공 좌표로 points 파일 갱신.
- 재개 가능: 최종 config 존재+구조검증(병원47·헬기장26·uav26) 통과 점은 자동 skip.
- 출력: scenarios/exp_train_pool/osrm_<name>_<sigcd>_osrm/(lat,lon)/config_*.yaml (gitignore 영역)
        + 좌표기록 scenarios/manifests/train_pool_points.json
- 매니페스트 조립(생성 완료 후 별도 실행): --assemble_manifest
        → scenarios/manifests/sigungu_osrm_train1000_manifest.json
          = 기존 250 항목(키·경로 그대로) ∪ 신규 성공분(키 <기존키>_t1/_t2/_t3, 절대경로).
          신규 700점 미만이면 실패 처리(--force 로 강행).

재사용 deps: cross_location_eval.gen_scenario_for_region(시나리오 생성),
             regen_sigungu_osrm.check_outputs(구조 검증).

예:
  # 스모크(종로구+양양군 6점만 생성; points 파일은 750점 전량 샘플링됨)
  PYTHONIOENCODING=utf-8 python src/sce_src/gen_train_pool_osrm.py --only 종로구,양양군
  # 전체 750점 생성(재개 가능)
  PYTHONIOENCODING=utf-8 python src/sce_src/gen_train_pool_osrm.py --workers 16
  # 매니페스트 조립(생성 완료 후)
  PYTHONIOENCODING=utf-8 python src/sce_src/gen_train_pool_osrm.py --assemble_manifest
"""
import argparse
import csv
import io
import json
import os
import re
import sys
import time
from contextlib import redirect_stdout
from multiprocessing import Pool

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
RL_DIR = os.path.join(REPO, "src", "rl_src")
for d in (THIS_DIR, RL_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import numpy as np
import shapefile
from pyproj import Transformer

from regen_sigungu_osrm import check_outputs  # 구조 검증(병원/헬기장/uav 행수) 재사용

OSRM_URL = os.environ.get("MCI_OSRM_URL", "http://127.0.0.1:5000")
EXP_PREFIX = "train_pool/osrm"
# 학습(시군구 OSRM)·holdout(gen_eval_holdout_osrm)과 완전 동일 파라미터
PARAMS = dict(incident_size=100, amb_count=30, uav_count=26, amb_velocity=50,
              uav_velocity=200, amb_handover_time=5.0, uav_handover_time=10.0,
              total_samples=1000, fixed_hos_num=47, uav_num=26)
N_HELI = 26                      # 헬기장 병원수 보장(=uav_num)
POINTS_PER_SIGUNGU = 3           # 시군구당 신규 점 수(250×3=750)
RADII_KM = (1.0, 0.5, 0.25, 0.125)   # 이격 반경 폴백 사다리
MUTUAL_MIN_KM = 0.1              # 같은 시군구 신규점 간 최소 이격(중복좌표 방지 sanity)
TRIES_PER_RADIUS = 20000         # 반경 단계당 rejection sampling 시도 상한

TRAIN_MANIFEST = os.path.join(REPO, "scenarios", "manifests", "sigungu_osrm_manifest.json")
HOLDOUT_POINTS = os.path.join(REPO, "scenarios", "manifests", "eval_holdout_points.json")
PLAN1NAT_POINTS = os.path.join(REPO, "scenarios", "manifests", "plan1nat_eval_points.json")
POINTS_PATH = os.path.join(REPO, "scenarios", "manifests", "train_pool_points.json")
MANIFEST_OUT = os.path.join(REPO, "scenarios", "manifests", "sigungu_osrm_train1000_manifest.json")


# ---------------------------------------------------------------- 기하 유틸(원형 승계)
def _rings(shape):
    parts = list(shape.parts) + [len(shape.points)]
    return [shape.points[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]


def _ring_contains(pt, ring):
    x, y = pt; inside = False; n = len(ring); j = n - 1
    for i in range(n):
        xi, yi = ring[i]; xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi):
            inside = not inside
        j = i
    return inside


def _contains(pt, rings):
    inside = False
    for r in rings:
        if _ring_contains(pt, r):
            inside = not inside
    return inside


def _min_dist_km(lat, lon, pts):
    """후보점 ↔ 제외집합 (N,2)[lat,lon] 최소 haversine 거리(km)."""
    la1, lo1 = np.radians(lat), np.radians(lon)
    la2, lo2 = np.radians(pts[:, 0]), np.radians(pts[:, 1])
    a = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return float(np.min(6371.0 * 2 * np.arcsin(np.sqrt(a))))


def _sample_excl(rings, bbox, rng, tr, excl, local_pts, radii=RADII_KM):
    """폴리곤 내부 + 제외집합과 radius 이상 이격한 점 1개 추출.

    거리검사(numpy, 저비용)를 폴리곤 판정(ray casting, 고비용)보다 먼저 수행.
    radius 는 radii 사다리를 따라 완화(초소형 시군구 폴백). (lat, lon, 사용반경) 반환.
    """
    xmin, ymin, xmax, ymax = bbox
    for radius in radii:
        for _ in range(TRIES_PER_RADIUS):
            x = rng.uniform(xmin, xmax); y = rng.uniform(ymin, ymax)
            lon, lat = tr.transform(x, y)
            lat, lon = round(lat, 6), round(lon, 6)
            if excl is not None and len(excl) and _min_dist_km(lat, lon, excl) < radius:
                continue
            if local_pts and _min_dist_km(lat, lon, np.asarray(local_pts)) < MUTUAL_MIN_KM:
                continue
            if _contains((x, y), rings):
                return lat, lon, radius
    raise RuntimeError(f"폴리곤 내부 이격점 추출 실패(radii={radii})")


# ---------------------------------------------------------------- 제외집합 로드
def load_exclusion_points():
    """(i) 학습 중심점 250 + (ii) holdout 1000(p0~p3) + plan1nat 85 → (N,2) 배열."""
    pts = []
    with open(TRAIN_MANIFEST, encoding="utf-8") as f:
        for key, cfg in json.load(f).items():
            m = re.search(r"\(([-\d.]+),([-\d.]+)\)", cfg)
            if not m:
                raise ValueError(f"학습 매니페스트 좌표 파싱 실패: {key} -> {cfg}")
            pts.append((float(m.group(1)), float(m.group(2))))
    n_train = len(pts)
    with open(HOLDOUT_POINTS, encoding="utf-8") as f:
        for v in json.load(f).values():
            pts.append((float(v["lat"]), float(v["lon"])))
    n_hold = len(pts) - n_train
    with open(PLAN1NAT_POINTS, encoding="utf-8") as f:
        for arr in json.load(f).values():
            for la, lo in arr:
                pts.append((float(la), float(lo)))
    n_p1n = len(pts) - n_train - n_hold
    print(f"[제외집합] 학습중심 {n_train} + holdout {n_hold} + plan1nat {n_p1n} "
          f"= {len(pts)}점", flush=True)
    return np.asarray(pts, dtype=float)


# ---------------------------------------------------------------- Pool 전역(제외집합)
_EXCL = None


def _init_pool(excl):
    global _EXCL
    _EXCL = excl


# ---------------------------------------------------------------- Phase A: 좌표 샘플링
def sample_worker(task):
    """시군구 1곳의 신규 3점 샘플링(제외집합·상호이격 제약)."""
    sigcd, name, sido, rings, bbox, seed = task
    tr = Transformer.from_crs(5179, 4326, always_xy=True)
    rng = np.random.default_rng(seed)
    out, local = [], []
    for tidx in range(1, POINTS_PER_SIGUNGU + 1):
        try:
            lat, lon, radius = _sample_excl(rings, bbox, rng, tr, _EXCL, local)
        except Exception as e:
            return dict(sigcd=sigcd, name=name, ok=False, err=str(e)[:160])
        local.append((lat, lon))
        out.append(dict(tidx=tidx, lat=lat, lon=lon, radius_km=radius))
    return dict(sigcd=sigcd, name=name, sido=sido, ok=True, points=out)


def build_points_file(args, excl):
    """sig.shp 250 시군구 × 3점 전량 샘플링 → train_pool_points.json 생성."""
    sf = shapefile.Reader(os.path.join(REPO, "scenarios", "sig.shp"), encoding="cp949")
    fields = [f[0] for f in sf.fields[1:]]
    ci_cd, ci_nm = fields.index("SIG_CD"), fields.index("SIG_KOR_NM")
    sido_of = {r["sigcd"]: r["sido"] for r in csv.DictReader(
        open(os.path.join(REPO, "results", "sigungu_by_sido.csv"), encoding="utf-8-sig"))}

    tasks = []
    for i, sr in enumerate(sf.shapeRecords()):
        sigcd = str(sr.record[ci_cd]).strip(); name = str(sr.record[ci_nm]).strip()
        tasks.append((sigcd, name, sido_of.get(sigcd, "미상"),
                      _rings(sr.shape), tuple(sr.shape.bbox), args.seed + i))
    print(f"[샘플링] 시군구 {len(tasks)}개 × {POINTS_PER_SIGUNGU}점 좌표 추출 "
          f"(1km 이격, 폴백 {RADII_KM})", flush=True)

    t0 = time.time(); points = {}; fallbacks = []
    with Pool(args.workers, initializer=_init_pool, initargs=(excl,)) as pool:
        for k, res in enumerate(pool.imap_unordered(sample_worker, tasks), 1):
            if not res["ok"]:
                raise RuntimeError(f"샘플링 실패: {res['name']}_{res['sigcd']} — {res['err']}")
            for p in res["points"]:
                key = f"{res['name']}_{res['sigcd']}_t{p['tidx']}"
                points[key] = dict(name=res["name"], sigcd=res["sigcd"], sido=res["sido"],
                                   lat=p["lat"], lon=p["lon"], radius_km=p["radius_km"],
                                   cfg=None, ok=False)
                if p["radius_km"] < RADII_KM[0]:
                    fallbacks.append((key, p["radius_km"]))
            if k % 50 == 0:
                print(f"  [{k}/{len(tasks)}] ({time.time()-t0:.0f}s)", flush=True)

    _write_points(points)
    print(f"[샘플링] 완료 {len(points)}점, wall={time.time()-t0:.0f}s → {POINTS_PATH}", flush=True)
    if fallbacks:
        print(f"  ⚠️ 반경 폴백 {len(fallbacks)}점(초소형 시군구): "
              f"{[(k, r) for k, r in fallbacks]}", flush=True)
    return points


def _write_points(points):
    """points 파일 원자적 저장(키 정렬로 diff 안정화)."""
    tmp = POINTS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(points.items())), f, ensure_ascii=False, indent=2)
    os.replace(tmp, POINTS_PATH)


# ---------------------------------------------------------------- Phase B: 시나리오 생성
def _verify_cfg(cfg):
    """최종 config 구조검증: 병원47·헬기장26·uav26 (성남시의료원 포함여부는 참고용)."""
    n_hos, n_heli, n_uav, has_sn = check_outputs(cfg)
    ok = (n_hos == PARAMS["fixed_hos_num"] and n_heli == N_HELI and n_uav == PARAMS["uav_num"])
    return ok, f"hos{n_hos}/heli{n_heli}/uav{n_uav}", has_sn


def gen_worker(task):
    """1점 시나리오 생성 + 구조검증. 실패 시 제약 만족 재샘플 재시도(최대 10회)."""
    key, name, sigcd, lat, lon, radius_km, rings, bbox, seed = task
    from cross_location_eval import gen_scenario_for_region
    tr = Transformer.from_crs(5179, 4326, always_xy=True)
    rng = np.random.default_rng(seed)
    short = f"{name}_{sigcd}"       # 같은 시군구 3점은 (lat,lon) 하위폴더로 분리
    radii = tuple(r for r in RADII_KM if r <= radius_km) or (radius_km,)
    last_err, used_radius = None, radius_km
    for attempt in range(10):
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                cfg = gen_scenario_for_region(
                    short, lat, lon, base_path=REPO, exp_prefix=EXP_PREFIX,
                    is_use_time=False, osrm_url=OSRM_URL, kakao_api_key=None,
                    departure_time=None, **PARAMS)
            ok, detail, has_sn = _verify_cfg(cfg)
            if not ok:
                raise RuntimeError(f"구조 불일치 {detail}")
            return dict(key=key, lat=lat, lon=lon, radius_km=used_radius,
                        cfg=cfg, ok=True, has_sn=has_sn, attempts=attempt + 1)
        except Exception as e:
            last_err = str(e)[:160]
            try:
                lat, lon, used_radius = _sample_excl(rings, bbox, rng, tr, _EXCL,
                                                     [(lat, lon)], radii=radii)  # 제약 만족 재추출
            except Exception as e2:
                last_err = f"resample fail: {e2}"; break
    return dict(key=key, lat=lat, lon=lon, radius_km=radius_km,
                cfg=None, ok=False, err=last_err)


def run_generation(args, excl, points):
    """points 파일 기준 생성 실행(검증 통과분 자동 skip = 재개)."""
    # 시군구 폴리곤(재샘플용) 로드
    sf = shapefile.Reader(os.path.join(REPO, "scenarios", "sig.shp"), encoding="cp949")
    fields = [f[0] for f in sf.fields[1:]]
    ci_cd = fields.index("SIG_CD")
    geo = {}
    for sr in sf.shapeRecords():
        geo[str(sr.record[ci_cd]).strip()] = (_rings(sr.shape), tuple(sr.shape.bbox))

    only = set(s.strip() for s in args.only.split(",") if s.strip()) if args.only else None
    tasks, skipped, ordered_keys = [], 0, []
    for j, (key, v) in enumerate(sorted(points.items())):
        # --only: 시군구명 또는 '이름_시군구코드' 매칭 / --limit: 앞에서 N점
        if only and v["name"] not in only and f"{v['name']}_{v['sigcd']}" not in only:
            continue
        ordered_keys.append(key)
        if args.limit and len(ordered_keys) > args.limit:
            ordered_keys.pop(); break
        # skip-done: 최종 config 존재 + 구조검증 통과 → 스킵(재개)
        if v.get("ok") and v.get("cfg") and os.path.exists(v["cfg"]):
            try:
                ok, _, _ = _verify_cfg(v["cfg"])
                if ok:
                    skipped += 1
                    continue
            except Exception:
                pass
        rings, bbox = geo[v["sigcd"]]
        tasks.append((key, v["name"], v["sigcd"], v["lat"], v["lon"],
                      v.get("radius_km", RADII_KM[0]), rings, bbox,
                      args.seed + 500000 + j))

    print(f"[생성] 대상 {len(tasks)}점 + skip {skipped}점 (총 {len(ordered_keys)}), "
          f"workers={args.workers}, OSRM={OSRM_URL}", flush=True)
    if not tasks:
        print("[생성] 남은 작업 없음(모두 완료).", flush=True)
        return

    t0 = time.time(); n_ok = n_fail = 0
    with Pool(args.workers, initializer=_init_pool, initargs=(excl,)) as pool:
        for k, res in enumerate(pool.imap_unordered(gen_worker, tasks), 1):
            v = points[res["key"]]
            v.update(lat=res["lat"], lon=res["lon"], cfg=res["cfg"], ok=res["ok"],
                     radius_km=res["radius_km"])
            if res["ok"]:
                n_ok += 1
            else:
                n_fail += 1
            if k % 10 == 0 or not res["ok"]:
                tag = "OK" if res["ok"] else f"FAIL({res.get('err')})"
                print(f"  [{k}/{len(tasks)}] {res['key']} {tag} ({time.time()-t0:.0f}s)",
                      flush=True)
            if k % 25 == 0:
                _write_points(points)   # 중간 저장(중단 시 재개 지점 보존)

    _write_points(points)
    total_ok = sum(1 for v in points.values() if v.get("ok"))
    print(f"\n[생성] 이번 run 성공 {n_ok}/{len(tasks)}, 실패 {n_fail}, "
          f"누적 성공 {total_ok}/{len(points)}, wall={time.time()-t0:.0f}s", flush=True)
    if n_fail:
        fails = [k for k, v in sorted(points.items()) if not v.get("ok")]
        print(f"  실패 키: {fails[:30]}{' ...' if len(fails) > 30 else ''}", flush=True)


# ---------------------------------------------------------------- 매니페스트 조립
def assemble_manifest(args):
    """기존 250 항목(그대로) ∪ 신규 성공분(_t1/_t2/_t3) → train1000 매니페스트.

    검증 통과분만 포함. 신규 700점 미만이면 exit 1(--force 로 강행).
    """
    with open(TRAIN_MANIFEST, encoding="utf-8") as f:
        base = json.load(f)
    with open(POINTS_PATH, encoding="utf-8") as f:
        points = json.load(f)

    new, bad = {}, []
    for key, v in sorted(points.items()):
        if not (v.get("ok") and v.get("cfg") and os.path.exists(v["cfg"])):
            bad.append((key, "미생성"))
            continue
        try:
            ok, detail, _ = _verify_cfg(v["cfg"])
        except Exception as e:
            ok, detail = False, str(e)[:80]
        if ok:
            new[key] = os.path.abspath(v["cfg"])
        else:
            bad.append((key, detail))

    print(f"[조립] 기존 {len(base)} + 신규 {len(new)}/{len(points)} "
          f"(제외 {len(bad)})", flush=True)
    if bad:
        print(f"  제외 목록: {bad[:20]}{' ...' if len(bad) > 20 else ''}", flush=True)
    if len(new) < args.min_new and not args.force:
        print(f"[조립] 신규 {len(new)} < 기준 {args.min_new} — 매니페스트 미작성 "
              f"(생성 재개 후 재시도하거나 --force).", flush=True)
        sys.exit(1)

    merged = dict(base); merged.update(new)
    with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"[조립] 완료: 총 {len(merged)}항목 → {MANIFEST_OUT}", flush=True)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16, help="병렬 워커 수(라우팅 부하 고려 16)")
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--limit", type=int, default=0, help="테스트용 앞 N점만 생성(0=전체)")
    ap.add_argument("--only", default=None,
                    help="특정 시군구만 생성(쉼표구분, 시군구명 또는 이름_코드). 스모크용")
    ap.add_argument("--assemble_manifest", action="store_true",
                    help="생성 완료 후 train1000 매니페스트 조립(생성은 수행 안 함)")
    ap.add_argument("--min_new", type=int, default=700,
                    help="조립 시 신규 최소 성공수(미만이면 실패, --force 로 강행)")
    ap.add_argument("--force", action="store_true", help="min_new 미만이어도 조립 강행")
    args = ap.parse_args()

    if args.assemble_manifest:
        assemble_manifest(args)
        return

    excl = load_exclusion_points()
    if os.path.exists(POINTS_PATH):
        with open(POINTS_PATH, encoding="utf-8") as f:
            points = json.load(f)
        print(f"[좌표] 기존 points 파일 재사용: {len(points)}점 ({POINTS_PATH})", flush=True)
    else:
        points = build_points_file(args, excl)
    run_generation(args, excl, points)


if __name__ == "__main__":
    main()
