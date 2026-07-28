# -*- coding: utf-8 -*-
"""증류 모델 선택과 분리한 시군구 250개 외부 테스트 시나리오 생성.

시군구마다 폴리곤 내부 좌표 1개를 새로 뽑되 다음 기존 좌표에서 원칙적으로 1 km 이상
떨어지게 한다.

* v10 학습 random4 1,000좌표
* 이미 모델 비교에 사용한 대표점 250좌표
* 과거 train-pool/plan1nat 등 ``*points.json``에 기록된 좌표

초소형 시군구는 1.0→0.5→0.25→0.125 km 순으로 완화하며 실제 사용 반경을 기록한다.
시나리오는 v10과 같은 incident100·AMB30·UAV26·병원47·seed0·OSRM 조건이다.
좌표와 시나리오는 재개 가능하고, 최종 매니페스트는 250개 구조 검증 후에만 기록한다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from contextlib import redirect_stdout
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import shapefile
from pyproj import Transformer

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parents[1]
RL_DIR = REPO / "src/rl_src"
for _d in (str(THIS_DIR), str(RL_DIR)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from regen_sigungu_osrm import check_outputs

OSRM_URL = os.environ.get("MCI_OSRM_URL", "http://127.0.0.1:5000")
EXP_PREFIX = "distill_external/osrm"
POINTS_PATH = REPO / "scenarios/manifests/distill_external_test250_points.json"
MANIFEST_PATH = REPO / "scenarios/manifests/distill_external_test250_osrm_manifest.json"
META_PATH = REPO / "scenarios/manifests/distill_external_test250_meta.json"
RADII_KM = (1.0, 0.5, 0.25, 0.125)
PARAMS = dict(
    incident_size=100,
    amb_count=30,
    uav_count=26,
    amb_velocity=50,
    uav_velocity=200,
    amb_handover_time=5.0,
    uav_handover_time=10.0,
    total_samples=1000,
    fixed_hos_num=47,
    uav_num=26,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _rings(shape):
    parts = list(shape.parts) + [len(shape.points)]
    return [shape.points[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]


def _ring_contains(pt, ring):
    x, y = pt
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi
        ):
            inside = not inside
        j = i
    return inside


def _contains(pt, rings):
    inside = False
    for ring in rings:
        if _ring_contains(pt, ring):
            inside = not inside
    return inside


def _min_dist_km(lat: float, lon: float, points: np.ndarray) -> float:
    if len(points) == 0:
        return math.inf
    la1, lo1 = np.radians(lat), np.radians(lon)
    la2, lo2 = np.radians(points[:, 0]), np.radians(points[:, 1])
    a = (
        np.sin((la2 - la1) / 2) ** 2
        + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    )
    return float(np.min(6371.0 * 2.0 * np.arcsin(np.sqrt(a))))


def _path_coord(value: str):
    m = re.search(r"\(([-\d.]+),([-\d.]+)\)", value)
    return (float(m.group(1)), float(m.group(2))) if m else None


def _walk_points(obj, out: list[tuple[float, float]]) -> None:
    if isinstance(obj, dict):
        if "lat" in obj and "lon" in obj:
            try:
                out.append((float(obj["lat"]), float(obj["lon"])))
            except (TypeError, ValueError):
                pass
        for value in obj.values():
            _walk_points(value, out)
    elif isinstance(obj, list):
        if (
            len(obj) == 2
            and all(isinstance(x, (int, float)) for x in obj)
            and 30 <= float(obj[0]) <= 40
            and 120 <= float(obj[1]) <= 135
        ):
            out.append((float(obj[0]), float(obj[1])))
        else:
            for value in obj:
                _walk_points(value, out)
    elif isinstance(obj, str):
        point = _path_coord(obj)
        if point is not None:
            out.append(point)


def load_exclusions() -> tuple[np.ndarray, list[dict]]:
    """기존 좌표 원장과 정본 매니페스트를 모두 읽고 중복 제거."""
    sources: list[Path] = [
        REPO / "scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json",
        REPO / "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json",
    ]
    sources.extend(
        p for p in sorted((REPO / "scenarios/manifests").glob("*points.json"))
        if p.resolve() != POINTS_PATH.resolve()
    )
    points: list[tuple[float, float]] = []
    source_meta = []
    for path in dict.fromkeys(sources):
        if not path.exists():
            continue
        obj = json.load(open(path, encoding="utf-8"))
        before = len(points)
        _walk_points(obj, points)
        source_meta.append({
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "coordinates_read": len(points) - before,
        })
    unique = sorted(set((round(a, 6), round(b, 6)) for a, b in points))
    return np.asarray(unique, dtype=float), source_meta


def _sample_point(rings, bbox, seed: int, exclusions: np.ndarray):
    tr = Transformer.from_crs(5179, 4326, always_xy=True)
    rng = np.random.default_rng(seed)
    xmin, ymin, xmax, ymax = bbox
    for radius in RADII_KM:
        for _ in range(30000):
            x = rng.uniform(xmin, xmax)
            y = rng.uniform(ymin, ymax)
            if not _contains((x, y), rings):
                continue
            lon, lat = tr.transform(x, y)
            lat, lon = round(lat, 6), round(lon, 6)
            if _min_dist_km(lat, lon, exclusions) >= radius:
                return lat, lon, radius
    raise RuntimeError("이격된 폴리곤 내부 좌표 추출 실패")


_EXCLUSIONS: np.ndarray | None = None


def _init_pool(exclusions):
    global _EXCLUSIONS
    _EXCLUSIONS = exclusions


def _sample_worker(task):
    sigcd, name, sido, rings, bbox, seed = task
    try:
        lat, lon, radius = _sample_point(rings, bbox, seed, _EXCLUSIONS)
        return {
            "ok": True,
            "key": f"{name}_{sigcd}_ext",
            "name": name,
            "sigcd": sigcd,
            "sido": sido,
            "lat": lat,
            "lon": lon,
            "radius_km": radius,
            "seed": seed,
            "cfg": None,
        }
    except Exception as exc:
        return {"ok": False, "key": f"{name}_{sigcd}_ext", "err": str(exc)}


def _valid_cfg(path: str | None) -> bool:
    if not path or not os.path.exists(path):
        return False
    try:
        n_hos, n_heli, n_uav, _ = check_outputs(path)
        return n_hos == 47 and n_heli == 26 and n_uav == 26
    except Exception:
        return False


def _scenario_worker(point):
    from cross_location_eval import gen_scenario_for_region

    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            cfg = gen_scenario_for_region(
                f"{point['name']}_{point['sigcd']}_ext",
                point["lat"],
                point["lon"],
                base_path=str(REPO),
                exp_prefix=EXP_PREFIX,
                is_use_time=False,
                osrm_url=OSRM_URL,
                kakao_api_key=None,
                departure_time=None,
                **PARAMS,
            )
        if not _valid_cfg(cfg):
            raise RuntimeError("생성물 구조가 hospital47/helipad26/uav26과 불일치")
        return {**point, "ok": True, "cfg": str(Path(cfg).resolve())}
    except Exception as exc:
        return {**point, "ok": False, "err": str(exc)[:500]}


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _shape_tasks(seed: int):
    sf = shapefile.Reader(str(REPO / "scenarios/sig.shp"), encoding="cp949")
    fields = [f[0] for f in sf.fields[1:]]
    ci_cd, ci_nm = fields.index("SIG_CD"), fields.index("SIG_KOR_NM")
    sido_of = {
        row["sigcd"]: row["sido"]
        for row in csv.DictReader(
            open(REPO / "results/sigungu_by_sido.csv", encoding="utf-8-sig")
        )
    }
    tasks = []
    for i, sr in enumerate(sf.shapeRecords()):
        sigcd = str(sr.record[ci_cd]).strip()
        name = str(sr.record[ci_nm]).strip()
        tasks.append((
            sigcd,
            name,
            sido_of.get(sigcd, "미상"),
            _rings(sr.shape),
            tuple(sr.shape.bbox),
            seed + i * 1009,
        ))
    if len(tasks) != 250:
        raise RuntimeError(f"시군구 shp 레코드 {len(tasks)} != 250")
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--resample", action="store_true", help="기존 좌표 원장을 폐기하고 재추출")
    ap.add_argument("--limit", type=int, default=0, help="스모크용 앞 N개만; 정본 manifest 미작성")
    args = ap.parse_args()
    t0 = time.time()
    exclusions, source_meta = load_exclusions()
    tasks = _shape_tasks(args.seed)
    if args.limit:
        tasks = tasks[:args.limit]
    print(
        f"[external-test] sigungu={len(tasks)} exclusions={len(exclusions)} "
        f"workers={min(args.workers,len(tasks))} OSRM={OSRM_URL}",
        flush=True,
    )

    if POINTS_PATH.exists() and not args.resample:
        stored = json.load(open(POINTS_PATH, encoding="utf-8"))
        points = list(stored.values())
        expected = {f"{x[1]}_{x[0]}_ext" for x in tasks}
        points = [x for x in points if x["key"] in expected]
        if len(points) != len(tasks):
            raise RuntimeError(
                f"기존 points가 요청 범위와 불일치 {len(points)} != {len(tasks)}; "
                "--resample로 다시 생성"
            )
        print(f"[external-test] 좌표 원장 재사용 {POINTS_PATH}", flush=True)
    else:
        points = []
        with Pool(
            min(args.workers, len(tasks)),
            initializer=_init_pool,
            initargs=(exclusions,),
        ) as pool:
            for i, result in enumerate(pool.imap_unordered(_sample_worker, tasks), 1):
                if not result["ok"]:
                    raise RuntimeError(f"좌표 추출 실패 {result['key']}: {result['err']}")
                points.append(result)
                if i % 50 == 0:
                    print(f"  [sample {i}/{len(tasks)}]", flush=True)
        points.sort(key=lambda x: x["sigcd"])
        _atomic_json(POINTS_PATH, {x["key"]: x for x in points})

    pending = [x for x in points if not _valid_cfg(x.get("cfg"))]
    completed = [x for x in points if _valid_cfg(x.get("cfg"))]
    point_map = {x["key"]: x for x in points}
    print(f"[external-test] scenario pending={len(pending)} reused={len(completed)}", flush=True)
    failed = []
    if pending:
        with Pool(min(args.workers, len(pending)), maxtasksperchild=1) as pool:
            for i, result in enumerate(pool.imap_unordered(_scenario_worker, pending), 1):
                point_map[result["key"]] = result
                if result["ok"]:
                    completed.append(result)
                else:
                    failed.append(result)
                if i % 10 == 0 or not result["ok"]:
                    print(
                        f"  [scenario {i}/{len(pending)}] {result['key']} "
                        f"{'OK' if result['ok'] else 'FAIL'}",
                        flush=True,
                    )
                _atomic_json(POINTS_PATH, dict(sorted(point_map.items())))
    if failed:
        fail_path = Path(str(POINTS_PATH) + ".failed.json")
        _atomic_json(fail_path, failed)
        raise RuntimeError(f"시나리오 생성 실패 {len(failed)}개: {fail_path}")
    if args.limit:
        print("[external-test] limit 모드는 정본 manifest를 만들지 않음", flush=True)
        return
    if len(completed) != 250 or len({x["sigcd"] for x in completed}) != 250:
        raise RuntimeError("외부 테스트 시군구 완전성 오류")
    manifest = {x["key"]: x["cfg"] for x in sorted(completed, key=lambda y: y["sigcd"])}
    coords = {(round(x["lat"], 6), round(x["lon"], 6)) for x in completed}
    excluded_coords = {(round(x[0], 6), round(x[1], 6)) for x in exclusions}
    overlap = coords & excluded_coords
    if overlap:
        raise RuntimeError(f"외부 테스트가 기존 좌표와 정확 중복 {len(overlap)}개")
    _atomic_json(MANIFEST_PATH, manifest)
    meta = {
        "schema_version": 1,
        "role": "final_external_test_only",
        "selection_prohibition": "모델·하이퍼파라미터·위임 임계값 선택에 사용 금지",
        "seed": args.seed,
        "n_sigungu": 250,
        "n_coordinates": 250,
        "n_unique_exclusion_coordinates": int(len(exclusions)),
        "exact_overlap_with_exclusions": 0,
        "minimum_radius_rule_km": list(RADII_KM),
        "radius_counts": {
            str(r): sum(float(x["radius_km"]) == r for x in completed) for r in RADII_KM
        },
        "exclusion_sources": source_meta,
        "points": str(POINTS_PATH.resolve()),
        "points_sha256": _sha256(POINTS_PATH),
        "manifest": str(MANIFEST_PATH.resolve()),
        "manifest_sha256": _sha256(MANIFEST_PATH),
        "scenario_params": PARAMS,
        "osrm_url": OSRM_URL,
        "wall_seconds": time.time() - t0,
    }
    _atomic_json(META_PATH, meta)
    print(
        f"[external-test] 완료 250개 wall={(time.time()-t0)/60:.1f}분 "
        f"→ {MANIFEST_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
