"""전국 RL 학습용 시군구당 무작위 4점 시나리오 일괄 생성(단위균등, OSRM).

2026-07-23 분할 변경: 과거 일반화 hold-out p0~p3를 새 정책의 학습 좌표로 사용하고,
시군구 representative_point 250개는 평가 전용으로 둔다.
- 샘플링: sig.shp 250 시군구 각각 폴리곤 내부 **무작위 점 4개**(기본값).
  → 전체 1,000점. 키는 <시군구>_<sigcd>_p0..p3.
- 파라미터는 평가 대표점과 **완전 일치**: incident100/amb30/uav_count26/**uav_num26**/
  **fixed_hos_num47**/vel50·200/handover5·10/total1000, OSRM(is_use_time=False). 좌표만 새것.
  (2026-07-02 성남시의료원 헬기장 정정: 자원 추가 원칙으로 46→47병원·헬기장 26 보장.)
- 실패(OSRM 경로 실패·병원수≠47) 시 같은 시군구 폴리곤서 재추출 재시도.
- 출력: scenarios/exp_eval_holdout/osrm_<name>_<sigcd>/(lat,lon)/config_*.yaml (학습과 분리)
        + 학습 매니페스트: scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json (1,000)
        + 시도별 부분집합(레거시 경로): scenarios/manifests/eval_holdout_sido/<sido>.json
        + 좌표기록: scenarios/manifests/sigungu_osrm_train1000_random4_points.json

예: PYTHONIOENCODING=utf-8 python src/sce_src/gen_eval_holdout_osrm.py --workers 48
"""
import argparse, csv, io, json, os, sys, time
from contextlib import redirect_stdout
from multiprocessing import Pool

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
RL_DIR = os.path.join(REPO, "src", "rl_src")
for d in (THIS_DIR, RL_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import shapefile
from pyproj import Transformer
import numpy as np

OSRM_URL = os.environ.get("MCI_OSRM_URL", "http://127.0.0.1:5000")
EXP_PREFIX = "eval_holdout/osrm"
PARAMS = dict(incident_size=100, amb_count=30, uav_count=26, amb_velocity=50,
              uav_velocity=200, amb_handover_time=5.0, uav_handover_time=10.0,
              total_samples=1000, fixed_hos_num=47, uav_num=26)


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


def _sample(rings, bbox, rng, tr):
    xmin, ymin, xmax, ymax = bbox
    for _ in range(200000):
        x = rng.uniform(xmin, xmax); y = rng.uniform(ymin, ymax)
        if _contains((x, y), rings):
            lon, lat = tr.transform(x, y)
            return round(lat, 6), round(lon, 6)
    raise RuntimeError("폴리곤 내부 점 추출 실패")


def _count_hospitals(cfg):
    import yaml
    with open(cfg, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    hp = c["entity_info"]["hospital"]["info_path"]
    if not os.path.isabs(hp):
        hp = os.path.join(REPO, hp)
    with open(hp, encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def worker(task):
    sigcd, name, sido, rings, bbox, seed, pidx, fixed = task
    from cross_location_eval import gen_scenario_for_region
    if fixed is not None:
        # 좌표 고정 재생성 모드(--points_from): 재샘플 금지, 같은 좌표로만 재시도
        lat, lon = fixed
        max_attempts, resample = 3, False
    else:
        tr = Transformer.from_crs(5179, 4326, always_xy=True)
        rng = np.random.default_rng(seed)
        lat, lon = _sample(rings, bbox, rng, tr)
        max_attempts, resample = 10, True
    short = f"{name}_{sigcd}"  # 같은 시군구 4점은 (lat,lon) 하위폴더로 분리
    last_err = None
    for attempt in range(max_attempts):
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                cfg = gen_scenario_for_region(
                    short, lat, lon, base_path=REPO, exp_prefix=EXP_PREFIX,
                    is_use_time=False, osrm_url=OSRM_URL, kakao_api_key=None,
                    departure_time=None, **PARAMS)
            n = _count_hospitals(cfg)
            if n != PARAMS["fixed_hos_num"]:
                raise RuntimeError(f"hospital {n}≠{PARAMS['fixed_hos_num']}")
            return dict(sigcd=sigcd, name=name, sido=sido, pidx=pidx, lat=lat, lon=lon,
                        cfg=cfg, ok=True, attempts=attempt + 1)
        except Exception as e:
            last_err = str(e)[:160]
            if resample:
                try:
                    lat, lon = _sample(rings, bbox, rng, tr)  # 재추출
                except Exception as e2:
                    last_err = f"resample fail: {e2}"; break
    return dict(sigcd=sigcd, name=name, sido=sido, pidx=pidx, lat=lat, lon=lon,
                cfg=None, ok=False, err=last_err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20260627)
    ap.add_argument("--points_per_sigungu", type=int, default=4,
                    help="시군구당 무작위 점 수(4=총1000). 단위균등·구역수비례 동시충족.")
    ap.add_argument("--limit", type=int, default=0, help="테스트용 시군구 N개만(0=전체250)")
    ap.add_argument("--points_from", default=None,
                    help="기존 sigungu_osrm_train1000_random4_points.json 경로 — 좌표 고정 재생성(재샘플 없음). "
                         "병원 풀 정정 등으로 시나리오만 다시 구울 때 사용.")
    ap.add_argument("--skip_done", action="store_true",
                    help="(points_from 전용) 최종 config 존재+병원수 일치 좌표 skip(재개)")
    args = ap.parse_args()

    if args.points_from:
        # 좌표 고정 재생성: 키='<name>_<sigcd>_p<idx>', ok=True 항목만.
        # --skip_done 이면 최종 config 존재+병원수 일치 좌표는 건너뜀(재개).
        # 이 모드에서는 실행 말미 manifest 를 '이번 run 결과'가 아니라 skip 포함
        # 전체 태스크의 디스크 상태로 재구성한다(부분 실패/재개 시 manifest 잘림 방지).
        with open(args.points_from, encoding="utf-8") as f:
            pts = json.load(f)
        tasks, skipped = [], []
        for key, v in pts.items():
            if not v.get("ok"):
                continue
            stem, pstr = key.rsplit("_p", 1)
            name, sigcd = stem.rsplit("_", 1)
            t = (sigcd, name, v["sido"], None, None, 0, int(pstr),
                 (v["lat"], v["lon"]))
            if args.skip_done and v.get("cfg") and os.path.exists(v["cfg"]):
                try:
                    if _count_hospitals(v["cfg"]) == PARAMS["fixed_hos_num"]:
                        skipped.append(dict(sigcd=sigcd, name=name, sido=v["sido"],
                                            pidx=int(pstr), lat=v["lat"], lon=v["lon"],
                                            cfg=v["cfg"], ok=True))
                        continue
                except Exception:
                    pass
            tasks.append(t)
        if args.limit:
            tasks = tasks[:args.limit]
        print(f"[gen_eval] 좌표 고정 재생성 {len(tasks)}점 + skip {len(skipped)}점 "
              f"(from {args.points_from}), workers={args.workers}, OSRM={OSRM_URL}", flush=True)
    else:
        skipped = []
        sf = shapefile.Reader(os.path.join(REPO, "scenarios", "sig.shp"), encoding="cp949")
        fields = [f[0] for f in sf.fields[1:]]
        ci_cd, ci_nm = fields.index("SIG_CD"), fields.index("SIG_KOR_NM")
        sido_of = {r["sigcd"]: r["sido"] for r in
                   csv.DictReader(open(os.path.join(REPO, "results", "sigungu_by_sido.csv"), encoding="utf-8-sig"))}

        recs = sf.shapeRecords()
        if args.limit:
            recs = recs[:args.limit]
        tasks = []
        for i, sr in enumerate(recs):
            sigcd = str(sr.record[ci_cd]).strip(); name = str(sr.record[ci_nm]).strip()
            sido = sido_of.get(sigcd, "미상")
            rings, bbox = _rings(sr.shape), tuple(sr.shape.bbox)
            for pidx in range(args.points_per_sigungu):
                tasks.append((sigcd, name, sido, rings, bbox, args.seed + i * 1000 + pidx, pidx, None))
        print(f"[gen_eval] 시군구 {len(recs)}개 × {args.points_per_sigungu}점 = {len(tasks)} 시나리오, "
              f"workers={args.workers}, OSRM={OSRM_URL}", flush=True)

    t0 = time.time(); results = []
    with Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(worker, tasks), 1):
            results.append(res)
            tag = "OK" if res["ok"] else f"FAIL({res.get('err')})"
            if k % 10 == 0 or not res["ok"]:
                print(f"  [{k}/{len(tasks)}] {res['name']}_{res['sigcd']} {tag} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    results.extend(skipped)  # 재개 모드: skip 좌표도 manifest/points 에 포함(잘림 방지)
    ok = [r for r in results if r["ok"]]
    key = lambda r: f"{r['name']}_{r['sigcd']}_p{r['pidx']}"
    # 매니페스트 A (전국)
    os.makedirs(os.path.join(REPO, "scenarios", "manifests", "eval_holdout_sido"), exist_ok=True)
    A = {key(r): r["cfg"] for r in ok}
    with open(os.path.join(REPO, "scenarios", "manifests", "sigungu_osrm_train1000_random4_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(A, f, ensure_ascii=False, indent=2)
    # 매니페스트 B (시도별)
    from collections import defaultdict
    B = defaultdict(dict)
    for r in ok:
        B[r["sido"]][key(r)] = r["cfg"]
    for sido, d in B.items():
        with open(os.path.join(REPO, "scenarios", "manifests", "eval_holdout_sido", f"{sido}.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    # 좌표 기록
    with open(os.path.join(REPO, "scenarios", "manifests", "sigungu_osrm_train1000_random4_points.json"), "w", encoding="utf-8") as f:
        json.dump({key(r): {k: r[k] for k in ("name", "sido", "lat", "lon", "cfg", "ok")} for r in results},
                  f, ensure_ascii=False, indent=2)

    n_total = len(tasks) + len(skipped)
    print(f"\n[gen_eval] 완료: 성공 {len(ok)}/{n_total} (이번 run {len(tasks)}, skip {len(skipped)}), "
          f"실패 {n_total-len(ok)}, wall={time.time()-t0:.0f}s", flush=True)
    print(f"  A(전국)={len(A)}, B 시도수={len(B)}", flush=True)
    fails = [r for r in results if not r["ok"]]
    if fails:
        print("  실패목록:", [f"{r['name']}_{r['sigcd']}" for r in fails], flush=True)


if __name__ == "__main__":
    main()
