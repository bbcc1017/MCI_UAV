# -*- coding: utf-8 -*-
"""시군구 250 OSRM 시나리오 재생성 (좌표 고정, 헬기장 정정 대응).

2026-07-02 병원 풀(엑셀 결합 데이터.xlsx) 성남시의료원 헬기장여부 0→1 정정으로
헬기장 병원이 25→26곳. **자원 추가 원칙**(기존 병원 삭제 금지): fixed_hos_num 46→47,
헬기장 보장 uav_num 25→26 으로 올려 기존 46곳 집합을 그대로 보존하고 성남시의료원을
추가한다(신규 47집합 ⊇ 구 46집합). 기존 좌표(osrm_pre20260702 백업과 동일)로 전량 재생성.

- 좌표·키: scenarios/manifests/sigungu_osrm_manifest.json 재사용(경로 불변 → manifest 무수정).
- 파라미터: incident100/amb30/**uav_count26/uav_num26/fixed_hos47**/
  vel50·200/handover5·10/total1000/seed0, is_use_time=False.
- 검증: 병원 47행·헬기장 26행·uav_info 26행, 성남시의료원 포함 여부 리포트.
- 로컬 OSRM(MCI_OSRM_URL, 기본 127.0.0.1:5000) 필요. 재개 가능(--skip_done).

예: PYTHONIOENCODING=utf-8 python src/sce_src/regen_sigungu_osrm.py --workers 32
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
for d in (THIS_DIR, os.path.join(REPO, "src", "rl_src")):
    if d not in sys.path:
        sys.path.insert(0, d)

OSRM_URL = os.environ.get("MCI_OSRM_URL", "http://127.0.0.1:5000")
PARAMS = dict(incident_size=100, amb_count=30, uav_count=26, amb_velocity=50,
              uav_velocity=200, amb_handover_time=5.0, uav_handover_time=10.0,
              total_samples=1000, random_seed=0, uav_num=26)
FIXED_HOS_NUM = 47


def parse_manifest(path):
    """[(key, name, lat, lon, cfg_path)] — name='종로구', 좌표는 경로에서 추출."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = []
    for key, cfg in d.items():
        m = re.search(r"\(([-\d.]+),([-\d.]+)\)", cfg)
        if not m:
            raise ValueError(f"좌표 파싱 실패: {key} -> {cfg}")
        out.append((key, key.rsplit("_", 1)[0], m.group(1), m.group(2), cfg))
    return out


def check_outputs(cfg_path):
    """생성물 구조 검증 → (병원수, 헬기장수, uav행수, 성남시의료원 포함여부)."""
    d = os.path.dirname(cfg_path)
    with open(os.path.join(d, "hospital_info.csv"), encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    n_hos = len(rows)
    n_heli = sum(1 for r in rows if str(r.get("헬기장 여부", "0")).strip() in ("1", "1.0"))
    has_sn = any("성남시의료원" in r.get("요양기관명", "") for r in rows)
    with open(os.path.join(d, "uav_info.csv"), encoding="utf-8-sig") as f:
        n_uav = sum(1 for _ in f) - 1
    return n_hos, n_heli, n_uav, has_sn


def worker(task):
    key, name, lat, lon, cfg_expected = task
    from make_csv_yaml_dynamic import ScenarioGenerator
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            gen = ScenarioGenerator(
                base_path=REPO, experiment_id=f"시군구/osrm/{name}",
                kakao_api_key=None, departure_time=None, osrm_url=OSRM_URL,
                is_use_time=False, fixed_hos_num=FIXED_HOS_NUM)
            cfg = gen.generate_scenario(latitude=float(lat), longitude=float(lon),
                                        is_use_time=False, **PARAMS)
        n_hos, n_heli, n_uav, has_sn = check_outputs(cfg)
        ok = (n_hos == FIXED_HOS_NUM and n_heli == PARAMS["uav_num"]
              and n_uav == PARAMS["uav_num"])
        err = None if ok else f"구조 불일치 hos{n_hos}/heli{n_heli}/uav{n_uav}"
        if ok and os.path.abspath(cfg) != os.path.abspath(cfg_expected):
            ok, err = False, f"경로 불일치: {cfg}"
        return dict(key=key, ok=ok, err=err, has_sn=has_sn)
    except Exception as e:
        return dict(key=key, ok=False, err=str(e)[:200], has_sn=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(
        REPO, "scenarios", "manifests", "sigungu_osrm_manifest.json"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="테스트용 N개만(0=전체)")
    ap.add_argument("--skip_done", action="store_true",
                    help="최종 config 존재+구조검증 통과 좌표는 skip(재개)")
    args = ap.parse_args()

    tasks = parse_manifest(args.manifest)
    if args.limit:
        tasks = tasks[:args.limit]
    if args.skip_done:
        kept = []
        for t in tasks:
            if os.path.exists(t[4]):
                try:
                    n_hos, n_heli, n_uav, _ = check_outputs(t[4])
                    if n_hos == FIXED_HOS_NUM and n_heli == PARAMS["uav_num"]:
                        continue
                except Exception:
                    pass
            kept.append(t)
        print(f"[regen] skip_done: {len(tasks)-len(kept)}개 skip", flush=True)
        tasks = kept

    print(f"[regen] 시군구 OSRM {len(tasks)}좌표 재생성, workers={args.workers}, "
          f"OSRM={OSRM_URL}", flush=True)
    t0 = time.time()
    results = []
    with Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(worker, tasks), 1):
            results.append(res)
            if k % 25 == 0 or not res["ok"]:
                tag = "OK" if res["ok"] else f"FAIL({res['err']})"
                print(f"  [{k}/{len(tasks)}] {res['key']} {tag} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    ok = [r for r in results if r["ok"]]
    n_sn = sum(1 for r in ok if r["has_sn"])
    print(f"\n[regen] 완료: 성공 {len(ok)}/{len(results)}, "
          f"성남시의료원 포함 {n_sn}/{len(ok)}, wall={time.time()-t0:.0f}s", flush=True)
    fails = [r for r in results if not r["ok"]]
    if fails:
        print("  실패:", [(r["key"], r["err"]) for r in fails], flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
