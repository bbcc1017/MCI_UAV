# -*- coding: utf-8 -*-
"""자연-H(가변 병원 수) OSRM 시나리오 3세트 생성 (v6 Track A — 시나리오 비의존 RL).

기존 고정47 세트(fixed_hos_num=47, cap+fill)와 달리 **자연 선정 H 를 유지**하고
47 초과분만 절단(max_hos_num=47, cap-only — make_csv_yaml_dynamic._apply_hos_count
v6 분기)한 가변-H 세트를 만든다. 실측(2026-07-19, road API 0회) 자연 H 는
33~51(중앙값 43) → 기대 H∈[33,47]. 좌표·파라미터는 기존 고정47 세트와 동일
(incident100/amb30/uav26/vel50·200/handover5·10/total1000/seed0, is_use_time=False)
이라 같은 좌표의 고정47 세트와 쌍비교가 성립한다.

세트(--set):
  sigungu = 시군구250 (sigungu_osrm_manifest 좌표 재사용)   → exp_시군구natural/
  sido    = 시도17   (sido_osrm_manifest 좌표)              → exp_시도natural/
  holdout = eval_holdout_A 의 *_p0 250점                     → exp_holdoutAnatural/
  all     = 셋 다 순차.
산출 매니페스트: scenarios/manifests/{sigungu,sido,holdoutA}_natural_osrm_manifest.json
(값=절대경로 관례 — RL 학습·평가 드라이버가 그대로 소비).

- 로컬 OSRM(MCI_OSRM_URL, 기본 127.0.0.1:5000) 필요. 재개 가능(--skip_done).
- 구조 검증: H≤47(초과 0건)·헬기장 26행·uav_info 26행·성남시의료원 포함(헬기장
  보장룰상 전 지역 필수 포함), 세트별 H 분포(min/median/max) 리포트.

예: PYTHONIOENCODING=utf-8 python src/sce_src/gen_sigungu_natural_osrm.py --set all --workers 24
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
MAX_HOS_NUM = 47   # 사용자 결정(2026-07-19): 상한 47 유지 — 초과 지역만 cap-only
N_HELI = 26

MANI_DIR = os.path.join(REPO, "scenarios", "manifests")
# set 이름 → (소스 매니페스트, 키 필터, 출력 experiment_id 접두, 출력 매니페스트)
SETS = {
    "sigungu": (os.path.join(MANI_DIR, "sigungu_osrm_manifest.json"), None,
                "시군구natural/osrm", os.path.join(MANI_DIR, "sigungu_natural_osrm_manifest.json")),
    "sido": (os.path.join(MANI_DIR, "sido_osrm_manifest.json"), None,
             "시도natural/osrm", os.path.join(MANI_DIR, "sido_natural_osrm_manifest.json")),
    "holdout": (os.path.join(MANI_DIR, "eval_holdout_A_manifest.json"), "_p0",
                "holdoutAnatural/osrm", os.path.join(MANI_DIR, "holdoutA_natural_osrm_manifest.json")),
}


def parse_source(manifest_path, key_suffix):
    """[(key, name, lat, lon, expected_cfg)] — 좌표는 소스 경로에서 추출(포맷 보존).

    name 은 출력 디렉터리 명: 시군구/시도는 key 의 코드 접미 제거 없이 key 그대로 쓰면
    동명구(중구 6곳)도 (lat,lon) 하위 디렉터리로 구분되지만, 디렉터리 가독성을 위해
    시군구는 '이름_코드' 전체(key)를 사용한다(홀드아웃 '_p0' 접미도 유지 — 유일성 보장).
    """
    with open(manifest_path, encoding="utf-8") as f:
        d = json.load(f)
    out = []
    for key, cfg in d.items():
        if key_suffix and not key.endswith(key_suffix):
            continue
        m = re.search(r"\(([-\d.]+),([-\d.]+)\)", cfg)
        if not m:
            raise ValueError(f"좌표 파싱 실패: {key} -> {cfg}")
        out.append((key, key, m.group(1), m.group(2)))
    return out


def expected_cfg_path(prefix, name, lat, lon):
    """ScenarioGenerator 의 experiment_id 정규화 규칙(공백→'_', osrm 모드 →
    exp_<base>_osrm)에 따른 최종 config 경로. 홀드아웃 키의 공백('수원시 장안구_…')이
    실제 디렉터리에선 '_'로 바뀌므로 동일 정규화를 적용해야 한다."""
    safe = re.sub(r"\s+", "_", name).strip("_")
    return os.path.join(REPO, "scenarios", f"exp_{prefix}", f"{safe}_osrm",
                        f"({lat},{lon})", f"config_({lat},{lon}).yaml")


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


def _valid(n_hos, n_heli, n_uav):
    """자연-H 불변식: H≤47(초과=cap 실패), 헬기장·uav 26 고정."""
    return (n_hos <= MAX_HOS_NUM) and (n_heli == N_HELI) and (n_uav == PARAMS["uav_num"])


def worker(task):
    key, name, lat, lon, prefix = task
    from make_csv_yaml_dynamic import ScenarioGenerator
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            gen = ScenarioGenerator(
                base_path=REPO, experiment_id=f"{prefix}/{name}",
                kakao_api_key=None, departure_time=None, osrm_url=OSRM_URL,
                is_use_time=False, fixed_hos_num=None, max_hos_num=MAX_HOS_NUM)
            cfg = gen.generate_scenario(latitude=float(lat), longitude=float(lon),
                                        is_use_time=False, **PARAMS)
        n_hos, n_heli, n_uav, has_sn = check_outputs(cfg)
        ok = _valid(n_hos, n_heli, n_uav)
        err = None if ok else f"구조 불일치 hos{n_hos}/heli{n_heli}/uav{n_uav}"
        exp = expected_cfg_path(prefix, name, lat, lon)
        if ok and os.path.abspath(cfg) != os.path.abspath(exp):
            ok, err = False, f"경로 불일치: {cfg} != {exp}"
        return dict(key=key, ok=ok, err=err, has_sn=has_sn, n_hos=n_hos, cfg=cfg)
    except Exception as e:
        return dict(key=key, ok=False, err=str(e)[:200], has_sn=False, n_hos=0, cfg=None)


def run_set(set_name, workers, limit, only, skip_done):
    src, suffix, prefix, out_manifest = SETS[set_name]
    coords = parse_source(src, suffix)
    if only:
        coords = [c for c in coords if only in c[0]]
    if limit:
        coords = coords[:limit]
    tasks = [(k, n, la, lo, prefix) for (k, n, la, lo) in coords]

    done = {}          # key -> (cfg, n_hos) — skip 포함 최종 매니페스트 재료
    if skip_done:
        kept = []
        for t in tasks:
            exp = expected_cfg_path(prefix, t[1], t[2], t[3])
            if os.path.exists(exp):
                try:
                    n_hos, n_heli, n_uav, _ = check_outputs(exp)
                    if _valid(n_hos, n_heli, n_uav):
                        done[t[0]] = (exp, n_hos)
                        continue
                except Exception:
                    pass
            kept.append(t)
        print(f"[natural:{set_name}] skip_done: {len(tasks)-len(kept)}개 skip", flush=True)
        tasks = kept

    print(f"[natural:{set_name}] {len(tasks)}좌표 생성(max_hos_num={MAX_HOS_NUM}, cap-only), "
          f"workers={workers}, OSRM={OSRM_URL}", flush=True)
    t0 = time.time()
    results = []
    if tasks:
        with Pool(workers) as pool:
            for k, res in enumerate(pool.imap_unordered(worker, tasks), 1):
                results.append(res)
                if k % 25 == 0 or not res["ok"]:
                    tag = f"OK(H={res['n_hos']})" if res["ok"] else f"FAIL({res['err']})"
                    print(f"  [{k}/{len(tasks)}] {res['key']} {tag} "
                          f"({time.time()-t0:.0f}s)", flush=True)

    ok = [r for r in results if r["ok"]]
    for r in ok:
        done[r["key"]] = (r["cfg"], r["n_hos"])
    fails = [r for r in results if not r["ok"]]
    n_sn = sum(1 for r in ok if r["has_sn"])

    # H 분포 리포트(신규 생성 + skip 합산)
    hs = sorted(h for (_c, h) in done.values())
    if hs:
        med = hs[len(hs) // 2]
        print(f"[natural:{set_name}] H 분포: n={len(hs)} min={hs[0]} med={med} "
              f"max={hs[-1]} | ==47: {sum(1 for h in hs if h == 47)} | >47: "
              f"{sum(1 for h in hs if h > 47)}", flush=True)
    print(f"[natural:{set_name}] 완료: 신규성공 {len(ok)}/{len(results)} (성남 {n_sn}/{len(ok)}), "
          f"누적 {len(done)}, wall={time.time()-t0:.0f}s", flush=True)
    if fails:
        print(f"  실패: {[(r['key'], r['err']) for r in fails]}", flush=True)

    # 매니페스트: 전 좌표 성공 시에만 기록(부분 기록 금지 — 소비측 오작동 방지)
    if not limit and not only and len(done) == len(coords):
        mani = {k: done[k][0] for (k, _n, _la, _lo) in coords}
        with open(out_manifest, "w", encoding="utf-8") as f:
            json.dump(mani, f, ensure_ascii=False, indent=1)
        print(f"[natural:{set_name}] 매니페스트 기록: {out_manifest} ({len(mani)}엔트리)", flush=True)
    elif limit or only:
        print(f"[natural:{set_name}] --limit/--only 실행 — 매니페스트 미기록", flush=True)
    else:
        print(f"[natural:{set_name}] ⚠️ 미완주({len(done)}/{len(coords)}) — 매니페스트 미기록"
              f" (skip_done 재실행으로 회수)", flush=True)
    return len(done) == len(coords)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="all", choices=["sigungu", "sido", "holdout", "all"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="테스트용 N개만(0=전체, 매니페스트 미기록)")
    ap.add_argument("--only", default=None, help="키 부분일치 필터(테스트용, 매니페스트 미기록)")
    ap.add_argument("--skip_done", action="store_true",
                    help="최종 config 존재+구조검증 통과 좌표는 skip(재개)")
    args = ap.parse_args()

    names = ["sigungu", "sido", "holdout"] if args.set == "all" else [args.set]
    all_ok = True
    for nm in names:
        all_ok = run_set(nm, args.workers, args.limit, args.only, args.skip_done) and all_ok
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
