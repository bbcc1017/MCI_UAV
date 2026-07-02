# -*- coding: utf-8 -*-
"""Kakao 시나리오 외과 패치 — 성남시의료원 헬기장 정정(0→1) 반영 (2026-07-02).

병원 풀 정정으로 전 좌표의 병원집합이 바뀌지만(헬기장 25→26곳 → 최근접 25 선택),
Kakao 전량 재생성은 quota 부담이 크다. 대신 **구 백업 시나리오의 도로 결과를
캐시로 서빙하며 정식 생성 코드경로를 그대로 재실행**한다:

- CachedKakaoGenerator 가 get_road_distance_kakao 를 캐시 우선으로 오버라이드.
  캐시 = 구 routes/{center2site,hos2site}/*.json(payload 보존·이름 키)
       + 구 hospital_info/amb_station_info CSV(정밀 거리값 오버레이).
- 안전센터 후보·유지 병원 = 구본과 동일 → 전부 캐시 히트(호출 0회).
  **신규 진입 병원(대부분 성남시의료원 1곳)만 실제 Kakao 호출**(좌표당 ~1회).
- 스냅/OSRM폴백 이력 레그도 이름 키로 히트(구 결과값 그대로) → 재실패 안 함.
- 출력물(hospital_info/uav_info/H2H/routes/config)은 생성기가 정식 재작성
  → 인덱스 재정렬·포맷 일치 자동 보장. OSRM 세트와 병원집합 동일(선정은 API 무관).

대상: 시군구 250(sigungu_kakao_manifest, _dep 접미사 rename 후처리 재사용)
      + 시도 17(plan1_manifest, _dep 경로 유지). 백업 kakao_pre20260702 필요.
재개: experiment_logs/patch_helipad_kakao_state.json (키별 완료 기록, --force 재실행).

예: PYTHONIOENCODING=utf-8 python src/sce_src/patch_helipad_kakao.py \
      --keys_file ~/.kakao_keys.txt --sets sigungu,sido
"""
import argparse
import copy
import glob
import io
import json
import os
import re
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
for d in (THIS_DIR, os.path.join(REPO, "src", "rl_src")):
    if d not in sys.path:
        sys.path.insert(0, d)

import pandas as pd

from make_csv_yaml_dynamic import ScenarioGenerator, slugify, ensure_dir
from gen_sigungu_kakao import load_keys, is_quota_error, postprocess_rename, _norm_name

OSRM_URL = os.environ.get("MCI_OSRM_URL", "http://127.0.0.1:5000")
DEP = "202607301400"  # 두 Kakao 세트 공통 departure_time (구 config 실측)
PARAMS = dict(incident_size=100, amb_count=30, uav_count=25, amb_velocity=50,
              uav_velocity=200, amb_handover_time=5.0, uav_handover_time=10.0,
              total_samples=1000, random_seed=0, uav_num=25)
FIXED_HOS_NUM = 46
STATE_PATH = os.path.join(REPO, "experiment_logs", "patch_helipad_kakao_state.json")


# ---------------------------------------------------------------- 캐시
class RouteCache:
    """구 좌표 디렉터리의 도로 결과를 (route_type, name) 키로 서빙.

    값 우선순위: 구 CSV 정밀값(road_dist/duration) > routes json meta(반올림값).
    payload(카카오/OSRM 원응답)는 json에서 보존해 신규 routes 재저장에 사용.
    """

    def __init__(self, old_coord_dir):
        self.entries = {"center2site": {}, "hos2site": {}}
        self.hits = 0
        for rt in ("center2site", "hos2site"):
            for fp in glob.glob(os.path.join(old_coord_dir, "routes", rt, "*.json")):
                try:
                    with open(fp, encoding="utf-8") as f:
                        j = json.load(f)
                    name = str(j["meta"].get("name"))
                    self.entries[rt].setdefault(name, []).append(j)
                except Exception:
                    continue
        # CSV 정밀값 오버레이 (이름 → (dist_km, dur_min))
        self.exact = {"center2site": {}, "hos2site": {}}
        hp = os.path.join(old_coord_dir, "hospital_info.csv")
        if os.path.exists(hp):
            df = pd.read_csv(hp, encoding="utf-8-sig")
            for _, r in df.iterrows():
                self.exact["hos2site"][str(r["요양기관명"])] = (
                    float(r["road_dist"]), float(r["road_duration"]))
        ap = os.path.join(old_coord_dir, "amb_station_info.csv")
        if os.path.exists(ap):
            df = pd.read_csv(ap, encoding="utf-8-sig")
            names = df["안전센터/소방서이름"].astype(str)
            if names.is_unique:  # 동명 센터가 있으면 오버레이 생략(json 값 사용)
                for _, r in df.iterrows():
                    self.exact["center2site"][str(r["안전센터/소방서이름"])] = (
                        float(r["init_distance"]), float(r["duration"]))

    def lookup(self, route_type, name):
        """(dist_km, dur_min, old_json|None) 또는 None."""
        lst = self.entries.get(route_type, {}).get(str(name))
        if not lst:
            return None
        j = lst[0]  # 이름은 세트 내 유일(병원 dedup·센터 실측 유일)
        meta = j.get("meta", {})
        d, t = meta.get("distance_km"), meta.get("duration_min")
        ex = self.exact.get(route_type, {}).get(str(name))
        if ex:
            d, t = ex
        if d is None or t is None:
            return None
        self.hits += 1
        return float(d), float(t), j


class CachedKakaoGenerator(ScenarioGenerator):
    """도로호출을 RouteCache 우선으로 서빙하는 ScenarioGenerator (Kakao 모드 전용)."""

    def attach_cache(self, cache: RouteCache):
        self._patch_cache = cache
        self.real_calls = []  # (route_type, name)

    def get_road_distance_kakao(self, start, end, max_retries=3, save_json_dir=None,
                                route_type=None, source_index=None, name=None,
                                start_label="start", goal_label="goal"):
        ent = getattr(self, "_patch_cache", None) and \
            self._patch_cache.lookup(route_type, name)
        if ent:
            dist_km, dur_min, old_json = ent
            if save_json_dir and old_json is not None:
                jj = copy.deepcopy(old_json)
                jj.setdefault("meta", {})["source_index"] = source_index
                jj["meta"]["patched_from_backup"] = True
                fname = f"{(source_index if source_index is not None else 0):03d}_{slugify(name)}.json"
                ensure_dir(save_json_dir)
                with open(os.path.join(save_json_dir, fname), "w", encoding="utf-8") as f:
                    json.dump(jj, f, ensure_ascii=False, indent=2)
            return dist_km, dur_min
        self.real_calls.append((route_type, str(name)))
        return super().get_road_distance_kakao(
            start, end, max_retries=max_retries, save_json_dir=save_json_dir,
            route_type=route_type, source_index=source_index, name=name,
            start_label=start_label, goal_label=goal_label)


# ---------------------------------------------------------------- 세트 정의
def parse_coord(cfg_path):
    m = re.search(r"\(([-\d.]+),([-\d.]+)\)", cfg_path)
    if not m:
        raise ValueError(f"좌표 파싱 실패: {cfg_path}")
    return m.group(1), m.group(2)


def build_jobs(sets):
    """[(set_name, key, exp_id, lat, lon, old_coord_dir, final_cfg, postprocess_name|None)]"""
    jobs = []
    if "sigungu" in sets:
        with open(os.path.join(REPO, "scenarios", "manifests", "sigungu_kakao_manifest.json"),
                  encoding="utf-8") as f:
            man = json.load(f)
        for key, cfg in man.items():
            lat, lon = parse_coord(cfg)
            name = key.rsplit("_", 1)[0]
            old_dir = os.path.dirname(cfg).replace("/kakao/", "/kakao_pre20260702/")
            jobs.append(("sigungu", key, f"시군구/kakao/{_norm_name(name)}_kakao",
                         lat, lon, old_dir, cfg, name))
    if "sido" in sets:
        with open(os.path.join(REPO, "scenarios", "manifests", "plan1_manifest.json"),
                  encoding="utf-8") as f:
            man = json.load(f)
        for region, cfg in man.items():
            lat, lon = parse_coord(cfg)
            old_dir = os.path.dirname(cfg).replace("/kakao/", "/kakao_pre20260702/")
            jobs.append(("sido", region, f"시도/kakao/exp_{region}",
                         lat, lon, old_dir, cfg, None))
    return jobs


# ---------------------------------------------------------------- 검증
def check_outputs(cfg_path):
    d = os.path.dirname(cfg_path)
    df = pd.read_csv(os.path.join(d, "hospital_info.csv"), encoding="utf-8-sig")
    n_hos, n_heli = len(df), int((df["헬기장 여부"] == 1).sum())
    has_sn = bool(df["요양기관명"].str.contains("성남시의료원").any())
    uav = pd.read_csv(os.path.join(d, "uav_info.csv"), encoding="utf-8-sig")
    sn_uav = bool(uav["요양기관명"].str.contains("성남시의료원").any())
    return n_hos, n_heli, len(uav), has_sn, sn_uav


def diff_sets(old_dir, new_cfg):
    old = pd.read_csv(os.path.join(old_dir, "hospital_info.csv"),
                      encoding="utf-8-sig")["요양기관명"].tolist()
    new = pd.read_csv(os.path.join(os.path.dirname(new_cfg), "hospital_info.csv"),
                      encoding="utf-8-sig")["요양기관명"].tolist()
    return sorted(set(new) - set(old)), sorted(set(old) - set(new))


# ---------------------------------------------------------------- 메인
def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state):
    ensure_dir(os.path.dirname(STATE_PATH))
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)


def patch_one(job, api_key):
    set_name, key, exp_id, lat, lon, old_dir, final_cfg, pp_name = job
    if not os.path.isdir(old_dir):
        return dict(key=key, ok=False, err=f"백업 없음: {old_dir}")
    cache = RouteCache(old_dir)
    gen = CachedKakaoGenerator(
        base_path=REPO, experiment_id=exp_id, kakao_api_key=api_key,
        departure_time=DEP, osrm_url=OSRM_URL, is_use_time=True,
        fixed_hos_num=FIXED_HOS_NUM)
    gen.attach_cache(cache)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cfg = gen.generate_scenario(latitude=float(lat), longitude=float(lon),
                                    is_use_time=True, **PARAMS)
        if pp_name is not None:  # 시군구: _dep 접미사 → _kakao 병합 후처리
            cfg = postprocess_rename(REPO, pp_name, DEP, lat, lon, None)
    n_hos, n_heli, n_uav, has_sn, sn_uav = check_outputs(cfg)
    entrants, leavers = diff_sets(old_dir, cfg)
    ok = (n_hos == FIXED_HOS_NUM and n_heli == PARAMS["uav_num"]
          and n_uav == PARAMS["uav_num"] and has_sn and sn_uav
          and os.path.abspath(cfg) == os.path.abspath(final_cfg))
    return dict(key=key, ok=ok, cfg=cfg, cache_hits=cache.hits,
                real_calls=gen.real_calls, entrants=entrants, leavers=leavers,
                err=None if ok else f"검증실패 hos{n_hos}/heli{n_heli}/uav{n_uav}/"
                                    f"성남{has_sn}/{sn_uav}/경로{cfg}")


def _pool_worker(arg):
    """Pool 워커 — 같은 exp_id(동명 구) 그룹을 순차 처리.

    동명 시군구(동구/서구/남구/중구 등)는 같은 원본 폴더(<name>_kakao_dep_<ts>)를
    공유하므로 병렬 생성 시 postprocess 의 rmtree 가 서로의 생성 중 파일을 지운다
    → 그룹 단위 직렬화로 원천 차단. quota 는 표식으로 반환(상태 기록은 부모가).
    """
    job_group, api_key = arg
    out = []
    for job in job_group:
        try:
            out.append(patch_one(job, api_key))
        except RuntimeError as e:
            out.append(dict(key=job[1], ok=False,
                            quota=is_quota_error(str(e)), err=str(e)[:200]))
        except Exception as e:
            out.append(dict(key=job[1], ok=False, err=str(e)[:200]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys_file", default=os.path.expanduser("~/.kakao_keys.txt"))
    ap.add_argument("--sets", default="sigungu,sido")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8,
                    help="병렬 좌표 수(좌표당 마스터 엑셀 로드가 지배 비용, 실호출은 ~1회)")
    ap.add_argument("--force", action="store_true", help="완료 기록 무시하고 재실행")
    args = ap.parse_args()

    keys = load_keys(args.keys_file)
    if not keys:
        sys.exit("키 파일에 유효한 키 없음")
    jobs = build_jobs(args.sets.split(","))
    if args.limit:
        jobs = jobs[:args.limit]
    state = load_state()
    todo = {j[1]: j for j in jobs if args.force or j[1] not in state}
    print(f"[patch] 대상 {len(jobs)}좌표 (skip {len(jobs)-len(todo)}, "
          f"키 {len(keys)}개, workers={args.workers})", flush=True)

    from multiprocessing import Pool
    t0, n_real_total, odd, fails = time.time(), 0, [], []
    for ki, api_key in enumerate(keys):  # quota 실패분은 다음 키로 라운드 재시도
        if not todo:
            break
        if ki > 0:
            print(f"  🔑 키 전환 #{ki} — 잔여 {len(todo)}좌표 재시도", flush=True)
        groups = {}
        for j in todo.values():
            groups.setdefault(j[2], []).append(j)  # exp_id 단위 그룹(동명 구 직렬화)
        batch = list(groups.values())
        n_jobs = sum(len(g) for g in batch)
        fails, i = [], 0
        with Pool(args.workers, maxtasksperchild=4) as pool:
            for res_group in pool.imap_unordered(
                    _pool_worker, [(g, api_key) for g in batch]):
              for res in res_group:
                i += 1
                if res["ok"]:
                    state[res["key"]] = res["cfg"]
                    save_state(state)
                    todo.pop(res["key"], None)
                    n_real = len(res["real_calls"])
                    n_real_total += n_real
                    if res["entrants"] != ["성남시의료원"] or n_real > 2:
                        odd.append((res["key"], res["entrants"], res["leavers"], n_real))
                    if i % 10 == 0 or n_real > 2:
                        print(f"  [{i}/{n_jobs}] {res['key']} OK 진입={res['entrants']} "
                              f"실호출={n_real} ({time.time()-t0:.0f}s)", flush=True)
                else:
                    fails.append(res)
                    print(f"  [{i}/{n_jobs}] {res['key']} FAIL"
                          f"{'(quota)' if res.get('quota') else ''}: {res['err']}", flush=True)
        if not any(f.get("quota") for f in fails):
            break  # quota 아닌 실패만 남음 → 키 전환 무의미

    done = sum(1 for j in jobs if j[1] in state)
    print(f"\n[patch] 완료 {done}/{len(jobs)}, 실호출 총 {n_real_total}, "
          f"wall={time.time()-t0:.0f}s", flush=True)
    if odd:
        print(f"  ⚠️ 비정형(진입≠성남 단독 또는 실호출>2) {len(odd)}건:", flush=True)
        for k, e, l, n in odd[:30]:
            print(f"    {k}: +{e} -{l} 호출{n}", flush=True)
    if done < len(jobs):
        print("  잔여 실패:", [(f['key'], f['err']) for f in fails][:20], flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
