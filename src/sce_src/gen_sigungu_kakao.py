# -*- coding: utf-8 -*-
"""시군구 250개 Kakao 시나리오 일괄 생성 (시군구 OSRM 짝).

기존 시군구 OSRM 시나리오(scenarios/exp_시군구/osrm/<name>_osrm/(lat,lon)/)의
**Kakao 라우팅 짝**을 만든다 — 동일 좌표·동일 병원선정, is_use_time=True(Kakao 교통시간).
좌표·이름은 scenarios/manifests/sigungu_osrm_manifest.json 에서 그대로 재사용한다
(동일 좌표여야 OSRM↔Kakao 라우팅축 비교쌍이 성립).

출력 구조 (OSRM 의 _osrm 접미사에 대응하는 _kakao 접미사):
  scenarios/exp_시군구/kakao/<name>_kakao/(lat,lon)/config_*.yaml
  scenarios/manifests/sigungu_kakao_manifest.json  (키=<이름>_<SIGCD>, 값=config 경로)

★ 출력 폴더명 후처리:
  make_csv_yaml_dynamic.py 는 kakao+departure_time 일 때 experiment_id 에 _dep_<ts> 를
  강제 부착한다(예 <name>_kakao_dep_202607301400). OSRM 의 _osrm 과 대칭인 _kakao 경로를
  얻기 위해, 생성 직후 폴더를 <name>_kakao 로 rename + config 내부 경로 문자열을 치환한다.
  (departure_time 은 Kakao API 호출에는 정상 사용되고 config 의 departure_time 필드에도 남는다.)

★ 병원선정 일치 (중요):
  OSRM 시군구 config 는 uav_num=25 로 생성됨(헬기장 병원 25곳 보장 → 전국 헬기장 끌어옴).
  동일 병원 집합을 얻으려면 Kakao 도 --uav_num 25 여야 한다(검증 완료: 종로구 46곳·헬기장 25곳
  완전 일치). 사용자 task 의 --uav_num 3 은 OSRM 실제값과 불일치하므로 기본값을 25 로 둔다.

★ 키 로테이션 / quota:
  --keys_file 의 비주석 줄을 순서대로 사용(첫 키=주키). Kakao 일일 quota 소진(429 재시도초과
  또는 'API limit'/'할당량' 에러) 감지 시 다음 키로 전환. 모든 키 소진 시 graceful 중단
  (재개 가능 — 일일한도는 시간 지나면 초기화). 키값은 로그에 마스킹.

★ 비용/QPS:
  H2H 는 마스터 엑셀(DISTANCE_MATRIX_FINAL.xlsx) 재사용 → Kakao 호출 0회.
  좌표당 호출 ≈ 안전센터 후보(~100-110) + 병원 46 ≈ 150여 회.
  Kakao QPS rate-limit 회피 위해 좌표 간 순차(저병렬). make_csv 내부에 호출당 sleep(0.05) 있음.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MAKE_SCRIPT = os.path.join(THIS_DIR, "make_csv_yaml_dynamic.py")


def mask_key(k: str) -> str:
    if not k:
        return "(none)"
    return f"{k[:2]}****(len={len(k)})"


def load_keys(keys_file: str):
    """비주석·비공백 줄만 키로 로드 (순서 유지)."""
    keys = []
    with open(keys_file, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            keys.append(s)
    return keys


def parse_manifest(manifest_path: str):
    """OSRM 매니페스트 → [(manifest_key, name, lat, lon)] 리스트.

    manifest_key = '<이름>_<SIGCD>' (예 종로구_11110). name = SIGCD 앞부분.
    좌표는 config 경로의 (lat,lon) 에서 추출.
    """
    with open(manifest_path, encoding="utf-8") as f:
        d = json.load(f)
    out = []
    for key, cfg_path in d.items():
        m = re.search(r"\(([-\d.]+),([-\d.]+)\)", cfg_path)
        if not m:
            raise ValueError(f"좌표 파싱 실패: {key} -> {cfg_path}")
        lat, lon = m.group(1), m.group(2)
        name = key.rsplit("_", 1)[0]  # '종로구_11110' -> '종로구'
        out.append((key, name, lat, lon))
    return out


def is_quota_error(text: str) -> bool:
    """Kakao 일일 quota 소진 / rate-limit 영구실패 신호 감지."""
    if not text:
        return False
    t = text.lower()
    signals = [
        "api limit has been exceeded",
        "일일 할당량",
        "할당량이 소진",
        "호출 한도 초과",
        "(429)",
        "status 429",
        "quota",
        "-10",  # kakao code -10 (limit exceeded)
        "인증 실패",  # 401 → 잘못된 키. 다음 키로 전환 가치 있음
        "(401)",
    ]
    return any(s in t for s in signals)


def final_config_path(base, name, lat, lon):
    """후처리(_kakao) 후 최종 config 경로."""
    folder = f"({lat},{lon})"
    return os.path.join(base, "scenarios", "exp_시군구", "kakao",
                        f"{name}_kakao", folder, f"config_{folder}.yaml")


def raw_exp_dir(base, name, dep):
    """make_csv 가 만드는 원본 폴더(_dep_ 접미사) 경로."""
    return os.path.join(base, "scenarios", "exp_시군구", "kakao",
                        f"{name}_kakao_dep_{dep}")


def postprocess_rename(base, name, dep, lat, lon, log):
    """원본 <name>_kakao_dep_<ts> → <name>_kakao 로 rename + config 내부 문자열 치환.

    config 안의 모든 경로/output_path 가 '<name>_kakao_dep_<ts>' 를 포함하므로
    '<name>_kakao' 로 전역 치환한다(experiment_id 고유 문자열이라 안전).
    Returns: 최종 config 경로.
    """
    raw_dir = raw_exp_dir(base, name, dep)
    final_dir = os.path.join(base, "scenarios", "exp_시군구", "kakao", f"{name}_kakao")
    old_token = f"{name}_kakao_dep_{dep}"
    new_token = f"{name}_kakao"

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"원본 생성 폴더 없음: {raw_dir}")

    # rename (이미 최종 폴더가 있으면 원본을 그 안으로 병합하지 않고 교체 — 재시도 안전)
    if os.path.isdir(final_dir):
        shutil.rmtree(final_dir)
    shutil.move(raw_dir, final_dir)

    # config 내부 문자열 치환
    folder = f"({lat},{lon})"
    cfg = os.path.join(final_dir, folder, f"config_{folder}.yaml")
    with open(cfg, encoding="utf-8") as f:
        text = f.read()
    if old_token in text:
        text = text.replace(old_token, new_token)
        with open(cfg, "w", encoding="utf-8") as f:
            f.write(text)
    return cfg


def gen_one(base, name, lat, lon, dep, api_key, args, log):
    """단일 좌표 생성 → make_csv subprocess 호출 → rename 후처리.

    Returns: ("ok", final_cfg) | ("quota", err_tail) | ("fail", err_tail)
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = "src/rl_src:src/sim_src"
    # 단일코어 핀 (공유 서버 CPU 과점 방지)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[v] = "1"
    # 키는 ENV 로 전달 (argv 노출 회피)
    env["KAKAO_API_KEY"] = api_key

    cmd = [
        args.python, MAKE_SCRIPT,
        "--base_path", ".",
        "--experiment_id", f"시군구/kakao/{name}_kakao",
        "--latitude", lat, "--longitude", lon,
        "--incident_size", str(args.incident_size),
        "--amb_count", str(args.amb_count),
        "--uav_count", str(args.uav_count),
        "--uav_num", str(args.uav_num),
        "--fixed_hos_num", str(args.fixed_hos_num),
        "--amb_velocity", str(args.amb_velocity),
        "--uav_velocity", str(args.uav_velocity),
        "--amb_handover_time", str(args.amb_handover_time),
        "--uav_handover_time", str(args.uav_handover_time),
        "--is_use_time", "True",
        "--departure_time", dep,
        # 키는 ENV(KAKAO_API_KEY) fallback 으로 전달 (argv 미노출)
    ]
    try:
        p = subprocess.run(cmd, env=env, cwd=base, capture_output=True,
                           text=True, timeout=args.per_coord_timeout)
    except subprocess.TimeoutExpired:
        return ("fail", f"timeout>{args.per_coord_timeout}s")

    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        tail = out.strip().splitlines()[-8:]
        tail_s = " | ".join(tail)
        if is_quota_error(out):
            return ("quota", tail_s)
        return ("fail", tail_s)

    # 성공 → rename 후처리
    try:
        final_cfg = postprocess_rename(base, name, dep, lat, lon, log)
    except Exception as e:
        return ("fail", f"postprocess: {e}")
    # API 호출 수 추출(로그용)
    n_calls = None
    mcall = re.search(r"API_CALL_COUNT:(\d+)", out)
    if mcall:
        n_calls = int(mcall.group(1))
    return ("ok", (final_cfg, n_calls))


def main():
    ap = argparse.ArgumentParser(description="시군구 250 Kakao 시나리오 일괄 생성 (재개가능·키로테이션)")
    ap.add_argument("--base_path", default=".")
    ap.add_argument("--python", default="/home/ryu/anaconda3/envs/UAV/bin/python")
    ap.add_argument("--osrm_manifest",
                    default="scenarios/manifests/sigungu_osrm_manifest.json")
    ap.add_argument("--out_manifest",
                    default="scenarios/manifests/sigungu_kakao_manifest.json")
    ap.add_argument("--keys_file", required=True, help="Kakao 키 파일(비주석 줄=키, 첫 줄=주키)")
    ap.add_argument("--departure_time", default="202607301400")
    ap.add_argument("--incident_size", type=int, default=100)
    ap.add_argument("--amb_count", type=int, default=30)
    ap.add_argument("--uav_count", type=int, default=25)
    ap.add_argument("--uav_num", type=int, default=25,
                    help="OSRM 시군구는 uav_num=25 로 생성됨(헬기장 25곳 보장). 동일 병원선정 위해 25 권장")
    ap.add_argument("--fixed_hos_num", type=int, default=46)
    ap.add_argument("--amb_velocity", type=int, default=50)
    ap.add_argument("--uav_velocity", type=int, default=200)
    ap.add_argument("--amb_handover_time", type=float, default=5.0)
    ap.add_argument("--uav_handover_time", type=float, default=10.0)
    ap.add_argument("--per_coord_timeout", type=int, default=900,
                    help="좌표당 make_csv subprocess 타임아웃(초)")
    ap.add_argument("--sleep_between", type=float, default=1.0,
                    help="좌표 간 대기(초, QPS 완화)")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N개만(테스트용)")
    ap.add_argument("--only", nargs="+", default=None,
                    help="특정 manifest_key 만(예: 종로구_11110)")
    ap.add_argument("--log_dir", default="experiment_logs")
    args = ap.parse_args()

    base = os.path.abspath(args.base_path)
    os.makedirs(os.path.join(base, args.log_dir), exist_ok=True)
    log_path = os.path.join(base, args.log_dir, "gen_sigungu_kakao.log")
    state_path = os.path.join(base, args.log_dir, "gen_sigungu_kakao_state.json")

    def log(msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    keys = load_keys(args.keys_file)
    if not keys:
        log("❌ 키가 없습니다. 중단.")
        sys.exit(1)
    log(f"키 {len(keys)}개 로드: " + ", ".join(mask_key(k) for k in keys))

    targets = parse_manifest(os.path.join(base, args.osrm_manifest))
    if args.only:
        targets = [t for t in targets if t[0] in set(args.only)]
    if args.limit:
        targets = targets[:args.limit]
    log(f"대상 {len(targets)}개 (departure_time={args.departure_time}, "
        f"uav_num={args.uav_num}, fixed_hos_num={args.fixed_hos_num})")

    # 기존 매니페스트 로드(누적)
    out_manifest_path = os.path.join(base, args.out_manifest)
    manifest = {}
    if os.path.exists(out_manifest_path):
        with open(out_manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

    key_idx = 0
    done = 0
    skipped = 0
    failed = []
    t0 = time.time()

    for i, (mkey, name, lat, lon) in enumerate(targets, 1):
        final_cfg = final_config_path(base, name, lat, lon)
        # skip: 최종 config 존재 시
        if os.path.exists(final_cfg):
            manifest[mkey] = final_cfg
            skipped += 1
            if skipped <= 5 or skipped % 25 == 0:
                log(f"[{i}/{len(targets)}] SKIP (이미 존재) {mkey}")
            continue

        # 키 소진 처리 루프
        result = None
        while key_idx < len(keys):
            cur_key = keys[key_idx]
            log(f"[{i}/{len(targets)}] {mkey} ({lat},{lon}) 생성 시작 "
                f"[key#{key_idx} {mask_key(cur_key)}]")
            status, payload = gen_one(base, name, lat, lon,
                                      args.departure_time, cur_key, args, log)
            if status == "ok":
                cfg, ncalls = payload
                manifest[mkey] = cfg
                done += 1
                log(f"[{i}/{len(targets)}] ✅ {mkey} 완료 (API≈{ncalls}) {cfg}")
                result = "ok"
                break
            elif status == "quota":
                log(f"[{i}/{len(targets)}] ⚠️ quota/키 소진 감지 [key#{key_idx} "
                    f"{mask_key(cur_key)}] → 다음 키로 전환. 사유: {payload}")
                key_idx += 1
                if key_idx >= len(keys):
                    log("🛑 모든 키 소진. graceful 중단 — 시간 지나 quota 초기화 후 동일 명령으로 재개 가능.")
                    result = "exhausted"
                    break
                # 다음 키로 같은 좌표 재시도
                continue
            else:  # fail (좌표/네트워크 문제 — 키 전환 안 함)
                log(f"[{i}/{len(targets)}] ❌ {mkey} 실패(키 무관): {payload}")
                failed.append(mkey)
                result = "fail"
                break

        # 매 좌표 후 매니페스트·상태 즉시 저장(재개 대비)
        os.makedirs(os.path.dirname(out_manifest_path), exist_ok=True)
        with open(out_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=False)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({
                "updated": datetime.now().isoformat(),
                "total": len(targets), "done": done, "skipped": skipped,
                "failed": failed, "active_key_idx": key_idx,
                "manifest_count": len(manifest),
            }, f, ensure_ascii=False, indent=2)

        if result == "exhausted":
            log(f"진행 요약: 신규완료 {done}, skip {skipped}, 실패 {len(failed)}, "
                f"매니페스트 {len(manifest)}/{len(targets)}")
            sys.exit(2)  # 재개 필요 신호

        if args.sleep_between > 0:
            time.sleep(args.sleep_between)

    log(f"🏁 전체 완료. 신규완료 {done}, skip {skipped}, 실패 {len(failed)}, "
        f"매니페스트 {len(manifest)}/{len(targets)}, wall={time.time()-t0:.0f}s")
    if failed:
        log(f"  실패 목록: {failed}")
    # 완료 마커
    with open(os.path.join(base, args.log_dir, "_sigungu_kakao_DONE.txt"), "w",
              encoding="utf-8") as f:
        f.write(f"done={done} skipped={skipped} failed={len(failed)} "
                f"manifest={len(manifest)}/{len(targets)}\n")


if __name__ == "__main__":
    main()
