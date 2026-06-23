"""17개 광역시도 Plan1 시나리오 일괄 생성 (Kakao API).

cross_location_eval.LOCATIONS 좌표를 단일 출처로 사용한다.
출력:
  scenarios/exp_<prefix>_<region>_dep_<departure_time>/(lat,lon)/config_*.yaml
  scenarios/manifests/plan1_manifest.json   (region -> config_path 매핑)

Kakao API rate limit 을 고려해 지역을 순차 생성한다.
키는 ENV KAKAO_API_KEY 에서 읽는다 (코드 하드코딩 금지).

예:
  python src/sce_src/gen_regions.py                 # 17개 전체
  python src/sce_src/gen_regions.py --regions 서울   # 1개만 (스모크 테스트)
"""
import argparse
import json
import os
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir, "rl_src"))
if RL_DIR not in sys.path:
    sys.path.insert(0, RL_DIR)

# LOCATIONS / gen_scenario_for_region / Pass1 헬퍼를 단일 출처(cross_location_eval)에서 가져온다.
from cross_location_eval import (LOCATIONS, gen_scenario_for_region,
                                 natural_hos_count_for_region)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--departure_time", default="202605261530",
                   help="Kakao 길찾기 출발시각 (12자리 YYYYMMDDHHMM)")
    p.add_argument("--incident_size", type=int, default=100)
    p.add_argument("--amb_count", type=int, default=30, help="AMB 런타임 대수(YAML amb_num)")
    p.add_argument("--uav_count", type=int, default=25, help="UAV 생성 superset 상한(헬기장 병원당 1대)")
    p.add_argument("--uav_num", type=int, default=3, help="UAV 런타임 대수(YAML uav_num)")
    p.add_argument("--amb_velocity", type=int, default=50)
    p.add_argument("--uav_velocity", type=int, default=200)
    p.add_argument("--amb_handover_time", type=float, default=5.0)
    p.add_argument("--uav_handover_time", type=float, default=10.0)
    p.add_argument("--total_samples", type=int, default=1000,
                   help="config YAML 의 totalSamples (휴리스틱/평가 ep 수)")
    p.add_argument("--exp_prefix", default="plan1")
    p.add_argument("--base_path", default=".")
    p.add_argument("--regions", nargs="+", default=None,
                   help="부분 생성 (기본: 17개 전체). 예: --regions 서울 부산")
    p.add_argument("--manifest", default="scenarios/manifests/plan1_manifest.json")
    p.add_argument("--fixed_hos_num", type=int, default=None,
                   help="[구호환] 모든 지역 hos_num cap (가까운 N개로 잘라냄). min 모드와 동시 사용 불가")
    p.add_argument("--min_hos_mode", choices=["auto", "fixed", "off"], default="auto",
                   help="auto: Pass1로 H_max 자동산출 후 floor(권장). fixed: --min_hos_num 값 floor. off: floor 안 함(동적)")
    p.add_argument("--min_hos_num", type=int, default=None,
                   help="min_hos_mode=fixed 일 때 floor 값")
    p.add_argument("--max_hos_cap", type=int, default=None,
                   help="auto 모드 H_max 상한 안전장치(초과 시 클램프)")
    return p.parse_args()


def main():
    args = parse_args()
    base = os.path.abspath(args.base_path)

    if not os.environ.get("KAKAO_API_KEY"):
        raise SystemExit(
            "환경변수 KAKAO_API_KEY 가 필요합니다. "
            "예: export KAKAO_API_KEY=<your_key>")

    locations = LOCATIONS if not args.regions \
        else [l for l in LOCATIONS if l[0] in args.regions]
    print(f"[gen_regions] {len(locations)}개 지역 시나리오 생성 "
          f"(departure_time={args.departure_time}, exp_prefix={args.exp_prefix})")

    # ===== Pass1: H_max 산출 (road API 0회) → Pass2 min_hos_num 결정 =====
    min_hos_num = None
    if args.fixed_hos_num is not None:
        if args.min_hos_mode != "off":
            print(f"[gen_regions] --fixed_hos_num={args.fixed_hos_num} 지정 → cap 모드 "
                  f"(min_hos_mode={args.min_hos_mode} 무시)")
    elif args.min_hos_mode == "auto":
        import statistics
        print(f"[Pass1] {len(locations)}개 지역 자연 선정 병원 수 산출 중 (road API 0회)...")
        natural, pool = {}, {}
        for short_name, full_name, lat, lon in locations:
            c, psz = natural_hos_count_for_region(
                short_name, lat, lon,
                incident_size=args.incident_size, uav_num=args.uav_num,
                base_path=base, exp_prefix=args.exp_prefix)
            natural[short_name] = c
            pool[short_name] = psz
        max_nat = max(natural.values())
        min_pool = min(pool.values())
        H_max = min(max_nat, min_pool)  # 풀<max(natural) 희소지역 보호(전국 풀이라 보통 비활성)
        if args.max_hos_cap is not None:
            H_max = min(H_max, args.max_hos_cap)
        min_hos_num = H_max
        med = int(statistics.median(natural.values()))
        argmax_region = max(natural, key=natural.get)
        print(f"[Pass1] natural 병원수 min/median/max = "
              f"{min(natural.values())}/{med}/{max_nat} (max: {argmax_region})")
        if max_nat > min_pool:
            print(f"[Pass1] ⚠️ 최소 풀({min_pool}) < max(natural)({max_nat}) → H_max를 {min_pool}로 클램프")
        if args.max_hos_cap is not None and min(max_nat, min_pool) > args.max_hos_cap:
            print(f"[Pass1] ⚠️ --max_hos_cap={args.max_hos_cap} 적용 (일부 지역 cap-down 가능)")
        print(f"[Pass1] ★ H_max = {H_max} → Pass2 min_hos_num={H_max}")
        print(f"[Pass1] 지역별 natural: {natural}")
    elif args.min_hos_mode == "fixed":
        if args.min_hos_num is None:
            raise SystemExit("--min_hos_mode fixed 인데 --min_hos_num 미지정")
        min_hos_num = args.min_hos_num
        print(f"[gen_regions] min_hos_num={min_hos_num} (fixed floor)")
    # min_hos_mode == "off": min_hos_num=None (동적)

    manifest_path = os.path.join(base, args.manifest)
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

    t0 = time.time()
    ok, fail = [], []
    for i, (short_name, full_name, lat, lon) in enumerate(locations, 1):
        print(f"\n[{i}/{len(locations)}] {short_name} {full_name} ({lat},{lon})")
        ts = time.time()
        try:
            config_path = gen_scenario_for_region(
                short_name, lat, lon,
                incident_size=args.incident_size,
                amb_count=args.amb_count,
                uav_count=args.uav_count,
                uav_num=args.uav_num,
                amb_velocity=args.amb_velocity,
                uav_velocity=args.uav_velocity,
                amb_handover_time=args.amb_handover_time,
                uav_handover_time=args.uav_handover_time,
                total_samples=args.total_samples,
                base_path=base,
                exp_prefix=args.exp_prefix,
                is_use_time=True,
                osrm_url=None,
                kakao_api_key=None,  # ENV(KAKAO_API_KEY) fallback
                departure_time=args.departure_time,
                fixed_hos_num=args.fixed_hos_num,
                min_hos_num=min_hos_num,
            )
            manifest[short_name] = config_path
            ok.append(short_name)
            print(f"  ✅ {short_name}  ({time.time()-ts:.0f}s)  {config_path}")
        except Exception as e:
            fail.append(short_name)
            print(f"  ❌ {short_name} 생성 실패: {e}")

        # 중단 대비 — 매 지역 후 manifest 즉시 저장
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[gen_regions] 완료. 성공 {len(ok)}  실패 {len(fail)}  "
          f"wall={time.time()-t0:.0f}s")
    if fail:
        print(f"  실패 지역: {fail}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
