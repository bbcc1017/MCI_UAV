"""plan1nat 일반화 평가용 시나리오 일괄 생성 (피드백 3).

sample_region_points.py 가 뽑은 지역별 랜덤 평가점(plan1nat_eval_points.json)
각각에 대해 Kakao API 시나리오를 생성한다. 학습된 전국 단일 정책(plan1nat_f3)
을 학습 좌표가 아닌 hold-out 지점에서 평가하기 위한 것.

핵심 제약:
  - fixed_hos_num 은 학습(plan1nat)과 동일해야 obs 차원이 맞다 (기본 46).
  - 생성 후 hospital 수가 fixed_hos_num 과 다르면(원격 지점에서 풀 부족 등)
    그 점을 버리고 같은 지역 폴리곤에서 새 점을 재추출해 재시도한다.
  - Kakao 경로탐색 실패(해상/도로망 밖 등)도 재추출 사유.

출력:
  scenarios/exp_<prefix>_<region>_dep_<dep>/(lat,lon)/config_*.yaml
  scenarios/plan1nat_eval_manifest.json   {region: [config_path, ...]}

키는 ENV KAKAO_API_KEY. 예:
  python src/sce_src/gen_eval_points.py --fixed_hos_num 46
"""
import argparse
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
RL_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir, "rl_src"))
for d in (THIS_DIR, RL_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import numpy as np
import shapefile
from pyproj import Transformer

from cross_location_eval import gen_scenario_for_region
from sample_region_points import KOR_TO_SHORT, _shape_rings, _polygon_contains


def count_hospitals(config_path):
    """생성된 config 의 hospital info(hospital_info.csv) 병원 수."""
    import yaml
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    hpath = cfg["entity_info"]["hospital"]["info_path"]
    if not os.path.isabs(hpath):
        hpath = os.path.join(REPO, hpath)
    with open(hpath, encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def load_region_shapes(shp_path):
    """{short_name: (rings, bbox)} — 실패 시 폴리곤 재추출용."""
    sf = shapefile.Reader(shp_path, encoding="cp949")
    fields = [f[0] for f in sf.fields[1:]]
    kor_idx = fields.index("CTP_KOR_NM")
    out = {}
    for sr in sf.shapeRecords():
        short = KOR_TO_SHORT.get(sr.record[kor_idx].strip())
        if short:
            out[short] = (_shape_rings(sr.shape), sr.shape.bbox)
    return out


def resample_point(region_shape, rng, transformer):
    rings, (xmin, ymin, xmax, ymax) = region_shape
    for _ in range(100000):
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        if _polygon_contains((x, y), rings):
            lon, lat = transformer.transform(x, y)
            return [round(lat, 6), round(lon, 6)]
    raise RuntimeError("재추출 실패")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--points", default=os.path.join(REPO, "scenarios",
                                                    "manifests", "plan1nat_eval_points.json"))
    p.add_argument("--shp", default=os.path.join(REPO, "scenarios", "ctprvn.shp"))
    p.add_argument("--manifest", default=os.path.join(REPO, "scenarios",
                                                      "plan1nat_eval_manifest.json"))
    p.add_argument("--exp_prefix", default="plan1nat_eval")
    p.add_argument("--departure_time", default="202605261530")
    p.add_argument("--fixed_hos_num", type=int, default=46,
                   help="학습(plan1nat)과 동일해야 함 — obs 차원 일치")
    p.add_argument("--incident_size", type=int, default=100)
    p.add_argument("--amb_count", type=int, default=30)
    p.add_argument("--uav_count", type=int, default=25)
    p.add_argument("--amb_velocity", type=int, default=40)
    p.add_argument("--uav_velocity", type=int, default=80)
    p.add_argument("--amb_handover_time", type=float, default=10.0)
    p.add_argument("--uav_handover_time", type=float, default=15.0)
    p.add_argument("--total_samples", type=int, default=1000)
    p.add_argument("--max_retry", type=int, default=8,
                   help="점당 재추출 재시도 한도")
    p.add_argument("--regions", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=90909)
    return p.parse_args()


def collect_keys():
    """환경변수에서 Kakao 키를 모은다 — KAKAO_API_KEY, KAKAO_API_KEY_2, _3, ...

    여러 키를 순환 사용하면 키 1개당 일일 할당량(~24 시나리오)을 합산할 수 있다.
    """
    keys, seen = [], set()
    for name in ["KAKAO_API_KEY"] + [f"KAKAO_API_KEY_{i}" for i in range(2, 11)]:
        v = os.environ.get(name)
        if v and v not in seen:
            keys.append(v)
            seen.add(v)
    return keys


def gen_one(region, lat, lon, args, key):
    """단일 점 시나리오 생성 → config_path. 예외는 호출부에서 처리."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        cfg = gen_scenario_for_region(
            region, lat, lon,
            incident_size=args.incident_size, amb_count=args.amb_count,
            uav_count=args.uav_count, amb_velocity=args.amb_velocity,
            uav_velocity=args.uav_velocity,
            amb_handover_time=args.amb_handover_time,
            uav_handover_time=args.uav_handover_time,
            total_samples=args.total_samples, base_path=REPO,
            exp_prefix=args.exp_prefix, is_use_time=True, osrm_url=None,
            kakao_api_key=key, departure_time=args.departure_time,
            fixed_hos_num=args.fixed_hos_num)
    return cfg


def main():
    args = parse_args()
    keys = collect_keys()
    if not keys:
        raise SystemExit("환경변수 KAKAO_API_KEY(_2,_3,...) 필요.")
    print(f"[gen_eval_points] Kakao 키 {len(keys)}개 — 소진 시 순환")

    with open(args.points, encoding="utf-8") as f:
        points = json.load(f)
    if args.regions:
        points = {r: points[r] for r in args.regions if r in points}

    shapes = load_region_shapes(args.shp)
    rng = np.random.default_rng(args.seed)
    transformer = Transformer.from_crs(5179, 4326, always_xy=True)

    # manifest 구조: {region: {slot_str: config_path}} — slot 단위 resume.
    # (좌표를 키로 쓰지 않는 것은 실패 시 재추출로 좌표가 바뀌기 때문)
    manifest = {}
    if os.path.exists(args.manifest):
        with open(args.manifest, encoding="utf-8") as f:
            manifest = json.load(f)

    t0 = time.time()
    total_ok = total_fail = 0
    quota_hit = False
    key_idx = 0          # 현재 사용 중인 키 (소진 시 다음으로 순환)
    for region, pts in points.items():
        if quota_hit:
            break
        manifest.setdefault(region, {})
        for slot, pt in enumerate(pts):
            sk = str(slot)
            # 이미 생성됐고 파일도 존재하면 건너뜀 (할당량 리셋 후 이어받기)
            if sk in manifest[region] and os.path.exists(manifest[region][sk]):
                print(f"[{region} #{slot}] 이미 생성됨 — 건너뜀")
                continue
            lat, lon = pt
            ok = False
            resamples = 0
            while True:
                ts = time.time()
                try:
                    cfg = gen_one(region, lat, lon, args, keys[key_idx])
                    n_hos = count_hospitals(cfg)
                    if n_hos != args.fixed_hos_num:
                        raise RuntimeError(
                            f"hospital 수 {n_hos}≠{args.fixed_hos_num}")
                    manifest[region][sk] = cfg
                    print(f"[{region} #{slot}] ✅ ({lat},{lon}) "
                          f"hos={n_hos} 키{key_idx+1} ({time.time()-ts:.0f}s)")
                    ok = True
                    break
                except Exception as e:
                    # 키 할당량 소진 → 다음 키로 순환 (같은 점 재시도, 재추출 아님)
                    if "API limit has been exceeded" in str(e):
                        if key_idx + 1 < len(keys):
                            key_idx += 1
                            print(f"[{region} #{slot}] 🔑 키{key_idx} 소진 "
                                  f"→ 키{key_idx+1} 전환")
                            continue
                        print(f"[{region} #{slot}] ⛔ 전체 키({len(keys)}개) "
                              f"소진 — 중단 (리셋 후 재실행 시 이어받음)")
                        quota_hit = True
                        break
                    # 좌표 불량(경로탐색 실패 등) → 같은 지역에서 재추출
                    resamples += 1
                    if resamples > args.max_retry:
                        break  # 한도 초과 — 아래 공통 실패 처리로
                    print(f"[{region} #{slot}] ⚠️ ({lat},{lon}) 실패: {e} "
                          f"— 재추출 ({resamples}/{args.max_retry})")
                    lat, lon = resample_point(shapes[region], rng, transformer)
            total_ok += ok
            total_fail += (not ok and not quota_hit)
            if not ok and not quota_hit:
                print(f"[{region} #{slot}] ❌ {args.max_retry}회 재시도 실패")
            # 중단 대비 즉시 저장
            with open(args.manifest, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            if quota_hit:
                break

    n_done = sum(len(v) for v in manifest.values())
    print(f"\n[gen_eval_points] 성공 {total_ok} 실패 {total_fail} "
          f"누적 {n_done}개  wall={time.time()-t0:.0f}s  manifest={args.manifest}")
    if quota_hit:
        print("⛔ 할당량 소진으로 조기 종료 — 리셋(매일 자정 KST) 후 같은 명령을 "
              "다시 실행하면 남은 점만 이어서 생성합니다.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
