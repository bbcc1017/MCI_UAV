"""v6 Track A5: 규모·자원 도메인 랜덤화 학습 매니페스트 생성기.

기존 학습 매니페스트(dict[지역, config경로])에 지역당 N 개의 '변형' 엔트리를 덧붙인다.
변형은 같은 시나리오 config 를 재사용하되, 자원·규모 노브를 빌드시 env 로 주입하는
overrides(FeatureMultiRegionEnv 신 스키마)를 갖는다. 학습 시 reset() 마다 지역×변형을
무작위 샘플 → 도메인 랜덤화(규모·자원 분포에 강건한 단일 정책).

축 값의 출처(CLAUDE.md):
  * 자원·부하 트레이드오프 런타임 노브(시나리오 재생성 불요, 빌드시 env 소비):
      MCI_INCIDENT_SIZE  — 사고규모(부하). 저부하 단일 규모(용량 ~6~15배)만 학습하면 규모
                           일반화가 약함 → 규모 혼재 학습(보상 pdrwog 는 규모 불변 0~1 이라 안전).
                           "중스트레스 regime N200~350 서 RL 격차 폭발" → 50~300 범위.
      MCI_CAPA_SCALE     — 수술실수·병상수·max_send ×s(병원당 min1). s<1 이면 용량 게이트가
                           바인딩되어 목적지 분산(부하균형)의 가치가 드러남. {0.5, 1.0}.
      MCI_AMB_NUM        — AMB 대수(풀에서 도로소요 오름차순 슬라이스). {10, 20, 30}.
      MCI_UAV_NUM        — UAV 출발지 대수(병원집합·헬기장 26 불변, 출발지만 슬라이스). {5, 13, 26}.
  * amb=0/uav=0 은 action(mode 축) 차원이 갈리므로 제외(양쪽 ≥1).

기본 조합(incident=100, amb=30, uav=26, capa=1.0)=base 시나리오 파라미터와 동일 →
뽑히면 재샘플(base 엔트리와의 중복 방지). 시드 고정 → 재현성.

예:
  python src/rl_src/gen_randomized_manifest.py \\
    --base scenarios/manifests/sigungu_mixed47_manifest.json \\
    --out  scenarios/manifests/sigungu_mixed47_rand_manifest.json \\
    --variants_per_region 1 --seed 0
"""
import argparse
import json
import os
import random
from collections import Counter

# 축별 후보값 (CLAUDE.md 자원·부하 트레이드오프 노브 + 중스트레스 regime N200~350).
_INCIDENT = [50, 100, 200, 300]
_AMB = [10, 20, 30]
_UAV = [5, 13, 26]
_CAPA = [0.5, 1.0]
# base 시나리오 기본값 조합(시군구/시도 고정 파라미터) — 이 조합은 base 엔트리와 중복.
_DEFAULT = (100, 30, 26, 1.0)
_OVERRIDE_KEYS = {"MCI_INCIDENT_SIZE", "MCI_AMB_NUM", "MCI_UAV_NUM", "MCI_CAPA_SCALE"}


def _sample_variant(rng: random.Random):
    """(incident, amb, uav, capa) 축별 독립 샘플. 기본조합이면 재샘플(중복 방지)."""
    while True:
        combo = (rng.choice(_INCIDENT), rng.choice(_AMB),
                 rng.choice(_UAV), rng.choice(_CAPA))
        if combo != _DEFAULT:
            return combo


def build_randomized(base: dict, variants_per_region: int, seed: int) -> dict:
    """base 매니페스트(dict[지역, 경로str]) → base 전부 보존 + 지역당 변형 엔트리 부착.

    변형 키=`<지역>_v<i>`, 값=dict{"path": <base 경로>, "overrides": {노브 env}}.
    시드 고정 → 재현성(같은 base·vpr·seed → 동일 dict).
    """
    if variants_per_region < 0:
        raise ValueError(f"variants_per_region >=0 필요 (got {variants_per_region})")
    rng = random.Random(seed)
    out = dict(base)  # base 엔트리(문자열 값) 그대로 보존 — 구 스키마 불변
    for region, path in base.items():
        if not isinstance(path, str):
            raise ValueError(f"base 매니페스트 값이 문자열이 아님 [{region}]: "
                             f"{type(path).__name__} — 이미 변형된 매니페스트에 재적용?")
        for vi in range(variants_per_region):
            inc, amb, uav, capa = _sample_variant(rng)
            out[f"{region}_v{vi}"] = {
                "path": path,
                "overrides": {
                    "MCI_INCIDENT_SIZE": inc,
                    "MCI_AMB_NUM": amb,
                    "MCI_UAV_NUM": uav,
                    "MCI_CAPA_SCALE": capa,
                },
            }
    return out


def _validate(base: dict, out: dict, variants_per_region: int) -> None:
    """산출 매니페스트 스키마·불변식 self-check(생성 직후)."""
    exp = len(base) * (1 + variants_per_region)
    assert len(out) == exp, f"엔트리수 {len(out)} != 기대 {exp}"
    for region, path in base.items():                       # base 보존
        assert out[region] == path, f"base 엔트리 훼손 [{region}]"
    n_var = 0
    for k, v in out.items():
        if not isinstance(v, dict):
            continue                                        # base(문자열)
        n_var += 1
        assert set(v.keys()) == {"path", "overrides"}, f"변형 스키마 오류 [{k}]: {set(v.keys())}"
        assert isinstance(v["path"], str), f"변형 path 비문자열 [{k}]"
        ov = v["overrides"]
        assert set(ov.keys()) == _OVERRIDE_KEYS, f"변형 overrides 키 오류 [{k}]: {set(ov.keys())}"
        assert int(ov["MCI_AMB_NUM"]) >= 1 and int(ov["MCI_UAV_NUM"]) >= 1, f"amb/uav <1 [{k}]"
        combo = (ov["MCI_INCIDENT_SIZE"], ov["MCI_AMB_NUM"],
                 ov["MCI_UAV_NUM"], ov["MCI_CAPA_SCALE"])
        assert combo != _DEFAULT, f"기본조합 변형 존재(중복) [{k}]"
    assert n_var == len(base) * variants_per_region, \
        f"변형수 {n_var} != {len(base) * variants_per_region}"


def _print_axis_hist(out: dict) -> None:
    ci, ca, cu, cc = Counter(), Counter(), Counter(), Counter()
    for v in out.values():
        if isinstance(v, dict):
            o = v["overrides"]
            ci[o["MCI_INCIDENT_SIZE"]] += 1
            ca[o["MCI_AMB_NUM"]] += 1
            cu[o["MCI_UAV_NUM"]] += 1
            cc[o["MCI_CAPA_SCALE"]] += 1
    print(f"  incident : {dict(sorted(ci.items()))}")
    print(f"  amb      : {dict(sorted(ca.items()))}")
    print(f"  uav      : {dict(sorted(cu.items()))}")
    print(f"  capa     : {dict(sorted(cc.items()))}")


def main():
    ap = argparse.ArgumentParser(description="v6 A5 규모·자원 도메인 랜덤화 매니페스트 생성기")
    ap.add_argument("--base", default="scenarios/manifests/sigungu_mixed47_manifest.json")
    ap.add_argument("--out", default="scenarios/manifests/sigungu_mixed47_rand_manifest.json")
    ap.add_argument("--variants_per_region", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.base, encoding="utf-8") as f:
        base = json.load(f)
    out = build_randomized(base, args.variants_per_region, args.seed)
    _validate(base, out, args.variants_per_region)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[gen_randomized] {args.base} ({len(base)}) → {args.out} "
          f"({len(out)} 엔트리, 변형 {len(out) - len(base)}, "
          f"vpr={args.variants_per_region}, seed={args.seed})")
    _print_axis_hist(out)


if __name__ == "__main__":
    main()
