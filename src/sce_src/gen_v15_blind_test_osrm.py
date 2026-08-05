# -*- coding: utf-8 -*-
"""v15 정책 동결 후 생성하는 신규 블라인드 시군구250 테스트셋.

기존 외부셋 생성기의 이격·구조 검증을 그대로 재사용하되 출력 경로를 분리한다.
기존 ``*points.json`` 전체가 제외원에 포함되므로 v10 외부셋 좌표도 재사용하지 않는다.
"""
from pathlib import Path

import gen_distill_external_test_osrm as base

REPO = Path(__file__).resolve().parents[2]
base.EXP_PREFIX = "v15_blind/osrm"
base.POINTS_PATH = REPO / "scenarios/manifests/v15_blind250_points.json"
base.MANIFEST_PATH = REPO / "scenarios/manifests/v15_blind250_osrm_manifest.json"
base.META_PATH = REPO / "scenarios/manifests/v15_blind250_meta.json"


if __name__ == "__main__":
    base.main()
