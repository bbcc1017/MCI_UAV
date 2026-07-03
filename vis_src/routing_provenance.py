#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""시도·시군구 사고좌표(대표점)별 라우팅 provenance 집계 → CSV.

각 시나리오(= 사고좌표 1개)에 대해:
  - road_mode: 그 시나리오 세트의 provider (kakao / osrm)
  - site_snap_m: 대표점 → 최근접도로 스냅거리(m). OSRM /route waypoint.distance 로 측정
                 (좌표 고유값이라 kakao/osrm 동일 → OSRM 세트에서 한 번만 재고 두 row에 공유)
  - n_route_legs / n_leg_kakao / n_leg_osrm_fallback: 개별 경로 JSON의 meta.api_provider 전수 스캔
    (kakao 모드에서 provider=osrm 인 레그 = OSRM 라우팅 폴백. route_adjustments.json 은
     시도 kakao엔 없고 시군구도 151/250 뿐이라 신뢰 불가 → JSON provider 가 ground truth)
  - n_kakao_snap_legs: 스냅좌표 재투입(라우팅은 여전히 Kakao) 레그 수. route_adjustments.json 이
    있을 때만(시군구 일부) 채움, 없으면 공백(JSON provider 로는 스냅여부 구분 불가)
  - has_osrm_fallback: kakao 모드에서 n_leg_osrm_fallback>0 이면 Y, osrm 모드는 N/A

출력: results/map/routing_provenance.csv
사용법: PYTHONIOENCODING=utf-8 python3 vis_src/routing_provenance.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (scope, road_mode) -> manifest 파일
MANIFESTS = {
    ("시군구", "kakao"): "scenarios/manifests/sigungu_kakao_manifest.json",
    ("시군구", "osrm"): "scenarios/manifests/sigungu_osrm_manifest.json",
    ("시도", "kakao"): "scenarios/manifests/plan1_manifest.json",       # 시도 Kakao(plan1)
    ("시도", "osrm"): "scenarios/manifests/sido_osrm_manifest.json",
}

_COORD_RE = re.compile(r"\(([-\d.]+),\s*([-\d.]+)\)")
_PROV_RE = re.compile(rb'"api_provider"\s*:\s*"(\w+)"')


def _hav(a, b):
    la1, lo1, la2, lo2 = map(radians, [a[0], a[1], b[0], b[1]])
    h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371000.0 * asin(min(1.0, sqrt(h)))


def _provider_of(route_json_path):
    """개별 경로 JSON 의 meta.api_provider 를 앞부분만 읽어 빠르게 판별."""
    try:
        with open(route_json_path, "rb") as f:
            head = f.read(300)
        m = _PROV_RE.search(head)
        return m.group(1).decode() if m else None
    except OSError:
        return None


def _site_snap_from_osrm(scen_dir, site):
    """OSRM 세트 경로 JSON 하나에서 대표점(site)에 가장 가까운 waypoint 의 스냅거리(m)."""
    for rj in glob.glob(os.path.join(scen_dir, "routes", "*", "*.json")):
        try:
            d = json.load(open(rj, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        wps = d.get("payload", {}).get("osrm_response", {}).get("waypoints", [])
        best = None
        for w in wps:
            loc = w.get("location")
            if not loc:
                continue
            dd = _hav(site, (loc[1], loc[0]))
            if best is None or dd < best[0]:
                best = (dd, w.get("distance"))
        if best and best[1] is not None:
            return round(float(best[1]), 1)
    return None


def load_manifest(scope, road_mode):
    """manifest → [(region, sigcd, lat, lon, scen_dir), ...]."""
    fp = ROOT / MANIFESTS[(scope, road_mode)]
    m = json.loads(fp.read_text(encoding="utf-8"))
    out = []
    for key, cfg in m.items():
        if scope == "시군구":
            region, _, sigcd = key.rpartition("_")
            if not region:
                region, sigcd = key, ""
        else:
            region, sigcd = key, ""
        mt = _COORD_RE.search(cfg)
        if not mt:
            continue
        lat, lon = float(mt.group(1)), float(mt.group(2))
        scen_dir = os.path.dirname(cfg)  # config 가 있는 (lat,lon) 폴더
        out.append((region, sigcd, lat, lon, scen_dir))
    return out


def scan_kakao_scenario(scen_dir):
    """kakao 시나리오의 개별 경로 JSON 을 전수 스캔 → provider 카운트."""
    n_kakao = n_osrm = 0
    for rj in glob.glob(os.path.join(scen_dir, "routes", "*", "*.json")):
        p = _provider_of(rj)
        if p == "osrm":
            n_osrm += 1
        elif p == "kakao":
            n_kakao += 1
    return n_kakao, n_osrm


def read_route_adjust(scen_dir):
    """route_adjustments.json 이 있으면 (site_offset_m, n_kakao_snap_legs) 반환, 없으면 (None,None)."""
    fp = os.path.join(scen_dir, "route_adjustments.json")
    if not os.path.exists(fp):
        return None, None
    try:
        d = json.load(open(fp, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    site_off = (d.get("site") or {}).get("offset_m")
    n_snap = d.get("n_kakao_snap_legs")
    return site_off, n_snap


def main():
    ap = argparse.ArgumentParser(description="라우팅 provenance CSV 생성")
    ap.add_argument("--out", default=str(ROOT / "results/map/routing_provenance.csv"))
    args = ap.parse_args()

    # 1) OSRM 세트에서 (scope,region)별 site_snap_m 를 먼저 계산 (좌표 고유값 → 양 provider 공유)
    site_snap = {}  # (scope, region, sigcd) -> m
    for scope in ("시군구", "시도"):
        for region, sigcd, lat, lon, scen_dir in load_manifest(scope, "osrm"):
            site_snap[(scope, region, sigcd)] = _site_snap_from_osrm(scen_dir, (lat, lon))

    rows = []
    for scope in ("시군구", "시도"):
        for road_mode in ("kakao", "osrm"):
            for region, sigcd, lat, lon, scen_dir in load_manifest(scope, road_mode):
                snap = site_snap.get((scope, region, sigcd))
                n_legs = n_kakao = n_osrm_fb = n_snap_legs = None
                has_fb = "N/A(osrm mode)"
                if road_mode == "kakao":
                    n_kakao, n_osrm_fb = scan_kakao_scenario(scen_dir)
                    n_legs = n_kakao + n_osrm_fb
                    ra_site, n_snap_legs = read_route_adjust(scen_dir)
                    if snap is None and ra_site is not None:
                        snap = ra_site
                    has_fb = "Y" if n_osrm_fb and n_osrm_fb > 0 else "N"
                else:  # osrm: 전부 osrm 네이티브 (폴백 개념 없음)
                    all_j = glob.glob(os.path.join(scen_dir, "routes", "*", "*.json"))
                    n_legs = len(all_j)
                rows.append({
                    "scope": scope,
                    "region": region,
                    "sigcd": sigcd,
                    "lat": lat,
                    "lon": lon,
                    "road_mode": road_mode,
                    "site_snap_m": "" if snap is None else snap,
                    "n_route_legs": "" if n_legs is None else n_legs,
                    "n_leg_kakao": "" if n_kakao is None else n_kakao,
                    "n_leg_osrm_fallback": "" if n_osrm_fb is None else n_osrm_fb,
                    "n_kakao_snap_legs": "" if n_snap_legs is None else n_snap_legs,
                    "has_osrm_fallback": has_fb,
                })

    cols = ["scope", "region", "sigcd", "lat", "lon", "road_mode", "site_snap_m",
            "n_route_legs", "n_leg_kakao", "n_leg_osrm_fallback",
            "n_kakao_snap_legs", "has_osrm_fallback"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # 요약 출력
    kak_sgg = [r for r in rows if r["scope"] == "시군구" and r["road_mode"] == "kakao"]
    kak_sido = [r for r in rows if r["scope"] == "시도" and r["road_mode"] == "kakao"]
    fb_sgg = [r for r in kak_sgg if r["has_osrm_fallback"] == "Y"]
    fb_sido = [r for r in kak_sido if r["has_osrm_fallback"] == "Y"]
    print(f"[OK] {out}  (총 {len(rows)} rows)")
    print(f"  시군구 Kakao {len(kak_sgg)}개 중 OSRM 폴백 발생 = {len(fb_sgg)}개")
    for r in sorted(fb_sgg, key=lambda x: -x["n_leg_osrm_fallback"]):
        print(f"    - {r['region']}({r['sigcd']}) 폴백 {r['n_leg_osrm_fallback']}/{r['n_route_legs']} 레그, "
              f"site_snap={r['site_snap_m']}m")
    print(f"  시도 Kakao {len(kak_sido)}개 중 OSRM 폴백 발생 = {len(fb_sido)}개")
    for r in sorted(fb_sido, key=lambda x: -x["n_leg_osrm_fallback"]):
        print(f"    - {r['region']} 폴백 {r['n_leg_osrm_fallback']}/{r['n_route_legs']} 레그, "
              f"site_snap={r['site_snap_m']}m")


if __name__ == "__main__":
    main()
