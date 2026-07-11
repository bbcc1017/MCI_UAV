# -*- coding: utf-8 -*-
"""vWorld WFS 수집 — 씬 현실성 통합(2026-07-02).

① ngii : 수치지형도 1:5000 도로중심선(lt_l_n3a0020000, 전국) → nationwide/ngiiroads/<구>.txt
   라인 "rdln rvwd onsd lat lon lat lon ..." (차로수/도로폭m/일방) — 임포터 NgiiRoadWidth가 읽어
   OSM lanes=0 way의 클래스 기본폭을 실측으로 교체.
② hd   : 정밀도로지도 5종(차선/보도면/신호등/가드레일/방지턱) → nationwide/hdmap/<layer>/<구>.geojson
   원본 보존(후속 차선 오버레이·소품 실좌표 배치 임포터용).

쿼드트리 분할(요청당 1000피처 캡 도달 시 4분할), id 중복 제거, 재개가능(기존 파일 skip).
키: 환경변수 VWORLD_API_KEY.

사용:
  python tools/hdmap_fetch.py ngii [--only a,b] [--force]
  python tools/hdmap_fetch.py hd --only 구1,구2 [--force]
"""
import argparse
import json
import os
import time

import requests

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(TOOLS, "nationwide")
SGG_JSON = os.path.join(ROOT, "sgg.json")
NGII_DIR = os.path.join(ROOT, "ngiiroads")
HD_DIR = os.path.join(ROOT, "hdmap")
KEY = os.environ.get("VWORLD_API_KEY")
WFS = "https://api.vworld.kr/req/wfs"
S = requests.Session()
S.headers["User-Agent"] = "MCI-UAV-research/1.0"

HD_LAYERS = [
    "lt_l_b2surfacelinemark",        # 노면선표시(차선)
    "lt_c_a4subsidiarysection",      # 보도구간(면)
    "lt_p_c1trafficlight",           # 신호등(점)
    "lt_l_c3vehicleprotectionsafety",  # 차량방호(가드레일)
    "lt_c_c4speedbump",              # 과속방지턱
    # ↓ 도로 3D 전환(2026-07-11): 전 레이어 z(고도) 포함 — 고가/IC·JC 실높이 소스
    "lt_c_a3drivewaysection",        # 차도구간(면) — 도로 표면 폴리곤
    "lt_l_a2link",                   # 주행경로링크(차로수/좌우인접/노드 위상)
    "lt_p_a1node",                   # 주행경로노드(분기/합류점)
    "lt_c_b3surfacemark",            # 노면표시(면: 화살표/횡단보도/도류대)
    "lt_p_b1safetysign",             # 안전표지(점 — 표지판 프롭)
    "lt_l_c5heightbarrier",          # 높이장벽(중앙분리대 등)
    "lt_p_c6postpoint",              # 지주(표지판/신호 폴)
    "lt_c_a5parkinglot",             # 주차장(면)
]


def wfs(layer, bbox, maxf=1000):
    params = {
        "SERVICE": "WFS", "VERSION": "1.1.0", "REQUEST": "GetFeature",
        "TYPENAME": layer,
        "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},EPSG:4326",
        "SRSNAME": "EPSG:4326", "OUTPUT": "application/json",
        "MAXFEATURES": str(maxf), "KEY": KEY, "DOMAIN": "localhost",
    }
    for attempt in range(4):
        try:
            r = S.get(WFS, params=params, timeout=40)
            r.raise_for_status()
            return r.json().get("features", [])
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                print(f"  포기 {layer} {bbox}: {str(e)[:60]}", flush=True)
                return []
            time.sleep(1.5 * (attempt + 1))
    return []


def harvest(layer, bbox, seen, out, depth=0):
    """쿼드트리 수집 — 1000 가득 차면 4분할(깊이 8 상한)."""
    feats = wfs(layer, bbox)
    time.sleep(0.12)
    if len(feats) >= 1000 and depth < 8:
        mla = (bbox[0] + bbox[2]) / 2
        mlo = (bbox[1] + bbox[3]) / 2
        for sub in ((bbox[0], bbox[1], mla, mlo), (bbox[0], mlo, mla, bbox[3]),
                    (mla, bbox[1], bbox[2], mlo), (mla, mlo, bbox[2], bbox[3])):
            harvest(layer, sub, seen, out, depth + 1)
        return
    for f in feats:
        fid = f.get("id") or (f.get("properties") or {}).get("ufid") or json.dumps(f.get("geometry", {}))[:80]
        if fid in seen:
            continue
        seen.add(fid)
        out.append(f)


def _num(v, cast=float):
    try:
        return cast(float(v))
    except Exception:  # noqa: BLE001
        return 0


def cmd_ngii(args):
    os.makedirs(NGII_DIR, exist_ok=True)
    sggs = json.load(open(SGG_JSON, encoding="utf-8"))
    if args.only:
        names = set(args.only.split(","))
        sggs = [s for s in sggs if s["name"] in names]
    print(f"[ngii] 대상 {len(sggs)}개 구 → {NGII_DIR}", flush=True)
    for i, s in enumerate(sggs):
        out_path = os.path.join(NGII_DIR, f"{s['name']}.txt")
        if os.path.exists(out_path) and not args.force:
            continue
        seen, feats = set(), []
        harvest("lt_l_n3a0020000", tuple(s["bbox"]), seen, feats)
        lines = []
        for f in feats:
            pr = f.get("properties") or {}
            rdln = _num(pr.get("rdln"), int)
            rvwd = _num(pr.get("rvwd"))
            onsd = _num(pr.get("onsd"), int)
            geom = f.get("geometry") or {}
            paths = ([geom.get("coordinates")] if geom.get("type") == "LineString"
                     else geom.get("coordinates") or [])
            for path in paths:
                if not path or len(path) < 2:
                    continue
                coords = " ".join(f"{p[1]:.7f} {p[0]:.7f}" for p in path if isinstance(p, (list, tuple)) and len(p) >= 2)
                if coords:
                    lines.append(f"{rdln} {rvwd:.1f} {onsd} {coords}")
        with open(out_path, "w", encoding="utf-8") as fp:
            fp.write("\n".join(lines))
        print(f"[ngii] {s['name']}: 피처 {len(feats)} → 선 {len(lines)} ({os.path.getsize(out_path)//1024}KB)", flush=True)
        if (i + 1) % 10 == 0:
            print(f"[ngii] 진행 {i + 1}/{len(sggs)}", flush=True)
    print("[ngii] 완료", flush=True)


def cmd_hd(args):
    sggs = json.load(open(SGG_JSON, encoding="utf-8"))
    if args.only:
        names = set(args.only.split(","))
        sggs = [s for s in sggs if s["name"] in names]
    print(f"[hd] 대상 {len(sggs)}개 구 × {len(HD_LAYERS)}레이어", flush=True)
    for layer in HD_LAYERS:
        ldir = os.path.join(HD_DIR, layer)
        os.makedirs(ldir, exist_ok=True)
        for s in sggs:
            out_path = os.path.join(ldir, f"{s['name']}.geojson")
            if os.path.exists(out_path) and not args.force:
                continue
            seen, feats = set(), []
            harvest(layer, tuple(s["bbox"]), seen, feats)
            with open(out_path, "w", encoding="utf-8") as fp:
                json.dump({"type": "FeatureCollection", "features": feats}, fp, ensure_ascii=False)
            print(f"[hd] {layer}/{s['name']}: {len(feats)}피처", flush=True)
    print("[hd] 완료", flush=True)


def main():
    if not KEY:
        raise SystemExit("VWORLD_API_KEY 환경변수가 없습니다.")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ngii")
    a.add_argument("--only", default=None)
    a.add_argument("--force", action="store_true")
    a.set_defaults(func=cmd_ngii)
    b = sub.add_parser("hd")
    b.add_argument("--only", default=None)
    b.add_argument("--force", action="store_true")
    b.set_defaults(func=cmd_hd)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
