# -*- coding: utf-8 -*-
"""공역(비행금지/관제권/비행제한/위험구역) → 전국 단일 airspace.json (vWorld WFS).

data.go.kr '국토교통부_항공정보도'는 결국 vWorld WMS/WFS로 링크되며, 우리는 이미
건물(lt_c_bldginfo)에 vWorld WFS를 쓰고 있다. 공역도 같은 키(VWORLD_API_KEY)로
GeoJSON MultiPolygon(EPSG:4326)을 그대로 받는다 — data.go.kr 키 불필요.

레이어(검증: 전국 비행금지 15·관제권 53·비행제한 83·위험 32 = 183개, 1콜/레이어):
  lt_c_aisprhc 비행금지  prh_lbl_1=코드 prh_lbl_2=상한 prh_lbl_3=하한
  lt_c_aisctrc 관제권    ctr_lbl_1=명칭
  lt_c_aisresc 비행제한  res_lbl_1=코드 res_lbl_2=상한 res_lbl_3=하한
  lt_c_aisdngc 위험구역  dng_lbl_1=코드 dng_lbl_2=상한 dng_lbl_3=하한

출력 airspace.json (Unity NoFlyZoneManager가 런타임 로드):
  {"zones":[{"type","code","label","alt_lo_m","alt_hi_m","alt_lo_raw","alt_hi_raw",
             "rings":[[lon,lat,lon,lat,...], ...]}], ...}
  rings = MultiPolygon의 외곽 링들(각각 닫힌 평면 배열). 고도는 ft→m 변환, UNL=None.

사용:
  python tools/airspace_fetch.py            # tools/nationwide/airspace.json 생성
  python tools/airspace_fetch.py --unity-copy   # + Unity Assets/Scenes/Regions/로 복사
키: 환경변수 VWORLD_API_KEY.
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

KEY = os.environ.get("VWORLD_API_KEY")
WFS_URL = "https://api.vworld.kr/req/wfs"
TOOLS = os.path.dirname(os.path.abspath(__file__))
OUT_DEFAULT = os.path.join(TOOLS, "nationwide", "airspace.json")
UNITY_DST = os.path.join(
    TOOLS, os.pardir, "external", "ml-agents", "UAV_test",
    "Assets", "Scenes", "Regions", "airspace.json")

# 전국 bbox (lat,lon,lat,lon — WFS 1.1.0 + EPSG:4326 축순서 lat,lon). 서해5도~독도~마라도 포함.
KOREA_BBOX = "33.0,124.5,38.7,131.9"

LAYERS = [
    # (type, typeName, 코드필드, 상한필드, 하한필드)
    ("prohibited", "lt_c_aisprhc", "prh_lbl_1", "prh_lbl_2", "prh_lbl_3"),
    ("control",    "lt_c_aisctrc", "ctr_lbl_1", None,        None),
    ("restricted", "lt_c_aisresc", "res_lbl_1", "res_lbl_2", "res_lbl_3"),
    ("danger",     "lt_c_aisdngc", "dng_lbl_1", "dng_lbl_2", "dng_lbl_3"),
]

FT_TO_M = 0.3048


def parse_alt(raw):
    """고도 라벨 → 미터. GND/SFC=0, UNL=None(무제한), FLxxx=xxx*100ft, 'n 000 AMSL/AGL'=ft."""
    if not raw:
        return None, None
    s = str(raw).strip().upper()
    if s in ("GND", "SFC", "0"):
        return 0.0, s
    if s in ("UNL", "UNLTD", "UNLIMITED"):
        return None, s
    if s.startswith("FL"):
        digits = "".join(ch for ch in s[2:] if ch.isdigit())
        return (int(digits) * 100 * FT_TO_M, s) if digits else (None, s)
    digits = "".join(ch for ch in s if ch.isdigit())  # "6 000 AMSL" → 6000
    if digits:
        return round(int(digits) * FT_TO_M, 1), s
    return None, s


def wfs_geojson(type_name, max_features=1000):  # vWorld WFS 상한 1000 (초과 시 ServiceException)
    params = {
        "SERVICE": "WFS", "VERSION": "1.1.0", "REQUEST": "GetFeature",
        "TYPENAME": type_name,
        "BBOX": KOREA_BBOX + ",EPSG:4326",
        "SRSNAME": "EPSG:4326", "OUTPUT": "application/json",
        "MAXFEATURES": str(max_features), "KEY": KEY, "DOMAIN": "localhost",
    }
    url = WFS_URL + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
            print(f"  retry {type_name}: {e}", flush=True)


def outer_rings(geom):
    """MultiPolygon/Polygon → 외곽 링 평면 배열 목록([lon,lat,lon,lat,...])."""
    if not geom:
        return []
    t = geom.get("type")
    coords = geom.get("coordinates") or []
    polys = coords if t == "MultiPolygon" else ([coords] if t == "Polygon" else [])
    rings = []
    for poly in polys:
        if not poly:
            continue
        ext = poly[0]  # 외곽 링(홀 무시 — 공역은 사실상 홀 없음)
        if len(ext) < 3:
            continue
        flat = []
        for c in ext:
            flat.append(round(c[0], 7))
            flat.append(round(c[1], 7))
        rings.append(flat)
    return rings


def build():
    zones = []
    summary = []
    for typ, tn, fc, fhi, flo in LAYERS:
        js = wfs_geojson(tn)
        feats = js.get("features", [])
        cnt = 0
        for f in feats:
            pr = f.get("properties", {}) or {}
            rings = outer_rings(f.get("geometry"))
            if not rings:
                continue
            code = pr.get(fc) or "-"
            hi_m, hi_raw = parse_alt(pr.get(fhi)) if fhi else (None, None)
            lo_m, lo_raw = parse_alt(pr.get(flo)) if flo else (0.0, "GND")
            zones.append({
                "type": typ,
                "code": str(code).strip(),
                "label": {"prohibited": "비행금지", "control": "관제권",
                          "restricted": "비행제한", "danger": "위험"}[typ],
                "alt_lo_m": lo_m, "alt_hi_m": hi_m,
                "alt_lo_raw": lo_raw, "alt_hi_raw": hi_raw,
                "rings": rings,
            })
            cnt += 1
        summary.append(f"{typ}({tn}): {cnt}")
        print(f"[airspace] {typ:11s} {tn}: {cnt}개 구역", flush=True)
        time.sleep(0.3)
    return zones, summary


def main():
    if not KEY:
        sys.exit("VWORLD_API_KEY 환경변수가 없습니다.")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--unity-copy", action="store_true",
                    help="Unity Assets/Scenes/Regions/airspace.json 으로도 복사")
    args = ap.parse_args()

    zones, summary = build()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    doc = {"crs": "EPSG:4326", "note": "rings=[lon,lat,...] 외곽링; alt 미터(UNL=null)",
           "zones": zones}
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(doc, fp, ensure_ascii=False)
    print(f"\n[airspace] 총 {len(zones)}개 구역 → {args.out} "
          f"({os.path.getsize(args.out) / 1024:.0f}KB) | {', '.join(summary)}", flush=True)

    if args.unity_copy:
        dst = os.path.abspath(UNITY_DST)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as fp:
            json.dump(doc, fp, ensure_ascii=False)
        print(f"[airspace] Unity 복사 → {dst}", flush=True)


if __name__ == "__main__":
    main()
