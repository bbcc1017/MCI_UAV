#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""전국 대표점·병원·안전센터 위치를 한 눈에 보는 OpenStreetMap(Leaflet) HTML 생성기.

시뮬레이션 결과와 무관하게 "점 위치"만 그린다. 4개 레이어를 각각 토글할 수 있다:
  - 시도 대표점 (17개, cross_location_eval.LOCATIONS)      → 큰 별 모양 점
  - 시군구 대표점 (250개, sigungu_osrm_eval250_representative_manifest.json 좌표)  → 작은 점
  - 병원 (548개, 엑셀 결합 데이터.xlsx)                     → 마커(빨강 H, 헬기장은 금색 링)
  - 안전센터/소방서 (997개, 안전센터와 소방서.csv)          → 마커(파랑 🚑)

외부 이미지 없이 CDN Leaflet + divIcon(CSS)만 사용해 자체 완결형 HTML 하나를 만든다.
결과물: results/map/mci_map.html
사용법: PYTHONIOENCODING=utf-8 python3 vis_src/build_map.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd

# 리포 루트(= 이 파일의 상위 디렉터리) 기준 상대경로 기본값
ROOT = Path(__file__).resolve().parents[1]

# 시도 17 대표점(단일 진실원): 좌표 재계산 없이 코드에서 그대로 읽어온다.
from importlib import util as _util  # noqa: E402


def _load_locations(cross_eval_py: Path):
    """cross_location_eval.py 의 LOCATIONS 리스트를 임포트한다(의존성 최소)."""
    spec = _util.spec_from_file_location("_cle", cross_eval_py)
    mod = _util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return list(mod.LOCATIONS)


# 종별코드 → 사람이 읽는 종별명(HIRA 코드 관례)
KIND_NAME = {1: "상급종합병원", 11: "종합병원", 21: "병원"}

_COORD_RE = re.compile(r"\(([\d.]+),\s*([\d.]+)\)")


def load_sido(cross_eval_py: Path) -> list[dict]:
    out = []
    for name, city, lat, lon in _load_locations(cross_eval_py):
        out.append({"n": name, "city": city, "lat": lat, "lon": lon})
    return out


def load_sigungu(manifest: Path) -> list[dict]:
    m = json.loads(manifest.read_text(encoding="utf-8"))
    out = []
    for key, cfg_path in m.items():
        # 키: "종로구_11110" (동명 구 구분용 시군구코드 포함)
        name, _, code = key.rpartition("_")
        if not name:
            name, code = key, ""
        # 좌표는 config 경로의 "(lat,lon)" 에서 파싱
        mt = _COORD_RE.search(cfg_path)
        if not mt:
            continue
        lat, lon = float(mt.group(1)), float(mt.group(2))
        out.append({"n": name, "code": code, "lat": lat, "lon": lon})
    return out


def load_hospitals(xlsx: Path) -> list[dict]:
    df = pd.read_excel(xlsx)
    out = []
    for _, r in df.iterrows():
        lat, lon = r.get("y좌표"), r.get("x좌표")
        if pd.isna(lat) or pd.isna(lon):
            continue
        out.append({
            "n": str(r.get("요양기관명", "")),
            "lat": float(lat),
            "lon": float(lon),
            "heli": int(r.get("헬기장 여부", 0) or 0),
            "kind": KIND_NAME.get(int(r.get("종별코드", 0) or 0), "기타"),
            "sido": str(r.get("시도코드명", "")),
            "sgg": str(r.get("시군구코드명", "")),
            "oror": int(r.get("수술실병상수", 0) or 0),   # 수술실병상수
            "erbed": int(r.get("응급실병상수", 0) or 0),  # 응급실병상수
        })
    return out


def load_centers(csv_path: Path) -> list[dict]:
    out = []
    with open(csv_path, encoding="cp949") as f:
        for r in csv.DictReader(f):
            lat, lon = r.get("y좌표"), r.get("x좌표")
            if not lat or not lon:
                continue
            out.append({
                "n": str(r.get("기관명", "")).strip(),
                "lat": float(lat),
                "lon": float(lon),
                "hq": str(r.get("상위 본부명", "")).strip(),
                "addr": str(r.get("주소", "")).strip(),
            })
    return out


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>MCI 전국 위치도 — 시도·시군구 대표점 / 병원 / 안전센터</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
<style>
  html, body {{ margin: 0; height: 100%; font-family: "Malgun Gothic","Apple SD Gothic Neo",sans-serif; }}
  #map {{ position: absolute; inset: 0; }}

  /* divIcon 마커 공통 */
  .pin {{ display:flex; align-items:center; justify-content:center;
         width:22px; height:22px; border-radius:50% 50% 50% 0;
         transform: rotate(-45deg); box-shadow:0 1px 4px rgba(0,0,0,.4);
         border:2px solid #fff; font-size:11px; font-weight:700; color:#fff; }}
  .pin > span {{ transform: rotate(45deg); }}
  .pin-hos  {{ background:#e53935; }}                    /* 병원 = 빨강 */
  .pin-heli {{ background:#e53935; border-color:#ffd54f; box-shadow:0 0 0 3px #ffd54f88,0 1px 4px rgba(0,0,0,.4); }} /* 헬기장 병원 */
  .pin-cen  {{ background:#1e88e5; }}                    /* 안전센터 = 파랑 */

  /* 범례 */
  .legend {{ background:rgba(255,255,255,.94); padding:10px 12px; border-radius:8px;
            box-shadow:0 1px 6px rgba(0,0,0,.3); line-height:1.7; font-size:13px; }}
  .legend h4 {{ margin:0 0 6px; font-size:14px; }}
  .legend .row {{ display:flex; align-items:center; gap:8px; }}
  .swatch {{ display:inline-block; width:16px; height:16px; border-radius:50%; border:1px solid #666; flex:0 0 auto; }}
  .sw-sido {{ background:#ff8f00; width:18px; height:18px; border:2px solid #4e342e; }}
  .sw-sgg  {{ background:#8e24aa; width:11px; height:11px; }}
  .sw-hos  {{ background:#e53935; }}
  .sw-heli {{ background:#e53935; box-shadow:0 0 0 3px #ffd54f; }}
  .sw-cen  {{ background:#1e88e5; }}
  .leaflet-popup-content {{ font-size:13px; line-height:1.5; }}
  .cnt {{ color:#888; font-weight:400; }}
</style>
</head>
<body>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<script>
const DATA = {data_json};

const map = L.map('map', {{ preferCanvas: true }});

// 베이스 타일: OSM 표준 + CartoDB Positron(밝은 배경)
const osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom: 19, attribution: '&copy; OpenStreetMap contributors' }});
const positron = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  maxZoom: 20, attribution: '&copy; OpenStreetMap contributors &copy; CARTO' }});
positron.addTo(map);

function pinIcon(cls, label) {{
  return L.divIcon({{ className:'', html:'<div class="pin '+cls+'"><span>'+label+'</span></div>',
    iconSize:[22,22], iconAnchor:[11,22], popupAnchor:[0,-20] }});
}}
const icoHos  = pinIcon('pin-hos',  'H');
const icoHeli = pinIcon('pin-heli', 'H');
const icoCen  = pinIcon('pin-cen',  '🚑');

// --- 시도 대표점 (큰 별점) ---
const sidoLayer = L.layerGroup();
DATA.sido.forEach(d => {{
  L.circleMarker([d.lat, d.lon], {{
    radius:9, color:'#4e342e', weight:2, fillColor:'#ff8f00', fillOpacity:.95 }})
    .bindPopup('<b>[시도] '+d.n+'</b><br>대표점: '+d.city+'<br>'+d.lat.toFixed(4)+', '+d.lon.toFixed(4))
    .addTo(sidoLayer);
}});

// --- 시군구 대표점 (작은 점) ---
const sggLayer = L.layerGroup();
DATA.sigungu.forEach(d => {{
  L.circleMarker([d.lat, d.lon], {{
    radius:4, color:'#4a148c', weight:1, fillColor:'#8e24aa', fillOpacity:.9 }})
    .bindPopup('<b>[시군구] '+d.n+'</b>'+(d.code?' ('+d.code+')':'')+'<br>'+d.lat.toFixed(4)+', '+d.lon.toFixed(4))
    .addTo(sggLayer);
}});

// --- 병원 (마커) ---
const hosLayer = L.layerGroup();
DATA.hospitals.forEach(d => {{
  const heli = d.heli === 1;
  L.marker([d.lat, d.lon], {{ icon: heli ? icoHeli : icoHos }})
    .bindPopup('<b>🏥 '+d.n+'</b><br>'+d.kind+(heli?' · <b>헬기장 O</b>':'')
      +'<br>'+d.sido+' '+d.sgg
      +'<br>수술실병상 '+d.oror+' · 응급실병상 '+d.erbed
      +'<br>'+d.lat.toFixed(4)+', '+d.lon.toFixed(4))
    .addTo(hosLayer);
}});

// --- 안전센터/소방서 (마커) ---
const cenLayer = L.layerGroup();
DATA.centers.forEach(d => {{
  L.marker([d.lat, d.lon], {{ icon: icoCen }})
    .bindPopup('<b>🚑 '+d.n+'</b><br>'+d.hq+'<br>'+d.addr
      +'<br>'+d.lat.toFixed(4)+', '+d.lon.toFixed(4))
    .addTo(cenLayer);
}});

sidoLayer.addTo(map); sggLayer.addTo(map); hosLayer.addTo(map); cenLayer.addTo(map);

L.control.layers(
  {{ 'OSM 표준': osm, 'CartoDB Positron(밝게)': positron }},
  {{
    ['★ 시도 대표점 <span class="cnt">('+DATA.sido.length+')</span>']: sidoLayer,
    ['● 시군구 대표점 <span class="cnt">('+DATA.sigungu.length+')</span>']: sggLayer,
    ['🏥 병원 <span class="cnt">('+DATA.hospitals.length+')</span>']: hosLayer,
    ['🚑 안전센터 <span class="cnt">('+DATA.centers.length+')</span>']: cenLayer,
  }},
  {{ collapsed:false }}
).addTo(map);

// 범례
const legend = L.control({{ position:'bottomleft' }});
legend.onAdd = function() {{
  const div = L.DomUtil.create('div', 'legend');
  div.innerHTML =
    '<h4>MCI 전국 위치도</h4>'
    + '<div class="row"><span class="swatch sw-sido"></span> 시도 대표점 ('+DATA.sido.length+')</div>'
    + '<div class="row"><span class="swatch sw-sgg"></span> 시군구 대표점 ('+DATA.sigungu.length+')</div>'
    + '<div class="row"><span class="swatch sw-hos"></span> 병원 ('+DATA.hospitals.length+')</div>'
    + '<div class="row"><span class="swatch sw-heli"></span> └ 헬기장 병원 (금색 링)</div>'
    + '<div class="row"><span class="swatch sw-cen"></span> 안전센터/소방서 ('+DATA.centers.length+')</div>';
  return div;
}};
legend.addTo(map);

// 전체 점에 맞춰 화면 맞춤
const allPts = []
  .concat(DATA.sido, DATA.sigungu, DATA.hospitals, DATA.centers)
  .map(d => [d.lat, d.lon]);
map.fitBounds(L.latLngBounds(allPts).pad(0.03));
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="전국 위치 OpenStreetMap HTML 생성")
    ap.add_argument("--xlsx", default=str(ROOT / "scenarios/엑셀 결합 데이터.xlsx"))
    ap.add_argument("--centers", default=str(ROOT / "scenarios/안전센터와 소방서.csv"))
    ap.add_argument("--cross_eval", default=str(ROOT / "src/rl_src/cross_location_eval.py"))
    ap.add_argument("--sigungu_manifest",
                    default=str(ROOT / "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json"))
    ap.add_argument("--out", default=str(ROOT / "results/map/mci_map.html"))
    args = ap.parse_args()

    sido = load_sido(Path(args.cross_eval))
    sigungu = load_sigungu(Path(args.sigungu_manifest))
    hospitals = load_hospitals(Path(args.xlsx))
    centers = load_centers(Path(args.centers))

    data = {"sido": sido, "sigungu": sigungu, "hospitals": hospitals, "centers": centers}
    html = HTML_TEMPLATE.format(
        data_json=json.dumps(data, ensure_ascii=False, separators=(",", ":")))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    print(f"[OK] {out}")
    print(f"     시도 {len(sido)} · 시군구 {len(sigungu)} · 병원 {len(hospitals)}"
          f" (헬기장 {sum(h['heli'] for h in hospitals)}) · 안전센터 {len(centers)}")


if __name__ == "__main__":
    main()
