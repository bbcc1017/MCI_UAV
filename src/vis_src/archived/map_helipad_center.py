"""헬기장 병원 25 / 헬기장 중점 / 17 광역시도청 / 전국 헬기장 (V-world) 시각화.

각 카테고리는 별도 FeatureGroup 으로 묶여 우상단 LayerControl 에서 on/off.

사용:
    python src/vis_src/map_helipad_center.py
출력:
    results/map_helipad_center.html
"""
import argparse
from pathlib import Path

import folium
import pandas as pd
from folium.plugins import MarkerCluster

HELIPAD_CENTER = (36.245107096, 127.462534992)

# 시나리오 폴더에서 거리 정보 가져오는 기본 경로
DEFAULT_SCENARIO_DIR = "scenarios/exp_helipad_center_uav/(36.245107096,127.462534992)"

REGIONS = [
    ("서울", "서울특별시청",         37.5666, 126.9784),
    ("부산", "부산광역시청",         35.1798, 129.0750),
    ("대구", "대구광역시청(산격)",    35.8894, 128.6087),
    ("인천", "인천광역시청",         37.4563, 126.7052),
    ("광주", "광주광역시청",         35.1601, 126.8515),
    ("대전", "대전광역시청",         36.3505, 127.3845),
    ("울산", "울산광역시청",         35.5398, 129.3114),
    ("세종", "세종특별자치시청",      36.4800, 127.2890),
    ("경기", "경기도청(광교)",       37.2893, 127.0535),
    ("강원", "강원특별자치도청",      37.8845, 127.7297),
    ("충북", "충청북도청",           36.6359, 127.4913),
    ("충남", "충청남도청",           36.6588, 126.8315),
    ("전북", "전북특별자치도청",      35.8203, 127.1088),
    ("전남", "전라남도청",           34.8160, 126.4623),
    ("경북", "경상북도청",           36.5759, 128.7067),
    ("경남", "경상남도청",           35.2277, 128.6811),
    ("제주", "제주특별자치도청",      33.4890, 126.4983),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hospital_data", default=None,
                   help="병원 엑셀 경로. 기본: scenarios/엑셀 결합 데이터.xlsx")
    p.add_argument("--scenario_dir", default=None,
                   help="hospital_info_euc.csv / distance_Hos2Site_euc.csv 위치")
    p.add_argument("--helipad_csv", default=None,
                   help="전국 헬기장 csv. 기본: helipad_location.csv")
    p.add_argument("--out_html", default="results/map_helipad_center.html")
    p.add_argument("--zoom_start", type=int, default=7)
    return p.parse_args()


def load_hospital_with_distance(repo_root: Path, hospital_xlsx: Path, scenario_dir: Path):
    """엑셀에서 25개 헬기장 추출 + 시나리오 distance 파일과 매칭."""
    df_full = pd.read_excel(hospital_xlsx, engine="openpyxl")
    helipads = df_full[df_full["헬기장 여부"] == 1].copy().reset_index(drop=True)

    info = pd.read_csv(scenario_dir / "hospital_info_euc.csv", encoding="utf-8-sig")
    dist = pd.read_csv(scenario_dir / "distance_Hos2Site_euc.csv", encoding="utf-8-sig")
    info = info.merge(dist, on="Index")
    info_lookup = info.set_index("요양기관명")["distance"].to_dict()
    helipads["distance_km"] = helipads["요양기관명"].map(info_lookup)
    return helipads


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    hospital_xlsx = Path(args.hospital_data) if args.hospital_data else \
        repo_root / "scenarios" / "엑셀 결합 데이터.xlsx"
    scenario_dir = Path(args.scenario_dir) if args.scenario_dir else \
        repo_root / DEFAULT_SCENARIO_DIR
    helipad_csv = Path(args.helipad_csv) if args.helipad_csv else \
        repo_root / "helipad_location.csv"

    helipads = load_hospital_with_distance(repo_root, hospital_xlsx, scenario_dir)
    print(f"헬기장 병원: {len(helipads)}개, distance 매칭 완료")

    m = folium.Map(
        location=HELIPAD_CENTER,
        zoom_start=args.zoom_start,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Layer 1: 헬기장 병원 (Tier3 빨강, Tier2 주황) + 점선 연결
    fg_hospitals = folium.FeatureGroup(name=f"헬기장 병원 ({len(helipads)})", show=True)
    for _, row in helipads.iterrows():
        is_tier3 = int(row["종별코드"]) == 1
        color = "red" if is_tier3 else "orange"
        tier_label = "Tier3 (상급종합)" if is_tier3 else "Tier2"
        dist_km = row.get("distance_km")
        dist_str = f"{dist_km:.2f} km" if pd.notna(dist_km) else "N/A"
        popup_html = (
            f"<b>{row['요양기관명']}</b><br>"
            f"종별: {tier_label}<br>"
            f"응급실병상수: {int(row['응급실병상수']) if pd.notna(row['응급실병상수']) else '-'}<br>"
            f"helipad_center 까지: <b>{dist_str}</b><br>"
            f"좌표: ({row['y좌표']:.4f}, {row['x좌표']:.4f})"
        )
        folium.Marker(
            location=(row["y좌표"], row["x좌표"]),
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['요양기관명']} ({dist_str})",
            icon=folium.Icon(color=color, icon="plus", prefix="fa"),
        ).add_to(fg_hospitals)
        # 점선
        folium.PolyLine(
            locations=[HELIPAD_CENTER, (row["y좌표"], row["x좌표"])],
            color=color, weight=1.5, opacity=0.7, dash_array="6, 8",
        ).add_to(fg_hospitals)
    fg_hospitals.add_to(m)

    # Layer 2: helipad_center
    fg_center = folium.FeatureGroup(name="헬기장 중점 (helipad_center)", show=True)
    folium.Marker(
        location=HELIPAD_CENTER,
        popup=folium.Popup(
            f"<b>헬기장 중점</b><br>({HELIPAD_CENTER[0]:.6f}, {HELIPAD_CENTER[1]:.6f})",
            max_width=300),
        tooltip="helipad_center",
        icon=folium.Icon(color="darkred", icon="star", prefix="fa"),
    ).add_to(fg_center)
    folium.CircleMarker(
        location=HELIPAD_CENTER,
        radius=6, color="black", fill=True, fill_opacity=0.8,
    ).add_to(fg_center)
    fg_center.add_to(m)

    # Layer 3: 광역시도청 17
    fg_regions = folium.FeatureGroup(name=f"광역시도청 ({len(REGIONS)})", show=True)
    for short_name, full_name, lat, lon in REGIONS:
        folium.Marker(
            location=(lat, lon),
            popup=folium.Popup(
                f"<b>{short_name}</b><br>{full_name}<br>({lat:.4f}, {lon:.4f})",
                max_width=300),
            tooltip=f"{short_name} - {full_name}",
            icon=folium.Icon(color="green", icon="building", prefix="fa"),
        ).add_to(fg_regions)
    fg_regions.add_to(m)

    # Layer 4: 전국 헬기장 (V-world LT_P_AISHCSTRIP)
    helipad_all = pd.read_csv(helipad_csv, encoding="utf-8-sig")
    print(f"전국 헬기장: {len(helipad_all)}개")
    fg_helipad = folium.FeatureGroup(name=f"전국 헬기장 V-world ({len(helipad_all)})", show=False)
    cluster = MarkerCluster().add_to(fg_helipad)
    for _, row in helipad_all.iterrows():
        try:
            lat = float(row["y"])
            lon = float(row["x"])
        except (ValueError, TypeError):
            continue
        popup_html = (
            f"<b>{row.get('str_nam', '')}</b><br>"
            f"운영: {row.get('org_nam', '')}<br>"
            f"주소: {row.get('str_adr', '')}<br>"
            f"유형: {row.get('stt_cde', '')}<br>"
            f"좌표: ({lat:.4f}, {lon:.4f})"
        )
        folium.Marker(
            location=(lat, lon),
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=row.get("str_nam", ""),
            icon=folium.Icon(color="purple", icon="helicopter", prefix="fa"),
        ).add_to(cluster)
    fg_helipad.add_to(m)

    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    out_path = (repo_root / args.out_html).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
