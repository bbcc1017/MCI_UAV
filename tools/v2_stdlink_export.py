"""표준노드링크(MOCT_LINK) → 강남 구역 링크 중점 bin (도로별 실시간 교통량 매칭용).
ITS 소통정보의 linkId(=표준링크 LINK_ID)와 매칭되는 geometry를 제공. Unity 가 각 LGV2 링크를
최근접 표준링크에 붙여 그 링크의 실측 speed(혼잡)를 per-road NPC 밀도/속도에 반영.

MOCT_LINK CRS = ITRF2000 Central Belt = EPSG:5186 (우리 LGV2/타일과 동일 좌표계).
출력: tools/nationwide_v2/lanegraph/<region>.stdlink.bin (STDL)
 · 앵커는 LGV2/WLK2 와 달리 **링크 중점 평균을 자체 계산**해 헤더에 싣는다(자기기술) — 소비측은
   STDL 헤더의 anchorE/N 을 읽어서 복원할 것. LGV2 앵커를 가정하면 어긋난다.
실행(기본=강남 9타일): PYTHONIOENCODING=utf-8 <UAV python> tools/v2_stdlink_export.py
다른 구: --shp <MOCT_LINK.shp> --bbox E0 N0 E1 N1 --region <이름>
"""
import argparse
import os
import struct

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 전국 표준노드링크 원본(ITS 배포본을 내려받은 로컬 경로 — 박스마다 다르므로 --shp 로 덮어쓴다)
DEF_SHP = r"C:\Users\User\Downloads\[2026-07-16]NODELINKDATA\[2026-07-16]NODELINKDATA\MOCT_LINK.shp"
DEF_REGION = "seoul_gangnamgu"
DEF_BBOX = (202000, 543000, 205000, 546000)   # 강남 9타일 구역(EPSG:5186)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shp", default=DEF_SHP, help="MOCT_LINK.shp 경로")
    ap.add_argument("--region", default=DEF_REGION, help="출력 파일명 접두(<region>.stdlink.bin)")
    ap.add_argument("--bbox", nargs=4, type=float, default=list(DEF_BBOX),
                    metavar=("E0", "N0", "E1", "N1"), help="EPSG:5186 추출 범위")
    ap.add_argument("--out_dir", default=os.path.join(REPO, "tools", "nationwide_v2", "lanegraph"))
    a = ap.parse_args()
    E0, N0, E1, N1 = a.bbox
    OUT = os.path.join(a.out_dir, a.region + ".stdlink.bin")
    if not os.path.exists(a.shp):
        raise SystemExit(f"[stdlink] MOCT_LINK 원본이 없다: {a.shp}\n"
                         f"          ITS 표준노드링크 배포본을 받아 --shp 로 지정할 것.")

    import geopandas as gpd
    g = gpd.read_file(a.shp, bbox=(E0, N0, E1, N1))
    print(f"[stdlink] bbox 내 표준링크 {len(g)}개 (CRS={g.crs.to_epsg()})")

    ids, midE, midN, spd = [], [], [], []
    for _, row in g.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        try:
            lid = int(str(row["LINK_ID"]).strip())
        except (ValueError, TypeError):
            continue
        c = geom.interpolate(0.5, normalized=True)   # 링크 중점
        ids.append(lid); midE.append(c.x); midN.append(c.y)
        try:
            spd.append(float(row.get("MAX_SPD", 0) or 0))
        except (ValueError, TypeError):
            spd.append(0.0)

    n = len(ids)
    aE = float(np.mean(midE)) if n else 0.0
    aN = float(np.mean(midN)) if n else 0.0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as fo:
        fo.write(b"STDL")
        fo.write(struct.pack("<I", 1))
        fo.write(struct.pack("<I", n))
        fo.write(struct.pack("<2d", aE, aN))
        fo.write(np.asarray(ids, dtype="<i8").tobytes())
        fo.write((np.asarray(midE) - aE).astype("<f4").tobytes())
        fo.write((np.asarray(midN) - aN).astype("<f4").tobytes())
        fo.write(np.asarray(spd, dtype="<f4").tobytes())
    print(f"[stdlink] {n}개 링크 → {OUT} ({os.path.getsize(OUT)}B), anchor=({aE:.0f},{aN:.0f})")
    # 샘플
    for i in range(min(3, n)):
        print(f"  linkId={ids[i]} mid=({midE[i]:.0f},{midN[i]:.0f}) maxspd={spd[i]:.0f}")


if __name__ == "__main__":
    main()
