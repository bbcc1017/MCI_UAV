"""region_index.json 재생성 — 새 지역 임포트 후 1회 실행.

tools/vw_*/tiles_manifest.json 의 mosaic epsg3857_bounds 유니온을
위경도 bbox(S W N E)로 역변환해 Assets/Scenes/Regions/region_index.json 갱신.
jeju 3밴드(vw_jeju_n/m/s)는 'jeju' 하나로 유니온.
지역 씬(.unity)이 실제 존재하는 항목만 포함.

사용: python tools/build_region_index.py [--nationwide]
  --nationwide : tools/nationwide/sgg/vw_*(시군구 전국 빌드)만 스캔 — 구 도시 단위 vw_* 제외.
                 전국 빌드 컷오버 후에는 항상 이 모드를 사용.
"""
import json
import math
import os
import sys

R = 6378137.0
TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
REGIONS_DIR = os.path.join(
    REPO, "external", "ml-agents", "UAV_test", "Assets", "Scenes", "Regions")
OUT = os.path.join(REGIONS_DIR, "region_index.json")
NATIONWIDE = "--nationwide" in sys.argv
SGG_ROOT = os.path.join(TOOLS, "nationwide", "sgg")


def inv_merc(x, y):
    lon = math.degrees(x / R)
    lat = math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2)
    return lat, lon


def region_name(dirname):
    name = dirname[3:]  # strip vw_
    # 구(도시 단위) jeju 3밴드만 유니온 — 전국 빌드의 jeju_jejusi 등은 그대로
    if not NATIONWIDE and name.startswith("jeju"):
        return "jeju"
    return name


def scan_dirs():
    if NATIONWIDE:
        if not os.path.isdir(SGG_ROOT):
            sys.exit(f"{SGG_ROOT} 없음")
        return [(d, os.path.join(SGG_ROOT, d)) for d in sorted(os.listdir(SGG_ROOT))
                if d.startswith("vw_")]
    return [(d, os.path.join(TOOLS, d)) for d in sorted(os.listdir(TOOLS))
            if d.startswith("vw_")]


def main():
    index = {}
    for d, dpath in scan_dirs():
        mf = os.path.join(dpath, "tiles_manifest.json")
        if not os.path.isfile(mf):
            print(f"skip {d}: tiles_manifest.json 없음")
            continue
        name = region_name(d)
        if not os.path.isfile(os.path.join(REGIONS_DIR, name + ".unity")):
            print(f"skip {d}: 지역 씬 {name}.unity 없음")
            continue
        with open(mf, encoding="utf-8") as f:
            man = json.load(f)
        xs, ys = [], []
        for m in man["mosaics"]:
            x0, y0, x1, y1 = m["epsg3857_bounds"]
            xs += [x0, x1]
            ys += [y0, y1]
        if xs:
            s, w = inv_merc(min(xs), min(ys))
            n, e = inv_merc(max(xs), max(ys))
        else:
            # 건물 전용 시군구(블록 미보유, 영상은 이웃 씬) — buildings.txt에서 bbox 도출
            bt = os.path.join(dpath, "buildings.txt")
            if not os.path.isfile(bt):
                print(f"skip {d}: 모자이크/건물 모두 없음")
                continue
            s, w, n, e = 90.0, 180.0, -90.0, -180.0
            with open(bt, encoding="utf-8") as f:
                for line in f:
                    tok = line.split()
                    for i in range(1, len(tok) - 1, 2):
                        lon = float(tok[i]); lat = float(tok[i + 1])
                        s = min(s, lat); n = max(n, lat)
                        w = min(w, lon); e = max(e, lon)
            if s > n:
                print(f"skip {d}: 건물 없음")
                continue
        if name in index:  # jeju 밴드 유니온
            ps, pw, pn, pe = index[name]
            s, w, n, e = min(s, ps), min(w, pw), max(n, pn), max(e, pe)
        index[name] = [s, w, n, e]
        print(f"{name}: S{s:.4f} W{w:.4f} N{n:.4f} E{e:.4f}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1, ensure_ascii=False)
    print(f"\n{len(index)}개 지역 → {OUT}")


if __name__ == "__main__":
    main()
