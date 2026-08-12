"""Copernicus GLO-30 DEM(DSM) → 도심용 매끄러운 지면장으로 재가공.

문제: GLO-30 은 DSM(표면모델)이라 도심에서 건물·수목 덩어리가 지면 고도에 섞여 있다.
30m 격자를 bilinear 로 읽으면 그 덩어리가 도로 리본·건물 바닥을 뚫고 올라와 "지형이 도로/건물과
뒤섞이는" 고질적 아티팩트가 생긴다. 해상도를 올릴 소스가 없으므로(국토지리원 5m DEM 은 신청 필요),
DSM→DTM 근사 + 업샘플로 **지면장 자체를 매끄럽게** 만든다.

파이프라인(격자 좌표계·원점 불변, 격자 간격만 세분):
  1. 회색조 형태학적 **opening**(min 필터 → max 필터, 반경 r셀 ≈ 60m)
     - 폭 ~120m 보다 좁은 **양(+) 돌출**만 제거 = 건물·블록 덩어리. 산은 그보다 넓어 살아남는다.
     - 도로·골목은 국소 최저점이라 min 필터에 보존된다(내려가지 않음).
  2. 하강량 상한(--max-drop, 기본 12m) — 좁고 뾰족한 실제 봉우리(문학산·마니산류)가 깎이는 것을 막는다.
  3. 가우시안 스무딩(σ 셀) — bilinear 셀 경계의 C1 꺾임(정사메시 22m 패싯의 주름 원인) 완화.
  4. 정수배 업샘플(--scale, 기본 3 → 약 10m 격자) + 마무리 가우시안.

출력은 입력과 동일 포맷(float32 LE bin + json). 원본은 terrain_raw/ 로 백업하며, 이미 백업이
있으면 **항상 백업본을 입력으로** 삼는다(반복 실행이 누적 스무딩되지 않게).

업샘플은 격자 셀 수를 scale² 배로 불린다. 도서·군 단위 지역은 원 격자부터 크고(인천 옹진군
26M 셀), ×9 를 하면 런타임 TerrainHeight 가 지역당 수백 MB 를 물게 된다 — --max-cells 로
출력 셀 수 상한을 두고 넘치면 scale 을 3→2→1 로 자동 낮춘다(스무딩 자체는 그대로 적용).

사용:
  python tools/terrain_smooth.py --regions seoul_gangnamgu
  python tools/terrain_smooth.py --seoul                 # 서울 25개 구
  python tools/terrain_smooth.py --metro                 # 특별시·광역시·특별자치시 전부
  python tools/terrain_smooth.py --sido busan,daegu      # 시도 접두어로 선택
  python tools/terrain_smooth.py --all --skip-done       # 전국(이미 스무딩된 지역 건너뜀)
  python tools/terrain_smooth.py --regions a,b --dry-run  # 통계만
"""

import argparse
import json
import os
import shutil
import numpy as np

TERRAIN_DIR = os.path.join(
    "external", "ml-agents", "UAV_test", "Assets", "Scenes", "Regions", "terrain")
RAW_DIR = os.path.join(
    "external", "ml-agents", "UAV_test", "Assets", "Scenes", "Regions", "terrain_raw")


def load(region, terrain_dir, raw_dir):
    """원본(백업본 우선) 격자와 메타를 읽는다."""
    src_dir = raw_dir if os.path.exists(
        os.path.join(raw_dir, region + ".bin")) else terrain_dir
    jpath = os.path.join(src_dir, region + ".json")
    bpath = os.path.join(src_dir, region + ".bin")
    if not (os.path.exists(jpath) and os.path.exists(bpath)):
        return None, None, None
    with open(jpath, encoding="utf-8") as fh:
        meta = json.load(fh)
    grid = np.fromfile(bpath, dtype="<f4")
    expected = int(meta["nrows"]) * int(meta["ncols"])
    if grid.size < expected:
        raise ValueError(f"{region}: bin 크기 부족 {grid.size} < {expected}")
    grid = grid[:expected].reshape(int(meta["nrows"]), int(meta["ncols"])).astype(np.float32)
    return grid, meta, src_dir


def shift_stack(grid, radius, op):
    """분리형 min/max 필터 — (2r+1)² 창을 행/열 1D 두 번으로 처리(메모리 절약)."""
    out = grid
    for axis in (0, 1):
        acc = out
        for offset in range(1, radius + 1):
            for sign in (-1, 1):
                shifted = np.roll(out, sign * offset, axis=axis)
                # 경계는 자기 값으로 클램프(roll 랩어라운드 제거)
                if axis == 0:
                    if sign > 0:
                        shifted[:offset, :] = out[:offset, :]
                    else:
                        shifted[-offset:, :] = out[-offset:, :]
                else:
                    if sign > 0:
                        shifted[:, :offset] = out[:, :offset]
                    else:
                        shifted[:, -offset:] = out[:, -offset:]
                acc = op(acc, shifted)
        out = acc
    return out


def gaussian(grid, sigma):
    """분리형 가우시안(반경 = 3σ)."""
    if sigma <= 0:
        return grid
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    out = grid
    for axis in (0, 1):
        padded = np.pad(out, ((radius, radius), (0, 0)) if axis == 0
                        else ((0, 0), (radius, radius)), mode="edge")
        acc = np.zeros_like(out)
        for index, weight in enumerate(kernel):
            if axis == 0:
                acc += weight * padded[index:index + out.shape[0], :]
            else:
                acc += weight * padded[:, index:index + out.shape[1]]
        out = acc
    return out


def upsample(grid, scale):
    """정수배 bilinear 업샘플 — 노드 정렬 유지((n-1)*scale+1)."""
    if scale <= 1:
        return grid
    rows, cols = grid.shape
    new_rows = (rows - 1) * scale + 1
    new_cols = (cols - 1) * scale + 1
    fi = np.linspace(0, rows - 1, new_rows, dtype=np.float32)
    fj = np.linspace(0, cols - 1, new_cols, dtype=np.float32)
    i0 = np.clip(np.floor(fi).astype(np.int32), 0, rows - 2)
    j0 = np.clip(np.floor(fj).astype(np.int32), 0, cols - 2)
    ti = (fi - i0)[:, None]
    tj = (fj - j0)[None, :]
    g00 = grid[np.ix_(i0, j0)]
    g01 = grid[np.ix_(i0, j0 + 1)]
    g10 = grid[np.ix_(i0 + 1, j0)]
    g11 = grid[np.ix_(i0 + 1, j0 + 1)]
    top = g00 * (1 - tj) + g01 * tj
    bottom = g10 * (1 - tj) + g11 * tj
    return (top * (1 - ti) + bottom * ti).astype(np.float32)


def effective_scale(shape, want, max_cells):
    """출력 셀 수가 max_cells 를 넘지 않는 최대 배수(≥1). 큰 격자에서 런타임 메모리 보호."""
    rows, cols = shape
    for scale in range(max(1, want), 0, -1):
        out = ((rows - 1) * scale + 1) * ((cols - 1) * scale + 1)
        if scale == 1 or out <= max_cells:
            return scale
    return 1


def process(region, args):
    grid, meta, src_dir = load(region, args.terrain_dir, args.raw_dir)
    if grid is None:
        print(f"[skip] {region}: DEM 없음")
        return False
    scale = effective_scale(grid.shape, args.scale, args.max_cells)

    opened = shift_stack(grid, args.radius, np.minimum)
    opened = shift_stack(opened, args.radius, np.maximum)
    # 실제 봉우리 보호 — 원본보다 max_drop 이상 내려가지 않게 한다.
    opened = np.maximum(opened, grid - args.max_drop)
    smoothed = gaussian(opened, args.sigma)
    # 가우시안은 뾰족한 봉우리를 opening 상한과 무관하게 더 깎는다(강남 구룡산 −19m 관측).
    # 총 하강량을 다시 묶고, 그 클램프 이음선은 뒤의 fine 가우시안이 부드럽게 만든다.
    smoothed = np.maximum(smoothed, grid - args.max_total_drop)
    fine = upsample(smoothed, scale)
    fine = gaussian(fine, args.fine_sigma)

    drop = grid - smoothed
    stats = (f"{region}: {grid.shape[0]}x{grid.shape[1]} → {fine.shape[0]}x{fine.shape[1]}"
             f" (x{scale}{'' if scale == args.scale else ' 셀상한'})"
             f" | 하강 평균 {drop.mean():+.2f}m 최대 {drop.max():.1f}m"
             f" | 상승 최대 {(-drop.min()):.1f}m"
             f" | 고도 {fine.min():.0f}~{fine.max():.0f}m")
    print("[stat] " + stats)
    if args.dry_run:
        return True

    os.makedirs(args.raw_dir, exist_ok=True)
    if src_dir != args.raw_dir:
        for extension in (".bin", ".json"):
            shutil.copy2(os.path.join(args.terrain_dir, region + extension),
                         os.path.join(args.raw_dir, region + extension))

    new_meta = dict(meta)
    new_meta["nrows"] = int(fine.shape[0])
    new_meta["ncols"] = int(fine.shape[1])
    new_meta["dlat"] = float(meta["dlat"]) / scale
    new_meta["dlon"] = float(meta["dlon"]) / scale
    new_meta["min_h"] = float(fine.min())
    new_meta["max_h"] = float(fine.max())
    new_meta["source"] = str(meta.get("source", "Copernicus GLO-30")) + " + DTM smooth v1"
    new_meta["smooth"] = {
        "radius_cells": args.radius,
        "max_drop_m": args.max_drop,
        "sigma_cells": args.sigma,
        "scale": scale,
        "fine_sigma_cells": args.fine_sigma,
    }

    fine.astype("<f4").tofile(os.path.join(args.terrain_dir, region + ".bin"))
    with open(os.path.join(args.terrain_dir, region + ".json"), "w", encoding="utf-8") as fh:
        json.dump(new_meta, fh, ensure_ascii=False, indent=1)
    return True


# 특별시·광역시·특별자치시 = 도심 DSM 편차가 가장 큰(=스무딩 효과가 가장 큰) 지역군.
METRO_PREFIXES = ("seoul", "busan", "daegu", "incheon",
                  "gwangju", "daejeon", "ulsan", "sejong")


def regions_by_prefix(terrain_dir, prefixes):
    """DEM 격자가 있는 지역 중 접두어(시도 슬러그)로 고른다. prefixes 빈 튜플=전부."""
    names = []
    for entry in sorted(os.listdir(terrain_dir)):
        if not entry.endswith(".json"):
            continue
        name = entry[:-5]
        if not prefixes or name.startswith(tuple(p + "_" for p in prefixes)):
            names.append(name)
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", default="")
    parser.add_argument("--seoul", action="store_true")
    parser.add_argument("--metro", action="store_true",
                        help="특별시·광역시·특별자치시(서울/부산/대구/인천/광주/대전/울산/세종)")
    parser.add_argument("--sido", default="", help="시도 슬러그 접두어 CSV(예: busan,daegu)")
    parser.add_argument("--all", action="store_true", dest="all_regions",
                        help="DEM 이 있는 전 지역")
    parser.add_argument("--skip-done", action="store_true", dest="skip_done",
                        help="terrain_raw 백업이 이미 있는(=스무딩 완료) 지역 건너뜀")
    parser.add_argument("--max-cells", type=int, default=8_000_000, dest="max_cells",
                        help="출력 격자 셀 수 상한 — 넘치면 --scale 을 자동으로 낮춘다")
    parser.add_argument("--radius", type=int, default=2, help="opening 반경(셀, 30m/셀)")
    parser.add_argument("--max-drop", type=float, default=12.0, dest="max_drop")
    parser.add_argument("--max-total-drop", type=float, default=15.0, dest="max_total_drop",
                        help="가우시안까지 포함한 총 하강 상한(m) — 실제 봉우리 보호")
    parser.add_argument("--sigma", type=float, default=1.2, help="조격자 가우시안 σ(셀)")
    parser.add_argument("--scale", type=int, default=3, help="업샘플 배수")
    parser.add_argument("--fine-sigma", type=float, default=2.0, dest="fine_sigma")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--terrain-dir", default=TERRAIN_DIR, dest="terrain_dir")
    parser.add_argument("--raw-dir", default=RAW_DIR, dest="raw_dir")
    args = parser.parse_args()

    regions = [r for r in args.regions.split(",") if r]
    if args.all_regions:
        regions = regions_by_prefix(args.terrain_dir, ())
    elif args.metro:
        regions = regions_by_prefix(args.terrain_dir, METRO_PREFIXES)
    elif args.sido:
        regions = regions_by_prefix(
            args.terrain_dir, tuple(s for s in args.sido.split(",") if s))
    elif args.seoul:
        regions = regions_by_prefix(args.terrain_dir, ("seoul",))
    if not regions:
        parser.error("--regions/--seoul/--metro/--sido/--all 중 하나 필요")
    if args.skip_done:
        before = len(regions)
        regions = [r for r in regions
                   if not os.path.exists(os.path.join(args.raw_dir, r + ".bin"))]
        print(f"[skip-done] 스무딩 완료 {before - len(regions)}개 제외 → {len(regions)}개 대상")

    done = 0
    for region in regions:
        if process(region, args):
            done += 1
    print(f"[done] {done}/{len(regions)} 지역 처리{' (dry-run)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
