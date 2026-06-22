#!/usr/bin/env python3
"""Fill zero hospital-to-hospital road distances with a local OSRM server.

The scenario generator currently builds distance_Hos2Hos_road.csv by looking up
hospital names in DISTANCE_MATRIX_FINAL.xlsx. If a selected hospital is absent
from that source matrix, the generated H2H road cell becomes 0. This tool uses
hospital coordinates from the current master hospital spreadsheet and OSRM's
table API to fill those zero cells.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


DEFAULT_MANIFEST = "scenarios/manifests/plan1_manifest.json"
DEFAULT_HOSPITAL_MASTER = "scenarios/엑셀 결합 데이터.xlsx"


def load_manifest(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be a region->config_path object: {path}")
    return {str(k): str(v) for k, v in data.items()}


def load_coords(path: Path) -> dict[str, tuple[float, float]]:
    df = pd.read_excel(path, engine="openpyxl")
    required = {"요양기관명", "x좌표", "y좌표"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    coords: dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        name = str(row["요양기관명"])
        lon = float(row["x좌표"])
        lat = float(row["y좌표"])
        coords[name] = (lon, lat)
    return coords


def request_table(
    osrm_url: str,
    coords: list[tuple[float, float]],
    timeout: float,
    retries: int,
) -> list[list[float | None]]:
    coord_text = ";".join(f"{lon:.7f},{lat:.7f}" for lon, lat in coords)
    url = f"{osrm_url.rstrip('/')}/table/v1/driving/{coord_text}"
    params = {"annotations": "distance"}

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if data.get("code") != "Ok":
                raise RuntimeError(f"OSRM table failed: code={data.get('code')} body={data}")
            distances = data.get("distances")
            if not distances:
                raise RuntimeError("OSRM table response has no distances")
            return distances
        except Exception as exc:  # noqa: BLE001 - retry boundary
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 * attempt, 10))
            else:
                raise RuntimeError(f"OSRM table failed after {retries} attempts: {exc}") from exc
    raise RuntimeError(f"OSRM table failed: {last_error}")


def iter_target_regions(manifest: dict[str, str], regions: Iterable[str] | None) -> Iterable[tuple[str, Path]]:
    selected = set(regions or [])
    for region, cfg in manifest.items():
        if selected and region not in selected:
            continue
        yield region, Path(cfg)


def fill_region(
    region: str,
    config_path: Path,
    coords_by_name: dict[str, tuple[float, float]],
    osrm_url: str,
    timeout: float,
    retries: int,
    dry_run: bool,
    backup: bool,
    recompute_all: bool,
) -> dict[str, int | str]:
    base = config_path.parent
    hospital_path = base / "hospital_info.csv"
    road_path = base / "distance_Hos2Hos_road.csv"

    hospitals = pd.read_csv(hospital_path, encoding="utf-8-sig")
    names = hospitals["요양기관명"].astype(str).tolist()
    missing_coord = [name for name in names if name not in coords_by_name]
    if missing_coord:
        raise ValueError(f"{region}: hospital coordinates missing: {missing_coord}")

    road = pd.read_csv(road_path, index_col=0)
    matrix = road.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = len(names)
    if matrix.shape != (n, n):
        raise ValueError(f"{region}: matrix shape {matrix.shape} != hospitals {(n, n)}")

    zero_pairs = [
        (i, j)
        for i in range(n)
        for j in range(n)
        if i != j and (pd.isna(matrix[i, j]) or float(matrix[i, j]) == 0.0)
    ]
    if not zero_pairs and not recompute_all:
        return {"region": region, "filled": 0, "zero_before": 0, "status": "skip"}

    coords = [coords_by_name[name] for name in names]
    distances_m = request_table(osrm_url, coords, timeout=timeout, retries=retries)
    if len(distances_m) != n or any(len(row) != n for row in distances_m):
        raise RuntimeError(f"{region}: OSRM table shape mismatch")

    target_pairs = [(i, j) for i in range(n) for j in range(n) if i != j] if recompute_all else zero_pairs
    filled = 0
    unreachable = 0
    for i, j in target_pairs:
        distance_m = distances_m[i][j]
        if distance_m is None:
            unreachable += 1
            continue
        matrix[i, j] = round(float(distance_m) / 1000.0, 3)
        filled += 1
    for i in range(n):
        matrix[i, i] = 0.0

    if not dry_run:
        if backup and not (road_path.with_suffix(road_path.suffix + ".bak")).exists():
            shutil.copy2(road_path, road_path.with_suffix(road_path.suffix + ".bak"))
        pd.DataFrame(matrix).to_csv(road_path, index=True, encoding="utf-8-sig")

    return {
        "region": region,
        "filled": filled,
        "unreachable": unreachable,
        "zero_before": len(zero_pairs),
        "status": "dry-run" if dry_run else "updated",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--hospital-master", default=DEFAULT_HOSPITAL_MASTER)
    parser.add_argument("--osrm-url", default="http://localhost:5000")
    parser.add_argument("--regions", nargs="*", help="Optional region names, e.g. 부산 강원")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument(
        "--recompute-all",
        action="store_true",
        help="Replace every non-diagonal H2H road cell with OSRM table distances.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(Path(args.manifest))
    coords_by_name = load_coords(Path(args.hospital_master))

    results = []
    for region, cfg in iter_target_regions(manifest, args.regions):
        result = fill_region(
            region=region,
            config_path=cfg,
            coords_by_name=coords_by_name,
            osrm_url=args.osrm_url,
            timeout=args.timeout,
            retries=args.retries,
            dry_run=args.dry_run,
            backup=not args.no_backup,
            recompute_all=args.recompute_all,
        )
        results.append(result)
        print(
            f"{result['status']} {region}: zero_before={result['zero_before']} "
            f"filled={result['filled']} unreachable={result.get('unreachable', 0)}",
            flush=True,
        )

    total_filled = sum(int(r["filled"]) for r in results)
    total_unreachable = sum(int(r.get("unreachable", 0)) for r in results)
    print(f"done: regions={len(results)} filled={total_filled} unreachable={total_unreachable}")
    return 0 if total_unreachable == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
