#!/usr/bin/env python3
"""Build DISTANCE_MATRIX_FINAL.xlsx from the current hospital master via OSRM."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


DEFAULT_HOSPITAL_MASTER = "scenarios/엑셀 결합 데이터.xlsx"
DEFAULT_OUTPUT = "scenarios/DISTANCE_MATRIX_FINAL.xlsx"
DEFAULT_OSRM_URL = "http://localhost:5000"


def read_hospitals(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    required = {"요양기관명", "x좌표", "y좌표"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    bad_coords = df[df[["x좌표", "y좌표"]].isna().any(axis=1)]
    if not bad_coords.empty:
        names = bad_coords["요양기관명"].astype(str).tolist()
        raise ValueError(f"missing coordinates for hospitals: {names[:10]}")

    df = df.copy()
    df["요양기관명"] = df["요양기관명"].astype(str)
    df["x좌표"] = pd.to_numeric(df["x좌표"], errors="raise")
    df["y좌표"] = pd.to_numeric(df["y좌표"], errors="raise")
    return df.reset_index(drop=True)


def osrm_table(
    osrm_url: str,
    coords: list[tuple[float, float]],
    source_indices: list[int],
    timeout: float,
    retries: int,
) -> list[list[float | None]]:
    coord_text = ";".join(f"{lon:.7f},{lat:.7f}" for lon, lat in coords)
    url = f"{osrm_url.rstrip('/')}/table/v1/driving/{coord_text}"
    params = {
        "annotations": "distance",
        "sources": ";".join(str(i) for i in source_indices),
        "destinations": "all",
    }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            if data.get("code") != "Ok":
                raise RuntimeError(f"OSRM table failed: code={data.get('code')} body={data}")
            distances = data.get("distances")
            if not isinstance(distances, list):
                raise RuntimeError("OSRM table response has no distances")
            return distances
        except Exception as exc:  # noqa: BLE001 - retry boundary
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 * attempt, 10))
            else:
                raise RuntimeError(f"OSRM table failed after {retries} attempts: {exc}") from exc

    raise RuntimeError(f"OSRM table failed: {last_error}")


def build_distance_matrix(
    osrm_url: str,
    coords: list[tuple[float, float]],
    chunk_size: int,
    timeout: float,
    retries: int,
) -> tuple[list[list[float]], int]:
    n = len(coords)
    matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
    unreachable = 0

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        sources = list(range(start, stop))
        distances = osrm_table(
            osrm_url=osrm_url,
            coords=coords,
            source_indices=sources,
            timeout=timeout,
            retries=retries,
        )
        if len(distances) != len(sources) or any(len(row) != n for row in distances):
            raise RuntimeError(
                f"OSRM table shape mismatch for rows {start}:{stop}: "
                f"{len(distances)} x {len(distances[0]) if distances else 0}, expected {len(sources)} x {n}"
            )

        for offset, row in enumerate(distances):
            i = start + offset
            for j, distance_m in enumerate(row):
                if i == j:
                    matrix[i][j] = 0.0
                elif distance_m is None or (isinstance(distance_m, float) and math.isnan(distance_m)):
                    matrix[i][j] = 0.0
                    unreachable += 1
                else:
                    matrix[i][j] = round(float(distance_m) / 1000.0, 2)
        print(f"  OSRM rows {start + 1}-{stop}/{n}", flush=True)

    return matrix, unreachable


def write_excel(output: Path, hospitals: pd.DataFrame, matrix: list[list[float]]) -> None:
    names = hospitals["요양기관명"].astype(str).tolist()
    matrix_df = pd.DataFrame(matrix, columns=names)
    matrix_df.insert(0, "Hospital / Hospital", names)

    info_df = hospitals.copy()
    if "Index" in info_df.columns:
        info_df = info_df.drop(columns=["Index"])
    info_df.insert(0, "Index", range(len(info_df)))

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        matrix_df.to_excel(writer, sheet_name="Distance_Matrix", index=False)
        info_df.to_excel(writer, sheet_name="Hospital_Info", index=False)


def backup_output(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup = path.with_name(f"{path.stem}.bak_{stamp}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital-master", default=DEFAULT_HOSPITAL_MASTER)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--osrm-url", default=DEFAULT_OSRM_URL)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    master_path = Path(args.hospital_master)
    output_path = Path(args.output)

    hospitals = read_hospitals(master_path)
    coords = list(zip(hospitals["x좌표"].astype(float), hospitals["y좌표"].astype(float)))

    duplicate_names = int(hospitals["요양기관명"].duplicated().sum())
    if duplicate_names:
        print(f"warning: duplicated hospital names={duplicate_names}; Excel labels preserve original names")

    backup = None if args.no_backup else backup_output(output_path)
    if backup:
        print(f"backup: {backup}")

    started = time.time()
    matrix, unreachable = build_distance_matrix(
        osrm_url=args.osrm_url,
        coords=coords,
        chunk_size=args.chunk_size,
        timeout=args.timeout,
        retries=args.retries,
    )
    write_excel(output_path, hospitals, matrix)

    elapsed = round(time.time() - started, 2)
    summary = {
        "hospital_master": str(master_path),
        "output": str(output_path),
        "rows": len(hospitals),
        "duplicate_hospital_names": duplicate_names,
        "unreachable_pairs_written_as_zero": unreachable,
        "elapsed_sec": elapsed,
        "backup": str(backup) if backup else None,
        "osrm_url": args.osrm_url,
        "chunk_size": args.chunk_size,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0 if unreachable == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
