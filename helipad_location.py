from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://api.vworld.kr/req/data"
DATASET = "LT_P_AISHCSTRIP"
DEFAULT_API_KEY = "8B988E0F-0BB8-37AE-ACF6-504F4FBC30EC"
DEFAULT_BBOX = (124.0, 31.0, 132.5, 39.5)
DEFAULT_OUTPUT = Path(__file__).with_name("helipad_location.csv")
MAX_PAGE_SIZE = 1000

PROPERTY_FIELDS = [
    "x",
    "y",
    "long",
    "lat",
    "org_nam",
    "str_use",
    "str_nam",
    "str_adr",
    "stt_cde",
    "alt_val",
    "int_len",
    "int_grd",
    "sht_grd",
    "lnd_siz",
    "pad_siz",
    "hor_rad",
    "com_dat",
    "use_dat",
    "use_typ",
]

BASE_FIELDS = [
    "feature_id",
    "geometry_type",
    "geometry_longitude",
    "geometry_latitude",
    "ag_geom",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch nationwide Korean helipad data from VWorld and write it to CSV.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VWORLD_API_KEY", DEFAULT_API_KEY),
        help="VWorld API key. Defaults to VWORLD_API_KEY or the provided project key.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path. Defaults to {DEFAULT_OUTPUT.name}.",
    )
    parser.add_argument(
        "--bbox",
        default=",".join(str(value) for value in DEFAULT_BBOX),
        help="WGS84 bounding box as min_lon,min_lat,max_lon,max_lat.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=MAX_PAGE_SIZE,
        help=f"Rows per request. VWorld allows up to {MAX_PAGE_SIZE}.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Seconds to wait between paged API requests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    try:
        parts = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("--bbox must contain numeric values") from exc

    if len(parts) != 4:
        raise ValueError("--bbox must be min_lon,min_lat,max_lon,max_lat")

    min_lon, min_lat, max_lon, max_lat = parts
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("--bbox minimum values must be smaller than maximum values")

    return min_lon, min_lat, max_lon, max_lat


def bbox_filter(bbox: tuple[float, float, float, float]) -> str:
    min_lon, min_lat, max_lon, max_lat = bbox
    return f"BOX({min_lon},{min_lat},{max_lon},{max_lat})"


def fetch_page(
    *,
    api_key: str,
    bbox: tuple[float, float, float, float],
    page: int,
    page_size: int,
    timeout: float,
) -> dict[str, Any]:
    params = {
        "service": "data",
        "version": "2.0",
        "request": "GetFeature",
        "data": DATASET,
        "key": api_key,
        "format": "json",
        "size": str(page_size),
        "page": str(page),
        "geomFilter": bbox_filter(bbox),
        "crs": "EPSG:4326",
    }
    url = f"{API_URL}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "MCI-UAV-helipad-fetcher/1.0"})

    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(f"VWorld HTTP error {exc.code}: {exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"VWorld request failed: {exc.reason}") from exc

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"VWorld returned non-JSON data: {body[:200]}") from exc

    response = payload.get("response", {})
    status = response.get("status")
    if status != "OK":
        error = response.get("error") or response
        raise RuntimeError(f"VWorld response status is {status!r}: {error}")

    return response


def get_features(response: dict[str, Any]) -> list[dict[str, Any]]:
    collection = response.get("result", {}).get("featureCollection", {})
    features = collection.get("features", [])
    if not isinstance(features, list):
        raise RuntimeError("VWorld response did not contain a feature list")
    return features


def fetch_all_features(
    *,
    api_key: str,
    bbox: tuple[float, float, float, float],
    page_size: int,
    sleep_seconds: float,
    timeout: float,
) -> list[dict[str, Any]]:
    first = fetch_page(
        api_key=api_key,
        bbox=bbox,
        page=1,
        page_size=page_size,
        timeout=timeout,
    )
    features = get_features(first)
    page_info = first.get("page", {})
    total_pages = int(page_info.get("total", 1))

    for page in range(2, total_pages + 1):
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        response = fetch_page(
            api_key=api_key,
            bbox=bbox,
            page=page,
            page_size=page_size,
            timeout=timeout,
        )
        features.extend(get_features(response))

    return features


def clean_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def format_coord(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.15g}"
    except (TypeError, ValueError):
        return str(value)


def feature_to_row(feature: dict[str, Any], fieldnames: list[str]) -> dict[str, Any]:
    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates") or []
    lon = coordinates[0] if len(coordinates) >= 1 else ""
    lat = coordinates[1] if len(coordinates) >= 2 else ""
    lon_text = format_coord(lon)
    lat_text = format_coord(lat)

    row = {
        "feature_id": clean_value(feature.get("id")),
        "geometry_type": clean_value(geometry.get("type")),
        "geometry_longitude": lon_text,
        "geometry_latitude": lat_text,
        "ag_geom": f"POINT({lon_text} {lat_text})" if lon_text and lat_text else "",
    }

    properties = feature.get("properties") or {}
    for key, value in properties.items():
        row[key] = clean_value(value)

    return {fieldname: row.get(fieldname, "") for fieldname in fieldnames}


def build_fieldnames(features: list[dict[str, Any]]) -> list[str]:
    seen = set(BASE_FIELDS + PROPERTY_FIELDS)
    extra_fields: list[str] = []

    for feature in features:
        properties = feature.get("properties") or {}
        for key in properties:
            if key not in seen:
                seen.add(key)
                extra_fields.append(key)

    return BASE_FIELDS + PROPERTY_FIELDS + extra_fields


def write_csv(features: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = build_fieldnames(features)

    with output.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for feature in features:
            writer.writerow(feature_to_row(feature, fieldnames))


def main() -> int:
    args = parse_args()

    if not args.api_key:
        print("Missing VWorld API key. Provide --api-key or VWORLD_API_KEY.", file=sys.stderr)
        return 2

    if args.page_size < 1 or args.page_size > MAX_PAGE_SIZE:
        print(f"--page-size must be between 1 and {MAX_PAGE_SIZE}.", file=sys.stderr)
        return 2

    try:
        bbox = parse_bbox(args.bbox)
        features = fetch_all_features(
            api_key=args.api_key,
            bbox=bbox,
            page_size=args.page_size,
            sleep_seconds=args.sleep,
            timeout=args.timeout,
        )
        write_csv(features, args.output)
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(features)} helipad records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
