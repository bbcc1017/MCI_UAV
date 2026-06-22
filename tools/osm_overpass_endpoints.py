# -*- coding: utf-8 -*-
"""Shared Overpass endpoint selection for MCI_UAV OSM fetch scripts."""

from __future__ import annotations

import os


DEFAULT_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]


def _normalize(url: str) -> str:
    url = url.strip().rstrip("/")
    if url.endswith("/api"):
        return f"{url}/interpreter"
    return url


def overpass_endpoints(defaults: list[str] | None = None) -> list[str]:
    configured = os.environ.get("MCI_OVERPASS_URL") or os.environ.get("OVERPASS_URL")
    if configured:
        return [_normalize(configured)]
    return list(defaults or DEFAULT_ENDPOINTS)


def overpass_endpoint(default: str = DEFAULT_ENDPOINTS[0]) -> str:
    return overpass_endpoints([default])[0]
