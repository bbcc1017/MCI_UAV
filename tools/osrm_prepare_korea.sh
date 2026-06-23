#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${OSRM_DATA_DIR:-${ROOT_DIR}/osrm_data}"
IMAGE="${OSRM_IMAGE:-ghcr.io/project-osrm/osrm-backend}"
PBF_URL="${OSM_PBF_URL:-https://download.geofabrik.de/asia/south-korea-latest.osm.pbf}"
PBF_FILE="${OSM_PBF_FILE:-$(basename "${PBF_URL}")}"
BASENAME="${OSRM_BASENAME:-${PBF_FILE%.osm.pbf}}"
PROFILE="${OSRM_PROFILE:-/opt/car.lua}"

mkdir -p "${DATA_DIR}"

DOCKER=(docker)
if ! docker ps >/dev/null 2>&1; then
  if sudo -n docker ps >/dev/null 2>&1; then
    DOCKER=(sudo -n docker)
  else
    echo "[osrm] cannot access Docker. Add this user to the docker group or run with sudo." >&2
    exit 1
  fi
fi

if [[ ! -f "${DATA_DIR}/${PBF_FILE}" ]]; then
  echo "[osrm] downloading ${PBF_URL}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --continue-at - -o "${DATA_DIR}/${PBF_FILE}" "${PBF_URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "${DATA_DIR}/${PBF_FILE}" "${PBF_URL}"
  else
    echo "[osrm] curl or wget is required" >&2
    exit 1
  fi
else
  echo "[osrm] using existing ${DATA_DIR}/${PBF_FILE}"
fi

echo "[osrm] image=${IMAGE}"
echo "[osrm] data=${DATA_DIR}"
echo "[osrm] pbf=${PBF_FILE}"
echo "[osrm] basename=${BASENAME}"

"${DOCKER[@]}" run --rm -t -v "${DATA_DIR}:/data" "${IMAGE}" \
  osrm-extract -p "${PROFILE}" "/data/${PBF_FILE}"

"${DOCKER[@]}" run --rm -t -v "${DATA_DIR}:/data" "${IMAGE}" \
  osrm-partition "/data/${BASENAME}.osrm"

"${DOCKER[@]}" run --rm -t -v "${DATA_DIR}:/data" "${IMAGE}" \
  osrm-customize "/data/${BASENAME}.osrm"

cat <<EOF
[osrm] ready.

Start local OSRM:
  OSRM_BASENAME=${BASENAME} docker compose -f docker-compose.osrm.yml up -d

Health check:
  curl 'http://localhost:5000/route/v1/driving/126.9784,37.5666;127.0276,37.4979?overview=false'
EOF
