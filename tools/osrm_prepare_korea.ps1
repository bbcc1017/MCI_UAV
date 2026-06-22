[CmdletBinding()]
param(
    [string]$DataDir = "",
    [string]$PbfUrl = "https://download.geofabrik.de/asia/south-korea-latest.osm.pbf",
    [string]$Image = "ghcr.io/project-osrm/osrm-backend",
    [string]$Profile = "/opt/car.lua"
)

$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).ProviderPath
if (-not $DataDir) {
    $DataDir = Join-Path $RootDir "osrm_data"
}
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
$DataDir = (Resolve-Path $DataDir).ProviderPath

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "docker command not found. Install Docker Desktop, then rerun this script."
}

$PbfFile = Split-Path $PbfUrl -Leaf
$BaseName = $PbfFile -replace "\.osm\.pbf$", ""
$PbfPath = Join-Path $DataDir $PbfFile

if (-not (Test-Path $PbfPath)) {
    Write-Host "[osrm] downloading $PbfUrl"
    Invoke-WebRequest -Uri $PbfUrl -OutFile $PbfPath
}
else {
    Write-Host "[osrm] using existing $PbfPath"
}

$Mount = "${DataDir}:/data"
Write-Host "[osrm] image=$Image"
Write-Host "[osrm] data=$DataDir"
Write-Host "[osrm] basename=$BaseName"

& docker run --rm -t -v $Mount $Image osrm-extract -p $Profile "/data/$PbfFile"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& docker run --rm -t -v $Mount $Image osrm-partition "/data/$BaseName.osrm"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& docker run --rm -t -v $Mount $Image osrm-customize "/data/$BaseName.osrm"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[osrm] ready."
Write-Host "Start: `$env:OSRM_BASENAME='$BaseName'; docker compose -f docker-compose.osrm.yml up -d"
