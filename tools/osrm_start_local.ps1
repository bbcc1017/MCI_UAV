[CmdletBinding()]
param(
    [int]$Port = 5000,
    [string]$BaseName = "south-korea-latest"
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).ProviderPath

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "docker command not found. Install Docker Desktop or use tools/osrm_tunnel_ryu.ps1 for the SSH-backed fallback."
}

$env:OSRM_PORT = [string]$Port
$env:OSRM_BASENAME = $BaseName

Push-Location $RootDir
try {
    & docker compose -f docker-compose.osrm.yml up -d
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $url = "http://localhost:$Port/route/v1/driving/126.9784,37.5666;127.0276,37.4979?overview=false"
    $response = Invoke-RestMethod -Uri $url -TimeoutSec 20
    if ($response.code -ne "Ok") {
        throw "OSRM health check failed: $($response | ConvertTo-Json -Compress)"
    }
    Write-Host "[osrm] local server ready: http://localhost:$Port"
    Write-Host "Use: `$env:MCI_OSRM_URL='http://localhost:$Port'"
}
finally {
    Pop-Location
}
