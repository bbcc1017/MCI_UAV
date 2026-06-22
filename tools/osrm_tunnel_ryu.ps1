[CmdletBinding()]
param(
    [string]$SshHost = "ryu",
    [int]$LocalPort = 5000,
    [int]$RemotePort = 5000
)

$ErrorActionPreference = "Stop"

$existing = Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "[osrm] localhost:$LocalPort is already listening."
}
else {
    $forward = "${LocalPort}:localhost:${RemotePort}"
    $proc = Start-Process -FilePath "ssh" -ArgumentList @("-N", "-L", $forward, $SshHost) -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 2
    Write-Host "[osrm] SSH tunnel started. pid=$($proc.Id), localhost:$LocalPort -> ${SshHost}:localhost:$RemotePort"
}

$url = "http://localhost:$LocalPort/route/v1/driving/126.9784,37.5666;127.0276,37.4979?overview=false"
$response = Invoke-RestMethod -Uri $url -TimeoutSec 20
if ($response.code -ne "Ok") {
    throw "OSRM tunnel health check failed: $($response | ConvertTo-Json -Compress)"
}

Write-Host "[osrm] ready through tunnel: http://localhost:$LocalPort"
Write-Host "Use: `$env:MCI_OSRM_URL='http://localhost:$LocalPort'"
