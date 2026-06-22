[CmdletBinding()]
param(
    [string]$OverpassUrl = "http://localhost:12345/api/interpreter",
    [string]$Only = "",
    [switch]$Force,
    [string[]]$Layers = @("roads2", "feat", "poi", "area", "infra"),
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).ProviderPath
$env:MCI_OVERPASS_URL = $OverpassUrl

$scripts = @{
    roads = "tools/osm_roads.py"
    roads2 = "tools/osm_roads2.py"
    feat = "tools/osm_features.py"
    poi = "tools/osm_poi.py"
    area = "tools/osm_areas.py"
    infra = "tools/osm_infra.py"
}

Push-Location $RootDir
try {
    foreach ($layer in $Layers) {
        if (-not $scripts.ContainsKey($layer)) {
            throw "Unknown OSM layer '$layer'. Valid layers: $($scripts.Keys -join ', ')"
        }
        $args = @($scripts[$layer], "fetch")
        if ($Only) { $args += @("--only", $Only) }
        if ($Force) { $args += "--force" }
        Write-Host "[osm] $layer via $OverpassUrl"
        & $Python @args
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
finally {
    Pop-Location
}
