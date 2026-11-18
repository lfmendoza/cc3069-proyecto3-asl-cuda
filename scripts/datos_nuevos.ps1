# Pruebas con datos nuevos: test interno, dataset externo y fotos del equipo.
# Necesita models\asl_mobilenetv3.pt (corrida oficial) y las fotos en
# .local\fotos-equipo\<persona>\ (asl-cuda fotos --persona <nombre>).
# Uso:  .\scripts\datos_nuevos.ps1 -Tag hansel-apple-m3 -GpuTag adrian-<gpu>
param([Parameter(Mandatory = $true)][string]$Tag, [Parameter(Mandatory = $true)][string]$GpuTag)
$ErrorActionPreference = "Continue"
$env:ASL_HOST_TAG = $Tag
$env:PYTHONUTF8 = "1"
$out = "results\$Tag"
$fotos = ".local\fotos-equipo"
New-Item -ItemType Directory -Force "$out\logs" | Out-Null

function Paso([string]$nombre, [string[]]$argumentos) {
    Write-Host "`n=== $nombre ===" -ForegroundColor Cyan
    uv run --no-sync asl-cuda @argumentos 2>&1 | Tee-Object -FilePath "$out\logs\$nombre.log"
    if ($LASTEXITCODE -ne 0) { throw "Falló el paso $nombre" }
}

Paso "10-eval-final" @("evaluate", "--split", "test", "external", "--prefix", "final")
foreach ($persona in Get-ChildItem -Directory $fotos) {
    Paso "11-fotos-$($persona.Name)" @("evaluate", "--split", $persona.FullName, "--name", "fotos-$($persona.Name)")
}
Paso "12-fotos-equipo" @("evaluate", "--split", $fotos, "--name", "fotos-equipo")
Paso "13-summarize" @("summarize")
Paso "14-informe" @("informe", "--tag", $GpuTag, "--cpu-tag", "fernando-ryzen-ai9", $Tag)
Write-Host "`nListo. Resultados en $out\eval" -ForegroundColor Green
