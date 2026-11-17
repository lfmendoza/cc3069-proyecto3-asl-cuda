# Mediciones en un equipo sin GPU NVIDIA (Windows / PowerShell).
# Uso:  .\scripts\corrida_cpu.ps1 -Tag hansel-cpu
param([Parameter(Mandatory = $true)][string]$Tag)
# "Continue": en PowerShell 5.1 la salida de stderr de un programa externo se vuelve
# error con "Stop" aunque el programa termine bien. Se valida con $LASTEXITCODE.
$ErrorActionPreference = "Continue"
$env:ASL_HOST_TAG = $Tag
$env:PYTHONUTF8 = "1"
$out = "results\$Tag"
New-Item -ItemType Directory -Force "$out\logs" | Out-Null

function Paso([string]$nombre, [string[]]$argumentos) {
    Write-Host "`n=== $nombre ===" -ForegroundColor Cyan
    uv run --no-sync asl-cuda @argumentos 2>&1 | Tee-Object -FilePath "$out\logs\$nombre.log"
    if ($LASTEXITCODE -ne 0) { throw "Falló el paso $nombre" }
}

Paso "00-env" @("env")
Paso "01-prep-secuencial" @("prep", "--workers", "1", "--force")
Paso "02-prep-paralelo" @("prep", "--workers", "0", "--force")
Paso "03-bench" @("bench", "--phases", "prep,augment,train,inference,pipeline", "--device", "cpu")
Paso "04-summarize" @("summarize")
Write-Host "`nListo. Resultados en $out" -ForegroundColor Green
