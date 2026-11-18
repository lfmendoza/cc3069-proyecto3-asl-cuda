# Pruebas con datos nuevos: test interno, dataset externo y fotos propias si las hay.
# Necesita models\asl_mobilenetv3.pt (corrida oficial) y las fotos en
# .local\fotos-equipo\<persona>\ (asl-cuda fotos --persona <nombre>).
# Uso:  .\scripts\datos_nuevos.ps1 -Tag hansel-apple-m3 -GpuTag adrian-<gpu>
param([Parameter(Mandatory = $true)][string]$Tag, [Parameter(Mandatory = $true)][string]$GpuTag)
$ErrorActionPreference = "Continue"
$env:ASL_HOST_TAG = $Tag
$env:PYTHONUTF8 = "1"
# La consola lee la salida de Python como UTF-8 (si no, los acentos salen mal en los logs).
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$out = "results\$Tag"
$fotos = ".local\fotos-equipo"
New-Item -ItemType Directory -Force "$out\logs" | Out-Null

function Paso([string]$nombre, [string[]]$argumentos) {
    Write-Host "`n=== $nombre ===" -ForegroundColor Cyan
    # "$_" deja los avisos de stderr como texto normal; el log se guarda en UTF-8.
    uv run --no-sync asl-cuda @argumentos 2>&1 | ForEach-Object { "$_" } | Tee-Object -Variable salida
    $codigo = $LASTEXITCODE
    $salida | Out-File -Encoding utf8 "$out\logs\$nombre.log"
    if ($codigo -ne 0) { throw "Falló el paso $nombre" }
}

Paso "10-eval-final" @("evaluate", "--split", "test", "external", "--prefix", "final")
if (Test-Path $fotos) {  # fotos propias, opcionales
    foreach ($persona in Get-ChildItem -Directory $fotos) {
        Paso "11-fotos-$($persona.Name)" @("evaluate", "--split", $persona.FullName, "--name", "fotos-$($persona.Name)")
    }
    Paso "12-fotos-equipo" @("evaluate", "--split", $fotos, "--name", "fotos-equipo")
}
Paso "13-summarize" @("summarize")
Paso "14-informe" @("informe", "--tag", $GpuTag, "--cpu-tag", "fernando-ryzen-ai9", $Tag)
Write-Host "`nListo. Resultados en $out\eval" -ForegroundColor Green
