# Corrida completa en la laptop con GPU NVIDIA (Windows / PowerShell).
# Uso:  .\scripts\corrida_gpu.ps1 -Tag adrian-rtx4060
# Deja todo en results\<Tag>\ : logs de consola, nvidia-smi cada segundo,
# mediciones por fase, dos entrenamientos completos y su evaluación.
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [switch]$SinBench,
    [switch]$SinEntrenamiento
)
# "Continue": en PowerShell 5.1 la salida de stderr de un programa externo se vuelve
# error con "Stop" aunque el programa termine bien. Se valida con $LASTEXITCODE.
$ErrorActionPreference = "Continue"
$env:ASL_HOST_TAG = $Tag
$env:PYTHONUTF8 = "1"
$out = "results\$Tag"
New-Item -ItemType Directory -Force "$out\logs" | Out-Null

$query = "timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm"
$smi = Start-Process -FilePath "nvidia-smi" -ArgumentList "--query-gpu=$query --format=csv -l 1" `
    -RedirectStandardOutput "$out\logs\nvidia-smi.csv" -NoNewWindow -PassThru

function Paso([string]$nombre, [string[]]$argumentos) {
    Write-Host "`n=== $nombre ===" -ForegroundColor Cyan
    $inicio = Get-Date
    uv run --no-sync asl-cuda @argumentos 2>&1 | Tee-Object -FilePath "$out\logs\$nombre.log"
    if ($LASTEXITCODE -ne 0) { throw "Falló el paso $nombre" }
    "{0}`t{1:n1} s" -f $nombre, ((Get-Date) - $inicio).TotalSeconds | Add-Content "$out\logs\duraciones.tsv"
}

try {
    nvidia-smi | Out-File "$out\logs\nvidia-smi-inicial.txt"
    Paso "00-env" @("env")
    Paso "01-prep" @("prep", "--workers", "0")
    if (-not $SinBench) {
        Paso "02-bench" @("bench", "--phases", "all")
    }
    if (-not $SinEntrenamiento) {
        Paso "03-train-base" @("train", "--config", "configs/train.yaml", "--device", "cuda", "--name", "base-cuda")
        Paso "04-train-robusto" @("train", "--config", "configs/train-robusto.yaml", "--device", "cuda", "--name", "robusto-cuda")
        Paso "05-train-aug-cpu" @("train", "--config", "configs/train.yaml", "--device", "cuda", "--augment-device", "cpu", "--max-epochs", "2", "--name", "base-cuda-aug-cpu")
        Paso "06-train-cpu-1epoca" @("train", "--config", "configs/train.yaml", "--device", "cpu", "--max-epochs", "1", "--name", "base-cpu-1epoca")
        Paso "07-eval-base" @("evaluate", "--checkpoint", "$out/train/base-cuda/model.pt", "--split", "test", "external", "--prefix", "base")
        Paso "08-eval-robusto" @("evaluate", "--checkpoint", "$out/train/robusto-cuda/model.pt", "--split", "test", "external", "--prefix", "robusto")
        Paso "09-select" @("select", "--runs", "base", "robusto")
    }
    Paso "10-summarize" @("summarize")
}
finally {
    Stop-Process -Id $smi.Id -ErrorAction SilentlyContinue
}
Write-Host "`nListo. Revise $out y suba los cambios en su rama." -ForegroundColor Green
