#!/usr/bin/env bash
# Igual que corrida_gpu.ps1, para Linux o WSL2 con GPU NVIDIA.
# Uso: bash scripts/corrida_gpu.sh adrian-rtx4060
set -euo pipefail
TAG="${1:?Indique el tag del equipo, p. ej. adrian-rtx4060}"
CPU_TAG="${2:-hansel-apple-m3}"
export ASL_HOST_TAG="$TAG"
OUT="results/$TAG"
mkdir -p "$OUT/logs"

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm \
  --format=csv -l 1 > "$OUT/logs/nvidia-smi.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT

paso() {
  local nombre=$1; shift
  echo -e "\n=== $nombre ==="
  local inicio=$SECONDS
  uv run --no-sync asl-cuda "$@" 2>&1 | tee "$OUT/logs/$nombre.log"
  printf '%s\t%s s\n' "$nombre" "$((SECONDS - inicio))" >> "$OUT/logs/duraciones.tsv"
}

nvidia-smi > "$OUT/logs/nvidia-smi-inicial.txt"
paso 00-env env
paso 01-prep prep --workers 0
paso 02-bench bench --phases all
paso 03-train-base train --config configs/train.yaml --device cuda --name base-cuda
paso 04-train-robusto train --config configs/train-robusto.yaml --device cuda --name robusto-cuda
paso 05-train-aug-cpu train --config configs/train.yaml --device cuda --augment-device cpu --max-epochs 2 --name base-cuda-aug-cpu
paso 06-train-cpu-1epoca train --config configs/train.yaml --device cpu --max-epochs 1 --name base-cpu-1epoca
paso 07-eval-base evaluate --checkpoint "$OUT/train/base-cuda/model.pt" --split test external --prefix base
paso 08-eval-robusto evaluate --checkpoint "$OUT/train/robusto-cuda/model.pt" --split test external --prefix robusto
paso 09-select select --runs base robusto
paso 10-summarize summarize
paso 11-informe informe --cpu-tag "$CPU_TAG"
