#!/usr/bin/env bash
# Mediciones en un equipo sin GPU NVIDIA (Linux, macOS o WSL2).
# Uso: bash scripts/corrida_cpu.sh hansel-cpu
set -euo pipefail
TAG="${1:?Indique el tag del equipo, p. ej. hansel-cpu}"
export ASL_HOST_TAG="$TAG"
OUT="results/$TAG"
mkdir -p "$OUT/logs"
paso() { local n=$1; shift; echo -e "\n=== $n ==="; uv run --no-sync asl-cuda "$@" 2>&1 | tee "$OUT/logs/$n.log"; }
paso 00-env env
paso 01-prep-secuencial prep --workers 1 --force
paso 02-prep-paralelo prep --workers 0 --force
paso 03-bench bench --phases prep,augment,train,inference,pipeline --device cpu
paso 04-summarize summarize
