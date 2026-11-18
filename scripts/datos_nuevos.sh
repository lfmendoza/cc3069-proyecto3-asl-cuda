#!/usr/bin/env bash
# Pruebas con datos nuevos: test interno, dataset externo y fotos propias si las hay.
# Necesita models/asl_mobilenetv3.pt (corrida oficial). Las fotos propias son opcionales:
# van en .local/fotos-equipo/<persona>/ (asl-cuda fotos --persona <nombre>).
# Uso: bash scripts/datos_nuevos.sh hansel-apple-m3 adrian-<gpu>
set -euo pipefail
TAG="${1:?Indique el tag de este equipo, p. ej. hansel-apple-m3}"
GPU_TAG="${2:?Indique el tag de la corrida oficial en GPU}"
export ASL_HOST_TAG="$TAG"
OUT="results/$TAG"
FOTOS=.local/fotos-equipo
mkdir -p "$OUT/logs"
paso() { local n=$1; shift; echo -e "\n=== $n ==="; uv run --no-sync asl-cuda "$@" 2>&1 | tee "$OUT/logs/$n.log"; }
paso 10-eval-final evaluate --split test external --prefix final
if [ -d "$FOTOS" ]; then  # fotos propias, opcionales
  for persona in "$FOTOS"/*/; do
    nombre=$(basename "$persona")
    paso "11-fotos-$nombre" evaluate --split "$persona" --name "fotos-$nombre"
  done
  paso 12-fotos-equipo evaluate --split "$FOTOS" --name fotos-equipo
fi
paso 13-summarize summarize
paso 14-informe informe --tag "$GPU_TAG" --cpu-tag fernando-ryzen-ai9 "$TAG"
echo -e "\nListo. Resultados en $OUT/eval"
