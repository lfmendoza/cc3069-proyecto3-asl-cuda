"""Copia al informe las tablas y figuras de las corridas.

El informe (informe/informe.qmd) incluye las tablas desde informe/tablas/ y
las figuras desde informe/figuras/. Este módulo las toma de results/<tag>/ para
que ningún número del PDF se copie a mano. Lo que todavía no existe se deja
como un aviso visible en el PDF para que no se pierda al revisar.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .constants import ROOT

INFORME = ROOT / "informe"

# Corrida oficial (laptop con GPU): misma máquina para CPU y GPU.
TABLAS_GPU = ["prep", "augment", "train", "train_lote_gpu", "inference", "pipeline", "kernel_bloques",
              "transferencias", "proyeccion_entrenamiento", "entrenamientos", "evaluacion", "fases"]
FIGURAS_GPU = ["speedup_fases", "prep_escalamiento", "speedup_augment", "speedup_train", "speedup_inference", "speedup_pipeline",
               "kernel_bloques", "monitor_bench", "monitor_base-cuda", "monitor_base-cuda-aug-cpu",
               "entrenamiento_base-cuda", "entrenamiento_robusto-cuda"]
# Capturas de la corrida oficial (evidencia/README.md).
EVIDENCIA = ["01-prep-cpu-administrador", "02-secuencial-cpu-administrador", "03-train-gpu-administrador",
             "04-train-nvidia-smi", "05-env", "06-aug-cpu-vs-gpu"]
# Segunda computadora, solo CPU.
TABLAS_CPU = ["prep", "augment", "train", "inference", "pipeline", "proyeccion_entrenamiento", "evaluacion", "fases"]
FIGURAS_CPU = ["prep_escalamiento", "monitor_bench"]


def _aviso_tabla(destino: Path, origen: str) -> None:
    destino.write_text(f"**[[Tabla pendiente: `{origen}`]]**\n", encoding="utf-8")


def _aviso_figura(destino: Path, origen: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.axis("off")
    ax.text(0.5, 0.5, f"Figura pendiente: {origen}", ha="center", va="center", fontsize=10, color="#52514e")
    fig.savefig(destino, dpi=120)
    plt.close(fig)


def _anchos(texto: str) -> str:
    """Ajusta la fila separadora de cada tabla al largo de su contenido.

    En tablas anchas pandoc reparte el ancho según la cantidad de guiones de
    esa fila; con todas iguales, la columna de nombres queda muy angosta.
    """
    lineas = texto.splitlines()
    for i, linea in enumerate(lineas):
        celdas = linea.strip().strip("|").split("|")
        if i == 0 or not linea.startswith("|") or not all(set(c.strip()) <= set("-:") and c.strip() for c in celdas):
            continue
        filas = [lineas[i - 1].strip().strip("|").split("|")]
        for fila in lineas[i + 1:]:
            if not fila.startswith("|"):
                break
            filas.append(fila.strip().strip("|").split("|"))
        filas = [f for f in filas if len(f) == len(celdas)]
        nuevas = []
        for j, celda in enumerate(celdas):
            largo = max(len(f[j].strip()) for f in filas)
            derecha = celda.strip().endswith(":")
            nuevas.append("-" * max(3, min(largo, 40) - derecha) + (":" if derecha else ""))
        lineas[i] = "|" + "|".join(nuevas) + "|"
    return "\n".join(lineas) + "\n"


def _copiar(origen: Path, destino: Path, es_figura: bool) -> bool:
    if origen.exists():
        if es_figura:
            shutil.copy2(origen, destino)
        else:
            destino.write_text(_anchos(origen.read_text(encoding="utf-8")), encoding="utf-8")
        return True
    (_aviso_figura if es_figura else _aviso_tabla)(destino, origen.relative_to(ROOT).as_posix())
    return False


def _grupo(tag: str, prefijo: str, tablas: list[str], figuras: list[str]) -> list[str]:
    base = ROOT / "results" / tag
    faltan = []
    for nombre in tablas:
        if not _copiar(base / "tablas" / f"{nombre}.md", INFORME / "tablas" / f"{prefijo}{nombre}.md", False):
            faltan.append(f"tablas/{nombre}.md")
    for nombre in figuras:
        if not _copiar(base / "figuras" / f"{nombre}.png", INFORME / "figuras" / f"{prefijo}{nombre}.png", True):
            faltan.append(f"figuras/{nombre}.png")
    return faltan


def preparar(gpu_tag: str, cpu_tag: str | None) -> list[str]:
    (INFORME / "tablas").mkdir(parents=True, exist_ok=True)
    (INFORME / "figuras").mkdir(parents=True, exist_ok=True)
    faltan = [f"{gpu_tag}/{f}" for f in _grupo(gpu_tag, "", TABLAS_GPU, FIGURAS_GPU)]

    # Matrices de confusión del modelo elegido con la mitad "selección" del dataset externo.
    eleccion = ROOT / "results" / gpu_tag / "eval" / "seleccion_modelo.json"
    elegido = json.loads(eleccion.read_text(encoding="utf-8"))["elegido"] if eleccion.exists() else None
    for split, nombre in (("test", "confusion_test"), ("external_final", "confusion_externo")):
        origen = ROOT / "results" / gpu_tag / "figuras" / f"confusion_{elegido or '<elegido>'}_{split}.png"
        if not _copiar(origen, INFORME / "figuras" / f"{nombre}.png", True):
            faltan.append(f"{gpu_tag}/figuras/{origen.name}")

    for nombre in EVIDENCIA:
        if not _copiar(ROOT / "evidencia" / f"{nombre}.png", INFORME / "figuras" / f"evidencia_{nombre}.png", True):
            faltan.append(f"../evidencia/{nombre}.png")

    if cpu_tag:
        faltan += [f"{cpu_tag}/{f}" for f in _grupo(cpu_tag, "cpu2_", TABLAS_CPU, FIGURAS_CPU)]
        # Hoja de miniaturas de las fotos del equipo (solo si hubo consentimiento y se generó).
        fotos = ROOT / "results" / cpu_tag / "eval" / "fotos-equipo_predicciones.png"
        if fotos.exists():
            shutil.copy2(fotos, INFORME / "figuras" / "fotos_equipo.png")
    return faltan
