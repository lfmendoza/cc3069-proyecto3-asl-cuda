"""Tablas de speedup y figuras a partir de results/<tag>/bench y results/<tag>/train.

Speedup:      S = T_base / T_variante
Eficiencia:   E = S / p                 (p = procesos o hilos usados)
Karp-Flatt:   e = (1/S - 1/p) / (1 - 1/p)
    Fracción serial estimada a partir de la medición. Si e se mantiene
    constante al crecer p, el límite es la parte secuencial del algoritmo
    (Amdahl); si crece con p, el límite es el overhead de paralelizar
    (arranque de procesos, comunicación, contención de disco).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .constants import ROOT  # noqa: E402

# Paleta categórica validada (orden fijo): secuencial, CPU paralelo, GPU.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
})


def _family(variant: str, device: str | float) -> str:
    v = variant.lower()
    if str(device) == "cuda" or "gpu" in v or "cuda" in v:
        return "gpu"
    if "secuencial" in v or "1 hilo" in v or v.endswith("1 hilos") or "pil" in v:
        return "seq"
    return "cpu"


COLORS = {"seq": BLUE, "cpu": ORANGE, "gpu": AQUA}
LABELS = {"seq": "Secuencial (1 núcleo)", "cpu": "CPU multinúcleo", "gpu": "GPU (CUDA)"}


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ("variant", "device", "threads", "workers", "block", "batch") if c in frame]
    grouped = frame.groupby("variant", sort=False)
    out = grouped.agg(images=("images", "first"), mean_s=("seconds", "mean"), std_s=("seconds", "std"),
                      runs=("seconds", "size")).reset_index()
    first = grouped[keep[1:]].first().reset_index() if len(keep) > 1 else None
    if first is not None:
        out = out.merge(first, on="variant")
    out["std_s"] = out["std_s"].fillna(0.0)
    out["cv_pct"] = 100 * out["std_s"] / out["mean_s"]
    out["images_per_s"] = out["images"] / out["mean_s"]
    out["family"] = [_family(v, d) for v, d in zip(out["variant"], out.get("device", ["cpu"] * len(out)), strict=True)]
    return out


def with_speedup(table: pd.DataFrame, base_variant: str) -> pd.DataFrame:
    base = table.loc[table["variant"] == base_variant, "mean_s"]
    if base.empty:
        return table
    t = table.copy()
    # normalizar por imagen: las variantes pueden procesar cantidades distintas
    per_image = t["mean_s"] / t["images"]
    base_per_image = float(base.iloc[0]) / float(t.loc[t["variant"] == base_variant, "images"].iloc[0])
    t["speedup"] = base_per_image / per_image
    return t


def to_markdown(table: pd.DataFrame, columns: dict[str, str]) -> str:
    t = table[[c for c in columns if c in table]].rename(columns=columns)
    header = "| " + " | ".join(t.columns) + " |\n|" + "|".join("---:" if t[c].dtype.kind in "fi" else "---" for c in t.columns) + "|\n"
    rows = []
    for row in t.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                cells.append(f"{value:,}")
            elif isinstance(value, float):
                cells.append("" if np.isnan(value) else (f"{value:.3f}" if abs(value) < 10 else f"{value:,.1f}"))
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return header + "\n".join(rows) + "\n"


COLS = {"variant": "Variante", "images": "Imágenes", "mean_s": "Tiempo medio (s)", "std_s": "Desv. est. (s)",
        "cv_pct": "CV (%)", "images_per_s": "Imágenes/s", "speedup": "Speedup"}


def _bar(table: pd.DataFrame, title: str, path: Path, value: str = "speedup", xlabel: str = "Speedup (×) respecto a la línea base") -> None:
    t = table.dropna(subset=[value]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 0.42 * len(t) + 1.0))
    bars = ax.barh(t["variant"], t[value], color=[COLORS[f] for f in t["family"]], height=0.62, edgecolor="white", linewidth=2)
    for bar, v in zip(bars, t[value], strict=True):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {v:,.1f}×" if value == "speedup" else f"  {v:,.0f}",
                va="center", color=INK, fontsize=8)
    if value == "speedup":
        ax.axvline(1, color=MUTED, linewidth=0.8)
        if t[value].max() / max(t[value].min(), 1e-9) > 40:
            ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    present = [f for f in ("seq", "cpu", "gpu") if f in set(t["family"])]
    if len(present) > 1:
        from matplotlib.patches import Patch

        ax.legend([Patch(color=COLORS[f]) for f in present], [LABELS[f] for f in present], loc="lower center",
                  bbox_to_anchor=(0.4, 1.0), ncols=len(present), fontsize=8)
        ax.set_title(title, loc="left", pad=22)
    ax.margins(x=0.18)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def prep_scaling(frame: pd.DataFrame, figures: Path, name: str = "prep_escalamiento.png") -> tuple[pd.DataFrame, str]:
    threads = frame[frame.get("kind", pd.Series(index=frame.index, dtype=object)) == "threads"]
    frame = frame.drop(threads.index)
    t = aggregate(frame)
    t["p"] = t["workers"].astype(int)
    t = t.sort_values("p")
    base = float(t.loc[t["p"] == 1, "mean_s"].iloc[0])
    t["speedup"] = base / t["mean_s"]
    t["efficiency"] = t["speedup"] / t["p"]
    p, s = t["p"].to_numpy(float), t["speedup"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        t["karp_flatt"] = np.where(p > 1, (1 / s - 1 / p) / (1 - 1 / p), np.nan)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(p, p, color=MUTED, linewidth=1, linestyle="--", label="Ideal (S = p)")
    serial = float(np.nanmedian(t["karp_flatt"])) if (p > 1).any() else 0.0
    grid = np.linspace(1, p.max(), 100)
    if serial > 0:
        ax.plot(grid, 1 / (serial + (1 - serial) / grid), color=BLUE, linewidth=1.2, label=f"Amdahl con f = {serial:.3f}")
    ax.plot(p, s, color=ORANGE, linewidth=2, marker="o", markersize=7, markeredgecolor="white", label="Medido")
    if not threads.empty:
        th = aggregate(threads)
        ax.scatter(th["workers"].astype(int), base / th["mean_s"], color=BLUE, s=60, marker="s", zorder=3,
                   edgecolor="white", label="Hilos (mismo código)")
        th["p"] = th["workers"].astype(int)
        th["speedup"] = base / th["mean_s"]
        th["efficiency"] = th["speedup"] / th["p"]
        t = pd.concat([t, th], ignore_index=True)
    ax.set_xticks(sorted(set(p)))
    ax.set_xlabel("Procesos")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Fase 1: decodificación paralela en CPU", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / name)
    plt.close(fig)
    md = to_markdown(t, {"variant": "Variante", "p": "Procesos/hilos", "mean_s": "Tiempo medio (s)", "std_s": "Desv. est. (s)", "images_per_s": "Imágenes/s",
                         "speedup": "Speedup", "efficiency": "Eficiencia", "karp_flatt": "Karp-Flatt e"})
    return t, md


def block_sizes(table: pd.DataFrame, figures: Path) -> str | None:
    kern = table[table["variant"].str.startswith("CUDA kernel ") & ~table["variant"].str.contains(r"\+")].copy()
    if kern.empty:
        return None
    kern["ms_por_lote"] = 1000 * kern["mean_s"] / np.ceil(kern["images"] / 64)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.bar(kern["block"], kern["images_per_s"] / 1000, color=AQUA, width=0.6, edgecolor="white", linewidth=2)
    ax.set_xlabel("Hilos por bloque (x × y)")
    ax.set_ylabel("Miles de imágenes/s")
    ax.set_title("Kernel de entrada según tamaño de bloque", loc="left")
    fig.tight_layout()
    fig.savefig(figures / "kernel_bloques.png")
    plt.close(fig)
    return to_markdown(kern, {"block": "Bloque", "threads": "Hilos/bloque", "mean_s": "Tiempo medio (s)",
                              "ms_por_lote": "ms por lote de 64", "images_per_s": "Imágenes/s", "cv_pct": "CV (%)"})


def monitor_figure(path: Path, out: Path, title: str) -> None:
    if not path.exists():
        return
    m = pd.read_csv(path)
    if m.empty:
        return
    cores = [c for c in m.columns if c.startswith("core_")]
    has_gpu = m["gpu_util"].notna().any() and (m["gpu_util"].astype(str) != "").any()
    fig, axes = plt.subplots(2 if has_gpu else 1, 1, figsize=(7.2, 4.6 if has_gpu else 2.8), sharex=True, squeeze=False)
    ax = axes[0, 0]
    image = ax.imshow(m[cores].T.to_numpy(float), aspect="auto", cmap="Blues", vmin=0, vmax=100,
                      extent=[m["t"].min(), m["t"].max(), len(cores) - 0.5, -0.5], interpolation="nearest")
    ax.set_ylabel("Núcleo lógico")
    ax.set_title(title, loc="left")
    ax.grid(False)
    fig.colorbar(image, ax=ax, pad=0.01, label="% uso")
    if has_gpu:
        ax2 = axes[1, 0]
        ax2.plot(m["t"], pd.to_numeric(m["gpu_util"], errors="coerce"), color=AQUA, linewidth=1.5, label="Uso SM (%)")
        mem = pd.to_numeric(m["gpu_mem_used_mb"], errors="coerce") / pd.to_numeric(m["gpu_mem_total_mb"], errors="coerce") * 100
        ax2.plot(m["t"], mem, color=BLUE, linewidth=1.5, label="Memoria usada (%)")
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("GPU (%)")
        ax2.legend(fontsize=8, loc="upper right", ncols=2)
        fig.colorbar(image, ax=ax2, pad=0.01).remove()
    axes[-1, 0].set_xlabel("Tiempo (s)")
    # marcar cambios de fase
    changes = list(m.index[m["phase"] != m["phase"].shift()]) + [m.index[-1]]
    span = float(m["t"].max() - m["t"].min()) or 1.0
    for start, end in zip(changes[:-1], changes[1:], strict=True):
        for a in axes[:, 0]:
            a.axvline(m.loc[start, "t"], color=MUTED, linewidth=0.5, alpha=0.6)
        width = float(m.loc[end, "t"] - m.loc[start, "t"])
        if width / span > 0.08:
            label = str(m.loc[start, "phase"]).split(":")[-1][:22]
            ax.text((m.loc[start, "t"] + m.loc[end, "t"]) / 2, -0.7, label, ha="center", va="bottom", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def training_figure(history: Path, out: Path) -> None:
    h = pd.read_csv(history)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axes[0].plot(h["epoch"], h["train_loss"], color=BLUE, linewidth=2, marker="o", markersize=5)
    axes[0].set_title("Pérdida de entrenamiento", loc="left")
    axes[0].set_xlabel("Época")
    axes[1].plot(h["epoch"], h["train_accuracy"], color=BLUE, linewidth=2, marker="o", markersize=5, label="Train (aumentado)")
    axes[1].plot(h["epoch"], h["val_macro_f1"], color=ORANGE, linewidth=2, marker="o", markersize=5, label="Validación macro F1")
    axes[1].set_title("Exactitud / macro F1", loc="left")
    axes[1].set_xlabel("Época")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def confusion_figure(matrix: np.ndarray, out: Path, title: str) -> None:
    from .constants import CLASSES

    rows = matrix.sum(1, keepdims=True)
    norm = np.divide(matrix, rows, out=np.zeros_like(matrix, dtype=float), where=rows > 0)
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    image = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), [c[:3] for c in CLASSES], rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(CLASSES)), [c[:3] for c in CLASSES], fontsize=6.5)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    ax.set_title(title, loc="left")
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="Proporción de la fila")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def summarize(tag: str, cpu_tag: str | None = None) -> None:
    base = ROOT / "results" / tag
    bench = base / "bench"
    tables, figures = base / "tablas", base / "figuras"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    report: dict = {"tag": tag}
    baselines = {
        "augment": "torchvision-PIL (Lab 3)",
        "train": "CPU 1 hilo (secuencial)",
        "pipeline": "Lab 3 secuencial (0 workers, 1 hilo)",
    }
    titles = {"augment": "Transformación de entrada (1160 imágenes)", "train": "Paso de entrenamiento (forward + backward + AdamW)",
              "pipeline": "Época completa de punta a punta"}

    if (bench / "bench_prep.csv").exists():
        t, md = prep_scaling(pd.read_csv(bench / "bench_prep.csv"), figures)
        (tables / "prep.md").write_text(md, encoding="utf-8")
        report["prep_best_speedup"] = float(t["speedup"].max())

    for phase, base_variant in baselines.items():
        path = bench / f"bench_{phase}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        if phase == "augment":
            raw = raw[~raw["variant"].str.startswith("H2D")]
        if phase == "train" and "sweep" in raw:
            sweep = raw[raw["sweep"] == True]  # noqa: E712
            raw = raw[raw["sweep"] != True]  # noqa: E712
            if not sweep.empty:
                (tables / "train_lote_gpu.md").write_text(to_markdown(aggregate(sweep), COLS | {"batch": "Lote"}), encoding="utf-8")
        t = with_speedup(aggregate(raw), base_variant)
        (tables / f"{phase}.md").write_text(to_markdown(t, COLS), encoding="utf-8")
        _bar(t, titles[phase], figures / f"speedup_{phase}.png")
        report[f"{phase}_speedups"] = dict(zip(t["variant"], t.get("speedup", pd.Series(dtype=float)).round(2), strict=False))
        if phase == "augment":
            md = block_sizes(aggregate(pd.read_csv(path)), figures)
            if md:
                (tables / "kernel_bloques.md").write_text(md, encoding="utf-8")
            h2d = aggregate(pd.read_csv(path))
            h2d = h2d[h2d["variant"].str.startswith("H2D")]
            if not h2d.empty:
                (tables / "transferencias.md").write_text(to_markdown(h2d, COLS), encoding="utf-8")

    inf = bench / "bench_inference.csv"
    if inf.exists():
        raw = pd.read_csv(inf)
        agg = aggregate(raw)
        extra = raw.groupby("variant", sort=False)[[c for c in ("p50_ms", "p95_ms", "fps_equivalent") if c in raw]].mean().reset_index()
        agg = agg.merge(extra, on="variant", how="left")
        batch = with_speedup(agg[agg["batch"] > 1], agg[agg["batch"] > 1]["variant"].iloc[0])
        (tables / "inference.md").write_text(
            to_markdown(batch, COLS) + "\n" + to_markdown(agg[agg["batch"] == 1], {"variant": "Variante", "p50_ms": "Latencia p50 (ms)",
                                                                                     "p95_ms": "Latencia p95 (ms)", "fps_equivalent": "FPS equivalentes"}),
            encoding="utf-8")
        _bar(batch, "Inferencia por lotes de 64", figures / "speedup_inference.png")

    pipe = bench / "bench_pipeline.csv"
    prep_report = ROOT / "data/cache/prep_report.json"
    if pipe.exists():
        t = aggregate(pd.read_csv(pipe))
        per_image = dict(zip(t["variant"], t["mean_s"] / t["images"], strict=True))
        train_images = 43500
        epochs = 14
        prep_seconds = None
        if prep_report.exists():
            prep_seconds = sum(s["seconds"] for s in json.loads(prep_report.read_text())["splits"].values() if not s.get("reused"))
        rows = []
        for variant, spi in per_image.items():
            hybrid = variant.startswith("Híbrido")
            total = spi * train_images * epochs + ((prep_seconds or 0.0) if hybrid else 0.0)
            rows.append({"variant": variant, "epoch_min": spi * train_images / 60, "total_h": total / 3600})
        proj = pd.DataFrame(rows)
        proj["speedup"] = proj["total_h"].iloc[0] / proj["total_h"]
        note = (f"\nProyección lineal a {train_images:,} imágenes por época y {epochs} épocas (presupuesto máximo de la receta). "
                + (f"Las variantes híbridas incluyen una sola vez la fase 1 medida en este equipo ({prep_seconds:.0f} s).\n" if prep_seconds else
                   "Corra `asl-cuda prep` en este equipo para sumar el costo real de la fase 1.\n"))
        (tables / "proyeccion_entrenamiento.md").write_text(
            to_markdown(proj, {"variant": "Variante", "epoch_min": "Min por época", "total_h": "Horas totales", "speedup": "Speedup"}) + note,
            encoding="utf-8")

    monitor_figure(bench / "monitor.csv", figures / "monitor_bench.png", "Uso de recursos durante las mediciones")
    for run_dir in sorted((base / "train").glob("*")) if (base / "train").exists() else []:
        if (run_dir / "history.csv").exists():
            training_figure(run_dir / "history.csv", figures / f"entrenamiento_{run_dir.name}.png")
        monitor_figure(run_dir / "monitor.csv", figures / f"monitor_{run_dir.name}.png", f"Recursos durante el entrenamiento ({run_dir.name})")

    for matrix_path in sorted((base / "eval").glob("*_confusion.npy")) if (base / "eval").exists() else []:
        name = matrix_path.name.removesuffix("_confusion.npy")
        if name.endswith(("test", "external", "external_final")):
            confusion_figure(np.load(matrix_path), figures / f"confusion_{name}.png", f"Matriz de confusión: {name}")

    if cpu_tag:
        other = ROOT / "results" / cpu_tag / "bench" / "bench_prep.csv"
        if other.exists():
            t, md = prep_scaling(pd.read_csv(other), figures, f"prep_escalamiento_{cpu_tag}.png")
            (tables / f"prep_{cpu_tag}.md").write_text(md, encoding="utf-8")

    (tables / "resumen.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Tablas en {tables.relative_to(ROOT)} | figuras en {figures.relative_to(ROOT)}")
