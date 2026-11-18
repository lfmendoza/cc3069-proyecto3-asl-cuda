"""Punto de entrada: `uv run asl-cuda <comando>`."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .constants import EXTERNAL_DATASET, ROOT, TRAIN_DATASET


def _tag(value: str | None) -> str:
    tag = value or os.environ.get("ASL_HOST_TAG")
    if not tag:
        raise SystemExit("Indique --tag (p. ej. adrian-rtx4060) o defina ASL_HOST_TAG para separar resultados por equipo.")
    return tag


def cmd_env(args) -> None:
    from .envinfo import environment

    info = environment()
    print(json.dumps(info, indent=2, default=str))
    gpu = info["gpu"]
    if gpu.get("available"):
        print(f"\nCUDA OK: {gpu['name']} (cc {gpu['capability']}, {gpu['memory_gb']} GB), driver {gpu.get('driver')}")
        try:
            import numpy as np
            from numba import cuda

            from .augment import IMAGENET_MEAN, IMAGENET_STD, cuda_kernel, grid_for, sample_params, warp_numpy
            from .config import load_train_config

            cfg = load_train_config()
            src = np.random.default_rng(0).integers(0, 256, (4, 200, 200, 3), dtype=np.uint8)
            params = sample_params(np.random.default_rng(1), 4, 200, 224, cfg.augment, True).packed()
            out = cuda.device_array((4, 3, 224, 224), np.float32)
            grid, block = grid_for(4, 224, 224, (16, 16))
            cuda_kernel()[grid, block](cuda.to_device(src), cuda.to_device(params), cuda.to_device(IMAGENET_MEAN),
                                       cuda.to_device(IMAGENET_STD), out)
            diff = float(abs(out.copy_to_host() - warp_numpy(src, params, 224)).max())
            print(f"Kernel Numba CUDA vs referencia NumPy: diferencia máxima {diff:.2e} -> {'OK' if diff < 1e-3 else 'REVISAR'}")
        except Exception as exc:  # noqa: BLE001
            print(f"Numba CUDA no pudo ejecutar el kernel: {exc!r}")
    else:
        print("\nSin GPU CUDA visible: solo se pueden correr las variantes CPU.")


def cmd_download(args) -> None:
    raw = ROOT / "data/raw"
    external = ROOT / "data/external/asl-alphabet-test"
    jobs = [(TRAIN_DATASET, raw, raw / "asl_alphabet_train"), (EXTERNAL_DATASET, external, external)]
    for slug, dest, marker in jobs:
        if marker.exists() and any(marker.iterdir()):
            print(f"{slug}: ya existe en {dest.relative_to(ROOT)}")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Descargando {slug} ...")
        subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"], check=True)
        nested = dest / dest.name
        if nested.is_dir():
            # el zip del dataset externo trae una copia idéntica anidada; se elimina para no contar doble
            shutil.rmtree(nested)


def cmd_prep(args) -> None:
    import numpy as np
    import pandas as pd

    from .config import load_data_config
    from .manifest import absolute_paths, build_manifest, check_cross_split_duplicates, manifest_digest
    from .prep import build_split_cache, load_shas

    dcfg = load_data_config(args.data)
    started = time.perf_counter()
    manifest = build_manifest(dcfg)
    cache_dir = dcfg.path("cache_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(cache_dir / "manifest.csv", index=False)
    print(f"Manifest: {len(manifest)} imágenes | " + ", ".join(f"{k}={v}" for k, v in manifest["split"].value_counts().items()))
    report = {"manifest_sha256": manifest_digest(manifest), "splits": {}}
    for split in ("train", "validation", "test"):
        part = manifest[manifest["split"] == split]
        meta = build_split_cache(absolute_paths(part), part["label"].to_numpy(), cache_dir, split, dcfg.cache_size,
                                 args.workers, manifest_digest(part), force=args.force)
        state = "reutilizada" if meta.get("reused") else f"{meta['seconds']:.1f}s, {meta['images_per_second']:.0f} img/s, {meta['workers']} procesos"
        print(f"  {split:<10} {len(part):>6} imágenes -> caché {state}")
        report["splits"][split] = meta
    order = ("train", "validation", "test")
    shas = np.concatenate([load_shas(cache_dir, s, dcfg.cache_size) for s in order])
    ordered = pd.concat([manifest[manifest["split"] == s] for s in order])
    report["cross_split_byte_duplicates"] = check_cross_split_duplicates(ordered, shas)
    report["total_seconds"] = time.perf_counter() - started
    (cache_dir / "prep_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Duplicados exactos entre particiones: {report['cross_split_byte_duplicates']}")


def _load_caches(dcfg, tcfg, splits=("train", "validation"), in_memory=False):
    from .prep import load_split

    data = {s: load_split(dcfg.path("cache_dir"), s, dcfg.cache_size, in_memory) for s in splits}
    if tcfg.max_per_class:
        import numpy as np

        from .prep import CachedSplit

        for s, d in data.items():
            # muestras equiespaciadas a lo largo de cada video, no los primeros frames
            keep = np.concatenate([idx[np.linspace(0, len(idx) - 1, min(len(idx), tcfg.max_per_class)).astype(int)]
                                   for idx in (np.flatnonzero(d.labels == c) for c in np.unique(d.labels))])
            data[s] = CachedSplit(np.asarray(d.images[np.sort(keep)]), d.labels[np.sort(keep)], d.meta)
    return data


def cmd_train(args) -> None:
    from .config import load_data_config, load_train_config
    from .monitor import maybe_monitor
    from .train import fit, resolve_device

    dcfg, tcfg = load_data_config(args.data), load_train_config(args.config)
    if args.augment_device:
        from dataclasses import replace

        tcfg = replace(tcfg, cuda=replace(tcfg.cuda, augment_device=args.augment_device))
    device = resolve_device(args.device)
    tag = _tag(args.tag)
    run_name = args.name or f"{Path(args.config).stem}-{device.type}"
    out_dir = ROOT / "results" / tag / "train" / run_name
    data = _load_caches(dcfg, tcfg, in_memory=args.in_memory)
    with maybe_monitor(out_dir / "monitor.csv" if not args.no_monitor else None) as monitor:
        with monitor.tag(f"train-{device.type}"):
            summary = fit(tcfg, data["train"], data["validation"], device, out_dir, max_epochs=args.max_epochs)
    if args.export:
        target = ROOT / args.export
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_dir / "model.pt", target)
        print(f"Modelo exportado a {target.relative_to(ROOT)}")
    print(json.dumps({k: summary[k] for k in ("device", "epochs_run", "best_epoch", "total_seconds")}, indent=2))


def cmd_evaluate(args) -> None:
    from .config import load_data_config, load_train_config
    from .evaluate import evaluate_split, folder_split
    from .prep import load_split
    from .train import resolve_device

    dcfg, tcfg = load_data_config(args.data), load_train_config(args.config)
    device = resolve_device(args.device)
    tag = _tag(args.tag)
    out_dir = ROOT / "results" / tag / "eval"
    checkpoint = ROOT / args.checkpoint
    targets = args.split or ["test"]
    for target in targets:
        if target in {"train", "validation", "test"}:
            data, name, paths, frame = load_split(dcfg.path("cache_dir"), target, dcfg.cache_size), target, None, None
        else:
            folder = dcfg.path("external_dir") if target == "external" else Path(target)
            data, frame = folder_split(folder, dcfg.cache_size)
            name, paths = ("external" if target == "external" else (args.name or folder.name)), frame["path"].tolist()
        jobs = [(name, data, paths)]
        if target == "external":
            # mitad "seleccion" para comparar recetas, mitad "final" para reportar una sola vez
            from .prep import CachedSplit

            rank = frame.groupby("label").cumcount().to_numpy()
            for part, mask in (("external_seleccion", rank % 2 == 0), ("external_final", rank % 2 == 1)):
                jobs.append((part, CachedSplit(data.images[mask], data.labels[mask], data.meta),
                             [p for p, m in zip(paths, mask, strict=True) if m]))
        for name, data, paths in jobs:
            name = f"{args.prefix}_{name}" if args.prefix else name
            result = evaluate_split(checkpoint, data, tcfg, device, out_dir, name, paths)
            print(f"{name:<20} n={result['images']:>6}  acc={result['accuracy']:.4f}  macroF1={result['macro_f1']:.4f}  "
                  f"top3={result['top3_accuracy']:.4f}  ({result['images_per_second']:.0f} img/s en {device})")


def cmd_select(args) -> None:
    tag = _tag(args.tag)
    base = ROOT / "results" / tag
    scores = {}
    for run in args.runs:
        path = base / "eval" / f"{run}_external_seleccion_metrics.json"
        if not path.exists():
            raise SystemExit(f"Falta {path.relative_to(ROOT)}: corra evaluate --prefix {run} --split test external")
        scores[run] = json.loads(path.read_text(encoding="utf-8"))
    best = max(scores, key=lambda r: (scores[r]["macro_f1"], scores[r]["accuracy"]))
    checkpoint = ROOT / scores[best]["checkpoint"]
    target = ROOT / args.export
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, target)
    decision = {"elegido": best, "criterio": "macro F1 en external_seleccion (435 fotos)",
                "candidatos": {r: {"macro_f1": v["macro_f1"], "accuracy": v["accuracy"], "checkpoint": v["checkpoint"]}
                               for r, v in scores.items()}}
    (base / "eval" / "seleccion_modelo.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    for run, v in scores.items():
        print(f"{run:<10} macroF1 selección = {v['macro_f1']:.4f}" + ("   <- elegido" if run == best else ""))
    print(f"Copiado a {target.relative_to(ROOT)}")


def cmd_predict(args) -> None:
    from .config import load_train_config
    from .evaluate import predict_files
    from .train import resolve_device

    rows = predict_files(ROOT / args.checkpoint, args.images, load_train_config(args.config), resolve_device(args.device))
    for row in rows:
        print(f"{row['path']}: {row['prediction']}  " + "  ".join(f"{c} {p:.2f}" for c, p in row["top"]))


def cmd_bench(args) -> None:
    from .bench import PHASES, run
    from .config import load_bench_config, load_data_config, load_train_config
    from .train import resolve_device

    phases = PHASES if args.phases == "all" else tuple(p.strip() for p in args.phases.split(","))
    unknown = set(phases) - set(PHASES)
    if unknown:
        raise SystemExit(f"Fases desconocidas: {sorted(unknown)}")
    run(list(phases), load_bench_config(args.bench), load_train_config(args.config), load_data_config(args.data),
        _tag(args.tag), resolve_device(args.device), quick=args.quick)


def cmd_summarize(args) -> None:
    from .summarize import summarize

    summarize(_tag(args.tag), cpu_tag=args.cpu_tag)


def cmd_demo(args) -> None:
    from .demo import run_demo

    run_demo(ROOT / args.checkpoint, args.device, args.camera, args.config)


def cmd_informe(args) -> None:
    from .informe import preparar

    faltan = preparar(_tag(args.tag), args.cpu_tag)
    print("Tablas y figuras copiadas a informe/tablas e informe/figuras")
    if faltan:
        print(f"Quedaron {len(faltan)} avisos de pendiente en el PDF:")
        for item in faltan:
            print(f"  results/{item}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="asl-cuda", description="ASL con preparación en CPU y entrenamiento en GPU")
    parser.add_argument("--data", default="configs/data.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("env", help="Hardware, versiones y prueba rápida del kernel CUDA").set_defaults(func=cmd_env)
    sub.add_parser("download", help="Descarga los datasets de Kaggle").set_defaults(func=cmd_download)

    p = sub.add_parser("prep", help="Fase 1 (CPU): manifest + caché uint8 decodificada en paralelo")
    p.add_argument("--workers", type=int, default=0, help="procesos (0 = todos los núcleos lógicos, 1 = secuencial)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_prep)

    p = sub.add_parser("train", help="Fase 2: entrenamiento (GPU o CPU)")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--device", default="auto")
    p.add_argument("--tag")
    p.add_argument("--name")
    p.add_argument("--max-epochs", type=int)
    p.add_argument("--augment-device", choices=["gpu", "cpu"])
    p.add_argument("--in-memory", action="store_true", help="cargar la caché completa en RAM en vez de memmap")
    p.add_argument("--export", help="copiar el mejor modelo a esta ruta (p. ej. models/asl_mobilenetv3.pt)")
    p.add_argument("--no-monitor", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="Métricas en test, dataset externo o una carpeta de fotos")
    p.add_argument("--checkpoint", default="models/asl_mobilenetv3.pt")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--device", default="auto")
    p.add_argument("--split", nargs="+", help="test | validation | external | ruta a carpeta")
    p.add_argument("--name", help="nombre para una carpeta de fotos")
    p.add_argument("--prefix", help="prefijo de los archivos de salida (p. ej. base, robusto)")
    p.add_argument("--tag")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("select", help="Elige el modelo final con la mitad 'seleccion' del dataset externo")
    p.add_argument("--tag")
    p.add_argument("--runs", nargs="+", default=["base", "robusto"], help="prefijos usados en evaluate")
    p.add_argument("--export", default="models/asl_mobilenetv3.pt")
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("predict", help="Predice la letra de imágenes sueltas")
    p.add_argument("images", nargs="+")
    p.add_argument("--checkpoint", default="models/asl_mobilenetv3.pt")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("bench", help="Mediciones por fase (secuencial, CPU paralelo, GPU)")
    p.add_argument("--phases", default="all", help="all o lista: prep,augment,train,inference,pipeline")
    p.add_argument("--bench", default="configs/bench.yaml")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--device", default="auto")
    p.add_argument("--tag")
    p.add_argument("--quick", action="store_true", help="versión mínima para probar que todo corre")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("summarize", help="Tablas de speedup y figuras a partir de results/<tag>")
    p.add_argument("--tag")
    p.add_argument("--cpu-tag", help="resultados CPU de otro equipo para comparar escalamiento")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("informe", help="Copia tablas y figuras de results/<tag> al informe")
    p.add_argument("--tag", help="equipo de la corrida oficial (GPU)")
    p.add_argument("--cpu-tag", nargs="+", help="computadoras de referencia sin GPU, en orden (cpu2_, cpu3_, ...)")
    p.set_defaults(func=cmd_informe)

    p = sub.add_parser("demo", help="Reconocimiento en vivo con la cámara")
    p.add_argument("--checkpoint", default="models/asl_mobilenetv3.pt")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--device", default="auto")
    p.add_argument("--camera", type=int, default=0)
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
