"""Mediciones por fase: secuencial vs CPU paralelo vs GPU.

Fases (cada una se puede correr por separado con --phases):

    prep       decodificación JPEG -> caché uint8, con 1..N procesos
    augment    transformación de entrada: torchvision/PIL (Lab 3), NumPy,
               Numba CPU, kernel CUDA (con y sin transferencias, varios tamaños de bloque)
    train      paso de entrenamiento aislado (forward + backward + AdamW)
    inference  throughput por lotes y latencia de una imagen
    pipeline   época completa sobre el subconjunto, de punta a punta

Todas usan el mismo subconjunto determinista de train. Cada medición se repite
`repeats` veces después de `warmup` corridas que se descartan (compilación JIT
de Numba, autotuning de cuDNN, caché de disco del sistema operativo).
"""

from __future__ import annotations

import gc
import json
import os
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .augment import IMAGENET_MEAN, IMAGENET_STD, launch_warp, sample_params, warp_cpu
from .config import BenchConfig, DataConfig, TrainConfig
from .constants import ROOT
from .envinfo import environment
from .manifest import absolute_paths, build_manifest, subsample
from .model import build_model, set_trainable
from .monitor import ResourceMonitor
from .prep import CachedSplit, decode_to_npy, decode_with_threads, resolve_workers
from .train import autocast, configure_backend, make_feeder

PHASES = ("prep", "augment", "train", "inference", "pipeline")


def quick_config(cfg: BenchConfig) -> BenchConfig:
    return replace(cfg, images_per_class=4, prep_images_per_class=8, repeats=1, pipeline_repeats=1, warmup=0, prep_workers=[1, 2],
                   train_batches=2, inference_batches=2, latency_samples=10,
                   threads_per_block=cfg.threads_per_block[:2], gpu_batch_sweep=[32])


class Recorder:
    def __init__(self, out_dir: Path, monitor: ResourceMonitor | None) -> None:
        self.out_dir = out_dir
        self.monitor = monitor
        self.rows: dict[str, list[dict]] = {}

    def measure(self, phase: str, variant: str, fn: Callable[[], dict], repeats: int, warmup: int, **fields) -> None:
        label = f"{phase}:{variant}"
        print(f"  {label} {fields if fields else ''}", flush=True)
        for i in range(warmup + repeats):
            gc.collect()
            ctx = self.monitor.tag(label) if self.monitor else _null()
            with ctx:
                result = fn()
            if i < warmup:
                continue
            row = {"phase": phase, "variant": variant, "repeat": i - warmup + 1, **fields, **result}
            if row.get("seconds") and row.get("images"):
                row["images_per_second"] = row["images"] / row["seconds"]
            self.rows.setdefault(phase, []).append(row)
        self.flush(phase)

    def flush(self, phase: str) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows[phase]).to_csv(self.out_dir / f"bench_{phase}.csv", index=False)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(fn: Callable[[], object], device: torch.device) -> float:
    _sync(device)
    started = time.perf_counter()
    fn()
    _sync(device)
    return time.perf_counter() - started


# ------------------------------------------------------------------ fases

def phase_prep(rec: Recorder, cfg: BenchConfig, prep_paths: list[str], paths: list[str], size: int,
               scratch: Path) -> np.ndarray:
    """Se mide con un subconjunto más grande que el resto de fases: con pocas
    imágenes el arranque de los procesos domina y la medición no dice nada."""
    target = scratch / "bench_prep_large.npy"
    workers = sorted({resolve_workers(w) for w in cfg.prep_workers})
    for w in workers:
        def run(w=w) -> dict:
            stats, _ = decode_to_npy(prep_paths, target, size, w)
            return {"images": stats.images, "seconds": stats.seconds, "load_imbalance": stats.load_imbalance,
                    "processes": len(stats.worker_busy_seconds)}
        rec.measure("prep", "secuencial" if w == 1 else f"procesos={w}", run, cfg.repeats, cfg.warmup, workers=w)
    import psutil

    threads = psutil.cpu_count(logical=False) or 4

    def with_threads() -> dict:
        return {"images": len(prep_paths), "seconds": decode_with_threads(prep_paths, size, threads)}
    rec.measure("prep", f"hilos={threads} (mismo código, GIL)", with_threads, cfg.repeats, cfg.warmup,
                workers=threads, kind="threads")
    target.unlink(missing_ok=True)
    small = scratch / "bench_prep.npy"
    decode_to_npy(paths, small, size, 0)
    return np.load(small)


def phase_augment(rec: Recorder, cfg: BenchConfig, tcfg: TrainConfig, images: np.ndarray, device: torch.device) -> None:
    import numba
    from PIL import Image

    from .secuencial import lab3_transform

    n, src_size = len(images), images.shape[1]
    out_size, batch = tcfg.image_size, cfg.batch_size
    starts = list(range(0, n, batch))
    params = [sample_params(np.random.default_rng([tcfg.seed, s]), min(batch, n - s), src_size, out_size,
                            tcfg.augment, True).packed() for s in starts]

    transform = lab3_transform(out_size, augment=True)

    def pil() -> dict:
        previous = torch.get_num_threads()
        torch.set_num_threads(1)  # ToTensor/Normalize usan hilos de torch; aquí debe ser 1 núcleo
        t = time.perf_counter()
        for s in starts:
            torch.stack([transform(Image.fromarray(img)) for img in images[s:s + batch]])
        elapsed = time.perf_counter() - t
        torch.set_num_threads(previous)
        return {"images": n, "seconds": elapsed}
    rec.measure("augment", "torchvision-PIL (Lab 3)", pil, cfg.repeats, cfg.warmup, device="cpu", threads=1)

    out = np.empty((batch, 3, out_size, out_size), np.float32)
    max_threads = numba.config.NUMBA_NUM_THREADS
    for threads in sorted({1, max_threads}):
        def numba_cpu(threads=threads) -> dict:
            numba.set_num_threads(threads)
            t = time.perf_counter()
            for s, p in zip(starts, params, strict=True):
                warp_cpu(images[s:s + batch], p, out[: len(p)])
            return {"images": n, "seconds": time.perf_counter() - t}
        rec.measure("augment", f"Numba CPU {threads} hilo{'s' if threads > 1 else ''}", numba_cpu, cfg.repeats, max(cfg.warmup, 1),
                    device="cpu", threads=threads)
    numba.set_num_threads(max_threads)

    if device.type != "cuda":
        return
    from numba import cuda

    d_mean, d_std = cuda.to_device(IMAGENET_MEAN), cuda.to_device(IMAGENET_STD)
    host_src = torch.from_numpy(images).pin_memory()
    d_src = [cuda.to_device(images[s:s + batch]) for s in starts]
    d_params = [cuda.to_device(p) for p in params]
    d_out = cuda.device_array((batch, 3, out_size, out_size), np.float32)

    for block in cfg.threads_per_block:
        block = tuple(block)

        def kernel_only(block=block) -> dict:
            start, end = cuda.event(timing=True), cuda.event(timing=True)
            start.record()
            for src, p in zip(d_src, d_params, strict=True):
                launch_warp(src, p, d_mean, d_std, d_out[: p.shape[0]], block)
            end.record()
            end.synchronize()
            return {"images": n, "seconds": cuda.event_elapsed_time(start, end) / 1000}
        rec.measure("augment", f"CUDA kernel {block[0]}x{block[1]}", kernel_only, cfg.repeats, max(cfg.warmup, 1),
                    device="cuda", block=f"{block[0]}x{block[1]}", threads=block[0] * block[1])

    block = tuple(tcfg.cuda.threads_per_block)
    out_host = torch.empty((batch, 3, out_size, out_size), dtype=torch.float32).pin_memory()

    def with_transfers() -> dict:
        h2d = kernel = d2h = 0.0
        t = time.perf_counter()
        for s, p in zip(starts, params, strict=True):
            m = len(p)
            e = [cuda.event(timing=True) for _ in range(4)]
            e[0].record()
            src = cuda.to_device(host_src[s:s + m].numpy())
            prm = cuda.to_device(p)
            e[1].record()
            launch_warp(src, prm, d_mean, d_std, d_out[:m], block)
            e[2].record()
            d_out[:m].copy_to_host(out_host[:m].numpy())
            e[3].record()
            e[3].synchronize()
            h2d += cuda.event_elapsed_time(e[0], e[1]) / 1000
            kernel += cuda.event_elapsed_time(e[1], e[2]) / 1000
            d2h += cuda.event_elapsed_time(e[2], e[3]) / 1000
        return {"images": n, "seconds": time.perf_counter() - t, "h2d_seconds": h2d, "kernel_seconds": kernel,
                "d2h_seconds": d2h}
    rec.measure("augment", "CUDA kernel + H2D + D2H", with_transfers, cfg.repeats, max(cfg.warmup, 1),
                device="cuda", block=f"{block[0]}x{block[1]}")

    def transfer(dtype) -> Callable[[], dict]:
        host = torch.empty((batch, 3, out_size, out_size) if dtype == torch.float32 else (batch, src_size, src_size, 3),
                           dtype=dtype).pin_memory()

        def run() -> dict:
            seconds = _timed(lambda: [host.to(device, non_blocking=True) for _ in starts], device)
            return {"images": n, "seconds": seconds, "bytes": host.numel() * host.element_size() * len(starts)}
        return run
    rec.measure("augment", "H2D lote uint8 200x200 (propuesta)", transfer(torch.uint8), cfg.repeats, cfg.warmup, device="cuda")
    rec.measure("augment", "H2D lote float32 224x224 (aumentar en CPU)", transfer(torch.float32), cfg.repeats, cfg.warmup, device="cuda")


def _batches(images: np.ndarray, labels: np.ndarray, tcfg: TrainConfig, batch: int, count: int,
             device: torch.device, channels_last: bool) -> list[tuple[torch.Tensor, torch.Tensor]]:
    data = CachedSplit(images, labels, {})
    feeder = make_feeder(data, tcfg, torch.device("cpu"), train=True, batch_size=batch)
    out = []
    while len(out) < count:
        for x, y in feeder:
            if len(y) < batch:
                continue
            x = x.clone().to(device)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            out.append((x, y.clone().to(device)))
            if len(out) == count:
                break
    return out


def _train_variant(tcfg: TrainConfig, device: torch.device, amp: bool, channels_last: bool, batches) -> Callable[[], dict]:
    model = build_model(tcfg.architecture, pretrained=False).to(device)
    finetune = tcfg.stages[-1]
    set_trainable(model, finetune.unfrozen_blocks)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=finetune.backbone_lr)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    def step(x, y):
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    def run() -> dict:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        seconds = _timed(lambda: [step(x, y) for x, y in batches], device)
        images = sum(len(y) for _, y in batches)
        extra = {"gpu_peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20} if device.type == "cuda" else {}
        return {"images": images, "seconds": seconds, "seconds_per_batch": seconds / len(batches), **extra}
    return run


def phase_train(rec: Recorder, cfg: BenchConfig, tcfg: TrainConfig, images: np.ndarray, labels: np.ndarray,
                device: torch.device) -> None:
    all_threads = os.cpu_count() or 1
    cpu = torch.device("cpu")
    cpu_batches = _batches(images, labels, tcfg, cfg.batch_size, cfg.train_batches, cpu, False)
    for threads in sorted({1, all_threads}):
        torch.set_num_threads(threads)
        name = "CPU 1 hilo (secuencial)" if threads == 1 else f"CPU {threads} hilos"
        rec.measure("train", name, _train_variant(tcfg, cpu, False, False, cpu_batches), cfg.repeats, max(cfg.warmup, 1),
                    device="cpu", threads=threads, batch=cfg.batch_size)
    torch.set_num_threads(all_threads)
    if device.type != "cuda":
        return
    # La propuesta original era AMP + channels_last; en la RTX 3050 con MobileNetV3 channels_last
    # resultó ~8 veces más lento, así que la propuesta final es AMP sin channels_last.
    variants = [("GPU fp32", False, False, False), ("GPU fp32 + cudnn.benchmark", False, False, True),
                ("GPU fp32 + channels_last", False, True, True), ("GPU AMP fp16 + channels_last", True, True, True),
                ("GPU AMP fp16 (propuesta)", True, False, True)]
    for name, amp, cl, bench in variants:
        torch.backends.cudnn.benchmark = bench
        batches = _batches(images, labels, tcfg, cfg.batch_size, cfg.train_batches, device, cl)
        rec.measure("train", name, _train_variant(tcfg, device, amp, cl, batches), cfg.repeats, max(cfg.warmup, 1),
                    device="cuda", batch=cfg.batch_size, amp=amp, channels_last=cl, cudnn_benchmark=bench)
    torch.backends.cudnn.benchmark = True
    for batch in cfg.gpu_batch_sweep:
        reps = max(1, (cfg.batch_size * cfg.train_batches) // batch)
        cl = tcfg.cuda.channels_last
        batches = _batches(images, labels, tcfg, batch, min(reps, max(1, len(images) // batch)), device, cl)
        rec.measure("train", f"GPU AMP lote {batch}", _train_variant(tcfg, device, True, cl, batches), cfg.repeats,
                    max(cfg.warmup, 1), device="cuda", batch=batch, amp=True, channels_last=cl, sweep=True)


def phase_inference(rec: Recorder, cfg: BenchConfig, tcfg: TrainConfig, images: np.ndarray, labels: np.ndarray,
                    device: torch.device) -> None:
    all_threads = os.cpu_count() or 1
    configs = [("CPU 1 hilo", torch.device("cpu"), 1, False), (f"CPU {all_threads} hilos", torch.device("cpu"), all_threads, False)]
    if device.type == "cuda":
        configs += [("GPU fp32", device, all_threads, False), ("GPU AMP fp16", device, all_threads, True)]
    for name, dev, threads, amp in configs:
        torch.set_num_threads(threads)
        cl = dev.type == "cuda" and tcfg.cuda.channels_last
        model = build_model(tcfg.architecture, pretrained=False).to(dev).eval()
        if cl:
            model = model.to(memory_format=torch.channels_last)
        count = cfg.inference_batches if dev.type == "cuda" else max(2, cfg.inference_batches // 4)
        batches = _batches(images, labels, tcfg, cfg.batch_size, count, dev, cl)
        single = [x[:1] for x, _ in batches][: cfg.latency_samples]
        while len(single) < cfg.latency_samples:
            single += single[: cfg.latency_samples - len(single)]

        @torch.inference_mode()
        def throughput(model=model, batches=batches, dev=dev, amp=amp) -> dict:
            def run():
                for x, _ in batches:
                    with autocast(dev, amp):
                        model(x)
            return {"images": sum(len(y) for _, y in batches), "seconds": _timed(run, dev)}

        @torch.inference_mode()
        def latency(model=model, single=single, dev=dev, amp=amp) -> dict:
            times = []
            for x in single:
                def one(x=x):
                    with autocast(dev, amp):
                        model(x)
                times.append(_timed(one, dev))
            arr = np.array(times) * 1000
            return {"images": len(single), "seconds": float(arr.sum() / 1000), "p50_ms": float(np.percentile(arr, 50)),
                    "p95_ms": float(np.percentile(arr, 95)), "fps_equivalent": float(1000 / np.percentile(arr, 50))}

        rec.measure("inference", f"{name} lote {cfg.batch_size}", throughput, cfg.repeats, max(cfg.warmup, 1),
                    device=dev.type, threads=threads, batch=cfg.batch_size, amp=amp)
        rec.measure("inference", f"{name} 1 imagen", latency, cfg.repeats, max(cfg.warmup, 1),
                    device=dev.type, threads=threads, batch=1, amp=amp)
    torch.set_num_threads(all_threads)


def phase_pipeline(rec: Recorder, cfg: BenchConfig, tcfg: TrainConfig, frame: pd.DataFrame, paths: list[str],
                   images: np.ndarray, device: torch.device) -> None:
    from .secuencial import make_loader, run_epoch

    all_threads = os.cpu_count() or 1
    labels = frame["label"].to_numpy()
    finetune = tcfg.stages[-1]

    def lab3(workers: int, threads: int) -> Callable[[], dict]:
        def run() -> dict:
            torch.set_num_threads(threads)
            model = build_model(tcfg.architecture, pretrained=False)
            set_trainable(model, finetune.unfrozen_blocks)
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=finetune.backbone_lr)
            loader = make_loader(frame, paths, tcfg.image_size, cfg.batch_size, True, workers, tcfg.seed, True)
            t = time.perf_counter()
            stats = run_epoch(model, loader, optimizer, nn.CrossEntropyLoss(), tcfg.gradient_clip_norm)
            stats["seconds"] = time.perf_counter() - t  # incluye arranque de los procesos del DataLoader
            return stats
        return run

    rec.measure("pipeline", "Lab 3 secuencial (0 workers, 1 hilo)", lab3(0, 1), cfg.pipeline_repeats, 0,
                device="cpu", threads=1, workers=0)
    rec.measure("pipeline", f"Lab 3 como se entregó (4 workers, {all_threads} hilos)", lab3(4, all_threads),
                cfg.pipeline_repeats, 0, device="cpu", threads=all_threads, workers=4)

    data = CachedSplit(images, labels, {})

    def hybrid(dev: torch.device, augment_device: str) -> Callable[[], dict]:
        def run() -> dict:
            torch.set_num_threads(all_threads)
            amp = dev.type == "cuda" and tcfg.cuda.amp
            cl = dev.type == "cuda" and tcfg.cuda.channels_last
            model = build_model(tcfg.architecture, pretrained=False).to(dev)
            set_trainable(model, finetune.unfrozen_blocks)
            if cl:
                model = model.to(memory_format=torch.channels_last)
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=finetune.backbone_lr)
            scaler = torch.amp.GradScaler(dev.type, enabled=amp)
            loss_fn = nn.CrossEntropyLoss()
            feeder = make_feeder(data, tcfg, dev, train=True, augment_device=augment_device)
            seen = 0

            def epoch():
                nonlocal seen
                for x, y in feeder:
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(dev, amp):
                        loss = loss_fn(model(x), y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), tcfg.gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    seen += len(y)
            seconds = _timed(epoch, dev)
            return {"images": seen, "seconds": seconds, "data_seconds": feeder.data_wait}
        return run

    rec.measure("pipeline", f"Híbrido CPU (caché + Numba CPU, {all_threads} hilos)", hybrid(torch.device("cpu"), "cpu"),
                cfg.pipeline_repeats, 1, device="cpu", threads=all_threads)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        rec.measure("pipeline", "Híbrido GPU, aumentación en CPU", hybrid(device, "cpu"), cfg.pipeline_repeats, 1,
                    device="cuda", augment="cpu")
        rec.measure("pipeline", "Híbrido GPU, kernel CUDA (propuesta)", hybrid(device, "gpu"), cfg.pipeline_repeats, 1,
                    device="cuda", augment="gpu")


# ------------------------------------------------------------------ entrada

def run(phases: list[str], cfg: BenchConfig, tcfg: TrainConfig, dcfg: DataConfig, tag: str, device: torch.device,
        quick: bool = False) -> Path:
    if quick:
        cfg = quick_config(cfg)
    out_dir = ROOT / "results" / tag / "bench"
    scratch = ROOT / "results" / tag / "scratch"
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_backend(device, tcfg)

    manifest = build_manifest(dcfg)
    frame = subsample(manifest[manifest["split"] == "train"], cfg.images_per_class).reset_index(drop=True)
    paths = absolute_paths(frame)
    labels = frame["label"].to_numpy()
    env = environment(device)
    env.update(bench_config=cfg.__dict__, images=len(paths), phases=phases, quick=quick)
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")
    print(f"Equipo: {env['cpu']} | {env['logical_cores']} hilos | GPU: {env['gpu'].get('name', 'no disponible')}")
    print(f"Subconjunto: {len(paths)} imágenes ({cfg.images_per_class} por clase) | fases: {', '.join(phases)}")

    monitor = ResourceMonitor(out_dir / "monitor.csv", interval=0.25).start()
    rec = Recorder(out_dir, monitor)
    try:
        target = scratch / "bench_prep.npy"
        if "prep" in phases:
            prep_frame = subsample(manifest[manifest["split"] == "train"], cfg.prep_images_per_class)
            images = phase_prep(rec, cfg, absolute_paths(prep_frame), paths, dcfg.cache_size, scratch)
        else:
            if not target.exists() or np.load(target, mmap_mode="r").shape[0] != len(paths):
                decode_to_npy(paths, target, dcfg.cache_size, 0)
            images = np.load(target)
        if "augment" in phases:
            phase_augment(rec, cfg, tcfg, images, device)
        if "train" in phases:
            phase_train(rec, cfg, tcfg, images, labels, device)
        if "inference" in phases:
            phase_inference(rec, cfg, tcfg, images, labels, device)
        if "pipeline" in phases:
            phase_pipeline(rec, cfg, tcfg, frame, paths, images, device)
    finally:
        monitor.stop()
    print(f"Resultados en {out_dir.relative_to(ROOT)}")
    return out_dir
