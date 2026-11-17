"""Fase 2: entrenamiento de la red. Mismo código para CPU y GPU.

En GPU se activan las optimizaciones que solo tienen sentido ahí:
precisión mixta (fp16 en Tensor Cores con GradScaler), layout channels_last
(lo prefieren las convoluciones de cuDNN con Tensor Cores), cudnn.benchmark
(elige el algoritmo de convolución más rápido para estas formas) y lotes
armados por el kernel Numba sin pasar por el CPU.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import TrainConfig
from .envinfo import environment
from .feeders import CpuFeeder, CudaFeeder
from .metrics import classification_metrics
from .model import build_model, count_parameters, parameter_groups, save_checkpoint, set_trainable
from .prep import CachedSplit


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Se pidió CUDA pero torch.cuda.is_available() es False. Revise el driver y el extra instalado.")
    return torch.device(name)


def configure_backend(device: torch.device, cfg: TrainConfig, threads: int | None = None) -> int:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = cfg.cuda.cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.manual_seed_all(cfg.seed)
    wanted = threads if threads is not None else cfg.cpu.threads
    wanted = wanted if wanted and wanted > 0 else (os.cpu_count() or 1)
    torch.set_num_threads(wanted)
    return wanted


def make_feeder(data: CachedSplit, cfg: TrainConfig, device: torch.device, train: bool,
                limit: int | None = None, batch_size: int | None = None, augment_device: str | None = None,
                block: tuple[int, int] | None = None):
    batch = batch_size or (cfg.batch_size if train else cfg.eval_batch_size)
    common = dict(batch_size=batch, out_size=cfg.image_size, augment=cfg.augment, train=train,
                  seed=cfg.seed, limit=limit)
    where = augment_device or cfg.cuda.augment_device
    if device.type == "cuda" and where == "gpu":
        return CudaFeeder(data, **common, device=device, channels_last=cfg.cuda.channels_last,
                          block=block or cfg.cuda.threads_per_block)
    return CpuFeeder(data, **common, device=device,
                     channels_last=cfg.cuda.channels_last if device.type == "cuda" else False)


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


@torch.inference_mode()
def predict(model: nn.Module, feeder, device: torch.device, amp: bool) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    truths, probs = [], []
    started = time.perf_counter()
    for images, labels in feeder:
        with autocast(device, amp):
            logits = model(images)
        probs.append(torch.softmax(logits.float(), dim=1))
        truths.append(labels)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    return torch.cat(truths).cpu().numpy(), torch.cat(probs).cpu().numpy(), seconds


def train_step(model, images, labels, loss_fn, optimizer, scaler, device, amp: bool, clip: float):
    optimizer.zero_grad(set_to_none=True)
    with autocast(device, amp):
        logits = model(images)
        loss = loss_fn(logits, labels)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), clip)
    scaler.step(optimizer)
    scaler.update()
    return logits, loss


def fit(cfg: TrainConfig, train_data: CachedSplit, val_data: CachedSplit, device: torch.device,
        out_dir: Path, max_epochs: int | None = None, limit: int | None = None, log=print) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    threads = configure_backend(device, cfg)
    amp = cfg.cuda.amp and device.type == "cuda"
    model = build_model(cfg.architecture, cfg.pretrained).to(device)
    if device.type == "cuda" and cfg.cuda.channels_last:
        model = model.to(memory_format=torch.channels_last)

    train_feeder = make_feeder(train_data, cfg, device, train=True, limit=limit)
    val_feeder = make_feeder(val_data, cfg, device, train=False, limit=limit)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    history: list[dict[str, Any]] = []
    best = {"macro_f1": -1.0, "state": None, "epoch": 0, "stage": ""}
    started = time.perf_counter()
    global_epoch = 0
    budget = max_epochs if max_epochs is not None else math.inf

    for stage in cfg.stages:
        if global_epoch >= budget:
            break
        set_trainable(model, stage.unfrozen_blocks)
        optimizer = torch.optim.AdamW(parameter_groups(model, stage.backbone_lr, stage.head_lr), weight_decay=cfg.weight_decay)
        epochs = int(min(stage.epochs, budget - global_epoch))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs * len(train_feeder)), eta_min=1e-6)
        stale = 0
        for local_epoch in range(1, epochs + 1):
            global_epoch += 1
            train_feeder.set_epoch(global_epoch)
            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            epoch_start = time.perf_counter()
            loss_sum = torch.zeros((), device=device)
            correct = torch.zeros((), device=device, dtype=torch.long)
            seen = 0
            for images, labels in train_feeder:
                logits, loss = train_step(model, images, labels, loss_fn, optimizer, scaler, device, amp, cfg.gradient_clip_norm)
                scheduler.step()
                loss_sum += loss.detach().float() * len(labels)
                correct += (logits.detach().argmax(1) == labels).sum()
                seen += len(labels)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_seconds = time.perf_counter() - epoch_start

            truth, probs, val_seconds = predict(model, val_feeder, device, amp)
            val = classification_metrics(truth, probs)
            row = {
                "epoch": global_epoch, "stage": stage.name, "stage_epoch": local_epoch,
                "train_loss": float(loss_sum) / seen, "train_accuracy": int(correct) / seen,
                "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"], "val_log_loss": val["log_loss"],
                "train_seconds": train_seconds, "val_seconds": val_seconds,
                "train_images_per_second": seen / train_seconds, "data_wait_seconds": train_feeder.data_wait,
                "lr": [g["lr"] for g in optimizer.param_groups],
                "trainable_parameters": count_parameters(model)[1],
                "gpu_peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
            }
            history.append(row)
            log(f"[{stage.name} {local_epoch}/{epochs}] loss={row['train_loss']:.4f} "
                f"train_acc={row['train_accuracy']:.4f} val_f1={val['macro_f1']:.4f} "
                f"t={train_seconds:.1f}s ({row['train_images_per_second']:.0f} img/s) espera_datos={train_feeder.data_wait:.1f}s")
            if val["macro_f1"] > best["macro_f1"]:
                best = {"macro_f1": val["macro_f1"], "state": copy.deepcopy(model.state_dict()),
                        "epoch": global_epoch, "stage": stage.name}
                stale = 0
            else:
                stale += 1
                if stage.unfrozen_blocks > 0 and stale >= cfg.early_stopping_patience:
                    log(f"Early stopping: {stale} épocas sin mejorar macro F1 de validación")
                    break

    total_seconds = time.perf_counter() - started
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    truth, probs, _ = predict(model, val_feeder, device, amp)
    final_val = classification_metrics(truth, probs)
    total, trainable = count_parameters(model)
    summary = {
        "device": str(device), "amp": amp, "channels_last": cfg.cuda.channels_last and device.type == "cuda",
        "augment_device": cfg.cuda.augment_device if device.type == "cuda" else "cpu",
        "torch_threads": threads, "epochs_run": global_epoch, "best_epoch": best["epoch"], "best_stage": best["stage"],
        "total_seconds": total_seconds, "train_images": train_feeder.count, "validation": final_val,
        "parameters_total": total, "parameters_trainable_last_stage": trainable,
        "config": asdict(cfg), "environment": environment(device),
    }
    save_checkpoint(model, out_dir / "model.pt", {
        "architecture": cfg.architecture, "image_size": cfg.image_size, "cache_size": train_data.images.shape[1],
        "eval_crop": cfg.augment.eval_crop, "best_epoch": best["epoch"], "validation_macro_f1": final_val["macro_f1"],
        "trained_on": str(device),
    })
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log(f"Listo en {total_seconds/60:.1f} min. Mejor macro F1 validación={final_val['macro_f1']:.4f} (época {best['epoch']})")
    return summary
