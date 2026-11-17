"""Muestreo de recursos en segundo plano (CPU por núcleo, RAM y GPU vía NVML).

Se usa como evidencia de qué hardware trabajó en cada fase. Cada fila lleva la
etiqueta de la fase activa, así las gráficas pueden separar "prep" de "train".
"""

from __future__ import annotations

import csv
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import psutil


class ResourceMonitor:
    def __init__(self, path: Path, interval: float = 0.5) -> None:
        self.path = path
        self.interval = interval
        self.phase = "idle"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None
        self.cores = psutil.cpu_count(logical=True) or 1
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # noqa: BLE001 - sin GPU NVIDIA el monitor sigue registrando CPU
            self._nvml = None

    def _gpu(self) -> list:
        if self._nvml is None:
            return [""] * 7
        n, h = self._nvml, self._handle
        try:
            util = n.nvmlDeviceGetUtilizationRates(h)
            mem = n.nvmlDeviceGetMemoryInfo(h)
            return [util.gpu, util.memory, round(mem.used / 2**20), round(n.nvmlDeviceGetPowerUsage(h) / 1000, 1),
                    n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU),
                    n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM), round(mem.total / 2**20)]
        except Exception:  # noqa: BLE001
            return [""] * 7

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        process = psutil.Process()
        psutil.cpu_percent(percpu=True)
        started = time.perf_counter()
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t", "phase", "cpu_total", "ram_used_gb", "proc_rss_gb", "gpu_util", "gpu_mem_util",
                             "gpu_mem_used_mb", "gpu_power_w", "gpu_temp_c", "gpu_sm_mhz", "gpu_mem_total_mb"]
                            + [f"core_{i}" for i in range(self.cores)])
            while not self._stop.wait(self.interval):
                cores = psutil.cpu_percent(percpu=True)
                try:
                    rss = sum(p.memory_info().rss for p in [process, *process.children(recursive=True)])
                except psutil.Error:
                    rss = process.memory_info().rss
                writer.writerow([round(time.perf_counter() - started, 2), self.phase, round(sum(cores) / len(cores), 1),
                                 round(psutil.virtual_memory().used / 2**30, 2), round(rss / 2**30, 2), *self._gpu(), *cores])
                handle.flush()

    def start(self) -> ResourceMonitor:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass

    @contextmanager
    def tag(self, phase: str):
        previous, self.phase = self.phase, phase
        try:
            yield
        finally:
            self.phase = previous

    def __enter__(self) -> ResourceMonitor:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


@contextmanager
def maybe_monitor(path: Path | None, interval: float = 0.5):
    if path is None:
        class _Null:
            @contextmanager
            def tag(self, phase: str):
                yield
        yield _Null()
        return
    with ResourceMonitor(path, interval) as monitor:
        yield monitor
