"""Datos del equipo y del software para dejarlos junto a cada medición."""

from __future__ import annotations

import os
import platform
import subprocess
from importlib import metadata
from typing import Any


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def cpu_name() -> str:
    system = platform.system()
    try:
        if system == "Windows":
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if system == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update(available=True, name=props.name, capability=f"{props.major}.{props.minor}",
                        multiprocessors=props.multi_processor_count, memory_gb=round(props.total_memory / 2**30, 2),
                        torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    except Exception as exc:  # noqa: BLE001 - solo es diagnóstico
        info["torch_error"] = repr(exc)
    try:
        import pynvml

        pynvml.nvmlInit()
        info["driver"] = pynvml.nvmlSystemGetDriverVersion()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info["power_limit_w"] = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
        pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        pass
    try:
        from numba import cuda

        info["numba_cuda_available"] = bool(cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        info["numba_error"] = repr(exc)
    return info


def environment(device: Any = None) -> dict[str, Any]:
    import psutil

    return {
        "host_tag": os.environ.get("ASL_HOST_TAG"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "device": str(device) if device is not None else None,
        "gpu": gpu_info(),
        "packages": {name: _version(name) for name in ("torch", "torchvision", "numba", "numba-cuda", "numpy", "pillow")},
        "commit": git_commit(),
    }
