# 5. Implementación con CUDA Python

<!-- Adrian: describir el entorno real (GPU, driver, CUDA) y completar con las
corridas de results/<equipo>/. -->

## 5.1 Entorno de ejecución

## 5.2 Componentes CUDA utilizados

| Componente | Librería | Dónde se usa |
|---|---|---|
| Kernel de entrada (`warp_kernel`) | Numba CUDA (`numba-cuda`) | `augment.py`, lanzado desde `feeders.py` |
| Interoperabilidad Numba ↔ PyTorch | `__cuda_array_interface__`, `cuda.external_stream` | `feeders.py` |
| Memoria pinned y copias asíncronas | PyTorch (`pin_memory`, `non_blocking=True`) | `feeders.py` |
| Eventos CUDA (sincronización y tiempos) | `torch.cuda.Event`, `numba.cuda.event` | `feeders.py`, `bench.py` |
| Convoluciones y productos matriciales | cuDNN y cuBLAS a través de PyTorch | `train.py` |
| Precisión mixta | `torch.autocast`, `torch.amp.GradScaler` | `train.py` |
| Telemetría de la GPU | NVML (`nvidia-ml-py`) y `nvidia-smi` | `monitor.py`, `scripts/corrida_gpu.*` |

## 5.3 Tiempos medidos en GPU

## 5.4 Configuración de hilos por bloque

## 5.5 Evidencia de uso de recursos
