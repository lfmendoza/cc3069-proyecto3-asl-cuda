# Evidencia de ejecución

Capturas tomadas durante las corridas oficiales. El nombre indica la fase y
la herramienta:

| Archivo | Qué debe verse |
|---|---|
| `01-prep-cpu-administrador.png` | Administrador de tareas → Rendimiento → CPU en vista de procesadores lógicos durante `asl-cuda prep`: todos los núcleos ocupados. |
| `02-secuencial-cpu-administrador.png` | Misma vista durante `pipeline: Lab 3 secuencial`: un solo núcleo ocupado. |
| `03-train-gpu-administrador.png` | Administrador de tareas → GPU NVIDIA, gráfica **Cuda** (o *Compute_0*) y memoria dedicada durante el entrenamiento. |
| `04-train-nvidia-smi.png` | `nvidia-smi` durante el entrenamiento: uso de GPU, memoria y el proceso `python.exe`. |
| `05-env.png` | Salida de `asl-cuda env` con la GPU detectada y el kernel validado. |
| `06-aug-cpu-vs-gpu.png` | Administrador de tareas durante `base-cuda-aug-cpu`: CPU al máximo y la GPU esperando. |

Los registros de texto (consola, `nvidia-smi` cada segundo y el monitor por
núcleo) están en `results/<equipo>/logs/` y `results/<equipo>/bench/monitor.csv`.
