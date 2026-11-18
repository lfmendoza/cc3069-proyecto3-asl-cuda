# Evidencia de ejecución

Capturas tomadas durante las corridas. Las de la laptop oficial (Ryzen 5
7235HS, RTX 3050 Laptop, Windows 11) son las que se usan en el informe.

| Archivo | Qué muestra |
|---|---|
| `01-prep-cpu-administrador.png` | Administrador de tareas, CPU por procesador lógico, durante la decodificación con 8 procesos: los 8 núcleos lógicos ocupados (93 % de uso medio según el monitor). |
| `02-secuencial-cpu-administrador.png` | La misma vista durante la época del Lab 3 en un hilo: ningún núcleo saturado (29 % de uso medio); Windows mueve el hilo entre núcleos. |
| `03-train-gpu-administrador.png` | GPU NVIDIA durante el entrenamiento: 95 % de uso y 1.7 GB de memoria dedicada. Windows 11 cuenta el trabajo de CUDA en el motor "3D" cuando está activada la programación de GPU acelerada por hardware. |
| `04-train-nvidia-smi.png` | `nvidia-smi` durante el entrenamiento: 46 W, 61 °C, 1.7 GB en uso y el proceso `python.exe` de tipo cómputo. |
| `05-env.png` | Salida de `asl-cuda env`: la GPU detectada (20 SM, 6 GB, driver 591.66, CUDA 13.0) y el kernel CUDA validado contra NumPy. |
| `06-aug-cpu-vs-gpu.png` | Entrenamiento con la transformación en el CPU (`base-cuda-aug-cpu`): CPU al 96 % y la GPU en 22 %, esperando datos. |
| `hansel-prep-secuencial.png` | MacBook M3, decodificación con un proceso. |
| `hansel-prep-paralelo.png` | MacBook M3, decodificación con 8 procesos: los 8 núcleos (4 de rendimiento y 4 de eficiencia) ocupados. |

Los registros de texto (consola, `nvidia-smi` cada segundo y el monitor por
núcleo) están en `results/<equipo>/logs/` y `results/<equipo>/bench/monitor.csv`.
