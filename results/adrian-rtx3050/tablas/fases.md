| Fase | Secuencial (ms/img) | Mejor variante CPU | CPU (ms/img) | Speedup CPU | GPU (ms/img) | Speedup GPU |
|---|---:|---|---:|---:|---:|---:|
| Decodificación (fase 1) | 0.677 | procesos=8 | 0.301 | 2.246 |  |  |
| Transformación de entrada | 5.719 | Numba CPU 8 hilos | 0.563 | 10.2 | 0.007 | 833.9 |
| Paso de entrenamiento | 16.3 | CPU 8 hilos | 8.054 | 2.027 | 3.352 | 4.871 |
| Inferencia por lotes | 5.118 | CPU 8 hilos lote 64 | 3.015 | 1.697 | 0.185 | 27.7 |
| Época completa | 22.7 | Híbrido CPU (caché + Numba CPU, 8 hilos) | 7.950 | 2.856 | 3.421 | 6.637 |
