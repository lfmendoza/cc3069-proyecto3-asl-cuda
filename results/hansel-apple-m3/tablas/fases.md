| Fase | Secuencial (ms/img) | Mejor variante CPU | CPU (ms/img) | Speedup CPU | GPU (ms/img) | Speedup GPU |
|---|---:|---|---:|---:|---:|---:|
| Decodificación (fase 1) | 0.244 | procesos=4 | 0.123 | 1.983 |  |  |
| Transformación de entrada | 1.458 | Numba CPU 8 hilos | 0.381 | 3.828 |  |  |
| Paso de entrenamiento | 22.8 | CPU 8 hilos | 66.8 | 0.341 |  |  |
| Inferencia por lotes | 17.2 | CPU 8 hilos lote 64 | 17.5 | 0.983 |  |  |
| Época completa | 22.0 | Híbrido CPU (caché + Numba CPU, 8 hilos) | 45.7 | 0.481 |  |  |
