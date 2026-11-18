| Fase | Secuencial (ms/img) | Mejor variante CPU | CPU (ms/img) | Speedup CPU | GPU (ms/img) | Speedup GPU |
|---|---:|---|---:|---:|---:|---:|
| Decodificación (fase 1) | 0.593 | procesos=8 | 0.275 | 2.160 |  |  |
| Transformación de entrada | 4.011 | Numba CPU 8 hilos | 0.501 | 8.012 | 0.006 | 646.8 |
| Paso de entrenamiento | 13.8 | CPU 8 hilos | 7.137 | 1.928 | 0.371 | 37.1 |
| Inferencia por lotes | 5.178 | CPU 8 hilos lote 64 | 2.955 | 1.752 | 0.167 | 31.1 |
| Época completa | 17.9 | Híbrido CPU (caché + Numba CPU, 8 hilos) | 7.721 | 2.318 | 0.454 | 39.4 |
