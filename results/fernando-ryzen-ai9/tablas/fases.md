| Fase | Secuencial (ms/img) | Mejor variante CPU | CPU (ms/img) | Speedup CPU | GPU (ms/img) | Speedup GPU |
|---|---:|---|---:|---:|---:|---:|
| Decodificación (fase 1) | 0.593 | procesos=8 | 0.151 | 3.919 |  |  |
| Transformación de entrada | 5.036 | Numba CPU 20 hilos | 0.404 | 12.5 |  |  |
| Paso de entrenamiento | 12.2 | CPU 20 hilos | 8.399 | 1.450 |  |  |
| Inferencia por lotes | 9.112 | CPU 20 hilos lote 64 | 4.004 | 2.276 |  |  |
| Época completa | 22.3 | Híbrido CPU (caché + Numba CPU, 20 hilos) | 8.104 | 2.748 |  |  |
