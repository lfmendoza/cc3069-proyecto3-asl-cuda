| Variante | Imágenes | Tiempo medio (s) | Desv. est. (s) | CV (%) | Imágenes/s | Speedup |
|----------------------------------------|-------:|---------------:|-------------:|-----:|---------:|------:|
| Lab 3 secuencial (0 workers, 1 hilo) | 1,160 | 26.3 | 0.234 | 0.889 | 44.0 | 1.000 |
| Lab 3 como se entregó (4 workers, 8 hilos) | 1,160 | 33.4 | 0.762 | 2.283 | 34.7 | 0.789 |
| Híbrido CPU (caché + Numba CPU, 8 hilos) | 1,160 | 9.222 | 0.105 | 1.135 | 125.8 | 2.856 |
| Híbrido GPU, aumentación en CPU | 1,160 | 4.657 | 0.009 | 0.198 | 249.1 | 5.656 |
| Híbrido GPU, kernel CUDA (propuesta) | 1,160 | 3.969 | 0.008 | 0.192 | 292.3 | 6.637 |
