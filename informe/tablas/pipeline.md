| Variante | Imágenes | Tiempo medio (s) | Desv. est. (s) | CV (%) | Imágenes/s | Speedup |
|----------------------------------------|-------:|---------------:|-------------:|-----:|---------:|------:|
| Lab 3 secuencial (0 workers, 1 hilo) | 1,160 | 20.8 | 0.231 | 1.114 | 55.9 | 1.000 |
| Lab 3 como se entregó (4 workers, 8 hilos) | 1,160 | 31.6 | 0.541 | 1.712 | 36.7 | 0.657 |
| Híbrido CPU (caché + Numba CPU, 8 hilos) | 1,160 | 8.956 | 0.046 | 0.515 | 129.5 | 2.318 |
| Híbrido GPU, aumentación en CPU | 1,160 | 1.313 | 0.133 | 10.1 | 883.3 | 15.8 |
| Híbrido GPU, kernel CUDA (propuesta) | 1,160 | 0.527 | 0.011 | 2.059 | 2,200.3 | 39.4 |
