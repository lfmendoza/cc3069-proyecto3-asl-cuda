| Variante | Imágenes | Tiempo medio (s) | Desv. est. (s) | CV (%) | Imágenes/s | Speedup |
|----------------------------------------|-------:|---------------:|-------------:|-----:|---------:|------:|
| Lab 3 secuencial (0 workers, 1 hilo) | 1,160 | 25.5 | 0.055 | 0.215 | 45.4 | 1.000 |
| Lab 3 como se entregó (4 workers, 8 hilos) | 1,160 | 75.0 | 0.381 | 0.508 | 15.5 | 0.340 |
| Híbrido CPU (caché + Numba CPU, 8 hilos) | 1,160 | 53.0 | 0.067 | 0.126 | 21.9 | 0.481 |
