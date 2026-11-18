| Variante | Min por época | Horas totales | Speedup |
|----------------------------------------|------------:|------------:|------:|
| Lab 3 secuencial (0 workers, 1 hilo) | 13.0 | 3.028 | 1.000 |
| Lab 3 como se entregó (4 workers, 8 hilos) | 19.7 | 4.607 | 0.657 |
| Híbrido CPU (caché + Numba CPU, 8 hilos) | 5.598 | 1.327 | 2.283 |
| Híbrido GPU, aumentación en CPU | 0.821 | 0.212 | 14.3 |
| Híbrido GPU, kernel CUDA (propuesta) | 0.329 | 0.097 | 31.1 |

Proyección lineal a 43,500 imágenes por época y 14 épocas (presupuesto máximo de la receta). Las variantes híbridas incluyen una sola vez la fase 1 medida en este equipo (73 s).
