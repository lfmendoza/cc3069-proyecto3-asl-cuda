| Variante | Min por época | Horas totales | Speedup |
|---|---:|---:|---:|
| Lab 3 secuencial (0 workers, 1 hilo) | 16.5 | 3.842 | 1.000 |
| Lab 3 como se entregó (4 workers, 8 hilos) | 20.9 | 4.868 | 0.789 |
| Híbrido CPU (caché + Numba CPU, 8 hilos) | 5.764 | 1.385 | 2.774 |
| Híbrido GPU, aumentación en CPU | 2.911 | 0.719 | 5.342 |
| Híbrido GPU, kernel CUDA (propuesta) | 2.481 | 0.619 | 6.209 |

Proyección lineal a 43,500 imágenes por época y 14 épocas (presupuesto máximo de la receta). Las variantes híbridas incluyen una sola vez la fase 1 medida en este equipo (144 s).
