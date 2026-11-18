| Variante | Min por época | Horas totales | Speedup |
|----------------------------------------|------------:|------------:|------:|
| Lab 3 secuencial (0 workers, 1 hilo) | 16.5 | 3.842 | 1.000 |
| Lab 3 como se entregó (4 workers, 8 hilos) | 20.9 | 4.868 | 0.789 |
| Híbrido CPU (caché + Numba CPU, 8 hilos) | 5.764 | 1.345 | 2.856 |
| Híbrido GPU, aumentación en CPU | 2.911 | 0.679 | 5.656 |
| Híbrido GPU, kernel CUDA (propuesta) | 2.481 | 0.579 | 6.637 |

Proyección lineal a 43,500 imágenes por época y 14 épocas (presupuesto máximo de la receta). Corra `asl-cuda prep` en este equipo para sumar el costo real de la fase 1.
