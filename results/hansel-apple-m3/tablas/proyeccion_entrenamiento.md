| Variante | Min por época | Horas totales | Speedup |
|---|---:|---:|---:|
| Lab 3 secuencial (0 workers, 1 hilo) | 16.0 | 3.722 | 1.000 |
| Lab 3 como se entregó (4 workers, 8 hilos) | 46.9 | 10.9 | 0.340 |
| Híbrido CPU (caché + Numba CPU, 8 hilos) | 33.2 | 7.742 | 0.481 |

Proyección lineal a 43,500 imágenes por época y 14 épocas (presupuesto máximo de la receta). Las variantes híbridas incluyen una sola vez la fase 1 medida en este equipo (19 s).
