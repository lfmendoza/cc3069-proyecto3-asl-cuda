| Variante | Min por época | Horas totales | Speedup |
|----------------------------------------|------------:|------------:|------:|
| Lab 3 secuencial (0 workers, 1 hilo) | 16.1 | 3.768 | 1.000 |
| Lab 3 como se entregó (4 workers, 20 hilos) | 9.698 | 2.263 | 1.665 |
| Híbrido CPU (caché + Numba CPU, 20 hilos) | 5.876 | 1.375 | 2.739 |

Proyección lineal a 43,500 imágenes por época y 14 épocas (presupuesto máximo de la receta). Las variantes híbridas incluyen una sola vez la fase 1 medida en este equipo (16 s).
