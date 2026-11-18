| Variante | Procesos/hilos | Tiempo medio (s) | Desv. est. (s) | Imágenes/s | Speedup | Eficiencia | Karp-Flatt e |
|---|---:|---:|---:|---:|---:|---:|---:|
| secuencial | 1 | 1.418 | 0.037 | 4,091.1 | 1.000 | 1.000 |  |
| procesos=2 | 2 | 1.019 | 0.086 | 5,691.7 | 1.391 | 0.696 | 0.438 |
| procesos=4 | 4 | 0.715 | 0.015 | 8,112.7 | 1.983 | 0.496 | 0.339 |
| procesos=8 | 8 | 0.721 | 0.026 | 8,047.4 | 1.967 | 0.246 | 0.438 |
| hilos=8 (mismo código, GIL) | 8 | 0.843 | 0.083 | 6,880.2 | 1.682 | 0.210 |  |
