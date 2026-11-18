| Variante | Procesos/hilos | Tiempo medio (s) | Desv. est. (s) | Imágenes/s | Speedup | Eficiencia | Karp-Flatt e |
|---|---:|---:|---:|---:|---:|---:|---:|
| secuencial | 1 | 3.439 | 0.126 | 1,686.4 | 1.000 | 1.000 |  |
| procesos=2 | 2 | 2.094 | 0.017 | 2,770.4 | 1.643 | 0.821 | 0.217 |
| procesos=4 | 4 | 1.642 | 0.013 | 3,532.2 | 2.095 | 0.524 | 0.303 |
| procesos=8 | 8 | 1.592 | 0.034 | 3,642.7 | 2.160 | 0.270 | 0.386 |
| hilos=4 (mismo código, GIL) | 4 | 1.910 | 0.023 | 3,037.3 | 1.801 | 0.450 |  |
