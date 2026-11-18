| Variante | Imágenes | Tiempo medio (s) | Desv. est. (s) | CV (%) | Imágenes/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| CPU 1 hilo lote 64 | 320 | 1.638 | 0.050 | 3.060 | 195.4 | 1.000 |
| CPU 8 hilos lote 64 | 320 | 0.965 | 0.009 | 0.940 | 331.6 | 1.697 |
| GPU fp32 lote 64 | 1,280 | 0.333 | 0.007 | 2.092 | 3,839.1 | 19.6 |
| GPU AMP fp16 lote 64 | 1,280 | 0.236 | 0.021 | 8.747 | 5,417.4 | 27.7 |

| Variante | Latencia p50 (ms) | Latencia p95 (ms) | FPS equivalentes |
|---|---:|---:|---:|
| CPU 1 hilo 1 imagen | 7.953 | 8.746 | 125.8 |
| CPU 8 hilos 1 imagen | 9.643 | 11.8 | 103.9 |
| GPU fp32 1 imagen | 8.593 | 13.9 | 116.5 |
| GPU AMP fp16 1 imagen | 11.0 | 17.4 | 91.1 |
