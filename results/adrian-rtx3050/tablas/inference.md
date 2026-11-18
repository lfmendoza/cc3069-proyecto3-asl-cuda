| Variante | Imágenes | Tiempo medio (s) | Desv. est. (s) | CV (%) | Imágenes/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| CPU 1 hilo lote 64 | 320 | 1.657 | 0.015 | 0.891 | 193.1 | 1.000 |
| CPU 8 hilos lote 64 | 320 | 0.946 | 0.013 | 1.372 | 338.4 | 1.752 |
| GPU fp32 lote 64 | 1,280 | 0.353 | 0.001 | 0.212 | 3,625.6 | 18.8 |
| GPU AMP fp16 lote 64 | 1,280 | 0.213 | 0.006 | 2.763 | 5,998.4 | 31.1 |

| Variante | Latencia p50 (ms) | Latencia p95 (ms) | FPS equivalentes |
|---|---:|---:|---:|
| CPU 1 hilo 1 imagen | 6.186 | 6.927 | 161.7 |
| CPU 8 hilos 1 imagen | 8.845 | 10.9 | 113.5 |
| GPU fp32 1 imagen | 6.556 | 12.3 | 152.6 |
| GPU AMP fp16 1 imagen | 8.503 | 16.4 | 117.6 |
