| Corrida | Configuración | Épocas | s por época (train) | Imágenes/s | Espera por datos (%) | Total (min) | Macro F1 validación | Memoria GPU pico (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base-cpu-1epoca | CPU, 8 hilos | 1 | 241.9 | 179.9 | 13.3 | 4.781 | 0.942 |  |
| base-cuda | GPU, AMP, aumentación en GPU | 11 | 28.0 | 1,677.1 | 21.0 | 5.648 | 0.986 | 749.9 |
| base-cuda-aug-cpu | GPU, AMP, aumentación en CPU | 2 | 58.3 | 748.1 | 64.5 | 2.380 | 0.946 | 719.4 |
| robusto-cuda | GPU, AMP, aumentación en GPU | 14 | 27.8 | 1,662.8 | 10.8 | 7.080 | 0.994 | 749.9 |
