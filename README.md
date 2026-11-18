# Proyecto 3 · Mejorando algoritmos con CUDA

**CC3069 Computación Paralela y Distribuida — Universidad del Valle de Guatemala**

Fernando Mendoza · Hansel · Adrian

Tomamos el clasificador del alfabeto de señas americano (ASL) del Laboratorio 3
de Data Science y lo reorganizamos en dos fases: el CPU, con todos sus
núcleos, se encarga de los archivos (leer y decodificar 66,700 JPEG una sola
vez), y la GPU se encarga de toda la matemática del entrenamiento, incluida la
transformación de entrada de la red, que escribimos como un kernel propio en
Numba CUDA.

## Resultados

Corrida oficial en una laptop con RTX 3050 6 GB Laptop GPU y Ryzen 5 7235HS
(4 núcleos, 8 hilos), Windows 11. Todas las variantes procesan las mismas
imágenes con el mismo modelo y cada speedup compara dos variantes en esa
misma laptop (`results/adrian-rtx3050/tablas/`).

| Fase | Línea base | Mejor variante | Speedup |
|---|---|---|---:|
| Decodificación (CPU) | 1 proceso | 8 procesos | 2.16× |
| Transformación de entrada | torchvision/PIL (Lab 3) | kernel Numba CUDA 16 × 16 | 647× |
| Paso de entrenamiento | CPU 1 hilo | GPU AMP fp16 | 37.1× |
| Inferencia por lotes de 64 | CPU 1 hilo | GPU AMP fp16 | 31.1× |
| Época completa | Lab 3 secuencial (20.8 s) | caché + kernel CUDA + GPU (0.53 s) | 39.4× |
| Entrenamiento completo (proyección) | Lab 3 secuencial (3.0 h) | caché + kernel CUDA + GPU (5.8 min) | 31.1× |

| Modelo final (receta robusta) | Test interno (11,600) | Dataset externo, mitad final (435) |
|---|---:|---:|
| macro F1 | 0.982 | 0.504 |

El entrenamiento real de la receta robusta (14 épocas) tomó 7.1 minutos en la
GPU.

## Cómo está organizado

```
configs/            datos (partición), receta de entrenamiento, protocolo de medición
src/asl_cuda/
  manifest.py       inventario y partición agrupada por ventanas de frames
  prep.py           FASE 1 (CPU): decodificación multiproceso → caché uint8 en memmap
  augment.py        kernel de entrada: homografía + bilineal + color + normalización
                    (Numba CUDA y Numba CPU con la misma función por píxel)
  feeders.py        armado de lotes: buffers pinned, copia asíncrona, kernel en el stream de PyTorch
  train.py          FASE 2 (GPU): entrenamiento con AMP (fp16) y cudnn.benchmark
  secuencial.py     versión original del Lab 3, usada como línea base
  bench.py          mediciones por fase (secuencial, CPU paralelo, GPU)
  summarize.py      speedup, eficiencia, Karp-Flatt, tablas y figuras
  informe.py        copia tablas y figuras de results/ al informe
  monitor.py        uso de CPU por núcleo y de GPU (NVML) durante cada fase
  evaluate.py       test interno, dataset externo y fotos propias
  demo.py           reconocimiento en vivo con la cámara
docs/               análisis, diseño y resultados por sección del enunciado
informe/            fuente del informe en PDF
results/<equipo>/   mediciones, logs, tablas y figuras de cada computadora
models/             modelo final exportado
notebooks/          recorrido completo del proyecto (entregable .ipynb)
```

## Instalación

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.12 (uv lo descarga si no
está). Se instala **uno** de los tres perfiles:

```bash
uv sync --extra cpu --extra demo      # sin GPU NVIDIA
uv sync --extra cu126 --extra demo    # GPU NVIDIA con driver >= 560 (CUDA 12.6)
uv sync --extra cu130 --extra demo    # GPU NVIDIA con driver >= 580 (CUDA 13.0); obligatorio en RTX 50xx
```

El driver instalado se ve en la esquina superior derecha de `nvidia-smi`
("CUDA Version"). Para comprobar que todo quedó bien:

```bash
uv run --no-sync asl-cuda env
```

En una máquina con GPU, el último renglón debe decir que el kernel CUDA
coincide con la referencia de NumPy.

## Datos

Se necesita una cuenta de Kaggle con token en `~/.kaggle/kaggle.json` o
`~/.kaggle/access_token`.

```bash
uv run --no-sync asl-cuda download     # grassknoted/asl-alphabet + danrasband/asl-alphabet-test
uv run --no-sync asl-cuda prep         # manifest, partición y caché (≈ 8 GB en data/cache)
```

## Uso

```bash
# entrenamiento (en GPU si está disponible)
uv run --no-sync asl-cuda train --config configs/train.yaml --tag <equipo>

# fotos propias: una por clase con la cámara (quedan en .local/fotos-equipo/<persona>, no se suben)
uv run --no-sync asl-cuda fotos --persona <nombre>

# evaluación del modelo final
uv run --no-sync asl-cuda evaluate --split test external --tag <equipo>
uv run --no-sync asl-cuda evaluate --split ruta/a/fotos --name fotos-equipo --tag <equipo>
uv run --no-sync asl-cuda predict foto1.jpg foto2.jpg

# mediciones y reporte
uv run --no-sync asl-cuda bench --tag <equipo>            # todas las fases
uv run --no-sync asl-cuda bench --quick --tag <equipo>    # prueba rápida
uv run --no-sync asl-cuda summarize --tag <equipo>
uv run --no-sync asl-cuda informe --tag <equipo> --cpu-tag <equipo-cpu> [...]   # tablas y figuras al informe

# demostración en vivo
uv run --no-sync asl-cuda demo
```

Las corridas completas están en `scripts/corrida_gpu.ps1` (o `.sh`),
`scripts/corrida_cpu.ps1` (o `.sh`) y `scripts/datos_nuevos.ps1` (o `.sh`),
que evalúa el modelo final con el test interno, el dataset externo y, si
existen, fotos propias en `.local/fotos-equipo/`.

## Informe

El informe se arma con [Quarto](https://quarto.org) a partir de `docs/` (una
sección por archivo). Las tablas y figuras no se copian a mano: `asl-cuda
informe` las toma de `results/<equipo>/` y las deja en `informe/tablas/` e
`informe/figuras/`.

```bash
quarto render informe/informe.qmd --to pdf
```

## Pruebas

```bash
uv run --no-sync pytest -q                                   # sin GPU: el kernel corre en el simulador de Numba
ASL_REAL_GPU=1 uv run --no-sync pytest tests/test_gpu.py -v   # con GPU real
```

## Contribuciones

| Integrante | Aporte |
|---|---|
| Fernando Mendoza | Algoritmo base (Lab 3), análisis de la solución secuencial, identificación de oportunidades y diseño del algoritmo paralelo; implementación del pipeline y del kernel. |
| Hansel | Medición de la versión secuencial y CPU multinúcleo, pruebas con datos nuevos (test interno y dataset externo), notebook, análisis comparativo e informe. |
| Adrian | Ejecución y validación en GPU, experimentos de configuración de hilos/bloques y tamaño de lote, entrenamiento final, evidencia de uso de recursos y sección de implementación CUDA. |
