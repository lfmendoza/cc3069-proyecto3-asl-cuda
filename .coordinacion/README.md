# Plan de trabajo del equipo

> Carpeta temporal de organización. Se borra antes de la entrega (ver
> `cierre.md`). Cada integrante tiene su archivo (`adrian.md`, `hansel.md`,
> `fernando.md`) con los pasos en orden, los comandos exactos y cómo saber
> que terminó; `convenciones.md` aplica a todos.

## Qué estamos entregando

El enunciado pide tomar el Lab 3 de Data Science (ASL), medir la versión
secuencial, diseñar e implementar una versión paralela con CUDA Python y
comparar. La rúbrica pesa:

| Puntos | Criterio | Dónde se cubre |
|---:|---|---|
| 40 | Refactorización: separación clara CPU (preparación de imágenes) / GPU (entrenamiento) | `src/asl_cuda/prep.py` (CPU), `feeders.py` + `augment.py` + `train.py` (GPU), notebook |
| 30 | Evidencia: logs y capturas de monitores mostrando el hardware en cada fase | `results/<equipo>/logs`, `monitor.csv`, `evidencia/*.png` |
| 30 | Reporte: speedup por fase, por qué archivos en CPU y matemática en GPU, librerías usadas | `docs/02`, `docs/05`, `docs/06`, `informe/informe.pdf` |

Entregables finales: código + notebook (`notebooks/proyecto3_asl_cuda.ipynb`
ejecutado con salidas), informe PDF, capturas y presentación.

## Estado actual

- Repositorio, entorno reproducible (`uv`), pruebas y CI.
- Pipeline completo: decodificación paralela en CPU, kernel Numba CUDA,
  entrenamiento en GPU con AMP, versión secuencial original del Lab 3,
  benchmarks por fase, resumen con speedup/Karp-Flatt, evaluación con datos
  nuevos y demo con cámara.
- Probado en CPU (Ryzen AI 9 365, 20 hilos): 23 pruebas pasan, el kernel CUDA
  coincide con la referencia en el simulador de Numba, la caché de 66,700
  imágenes se arma en 16 s con 20 procesos, un entrenamiento piloto alcanza
  0.91 de macro F1 con solo 150 imágenes por clase.
- Hansel: entorno y corrida CPU en la M3 integrados (PR #1).
- Fernando: corrida CPU limpia de referencia en la Ryzen AI 9 365
  (`results/fernando-ryzen-ai9/`).
- Informe redactado completo (secciones 1 a 6). Lo que depende de la corrida
  en GPU está marcado con `[[GPU: ...]]`; las tablas y figuras entran solas
  con `asl-cuda informe`, que ya corre al final de `scripts/corrida_gpu.ps1`.
  El PDF compila hoy con avisos de "pendiente" en esos lugares.
- Notebook entregable armado y probado en CPU.
- Presentación armada (20 láminas); los datos de GPU se llenan al final.
- Fotos del equipo: `asl-cuda fotos --persona <nombre>`; su evaluación es un
  solo script (`scripts/datos_nuevos.*`).
- **No probado todavía en GPU real.** Es lo primero que hace Adrian. Todo lo
  que falta depende de su corrida oficial: llenar las marcas `[[...]]`,
  evaluar las fotos con el modelo final y poner los números en la
  presentación.

## Reparto

| | Fernando | Hansel | Adrian |
|---|---|---|---|
| Rol | Algoritmo y diseño paralelo | Todo lo que no depende de NVIDIA | Todo lo que corre en la GPU |
| Secciones del informe | 1, 3, 4 | 2, 6 (con Adrian), ensamblado del PDF | 5, parte GPU de 6 |
| Código | `src/`, `configs/`, `tests/test_kernel.py` | notebook, pruebas nuevas | correcciones específicas de GPU (vía PR) |
| Datos/resultados | — | `results/hansel-*`, fotos del equipo | `results/adrian-*`, `models/`, `evidencia/` |
| Presentación | diseño y kernel | medición secuencial y datos nuevos | resultados GPU y demo en vivo |

## Cronograma por hitos

Los días cuentan desde que todos tienen acceso al repositorio. Si la fecha de
entrega es antes, se comprimen los hitos 2 y 3, nunca el 1.

| Hito | Día | Quién | Resultado verificable |
|---|---|---|---|
| H1 · Entornos | 1 | Adrian, Hansel | `asl-cuda env` corre; en la laptop de Adrian dice "kernel ... OK"; `pytest` pasa en ambos. Adrian además: `ASL_REAL_GPU=1 pytest tests/test_gpu.py` pasa. |
| H2 · Corridas oficiales | 2 | Adrian | `scripts/corrida_gpu.ps1` completo, `results/adrian-<gpu>/` y `models/asl_mobilenetv3.pt` en una rama con PR. Capturas en `evidencia/`. |
| H2 · Corrida CPU y fotos | 2 | Hansel (+ todos para las fotos) | `results/hansel-<cpu>/` en PR. Fotos de los tres en `.local/fotos-equipo/` (no se suben). |
| H3 · Datos nuevos | 3 | Hansel | Evaluación de fotos del equipo con el modelo final; párrafo de fotos en 6.5. |
| H3 · Sección 5 | 3 | Adrian | Marcas `[[GPU]]` de `docs/05` llenas; notebook ejecutado en GPU con salidas. |
| H4 · Informe | 4 | Fernando, revisión de los tres | Marcas `[[GPU]]` de `docs/02` y `docs/06` llenas; PDF compilado sin avisos. |
| H5 · Cierre | 5 | Todos | Ensayo de presentación, `cierre.md` ejecutado, PDF final, tag `v1.0`. |

Dependencias duras: H3 de Hansel necesita `models/asl_mobilenetv3.pt` de
Adrian. Si Adrian se atrasa, Hansel avanza con el notebook y la sección 2 con
sus propios números de CPU.

## Flujo de Git (para no chocar)

1. `main` siempre debe pasar CI. Nadie hace push directo a `main` excepto
   Fernando para integrar.
2. Cada quien trabaja en ramas con su nombre:
   `adrian/entorno-gpu`, `adrian/corrida-oficial`, `hansel/medicion-cpu`,
   `hansel/notebook`, `hansel/informe`, etc.
3. Antes de empezar el día: `git switch main && git pull && git switch <rama> && git rebase main`.
4. PR pequeño por hito; Fernando revisa y hace merge (squash no: se conserva
   la historia de cada quien).
5. Cada quien escribe solo en su zona (tabla de abajo). Si hace falta cambiar
   algo fuera de su zona, se abre un *issue* o se avisa al dueño.

| Zona | Dueño | Otros pueden… |
|---|---|---|
| `src/asl_cuda/`, `configs/`, `pyproject.toml`, `uv.lock` | Fernando | proponer arreglos por PR (Adrian para fallas exclusivas de GPU) |
| `tests/test_kernel.py`, `tests/test_data.py`, `tests/test_training.py` | Fernando | — |
| `tests/test_gpu.py` | Adrian | — |
| `tests/` archivos nuevos | Hansel | — |
| `docs/01`, `docs/03`, `docs/04` | Fernando | sugerir cambios en el PR |
| `docs/02`, `docs/06`, `notebooks/` | Fernando | Hansel escribe el párrafo de fotos de 6.5; Adrian sube el notebook ejecutado |
| `informe/tablas`, `informe/figuras` | se generan con `asl-cuda informe` | quien corra el comando en su PR |
| `docs/05`, `evidencia/`, `models/`, `scripts/corrida_gpu.*` | Adrian | — |
| `results/adrian-*/` | Adrian | nadie lo toca |
| `results/hansel-*/` | Hansel | nadie lo toca |
| `results/fernando-*/` | Fernando | nadie lo toca |
| `README.md` | Fernando | tabla de resultados la llena Hansel en su PR final |

Los resultados van en carpetas separadas por equipo (`--tag`), así que dos
personas nunca escriben el mismo archivo de resultados.

## Qué no se sube nunca

`data/` (≈ 9 GB), `.local/` (fotos del equipo), `results/*/scratch/`,
checkpoints fuera de `models/`, tokens de Kaggle, archivos de Nsight
(`*.nsys-rep`). El `.gitignore` ya lo cubre; revisar `git status` antes de
cada commit.

## Definición de terminado

- Ningún número en el informe se escribe a mano: todos salen de
  `results/<equipo>/tablas/*.md`, `eval/*.json` o `train/*/summary.json`, y
  el texto dice de qué equipo vienen.
- Speedup siempre comparado en la **misma computadora** (la de Adrian para
  los números oficiales). Los resultados de Hansel sirven como segunda
  computadora para hablar de escalamiento en CPU.
- Cada figura del informe tiene una oración que explica qué se ve y qué
  conclusión saca.
