# Adrian · ejecución en GPU

Tu parte es todo lo que necesita la laptop con NVIDIA: validar que el código
corre en hardware real, hacer las corridas oficiales (los speedup del informe
salen de tu computadora, CPU y GPU), entrenar el modelo final, juntar la
evidencia y escribir la sección 5.

Lee también `README.md` (plan y reglas de Git) y `convenciones.md`.

## Paso 1 · Entorno (Hito 1, día 1)

1. Actualiza el driver de NVIDIA (GeForce Experience o la página de NVIDIA).
   Abre una terminal y corre `nvidia-smi`. Anota el modelo de la GPU y el
   número de "CUDA Version" de la esquina superior derecha.
2. Instala Git y uv (PowerShell):
   ```powershell
   winget install --id Git.Git -e
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
3. Clona y crea tu rama:
   ```powershell
   git clone https://github.com/lfmendoza/cc3069-proyecto3-asl-cuda.git cc3069-proyecto3-asl-cuda
   cd cc3069-proyecto3-asl-cuda
   git switch -c adrian/entorno-gpu
   ```
4. Instala el perfil según tu "CUDA Version":
   - 13.0 o mayor → `uv sync --extra cu130 --extra demo`
   - 12.6 a 12.9 → `uv sync --extra cu126 --extra demo`
   - menor a 12.6 → actualiza el driver primero.
   - Si tu GPU es RTX 50xx, tiene que ser `cu130`.
5. Token de Kaggle: descarga `kaggle.json` desde tu cuenta (Settings → API)
   y guárdalo en `C:\Users\<tu-usuario>\.kaggle\kaggle.json`.
6. Define tu tag para toda la sesión (usa tu GPU real en el nombre):
   ```powershell
   $env:ASL_HOST_TAG = "adrian-rtx4060"
   ```
7. Verifica:
   ```powershell
   uv run --no-sync asl-cuda env
   uv run --no-sync pytest -q
   $env:ASL_REAL_GPU = "1"; uv run --no-sync pytest tests/test_gpu.py -v; Remove-Item Env:ASL_REAL_GPU
   ```
   **Terminado cuando:** `env` muestra tu GPU y la línea
   `Kernel Numba CUDA vs referencia NumPy: diferencia máxima ... -> OK`, y las
   4 pruebas de `test_gpu.py` pasan (no "skipped").

Si algo falla aquí, avisa a Fernando con el error completo antes de seguir.
Problemas conocidos:

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| `torch.cuda.is_available()` es False | se instaló el perfil `cpu` o el driver es viejo | `uv sync --extra cu130 --extra demo` (o `cu126`) y revisar el driver |
| `CUDA driver library cannot be found` / `cudaErrorInsufficientDriver` | driver menor al que pide el perfil | usar `cu126` o actualizar driver |
| `CUDA out of memory` | GPU de 4 GB | bajar `eval_batch_size` a 128 en `configs/train.yaml` solo en tu rama y avisar |
| Advertencias `NumbaPerformanceWarning: Grid size ... low occupancy` | lotes pequeños en las pruebas | se pueden ignorar |
| Errores con `external_stream` o `as_cuda_array` | versión de numba-cuda | mandar el error a Fernando; es un cambio de 2 líneas en `feeders.py` |

Haz commit de nada en este paso salvo que hayas tenido que corregir algo.

## Paso 2 · Datos y prueba rápida (día 1)

```powershell
uv run --no-sync asl-cuda download
uv run --no-sync asl-cuda prep
uv run --no-sync asl-cuda bench --quick
```
`prep` deja ≈ 8 GB en `data/cache`. `bench --quick` debe terminar en pocos
minutos sin errores. Borra el resultado de la prueba rápida:
`Remove-Item -Recurse results\$env:ASL_HOST_TAG`.

## Paso 3 · Corrida oficial (Hito 2, día 2)

Condiciones (anótalas, van en la sección 5):
- Laptop conectada a la corriente todo el tiempo.
- Modo de energía "Máximo rendimiento" (Configuración → Sistema → Energía) y,
  si la laptop tiene software del fabricante (Armoury Crate, Legion Vantage,
  etc.), modo rendimiento.
- Cerrar navegador, Discord, juegos, OneDrive sincronizando.
- No usar la laptop mientras corre.

```powershell
git switch main; git pull; git switch -c adrian/corrida-oficial
.\scripts\corrida_gpu.ps1 -Tag $env:ASL_HOST_TAG
```

Duración estimada: 1.5 a 2.5 horas. El script hace, en orden: `env`, `prep`,
todas las fases de `bench`, entrenamiento `base-cuda` y `robusto-cuda`, una
corrida corta con la aumentación en CPU, una época completa en CPU, la
evaluación de ambos modelos, la elección del modelo final
(`models/asl_mobilenetv3.pt`) y `summarize`. En paralelo guarda `nvidia-smi`
cada segundo en `results/<tag>/logs/nvidia-smi.csv`.

### Capturas mientras corre (30 de los puntos dependen de esto)

Ten abierto el Administrador de tareas. Guarda las capturas en `evidencia/`
con los nombres de `evidencia/README.md`:

1. **Durante `01-prep`** (dura poco, ~30 s; puedes repetir
   `uv run --no-sync asl-cuda prep --force` aparte solo para la captura):
   Rendimiento → CPU → clic derecho en la gráfica → "Cambiar gráfico a →
   Procesadores lógicos". Deben verse todos los núcleos arriba.
2. **Durante `02-bench`, cuando la consola diga
   `pipeline:Lab 3 secuencial`**: la misma vista; debe verse un solo núcleo
   trabajando.
3. **Durante `03-train-base`**: Rendimiento → GPU 1 (la NVIDIA). En una de las
   cuatro gráficas pequeñas cambia "3D" por **"Cuda"** (o "Compute_0").
   Debe verse alta, junto con la memoria dedicada ocupada.
4. **Durante `03-train-base`**, en otra terminal: `nvidia-smi` (captura con el
   proceso `python.exe` en la lista).
5. La salida de `00-env` (o córrelo de nuevo y captura).
6. **Durante `05-train-aug-cpu`**: Administrador de tareas con CPU al máximo y
   la gráfica Cuda más baja que en la captura 3. Es la evidencia de que la
   aumentación en CPU deja a la GPU esperando.

### Llevar los resultados al informe

El último paso del script (`11-informe`) copia tus tablas y figuras a
`informe/tablas/` e `informe/figuras/`. Como las capturas se guardan
mientras corre, cuando ya estén todas en `evidencia/` vuelve a correrlo:
```powershell
uv run --no-sync asl-cuda informe --cpu-tag fernando-ryzen-ai9 hansel-apple-m3
```
Al final imprime lo que todavía falta. Debe decir que no queda nada
pendiente de `results/<tu-tag>` ni de `evidencia/`.

### Revisión antes de subir

- `results/<tag>/logs/duraciones.tsv` tiene todos los pasos.
- `results/<tag>/tablas/` tiene `prep.md`, `augment.md`, `train.md`,
  `inference.md`, `pipeline.md`, `kernel_bloques.md`, `transferencias.md`,
  `proyeccion_entrenamiento.md`.
- `results/<tag>/eval/seleccion_modelo.json` dice qué modelo se eligió.
- Abre las figuras de `results/<tag>/figuras/` y revisa que se vean bien.
- `git status`: no debe aparecer nada de `data/` ni `scratch/`.

```powershell
git add results/$env:ASL_HOST_TAG models evidencia informe/tablas informe/figuras
git commit -m "data: corrida oficial en $env:ASL_HOST_TAG"
git push -u origin adrian/corrida-oficial
```
Abre el PR hacia `main`. **Avisa a Hansel** apenas esté el modelo:
lo necesita para probar las fotos del equipo.

## Paso 4 · Sección 5 del informe (Hito 3, día 3)

`docs/05-implementacion-cuda.md` ya está redactada completa; las tablas y
figuras entran solas con `asl-cuda informe`. Lo único que falta son los
datos de tu laptop, marcados con `[[GPU: ...]]`:

```powershell
git switch main; git pull; git switch -c adrian/seccion-5
git grep -n "\[\[" docs/05-implementacion-cuda.md
```

Cada marca dice exactamente qué dato va y de qué archivo sale. Reemplázala
por el número o por la oración que pide, con tus palabras (ver
`convenciones.md`). Si un resultado contradice lo que dice el texto (por
ejemplo, que 16 × 16 no sea de los más rápidos), cambia la oración: el
informe tiene que decir lo que mediste, y explicar por qué crees que pasó
también es análisis. Compila para revisar:
```powershell
quarto render informe/informe.qmd --to pdf
```
Las marcas `[[GPU: ...]]` de `docs/02` y `docs/06` las llena Fernando con tu
corrida; si notas algo que deba decirse ahí (el reloj de la GPU bajó, la
laptop se calentó, algo falló y se repitió), anótalo en el PR.

## Paso 6 · Notebook y demo (día 3–4)

- Cuando Hansel suba el notebook, ejecútalo en tu laptop para que quede con
  las salidas de la GPU:
  ```powershell
  uv run --no-sync jupyter nbconvert --to notebook --execute --inplace notebooks/proyecto3_asl_cuda.ipynb
  ```
  y súbelo en una rama `adrian/notebook-ejecutado`.
- Prueba la demo con cámara para la presentación: `uv run --no-sync asl-cuda demo`.
  Practica 3–4 letras que el modelo reconozca bien (revisa `per_class` en
  `results/<tag>/eval/`) y una palabra corta.

## Extra opcional (si sobra tiempo)

Perfil de una época con Nsight Systems para mostrar el traslape CPU/GPU:
```powershell
nsys profile -o results/$env:ASL_HOST_TAG/nsys/epoca --trace=cuda,osrt uv run --no-sync asl-cuda train --device cuda --max-epochs 1 --name nsys --no-monitor
```
No subas el `.nsys-rep` (es pesado); sube una captura de la línea de tiempo a
`evidencia/07-nsys-timeline.png` donde se vean las copias H2D, el kernel
`warp_kernel` y las convoluciones encadenadas.

## Último detalle (18/11)

La celda 21 del notebook (entrenamiento de 10 lotes) forzaba `channels_last`
en la GPU y por eso marcó 25 img/s. Fernando ya corrigió el código para que
use la configuración; falta volver a ejecutar el notebook (o solo esa celda,
después de las de arriba) en tu laptop y subirlo con las salidas nuevas:

```powershell
git pull
uv run --no-sync jupyter nbconvert --to notebook --execute --inplace notebooks/proyecto3_asl_cuda.ipynb
git add notebooks/proyecto3_asl_cuda.ipynb; git commit -m "notebook ejecutado en GPU sin channels_last"; git push
```
