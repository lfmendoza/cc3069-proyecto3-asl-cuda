# Hansel · medición secuencial, datos nuevos, notebook e informe

Tu parte es todo lo que no necesita NVIDIA: medir la versión secuencial y
CPU multinúcleo en una segunda computadora, probar el modelo con datos nuevos
(dataset externo y fotos del equipo), armar el notebook entregable, escribir
las secciones 2 y 6 y ensamblar el PDF.

Lee también `README.md` (plan y reglas de Git) y `convenciones.md`.

## Paso 1 · Entorno (Hito 1, día 1)

1. Instala Git y uv.
   - Windows (PowerShell):
     ```powershell
     winget install --id Git.Git -e
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   - Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Clona y prepara:
   ```bash
   git clone <url-del-repo> cc3069-proyecto3-asl-cuda
   cd cc3069-proyecto3-asl-cuda
   uv sync --extra cpu --extra demo
   ```
3. Token de Kaggle en `~/.kaggle/kaggle.json` (Windows:
   `C:\Users\<tu-usuario>\.kaggle\kaggle.json`).
4. Tag de tu equipo, con el modelo de tu CPU:
   `$env:ASL_HOST_TAG = "hansel-i7-1255u"` (PowerShell) o
   `export ASL_HOST_TAG=hansel-i7-1255u` (bash).
5. Verifica:
   ```bash
   uv run --no-sync asl-cuda env
   uv run --no-sync pytest -q
   uv run --no-sync asl-cuda download
   uv run --no-sync asl-cuda prep
   ```
   **Terminado cuando:** pasan las pruebas (las de GPU salen como *skipped*,
   es normal) y `prep` reporta las tres particiones con
   `Duplicados exactos entre particiones: 0`.

Para el informe necesitas Quarto (https://quarto.org) y una distribución de
LaTeX (`quarto install tinytex`).

## Paso 2 · Corrida CPU (Hito 2, día 2)

Mismas condiciones que Adrian: conectado a la corriente, modo máximo
rendimiento, todo cerrado.

```bash
git switch -c hansel/medicion-cpu
# Windows
.\scripts\corrida_cpu.ps1 -Tag $env:ASL_HOST_TAG
# Linux/macOS
bash scripts/corrida_cpu.sh $ASL_HOST_TAG
```
Tarda entre 30 y 60 minutos. Toma dos capturas del Administrador de tareas
(vista de procesadores lógicos): una durante `01-prep-secuencial` (un núcleo)
y otra durante `02-prep-paralelo` (todos). Guárdalas como
`evidencia/hansel-prep-secuencial.png` y `evidencia/hansel-prep-paralelo.png`.

```bash
git add results/$ASL_HOST_TAG evidencia/hansel-*
git commit -m "data: mediciones CPU en $ASL_HOST_TAG"
git push -u origin hansel/medicion-cpu
```
y abre el PR.

## Paso 3 · Fotos del equipo (día 2, los tres)

Organiza una sesión corta con Fernando y Adrian. Cada uno toma **una foto por
cada una de las 29 clases** (26 letras, `del`, `nothing` = fondo sin mano,
`space`), con la mano derecha, la mano centrada ocupando más o menos la mitad
del cuadro, en el lugar que sea (no hace falta pared lisa: queremos ver qué tan
bien generaliza). Cámara del celular o de la laptop.

Guarda así (esta carpeta **no se sube**, está en `.gitignore`):
```
.local/fotos-equipo/fernando/A_1.jpg ... Z_1.jpg, del_1.jpg, nothing_1.jpg, space_1.jpg
.local/fotos-equipo/hansel/...
.local/fotos-equipo/adrian/...
```
Pide a los tres su consentimiento por escrito (un mensaje basta) para
publicar las miniaturas en el informe. Si alguien no está de acuerdo, solo se
reportan las métricas.

## Paso 4 · Pruebas con datos nuevos (Hito 3, día 3)

Necesitas `models/asl_mobilenetv3.pt` (lo sube Adrian en su PR; si ya está en
`main`, haz `git pull`).

```bash
git switch main && git pull && git switch -c hansel/datos-nuevos
uv run --no-sync asl-cuda evaluate --split test external --prefix final
uv run --no-sync asl-cuda evaluate --split .local/fotos-equipo/fernando --name fotos-fernando
uv run --no-sync asl-cuda evaluate --split .local/fotos-equipo/hansel --name fotos-hansel
uv run --no-sync asl-cuda evaluate --split .local/fotos-equipo/adrian --name fotos-adrian
uv run --no-sync asl-cuda evaluate --split .local/fotos-equipo --name fotos-equipo
```
Esto corre en CPU sin problema (unos segundos). Deja en
`results/<tu-tag>/eval/` métricas, predicciones por foto y una hoja con
miniaturas (`fotos-*_predicciones.png`). Revisa las predicciones de cada foto
en los `_predictions.csv` y anota patrones: ¿qué letras confunde y con
cuáles? ¿Coinciden con las confusiones del test interno (`top_confusions` en
`final_test_metrics.json`)? ¿Influye el fondo o la luz?

Después corre `uv run --no-sync asl-cuda summarize` para generar las matrices
de confusión en `results/<tu-tag>/figuras/`. Sube `results/<tu-tag>/eval` y
las figuras en el PR de esta rama.

## Paso 5 · Notebook entregable (día 3)

Rama `hansel/notebook`. Archivo: `notebooks/proyecto3_asl_cuda.ipynb`. El
enunciado pide que el notebook "contenga el código original secuencial y la
nueva versión optimizada" y que **se vea clara la separación**: preparación de
datos en CPU y CUDA en el entrenamiento. El notebook no reimplementa nada:
importa el paquete `asl_cuda` y muestra cada pieza. Debe correr completo en
menos de 5 minutos con GPU y también en CPU (las celdas de GPU se saltan con
`if torch.cuda.is_available():`).

Estructura (una sección markdown por bloque, con 2–4 oraciones propias):

1. **Contexto.** Problema, dataset, qué se mide. Tabla del reparto del equipo.
2. **Entorno.** `from asl_cuda.envinfo import environment` e imprimir CPU,
   núcleos, GPU, versiones. `!nvidia-smi` dentro de un `if`.
3. **Datos y partición.** Leer `data/cache/manifest.csv`, tabla de conteos por
   partición y una cuadrícula con un ejemplo por clase (desde la caché).
4. **Versión original (secuencial).** Mostrar con `inspect.getsource` la
   función `lab3_transform` y `FileDataset` de `asl_cuda.secuencial`. Medir 2
   lotes con `make_loader(... workers=0)` y `run_epoch(..., max_batches=2)`
   con `torch.set_num_threads(1)` y mostrar `data_seconds` vs
   `compute_seconds`.
5. **Fase 1 · CPU.** Con 1,160 rutas (`subsample(..., 40)`), correr
   `decode_to_npy` con `workers=1` y con `workers=0` y mostrar el speedup.
   Explicar por qué procesos y no hilos (sección 4.2).
6. **Fase 2 · Kernel CUDA.** `inspect.getsource` de `_pixel` y
   `_warp_kernel`. Mostrar la configuración de lanzamiento con `grid_for`.
   Correr el kernel en un lote y comparar contra `warp_numpy` (diferencia
   máxima). Cuadrícula de 16 imágenes aumentadas (desnormalizar con
   `IMAGENET_MEAN/STD`).
7. **Fase 2 · Entrenamiento en GPU.** Mostrar el fragmento de `train_step`.
   No entrenar completo: cargar `results/<tag-adrian>/train/base-cuda/history.csv`
   y graficar; mostrar `summary.json` (tiempo total, img/s, memoria pico).
   Opcional: 20 pasos en vivo con `fit(..., limit=...)` para que se vea correr.
8. **Mediciones.** Cargar y mostrar las tablas de
   `results/<tag-adrian>/tablas/*.md` (con `IPython.display.Markdown`) y las
   figuras `speedup_*.png`, `prep_escalamiento.png`, `kernel_bloques.png`,
   `monitor_bench.png`.
9. **Datos nuevos.** Métricas de test, externo (mitad final) y fotos del
   equipo; hoja de miniaturas si hay consentimiento.
10. **Conclusiones.** 5–6 oraciones, las mismas ideas de la sección 6.

Antes de subir: *Kernel → Restart & Run All* en tu máquina para verificar que
corre en CPU; luego Adrian lo ejecuta en GPU y sube la versión con salidas.

## Paso 6 · Secciones 2 y 6 del informe (Hito 4, día 4)

Rama `hansel/informe`.

**`docs/02-medicion-secuencial.md`.** El procedimiento ya está escrito;
revísalo y ajusta si algo cambió. Completa:
- 2.2 Resultados: tablas de la versión secuencial y CPU multinúcleo de la
  **laptop de Adrian** (son las oficiales porque ahí también se mide la GPU) y
  de tu computadora como segunda referencia. Incluye la figura
  `prep_escalamiento.png` de ambos (cada una está en `results/<tag>/figuras/`
  de su equipo).
- 2.3 Factores: GIL (fila "hilos" de `prep.md`), núcleos físicos vs lógicos
  (dónde se aplana la curva), arranque de procesos en Windows (*spawn*),
  caché de disco del sistema operativo (por qué hay calentamiento), PIL
  procesando una imagen a la vez, `torch.set_num_threads` y el
  `DataLoader` con 4 procesos del Lab 3 que no siempre ayuda (comparar
  "Lab 3 secuencial" vs "Lab 3 como se entregó" en `pipeline.md`).
  Usa la columna Karp-Flatt: si `e` crece con los procesos, el límite es
  overhead y no la parte serial.

**`docs/06-analisis-conclusiones.md`.** 6.1–6.3 con tablas y figuras de la
corrida de Adrian; 6.4 en conjunto con Adrian (él escribe "cómo CUDA cambió
el tiempo" y "factores que limitan"). Tú escribes la primera pregunta:

> ¿Por qué es más eficiente procesar las imágenes/archivos con los múltiples
> núcleos del CPU y delegar exclusivamente la matemática de la red neuronal
> al GPU?

Puntos que deben aparecer, con números propios:
- La decodificación JPEG es secuencial dentro de cada archivo y llena de
  ramas (Huffman); la GPU no puede leer el disco. Entre archivos no hay
  dependencia → procesos del CPU (speedup medido en `prep.md`).
- Se decodifica una vez: 1.9 M decodificaciones del Lab 3 → 66,700.
- La red es aritmética densa y regular (convoluciones), ~0.11 GFLOP por
  imagen en forward; la GPU tiene miles de núcleos para eso (speedup en
  `train.md`).
- Mandar uint8 de 200×200 en lugar de float32 de 224×224 reduce 5 veces lo
  que cruza el PCIe (`transferencias.md`).
- La ablación `base-cuda-aug-cpu`: si el CPU hace la aumentación, la GPU
  espera (comparar `data_wait_seconds` e img/s con `base-cuda`).

6.5 Datos nuevos: tabla con test interno, externo mitad final, fotos por
integrante; discusión de las confusiones (M/N/S/X/T son puños parecidos) y
del cambio de dominio (fondo, luz, otra mano).

6.6 Conclusiones: 6–8 oraciones, una por hallazgo, cada una con su número.

**Ensamblar el PDF:**
```bash
quarto render informe/informe.qmd --to pdf
```
Copia a `informe/figuras/` solo las figuras que se usan y referéncialas desde
los `docs/*.md` con rutas relativas `../informe/figuras/<archivo>.png`.
Revisa el PDF completo: tablas que no se salgan del margen, figuras legibles,
ninguna sección vacía.

## Paso 7 · README y presentación (día 4–5)

- Llena la tabla de resultados del `README.md` con los números finales.
- Arma las diapositivas (Google Slides o PowerPoint, 12–15 láminas): cada
  quien presenta su parte. Usa las figuras de `results/<tag-adrian>/figuras/`.
