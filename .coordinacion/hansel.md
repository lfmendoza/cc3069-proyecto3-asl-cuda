# Hansel · fotos del equipo, datos nuevos y presentación

Lee también `README.md` (plan y reglas de Git) y `convenciones.md`.

## Estado

- **Pasos 1 y 2 (entorno y corrida CPU): hechos** (PR #1). Fernando limpió
  después dos detalles para la entrega: los logs `02-prep-paralelo.log` y
  `04-summarize.log` tenían rutas personales y un error de un intento
  anterior, y las capturas se recortaron a la ventana de historial de CPU y a
  los procesos de Python.
- **Notebook (antes paso 5): hecho** por Fernando en
  `notebooks/proyecto3_asl_cuda.ipynb`. Corre completo en CPU en ~30 s.
- **Secciones 2 y 6 (antes paso 6): redactadas.** La 2 ya usa tus números de
  la M3. Lo que depende de la GPU está marcado con `[[GPU: ...]]` y lo llena
  Fernando cuando entre la corrida de Adrian.

Te queda: fotos del equipo, evaluarlas con el modelo final, el párrafo de
fotos de la sección 6.5 y las diapositivas.

### Opcional pero recomendado: repetir la corrida CPU limpia

Tu corrida se hizo con otras aplicaciones abiertas (el monitor muestra los
núcleos de eficiencia ocupados 12–28 % aun con un hilo). Si tienes 40
minutos, repítela tal cual con todo cerrado, desde la terminal, conectada a
la corriente:

```bash
git switch main && git pull && git switch -c hansel/medicion-cpu-2
bash scripts/corrida_cpu.sh hansel-apple-m3
```

Reemplaza `results/hansel-apple-m3/` completo. Capturas nuevas: solo la
ventana del Monitor de Actividad y su ventana "Historial de CPU", sin
escritorio ni dock; la de paralelo tómala mientras `02-prep-paralelo` está
procesando la partición `train` (los primeros ~15 s). Si la repites, avisa a
Fernando: la sección 2.3 cita números tuyos que habría que actualizar.

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

Después corre:
```bash
uv run --no-sync asl-cuda summarize --tag hansel-apple-m3
uv run --no-sync asl-cuda informe --tag <tag-de-adrian> --cpu-tag hansel-apple-m3
```
El primero genera `results/hansel-apple-m3/tablas/evaluacion.md` y las
matrices de confusión; el segundo copia esa tabla al informe (sección 6.5).

Luego reemplaza en `docs/06-analisis-conclusiones.md` el bloque
`[[Fotos: ...]]` de la sección 6.5 por un párrafo propio con lo que
encontraste. Sube `results/hansel-apple-m3/eval`, `informe/tablas` y
`docs/06` en el PR de esta rama.

## Paso 5 · Presentación (día 4–5)

- Arma las diapositivas (Google Slides o PowerPoint, 12–15 láminas): cada
  quien presenta su parte. Usa las figuras de `results/<tag-adrian>/figuras/`.
