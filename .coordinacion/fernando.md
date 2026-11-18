# Fernando · algoritmo y diseño paralelo

## Hecho

- Estructura del repositorio, entorno con perfiles cpu/cu126/cu130, CI.
- `src/asl_cuda/` completo y probado en CPU; kernel probado en el simulador.
- Secciones 1, 3 y 4 (`docs/01`, `docs/03`, `docs/04`).
- Planes de trabajo de Hansel y Adrian.
- Secciones 2, 5 y 6 redactadas con marcas `[[GPU: ...]]` donde va un dato
  de la corrida oficial; tablas y figuras automáticas con `asl-cuda informe`.
- Notebook entregable `notebooks/proyecto3_asl_cuda.ipynb` (probado en CPU).
- Limpieza de los resultados de Hansel (logs y capturas).

## Pendiente

1. **Día 1 · Soporte a Adrian.** Estar disponible mientras Adrian corre
   `asl-cuda env` y `tests/test_gpu.py`. Si falla algo propio de la GPU
   (stream externo, interoperabilidad Numba–PyTorch, memoria), el arreglo va
   en una rama `fernando/fix-gpu-*` con PR, y Adrian vuelve a correr las
   pruebas antes del merge.
2. **Revisión de PRs.** Revisar que cada PR toque solo la zona de su autor,
   que no suba `data/`, `.local/` ni `scratch/`, que los mensajes de commit
   sigan el formato, y que CI pase. Merge sin squash.
3. **Día 3 · Llenar las marcas `[[GPU: ...]]` de `docs/02` y `docs/06`**
   con la corrida de Adrian (`git grep -n "\[\[" docs`), revisar que la
   sección 4 siga coincidiendo con lo medido (4.3 bloque, 4.4 traslape, 4.6
   recetas) y llenar la tabla de resultados del `README.md` con `fases.md`,
   `proyeccion_entrenamiento.md` y `evaluacion.md`.
4. **Día 4 · Revisión del informe completo** antes de que Hansel compile la
   versión final: números consistentes entre secciones, mismo equipo para
   todos los speedup, figuras citadas en el texto.
5. **Día 5 · Cierre** (ver `cierre.md`).
