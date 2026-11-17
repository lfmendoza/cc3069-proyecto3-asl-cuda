# Fernando · algoritmo y diseño paralelo

## Hecho

- Estructura del repositorio, entorno con perfiles cpu/cu126/cu130, CI.
- `src/asl_cuda/` completo y probado en CPU; kernel probado en el simulador.
- Secciones 1, 3 y 4 (`docs/01`, `docs/03`, `docs/04`).
- Planes de trabajo de Hansel y Adrian.

## Pendiente

1. **Día 1 · Soporte a Adrian.** Estar disponible mientras Adrian corre
   `asl-cuda env` y `tests/test_gpu.py`. Si falla algo propio de la GPU
   (stream externo, interoperabilidad Numba–PyTorch, memoria), el arreglo va
   en una rama `fernando/fix-gpu-*` con PR, y Adrian vuelve a correr las
   pruebas antes del merge.
2. **Revisión de PRs.** Revisar que cada PR toque solo la zona de su autor,
   que no suba `data/`, `.local/` ni `scratch/`, que los mensajes de commit
   sigan el formato, y que CI pase. Merge sin squash.
3. **Día 3 · Ajustar la sección 4 con los resultados reales.**
   - 4.3: agregar una oración con el tamaño de bloque que resultó mejor
     (`kernel_bloques.md` de Adrian) y si coincide con el razonamiento.
   - 4.4: confirmar con `data_wait_seconds` de `base-cuda` que el traslape
     funciona (debe ser una fracción pequeña del tiempo de la época).
   - 4.6: cambiar la tabla del piloto por la comparación completa
     `base` vs `robusto` en la mitad "selección" (`seleccion_modelo.json`).
   - Revisar que la sección 3 ("hipótesis") se responda en la sección 6.
4. **Día 4 · Revisión del informe completo** antes de que Hansel compile la
   versión final: números consistentes entre secciones, mismo equipo para
   todos los speedup, figuras citadas en el texto.
5. **Día 5 · Cierre** (ver `cierre.md`).
