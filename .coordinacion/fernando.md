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
- Corrida CPU de referencia limpia en la Ryzen (`results/fernando-ryzen-ai9/`).
- `asl-cuda fotos` y `scripts/datos_nuevos.*` para que la parte de fotos de
  Hansel sea tomar las fotos y correr un script (probado con 87 fotos).
- Presentación de 20 láminas con notas (carpeta privada, fuera del repo);
  se regenera con los números de Adrian.

## Hecho con la corrida oficial de Adrian (18/11)

- `pyproject.toml` y `uv.lock`: NumPy < 2.5 y bibliotecas de CUDA 13.0 en
  `cu130` (lo que Adrian tuvo que fijar a mano).
- Nota de `channels_last` en la sección 4.5 y menciones corregidas en 3, 4 y
  el README.
- Tabla de resultados del README llena.
- Logs de `results/adrian-rtx3050/logs` en UTF-8 y sin rutas personales; los
  scripts de PowerShell ya guardan los logs en UTF-8.
- Presentación con los números oficiales y guiones de los tres llenos.

## Pendiente

1. Cuando Hansel suba las fotos: revisar su PR y el párrafo de 6.5, y poner
   los aciertos de las fotos en la presentación (`numeros.json`).
2. **Cierre** (ver `cierre.md`): PDF final, borrar `.coordinacion`, tag
   `v1.0` y entregar en Canvas el PDF, el notebook y el enlace al repo.
