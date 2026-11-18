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

Te queda solo lo que necesita personas o el modelo final: tomar las fotos
(10 minutos con los tres), correr un script cuando Adrian suba el modelo y
escribir un párrafo. La presentación ya está armada (ver paso 5).

### Corrida CPU de referencia

Como tu corrida en la M3 se hizo con otras aplicaciones abiertas, Fernando
repitió la corrida completa en su laptop (Ryzen AI 9 365) en condiciones
limpias: `results/fernando-ryzen-ai9/`, sección 2.3 del informe. La M3 queda
en la sección 2.4 como comparación de arquitectura (núcleos de rendimiento y
de eficiencia), con la advertencia de las condiciones. No hace falta que
repitas nada.

## Paso 3 · Fotos del equipo (día 2, los tres, ~10 min)

Cada uno toma **una foto por cada una de las 29 clases** (26 letras, `del`,
`nothing` = fondo sin mano, `space`) con la cámara de la laptop. El comando
muestra qué seña toca y el mismo recuadro de la demo; se pone la mano
derecha dentro del recuadro y se presiona espacio:

```bash
uv run --no-sync asl-cuda fotos --persona hansel
uv run --no-sync asl-cuda fotos --persona fernando
uv run --no-sync asl-cuda fotos --persona adrian
```

Teclas: espacio = guardar y pasar a la siguiente, `b` = repetir la anterior,
`s` = saltar, `q` = salir. Si se corta, al volver a correrlo sigue donde iba.
Las fotos quedan en `.local/fotos-equipo/<persona>/A_1.jpg ...` (no se suben).
No hace falta pared lisa: queremos ver qué tan bien generaliza. Si alguien
prefiere el celular, basta con guardar sus fotos con esos mismos nombres en
su carpeta.

Pide a los tres su consentimiento por escrito (un mensaje basta) para
publicar las miniaturas en el informe. Si alguien no está de acuerdo, se
borra la hoja `fotos-equipo_predicciones.png` y solo se reportan las métricas.

## Paso 4 · Pruebas con datos nuevos (Hito 3, cuando el modelo esté en `main`)

```bash
git switch main && git pull && git switch -c hansel/datos-nuevos
bash scripts/datos_nuevos.sh hansel-apple-m3 adrian-rtx3050
```

Corre en CPU en un par de minutos: evalúa el modelo final con el test
interno, el dataset externo, las fotos de cada persona y las 87 juntas, y al
final actualiza las tablas del informe. Ya se probó de punta a punta con 87
fotos y el modelo piloto.

Después revisa `results/hansel-apple-m3/eval/fotos-equipo_predicciones.png`
(miniaturas, en rojo los errores) y los `fotos-*_predictions.csv`, y
reemplaza en `docs/06-analisis-conclusiones.md` el bloque `[[Fotos: ...]]`
de la sección 6.5 por un párrafo con lo que encontraste: qué letras
confunde y con cuáles, si coinciden con `top_confusions` de
`final_test_metrics.json`, y si los errores dependen del fondo o la luz.

Sube `results/hansel-apple-m3/eval`, `results/hansel-apple-m3/logs`,
`informe/` y `docs/06` en el PR de esta rama.

## Paso 5 · Presentación (día 4–5)

Fernando ya armó las diapositivas (20 láminas, una sección por integrante,
con notas del orador); te las comparte junto con tu guion. Los valores en
naranja (`[GPU]`, `[Fotos]`) se llenan cuando entren la corrida de Adrian y
tus fotos. Te toca ensayar tu parte (≈ 5 min) con el guion.
