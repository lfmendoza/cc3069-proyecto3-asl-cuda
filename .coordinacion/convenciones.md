# Convenciones del equipo

Aplican a todo lo que se sube al repositorio, lo escriba una persona o lo
genere una herramienta.

## Redacción (informe, docs, notebook, commits)

El informe tiene que leerse como lo que es: un trabajo de tres estudiantes que
midieron cosas en sus computadoras. Reglas:

- Español, primera persona del plural ("medimos", "decidimos", "nos dimos
  cuenta"). Tono técnico pero directo.
- Cada afirmación lleva el dato que la respalda: "la GPU pasó 4 % del tiempo
  esperando datos (`data_wait_seconds` = 1.2 s de 30 s)" y no "la GPU se
  aprovechó de forma eficiente".
- Párrafos cortos. Las listas son para cosas que de verdad son una lista;
  las explicaciones van en prosa.
- Evitar el relleno típico: "en resumen", "cabe destacar", "es importante
  mencionar", "en conclusión", "juega un papel crucial", "robusto",
  "integral", "sinergia", "potenciar", "aprovechar al máximo", "en el mundo
  actual", "un enfoque holístico", "sin duda". Nada de emojis.
- No poner negritas en cada oración ni títulos con dos puntos decorativos.
- Está bien decir lo que salió mal o lo que no esperábamos. Un resultado
  honesto (p. ej. "con 20 procesos fue más lento que con 8") es mejor que uno
  pulido; explicarlo vale puntos.
- Números con el formato del resto: separador de miles con coma (43,500),
  decimales con punto, unidades con espacio (3.5 s, 7.7 MB).
- Nombres de archivos, funciones y comandos en `código`.

## Limpieza

- Los mensajes de commit son solo el título y, si hace falta, un párrafo
  corto. Sin firmas ni líneas automáticas al final.
- En notebooks: limpiar celdas de prueba antes de subir; los comentarios del
  código suenan a quien lo escribió, no a un tutorial.
- Nada de rutas personales (`C:\Users\...`) en archivos que se suben.

## Commits

Formato corto en español, en presente, con prefijo:

```
feat: medición de la fase de inferencia en GPU
fix: sincronizar el stream antes de reutilizar el buffer pinned
docs: sección 5 con tiempos de la corrida oficial
data: resultados de la corrida oficial en adrian-rtx4060
```

Prefijos: `feat`, `fix`, `docs`, `data` (resultados, modelo, evidencia),
`test`, `chore`. Un commit por cambio lógico.

## Código

- Python 3.12, `ruff check src tests` sin errores.
- Comentarios en español, solo donde el *por qué* no es obvio.
- Toda medición nueva se agrega a `bench.py` (no scripts sueltos) para que
  quede registrada con el mismo formato.
- Ninguna ruta absoluta propia en el código ni en los resultados que se suben.

## Tags de equipo

`--tag <nombre>-<hardware>` en minúsculas, por ejemplo `adrian-rtx4060`,
`hansel-i7-1255u`, `fernando-ryzen9`. Se usa el mismo en todas las corridas de
esa computadora. También se puede definir una vez:
`$env:ASL_HOST_TAG = "adrian-rtx4060"` (PowerShell) o
`export ASL_HOST_TAG=adrian-rtx4060` (bash).
