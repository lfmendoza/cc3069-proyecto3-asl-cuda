# 2. Medición del tiempo de ejecución secuencial

## 2.1 Procedimiento

Todas las mediciones usan el mismo subconjunto de imágenes: 40 por clase
(1,160 en total) tomadas a intervalos iguales a lo largo de cada video de la
partición de entrenamiento, y 200 por clase (5,800) para la fase de
decodificación, donde con menos imágenes el arranque de los procesos tapa el
resultado. El subconjunto es determinista, así que cualquier computadora mide
exactamente las mismas imágenes.

Cada variante se ejecuta una vez como calentamiento (se descarta) y luego 5
veces seguidas (3 en la época completa). El calentamiento absorbe la
compilación JIT de Numba, la selección de algoritmos de cuDNN y la primera
lectura de los archivos desde el disco. Reportamos media, desviación estándar
y coeficiente de variación.

Condiciones: laptop conectada a la corriente, perfil de energía de máximo
rendimiento, sin otras aplicaciones abiertas y la misma versión del código
(commit registrado en `environment.json`).

Comando: `asl-cuda bench --tag <equipo>`; los datos crudos quedan en
`results/<equipo>/bench/`.

<!-- Hansel: completar con los resultados de results/<equipo>/tablas/ -->

## 2.2 Resultados

## 2.3 Factores que influyen en el rendimiento
