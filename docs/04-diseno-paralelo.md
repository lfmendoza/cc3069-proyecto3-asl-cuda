# 4. Diseño del algoritmo paralelo

## 4.1 Vista general

El programa quedó dividido en dos fases con una frontera clara: el CPU
entrega bytes `uint8` ya decodificados y la GPU hace toda la aritmética.

```
FASE 1 - CPU (una sola vez)                 FASE 2 - GPU (cada época, por lote de 64)
-------------------------------------       ---------------------------------------------
manifest + partición agrupada               CPU: gather de índices desde el memmap
            |                                    a un buffer pinned uint8 (7.7 MB)
            v                                          |
ProcessPoolExecutor (N procesos)                       v
  cada proceso, un bloque de rutas:         copia H2D asíncrona (stream de PyTorch)
  leer -> SHA-256 -> decodificar JPEG                  |
  -> EXIF -> RGB -> escribir en memmap                 v
            |                               kernel Numba: homografía + bilineal +
            v                               color + normalización + borrado
data/cache/<split>_200.images.npy             -> tensor float32 channels_last (38.5 MB)
  uint8, N x 200 x 200 x 3                             |
  + labels, sha256 y meta.json                         v
                                            MobileNetV3 forward/backward fp16 (cuDNN)
                                                       |
                                                       v
                                            AdamW + GradScaler; métricas en la GPU
```

`src/asl_cuda/prep.py` es la fase 1 y no importa PyTorch. `train.py`,
`feeders.py` y `augment.py` son la fase 2.

## 4.2 Fase 1: decodificación paralela en CPU

**Descomposición:** por datos. Las rutas de una partición se cortan en
bloques de `max(8, min(256, N / (4p)))` imágenes, es decir, unos cuatro
bloques por proceso. Con un solo bloque por proceso, el que reciba imágenes
más pesadas atrasa a todos; con bloques muy pequeños se paga demasiada
comunicación.

**Comunicación:** cada proceso abre el mismo `.npy` en modo `r+` (memory
map) y escribe en sus filas. Al proceso principal solo regresan los hashes,
errores y tiempos, unos cientos de bytes por bloque.

**Por qué procesos y no hilos:** lo probamos. Con el mismo código y 10 hilos
la decodificación de 5,800 imágenes tardó 6.6 s contra 3.5 s secuencial
(speedup 0.53); con 8 procesos tardó 0.82 s (speedup 4.3). Pillow libera el
GIL dentro del decodificador, pero todo lo que rodea a cada imagen (abrir el
archivo, el hash, EXIF, convertir a NumPy, el ciclo en Python) lo retiene, y
con imágenes de 200 × 200 esa parte pesa tanto como la decodificación misma.
Los hilos terminan peleándose el GIL. En Windows los procesos se crean con
*spawn*; por eso `prep.py` no importa torch (cada proceso nuevo tardaría
segundos en arrancar).

**Integridad:** se calcula el SHA-256 de cada archivo con los mismos bytes que
se decodifican y al final se verifica que ningún archivo idéntico aparezca en
dos particiones. Con la partición por ventanas el resultado fue 0.

**Resultado esperado:** speedup alto hasta el número de núcleos físicos y
luego un aplanamiento. Lo medimos con la fracción serial de Karp-Flatt (ver
sección 6).

## 4.3 Fase 2: kernel de entrada en Numba CUDA

### Qué calcula

Para cada imagen *b* del lote, el host genera 16 números (`augment.py`,
`sample_params`):

- una homografía 3 × 3 que lleva el centro de cada píxel de salida a una
  posición sub-píxel de la imagen fuente. Se arma como
  `H = S · A · P · N`, donde `N` normaliza la salida a [−1, 1], `P` es una
  perspectiva leve, `A` rotación y zoom, y `S` el recorte aleatorio sobre la
  fuente. Así, recorte, rotación, traslación, zoom y perspectiva cuestan lo
  mismo que una sola transformación;
- tres factores de color: brillo, contraste y saturación;
- un rectángulo de borrado (vacío si no aplica).

En evaluación la homografía es fija: el 87.5 % central de la imagen llevado a
224 × 224, que es el mismo campo visual de `Resize(256) + CenterCrop(224)`
del Lab 3.

### Distribución del trabajo

| Nivel | Dimensión | Tamaño | Qué representa |
|---|---|---|---|
| Hilo | (x, y) | 1 | un píxel de salida, sus 3 canales |
| Bloque | 16 × 16 × 1 | 256 hilos (8 warps) | un parche de 16 × 16 píxeles de una imagen |
| Grid | ceil(224/16) × ceil(224/16) × B | 14 × 14 × 64 = 12,544 bloques | todo el lote |

`cuda.grid(3)` entrega `(x, y, b)` directamente. Usamos la tercera
dimensión del grid para el índice de imagen porque así ningún hilo tiene que
dividir ni calcular módulos, y la condición de borde
(`x < 224 and y < 224 and b < B`) cubre lotes incompletos y bloques que no
dividen exacto el tamaño de salida.

Por qué 16 × 16:

- 256 hilos por bloque es múltiplo de 32 (warps completos) y deja que cada
  multiprocesador tenga varios bloques activos, lo que ayuda a esconder la
  latencia de memoria.
- Los 16 hilos consecutivos en `x` de un warp leen píxeles vecinos de la
  fuente y escriben posiciones contiguas en la salida `channels_last`
  (cada píxel son 3 floats seguidos), así que las escrituras se combinan
  bien.
- Un bloque cuadrado mantiene juntas en caché las filas de la fuente que
  usan los hilos, aunque haya rotación.

Este valor se eligió por razonamiento; el benchmark lo compara contra 8×8,
16×8, 32×8, 32×16 y 32×32 (`asl-cuda bench --phases augment`), que es el
mismo experimento del ejemplo de clase con 128, 256 y 512 hilos. El resultado
está en la sección 5.4.

### Trabajo por hilo

1. Aplica la homografía a `(x + 0.5, y + 0.5)` y divide entre `w`.
2. Toma los 4 vecinos de la fuente (con coordenadas recortadas al borde) y
   hace interpolación bilineal en los 3 canales: 12 lecturas `uint8`.
3. Aplica brillo, contraste y saturación, y recorta a [0, 1].
4. Normaliza con la media y desviación de ImageNet, o escribe 0 si el píxel
   cae en el rectángulo de borrado.
5. Escribe 3 `float32`.

Son unas 60 operaciones de punto flotante por 12 bytes leídos y 12
escritos: el kernel está limitado por memoria, no por cálculo. Por eso no
usamos memoria compartida: la región de la fuente que necesita un bloque
depende de la rotación de cada imagen y no se puede cargar en un mosaico fijo;
la caché L1/L2 ya aprovecha la localidad de los vecinos.

### Detalles que importan

- **Todo en float32.** En Numba un literal como `0.5` es float64 y convierte
  toda la expresión a doble precisión, que en una GPU de laptop corre a 1/32
  o 1/64 de la velocidad. Por eso las constantes están declaradas como
  `np.float32`.
- **Un solo bloque de parámetros.** Homografía, color y borrado viajan juntos
  en un arreglo `(B, 16)` float32: una copia H2D de 4 KB por lote en vez de
  tres.
- **Misma función, dos backends.** La matemática por píxel (`_pixel`) se
  compila como función de dispositivo CUDA y también como función de CPU con
  `numba.njit(parallel=True)`. Las pruebas comparan ambas contra una
  implementación NumPy independiente (`tests/test_kernel.py` exige una
  diferencia máxima menor a 1e-4 sobre valores ya normalizados, que van de
  −2.1 a 2.6; lo que queda es redondeo de float32). La versión CPU sirve para la variante "todo en CPU" y para la
  ablación.
- **Sin copia entre Numba y PyTorch.** El tensor de salida lo reserva
  PyTorch con `memory_format=channels_last` y se le pasa al kernel con
  `cuda.as_cuda_array(tensor, sync=False)`. El kernel se lanza en el mismo
  stream que usa PyTorch (`cuda.external_stream`), así que el orden entre el
  kernel y la red está garantizado sin sincronizar el CPU.

## 4.4 Superposición CPU–GPU

`CudaFeeder` (`feeders.py`) usa dos buffers *pinned* que se alternan. Para
el lote *k*:

1. el CPU espera el evento CUDA del buffer `k mod 2` (es decir, que la copia
   del lote *k − 2* ya terminó),
2. hace el *gather* de los índices del lote desde el memmap a ese buffer,
3. encola la copia H2D asíncrona, registra el evento, encola el kernel y
   entrega el tensor.

Como todo lo que se encola en la GPU es asíncrono, mientras la GPU ejecuta
el forward/backward del lote *k* el CPU ya está armando el *k + 1*. El tiempo
que el CPU pasa armando lotes queda registrado como `data_wait_seconds` en
el historial de cada época. Si el traslape funciona, ese tiempo debe ser una fracción
pequeña de la época; la sección 5.3 lo compara con la variante que hace la
transformación en el CPU.

Se envían `uint8` de 200 × 200 (7.7 MB por lote) en lugar de `float32` de
224 × 224 (38.5 MB): **5 veces menos tráfico por PCIe**. El benchmark mide
las dos copias (`H2D lote uint8` vs `H2D lote float32`).

## 4.5 Red neuronal en GPU

Se mantiene el mismo modelo, partición y receta del Lab 3 para que la
comparación sea justa. Lo que cambia en GPU:

| Técnica | Qué hace | Por qué ayuda |
|---|---|---|
| Precisión mixta (`torch.autocast` fp16 + `GradScaler`) | Convoluciones y matmul en fp16, acumulación y pesos en fp32 | Usa Tensor Cores y reduce a la mitad el tráfico de activaciones. `GradScaler` evita que gradientes pequeños se vuelvan 0 en fp16. |
| `channels_last` | Activaciones en NHWC | Es el layout que usan los kernels de cuDNN con Tensor Cores; evita transposiciones internas. |
| `cudnn.benchmark` | Prueba algoritmos de convolución la primera vez | Las formas de entrada no cambian, así que se elige el más rápido una vez. |
| Métricas acumuladas en GPU | Pérdida y aciertos como tensores | Un `.item()` por lote obliga a sincronizar el CPU con la GPU; se hace uno por época. |

## 4.6 Receta de entrenamiento

| | Lab 3 (robust-v3) | Este proyecto (`configs/train.yaml`) |
|---|---|---|
| Modelo | MobileNetV3-Small ImageNet | igual |
| Partición | ventanas de 100 frames con separadores | iguales ventanas; test = ventanas 17–20 completas (400 por clase) |
| Entrada | 224 × 224, normalización ImageNet | igual |
| Etapas | búsqueda de 8 corridas (33 épocas en total) | cabeza 2 épocas + ajuste fino de 4 bloques hasta 12 épocas |
| Optimización | AdamW, LR discriminativa, `ReduceLROnPlateau` | AdamW, LR discriminativa, coseno por paso |
| Parada | paciencia 3 sobre macro F1 | igual |

Pasamos de la búsqueda completa a la receta del ganador porque la búsqueda ya
se hizo en el Lab 3; repetirla no aporta nada a la comparación de tiempos.

### Variante para datos nuevos (`configs/train-robusto.yaml`)

La debilidad del Lab 3 era el desempeño fuera del dominio: el dataset tiene un
solo fondo, una persona y una iluminación. Para eso añadimos una segunda
receta con aumentación más fuerte (recorte desde 45 % del área, rotación
±15°, color ±40 %, saturación ±60 %, borrado con probabilidad 0.35), *label
smoothing* 0.1 y 8 bloques descongelados. En CPU no era práctico probarla
completa; en GPU cada corrida toma minutos.

Hicimos un piloto corto en CPU con 150 imágenes por clase y 4 épocas para
ver si la dirección era correcta, evaluando sobre el dataset externo
`danrasband/asl-alphabet-test` (870 fotos con fondos y manos distintas):

| Receta (piloto, 150 por clase, 4 épocas) | Test interno macro F1 | Externo macro F1 |
|---|---:|---:|
| `train.yaml` | 0.881 | 0.414 |
| `train-robusto.yaml` | 0.901 | 0.457 |

La mejora existe incluso con tan pocos datos. La comparación con datos
completos se hizo en GPU y se decidió con la mitad "selección" del dataset
externo; la mitad "final" se reporta una sola vez con el modelo elegido
(sección 6.5).

## 4.7 Cómo verificamos que el diseño es correcto

- `tests/test_kernel.py`: el kernel CUDA (en el simulador de Numba), el kernel
  de CPU y NumPy dan el mismo resultado para varios tamaños de bloque,
  incluyendo bloques que no dividen exacto la salida.
- `tests/test_data.py`: decodificar con 1 o con 3 procesos produce
  exactamente los mismos bytes; ninguna ventana queda repartida entre dos
  particiones.
- `tests/test_gpu.py` (solo con GPU): el kernel real contra NumPy y el
  `CudaFeeder` contra el `CpuFeeder` lote por lote.
- `asl-cuda env` corre el kernel en la GPU y lo compara con NumPy antes de
  cualquier medición.
