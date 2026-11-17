# 3. Identificación de oportunidades de paralelización

Partimos de la tabla de la sección 1 y para cada operación nos hicimos tres
preguntas: ¿los datos de una unidad de trabajo dependen de otra?, ¿cuánta
aritmética hay por byte que se mueve?, y ¿la operación necesita algo que solo
existe en el CPU (sistema de archivos, librerías de decodificación)?

| No. | Operación | Independencia | Aritmética por byte | ¿Dónde va? | Razón |
|--:|--------|--------|-------|---------|-----------------------------------|
| 1–2 | Inventario y partición | — | — | CPU, una vez | Se ejecuta una sola vez y tarda menos de un segundo. |
| 3–5 | Lectura, decodificación, EXIF/RGB | Total entre archivos | Baja, con ramas (Huffman) | **CPU, multiproceso** | Necesita el sistema de archivos y Pillow/libjpeg. La decodificación Huffman es secuencial dentro de un archivo y llena de saltos, mala para una GPU. Entre archivos no hay dependencia: se reparte por bloques entre procesos. Y se hace **una sola vez** (caché). |
| 6–9 | Remuestreo, color, normalización, borrado | Total entre píxeles y entre imágenes | Media, sin ramas | **GPU, kernel Numba CUDA** | Cada píxel de salida se calcula solo con su homografía y 4 vecinos de la fuente. Son 64 × 224 × 224 = 3.2 M salidas por lote, todas independientes. |
| 10 | Armado del lote | — | Copia | CPU (gather) + copia H2D | Con la caché basta un *gather* de memoria a un buffer fijo (*pinned*), y se envía uint8: 4 veces menos bytes que float32. |
| 11–13 | Forward, pérdida, backward | Paralelismo dentro de cada capa (canales, posiciones, lote) | Alta | **GPU, cuDNN/cuBLAS vía PyTorch** | Convoluciones y productos matriciales. Se usan las implementaciones de NVIDIA, que ya están optimizadas para Tensor Cores; escribirlas a mano no tendría sentido. |
| 14 | Recorte de gradiente y AdamW | Por parámetro | Media | GPU | 1.3 M parámetros actualizados de forma independiente; ya viven en la GPU. |
| 15 | Evaluación | Igual que 6–11 | Alta | GPU | Mismo camino que el entrenamiento, sin backward. |
| 16 | Selección y guardado | — | — | CPU | Una vez por época, trivial. |

## Lo que decidimos paralelizar y por qué

**En el CPU (núcleos múltiples): todo lo que toca archivos.** La
decodificación se reparte entre procesos. Usamos procesos y no hilos porque
Pillow retiene el GIL durante buena parte de la decodificación y con hilos el
speedup se queda cerca de 1. Cada proceso escribe sus imágenes directamente en
un arreglo `uint8` en disco abierto con *memory map*, así que el proceso
principal no tiene que recibir ni copiar imágenes.

**En la GPU: toda la matemática que alimenta y entrena la red.** Hay dos
partes:

1. **La transformación de entrada (operaciones 6–9).** El Lab 3 la hacía
   imagen por imagen con PIL. Es un problema de un hilo por píxel, sin
   dependencias y sin ramas, y su salida es precisamente el tensor que
   consume la primera capa de la red. La implementamos como un kernel propio
   con Numba CUDA. La justificación para dejarla en la GPU es la misma que
   para la red: es aritmética sobre tensores, cambia en cada época (por eso
   no se puede guardar en caché) y su salida ya queda en la memoria de la GPU
   donde se necesita.
2. **La red neuronal (operaciones 11–14).** Forward, backward y optimizador
   se ejecutan en la GPU con PyTorch, que llama a cuDNN y cuBLAS. Encima
   activamos precisión mixta (fp16 en Tensor Cores), el layout
   `channels_last` y `cudnn.benchmark`.

## Lo que no conviene llevar a la GPU

- **La decodificación JPEG.** Existe nvJPEG, pero el dataset se decodifica
  una sola vez (16 s con 20 procesos en nuestra prueba con las 66,700
  imágenes). Ganar segundos en algo que pasa una vez no justifica atar el
  pipeline a otra dependencia de NVIDIA.
- **La lectura de archivos.** La GPU no tiene acceso al disco; siempre pasa
  por el CPU.
- **Las decisiones por época** (early stopping, guardar checkpoint). Son
  pocas operaciones escalares.

## Hipótesis que vamos a comprobar con las mediciones

1. Eliminar la decodificación repetida reduce el tiempo por época aun sin GPU.
2. La decodificación paralela escala casi lineal hasta el número de núcleos
   físicos y después se aplana (núcleos lógicos compartidos, disco).
3. El kernel de entrada en GPU es al menos un orden de magnitud más rápido
   que torchvision/PIL y deja de ser un cuello de botella.
4. El paso de entrenamiento en GPU con AMP es más de 10 veces más rápido que
   en CPU multinúcleo.
5. Si la aumentación se queda en CPU mientras la red corre en GPU, la GPU
   queda esperando datos (lo verificamos con la variante "Híbrido GPU,
   aumentación en CPU").
