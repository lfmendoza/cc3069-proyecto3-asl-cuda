# 1. Análisis de la solución secuencial existente

## 1.1 Punto de partida

La solución que tomamos como base es la del Laboratorio 3 de Data Science
(reconocimiento del alfabeto ASL), en su versión final `robust-v3`. El
problema es clasificar una imagen de una mano en una de 29 clases: las 26
letras más `DELETE`, `NOTHING` y `SPACE`.

- **Datos:** dataset `grassknoted/asl-alphabet` de Kaggle, 87,000 JPEG de
  200 × 200 px, 3,000 por clase. Cada clase viene de un solo video, así que
  las imágenes con índice cercano son casi idénticas.
- **Modelo:** MobileNetV3-Small preentrenada en ImageNet (1.55 M de
  parámetros), con la última capa cambiada a 29 salidas.
- **Resultado del Lab 3:** macro F1 de 0.989 en validación y 0.930 en el
  conjunto de auditoría.
- **Costo:** el entrenamiento completo tomó **13,414 s (3.7 h)** en CPU, en
  una laptop con Ryzen AI 9 365 (20 hilos lógicos) usando 16 hilos de PyTorch
  y 4 procesos en el `DataLoader`.

El código original está en el repositorio del Lab 3. En este proyecto
copiamos a `src/asl_cuda/secuencial.py` las partes que se ejecutan en cada
época (dataset, transformaciones y ciclo de entrenamiento) sin cambiar su
lógica, para poder medirlas con los mismos datos que la versión nueva.

## 1.2 Partes principales del programa

| Módulo (Lab 3) | Qué hace |
|-------|----------------------|
| `data.py` | Recorre las carpetas del dataset, arma el *manifest* (ruta, clase, índice de frame, hashes) y define `ASLDataset`, que abre y transforma una imagen por cada acceso. |
| `robust.py` · partición | Agrupa los frames en ventanas de 100 y asigna ventanas completas a train, validation y audit, con ventanas de separación entre ellas para que frames vecinos no queden en particiones distintas. |
| `robust.py` · `robust_transform` | Cadena de transformaciones de torchvision sobre PIL: EXIF, RGB, recorte aleatorio, afín, color, desenfoque, perspectiva, `ToTensor`, normalización y borrado aleatorio. En evaluación: `Resize(256)` + `CenterCrop(224)`. |
| `robust.py` · modelos | Construye MobileNetV3-Small o ResNet18, congela y descongela bloques y arma grupos de parámetros con tasas de aprendizaje distintas para backbone y cabeza. |
| `robust.py` · `_train` | Ciclo de entrenamiento: AdamW, pérdida de entropía cruzada, recorte de gradiente, `ReduceLROnPlateau`, *early stopping* por macro F1 y guardado del mejor checkpoint. |
| `robust.py` · `train_robust` | Orquesta la búsqueda: 5 configuraciones de *screening*, 2 finalistas y un refinamiento final (*successive halving*). |
| `metrics.py` | Exactitud, exactitud balanceada, precisión, recall y F1 macro, top-3 y log loss. |
| `robust.py` · auditoría | Congela el modelo ganador (hash del checkpoint) y lo evalúa una sola vez sobre el conjunto de auditoría. |

## 1.3 Descomposición en operaciones

Flujo desde que una imagen está en disco hasta que se obtiene una predicción
o se actualizan los pesos:

| No. | Operación identificada | Descripción de su funcionamiento |
|--:|-----------|-------------------------|
| 1 | Inventario del dataset | Recorre 29 carpetas y lista 87,000 archivos. Extrae el índice de frame del nombre (`A1234.jpg` → 1234). |
| 2 | Partición agrupada | Calcula la ventana `(índice − 1) // 100` y la asigna a train, validation o audit según la configuración. Se ejecuta una vez. |
| 3 | Lectura del archivo | `Image.open` sobre la ruta: llamada al sistema de archivos y lectura de ~10–20 KB de JPEG. |
| 4 | Decodificación JPEG | Huffman + IDCT para reconstruir los 200 × 200 × 3 píxeles. Es la operación de CPU más cara del lado de los datos. |
| 5 | Orientación EXIF y conversión a RGB | Corrige la rotación si el archivo trae metadatos y garantiza 3 canales. |
| 6 | Transformaciones geométricas | Entrenamiento: `RandomResizedCrop`, `RandomAffine` y `RandomPerspective` remuestrean la imagen con interpolación. Evaluación: `Resize(256)` y `CenterCrop(224)`. Cada una recorre todos los píxeles. |
| 7 | Transformaciones de color | `ColorJitter` (brillo, contraste, saturación, tono) y `GaussianBlur` ocasional, píxel por píxel. |
| 8 | `ToTensor` y `Normalize` | Convierte de uint8 HWC a float32 CHW en [0, 1] y luego resta la media y divide entre la desviación de ImageNet. |
| 9 | Borrado aleatorio | Con probabilidad 0.10 pone en ruido un rectángulo de la imagen. |
| 10 | Armado del lote | El `DataLoader` apila 64 tensores de 3 × 224 × 224 en uno solo (`collate`). |
| 11 | Propagación hacia adelante | 13 bloques convolucionales de MobileNetV3 (convoluciones 1×1, *depthwise* 3×3/5×5, *squeeze-and-excitation*, *hardswish*) y la cabeza lineal. ~0.11 GFLOP por imagen. |
| 12 | Pérdida | Softmax + entropía cruzada sobre 29 clases. |
| 13 | Retropropagación | Gradientes de todas las capas descongeladas (1.27 M de parámetros en la fase principal). Aproximadamente el doble del costo del forward en esas capas. |
| 14 | Recorte de gradiente y AdamW | Norma global de los gradientes y actualización de cada parámetro con sus dos momentos. |
| 15 | Evaluación en validación | Repite 3–8 y 11 en modo inferencia sobre 11,600 imágenes al final de cada época y calcula las métricas. |
| 16 | Selección y guardado | Compara el macro F1 con el mejor anterior, guarda checkpoint, ajusta la tasa de aprendizaje y decide si detenerse. |

## 1.4 Operaciones repetitivas

El detalle importante es que el Lab 3 no guarda las imágenes ya
decodificadas: **cada época vuelve a abrir y decodificar todos los JPEG**.
Contando las épocas que realmente se ejecutaron en la búsqueda (33 de
entrenamiento y 41 pasadas de validación, incluida la evaluación inicial de
cada corrida):

| Operación repetitiva | Datos que procesa | Cantidad de repeticiones |
|------------|------------|------------|
| Lectura + decodificación JPEG (3–5) | 1 archivo de ~15 KB → 120,000 bytes (200 × 200 × 3) | 33 × 43,500 + 41 × 11,600 = **1,911,100** decodificaciones para solo 55,100 archivos distintos (~35 veces cada uno) |
| Transformaciones geométricas y de color (6–7) | 1 imagen PIL, 40,000 píxeles de entrada → 50,176 de salida | 1,435,500 en entrenamiento + 475,600 en evaluación |
| `ToTensor` + `Normalize` (8) | 150,528 valores float32 por imagen | 1,911,100 |
| Armado de lote (10) | 64 tensores → 1 tensor de 38.5 MB | ~29,900 lotes |
| Forward (11) | lote 64 × 3 × 224 × 224 | ~29,900 lotes (~1.91 M imágenes × 0.11 GFLOP ~212 TFLOP) |
| Backward + AdamW (13–14) | 1.27 M gradientes y parámetros por paso | 33 épocas × 680 pasos ~**22,400** actualizaciones |
| Dentro de cada convolución | multiplicaciones-suma por posición de salida, canal y filtro | ~56 M de multiplicaciones-suma por imagen en el forward |

## 1.5 Lo que esto nos dice

- Hay dos tipos de trabajo muy distintos. Las operaciones 3–5 tocan el disco
  y dependen de librerías de decodificación; las operaciones 6–14 son
  aritmética repetida sobre arreglos grandes y regulares.
- La decodificación es trabajo **redundante**: el resultado es el mismo en
  todas las épocas. Hacerla una sola vez elimina ~97 % de esas operaciones
  antes de pensar en paralelizar nada.
- Las transformaciones de la operación 6–9 se aplican **una imagen a la vez**
  en Python/PIL, aunque son independientes por píxel.
- El forward y el backward (11–14) dominan el tiempo total y son
  multiplicaciones de matrices y convoluciones: exactamente el tipo de cálculo
  para el que existe la GPU.

Estas observaciones son la base de las secciones 3 y 4.
