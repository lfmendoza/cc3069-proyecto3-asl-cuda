"""Reconocimiento en vivo con la cámara.

La mano se coloca dentro del recuadro. Cada cuadro se recorta, se lleva a
200x200 (la misma resolución de la caché) y pasa por el mismo kernel de
entrada que en evaluación. Para no "parpadear" entre letras se usa voto
mayoritario sobre los últimos cuadros; si una letra se mantiene estable se
agrega al texto (SPACE agrega espacio, DELETE borra, NOTHING no hace nada).

Teclas: q = salir, c = limpiar texto.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

from .config import load_train_config
from .constants import CLASSES
from .model import load_checkpoint
from .prep import CachedSplit
from .train import make_feeder, predict, resolve_device


def run_demo(checkpoint: Path, device_name: str, camera: int, config: str, window: int = 9, hold: int = 12) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Falta OpenCV: uv sync --extra <cpu|cu126|cu130> --extra demo") from exc

    cfg = load_train_config(config)
    device = resolve_device(device_name)
    model, meta = load_checkpoint(checkpoint, device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    size = int(meta.get("cache_size", 200))
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {camera}")

    buffer = np.zeros((1, size, size, 3), dtype=np.uint8)
    feeder = make_feeder(CachedSplit(buffer, np.zeros(1, dtype=np.int64), {}), cfg, device, train=False, batch_size=1)
    recent: deque[int] = deque(maxlen=window)
    text, stable, last_added = "", 0, None
    times: deque[float] = deque(maxlen=30)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        side = int(min(h, w) * 0.6)
        x0, y0 = w - side - 20, (h - side) // 2
        roi = cv2.cvtColor(frame[y0:y0 + side, x0:x0 + side], cv2.COLOR_BGR2RGB)
        buffer[0] = cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)

        started = time.perf_counter()
        _, probs, _ = predict(model, feeder, device, cfg.cuda.amp)
        times.append(time.perf_counter() - started)

        best = int(probs[0].argmax())
        recent.append(best)
        voted, votes = Counter(recent).most_common(1)[0]
        stable = stable + 1 if voted == best and votes >= window // 2 + 1 else 0
        label = CLASSES[voted]
        if stable >= hold and voted != last_added:
            if label == "SPACE":
                text += " "
            elif label == "DELETE":
                text = text[:-1]
            elif label != "NOTHING":
                text += label
            last_added, stable = voted, 0
        if label == "NOTHING":
            last_added = None

        ms = 1000 * float(np.mean(times))
        cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), (0, 200, 120), 2)
        cv2.putText(frame, f"{label}  {probs[0, voted]:.2f}", (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 120), 2)
        cv2.putText(frame, f"{device.type.upper()}  {ms:.1f} ms/cuadro", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, text[-40:], (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("ASL - CC3069", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            text = ""
    capture.release()
    cv2.destroyAllWindows()


def nombre_foto(clase: str) -> str:
    """A_1.jpg ... Z_1.jpg, del_1.jpg, nothing_1.jpg, space_1.jpg (lo que lee evaluate)."""
    return f"{clase.lower() if len(clase) > 1 else clase}_1.jpg".replace("delete_", "del_")


def capturar_fotos(destino: Path, camera: int) -> None:
    """Toma una foto por clase con el mismo recuadro que la demo.

    Se guarda solo el recuadro (como en la demo, la mano ocupa el centro).
    Teclas: espacio = guardar y pasar a la siguiente, b = volver a la
    anterior, s = saltar, q = salir. Las clases que ya tienen foto se saltan,
    así que se puede cortar y retomar.
    """
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Falta OpenCV: uv sync --extra <cpu|cu126|cu130> --extra demo") from exc

    destino.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {camera}")
    i = 0
    while i < len(CLASSES):
        archivo = destino / nombre_foto(CLASSES[i])
        if archivo.exists():
            i += 1
            continue
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        side = int(min(h, w) * 0.6)
        x0, y0 = w - side - 20, (h - side) // 2
        roi = frame[y0:y0 + side, x0:x0 + side].copy()
        cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), (0, 200, 120), 2)
        cv2.putText(frame, f"Seña: {CLASSES[i]}  ({i + 1}/{len(CLASSES)})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 200, 120), 2)
        cv2.putText(frame, "espacio = guardar   b = anterior   s = saltar   q = salir", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Fotos del equipo - CC3069", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            cv2.imwrite(str(archivo), roi, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"Guardada {archivo.name}")
            i += 1
        elif key == ord("s"):
            i += 1
        elif key == ord("b") and i > 0:
            i -= 1
            (destino / nombre_foto(CLASSES[i])).unlink(missing_ok=True)
    capture.release()
    cv2.destroyAllWindows()
    faltan = [c for c in CLASSES if not (destino / nombre_foto(c)).exists()]
    print(f"{len(CLASSES) - len(faltan)}/{len(CLASSES)} fotos en {destino}" + (f"; faltan {faltan}" if faltan else ""))
