"""Transformación de entrada de la red: remuestreo geométrico + color + normalización.

Cada imagen del lote se describe con unos pocos números generados en el host:

* una homografía 3x3 que lleva cada píxel de salida a una posición (sub-píxel)
  de la imagen fuente. Recorte aleatorio, rotación, zoom, traslación y
  perspectiva se componen en esa única matriz, así que el costo por píxel es el
  mismo sin importar cuántas transformaciones geométricas se apliquen;
* tres factores de color (brillo, contraste, saturación);
* un rectángulo de borrado (random erasing), vacío si no aplica.

Con eso, cada píxel de salida se calcula de forma independiente: es el caso
ideal para un hilo CUDA por píxel. La función `_pixel` contiene la matemática y
se compila dos veces: como función de dispositivo CUDA y como función nopython
para CPU. Así la versión GPU y la versión CPU paralela son la misma operación.
`warp_numpy` es una implementación vectorizada independiente que sirve como
referencia en las pruebas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import numpy as np
from numba import njit, prange

from .config import AugmentConfig

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Constantes float32: en CUDA un literal como 0.5 es float64 y arrastra toda la
# expresión a doble precisión, que en GPUs de consumo corre a 1/32 o 1/64 de velocidad.
F0 = np.float32(0.0)
F1 = np.float32(1.0)
HALF = np.float32(0.5)
INV255 = np.float32(1.0 / 255.0)
LUMA_R = np.float32(0.299)
LUMA_G = np.float32(0.587)
LUMA_B = np.float32(0.114)


@dataclass
class AugmentParams:
    homography: np.ndarray  # (B, 9) float32, salida -> fuente
    color: np.ndarray       # (B, 3) float32: brillo, contraste, saturación
    erase: np.ndarray       # (B, 4) int32: x0, y0, x1, y1 en coordenadas de salida

    def __len__(self) -> int:
        return len(self.homography)

    def packed(self) -> np.ndarray:
        """Los tres arreglos en un solo bloque float32 (B, 16) para una sola copia H2D."""
        out = np.zeros((len(self), 16), dtype=np.float32)
        out[:, :9] = self.homography
        out[:, 9:12] = self.color
        out[:, 12:16] = self.erase
        return out


def _pixel(src, b, x, y, params, mean, std, out):
    # params[b] = [h0..h8, brillo, contraste, saturación, ex0, ey0, ex1, ey1]
    fx = np.float32(x) + HALF
    fy = np.float32(y) + HALF
    w = params[b, 6] * fx + params[b, 7] * fy + params[b, 8]
    sx = (params[b, 0] * fx + params[b, 1] * fy + params[b, 2]) / w - HALF
    sy = (params[b, 3] * fx + params[b, 4] * fy + params[b, 5]) / w - HALF

    hs = src.shape[1]
    ws = src.shape[2]
    x0 = int(math.floor(sx))
    y0 = int(math.floor(sy))
    ax = sx - np.float32(x0)
    ay = sy - np.float32(y0)
    xa = min(max(x0, 0), ws - 1)
    xb = min(max(x0 + 1, 0), ws - 1)
    ya = min(max(y0, 0), hs - 1)
    yb = min(max(y0 + 1, 0), hs - 1)
    w00 = (F1 - ax) * (F1 - ay)
    w01 = ax * (F1 - ay)
    w10 = (F1 - ax) * ay
    w11 = ax * ay

    r = (w00 * np.float32(src[b, ya, xa, 0]) + w01 * np.float32(src[b, ya, xb, 0])
         + w10 * np.float32(src[b, yb, xa, 0]) + w11 * np.float32(src[b, yb, xb, 0])) * INV255
    g = (w00 * np.float32(src[b, ya, xa, 1]) + w01 * np.float32(src[b, ya, xb, 1])
         + w10 * np.float32(src[b, yb, xa, 1]) + w11 * np.float32(src[b, yb, xb, 1])) * INV255
    bl = (w00 * np.float32(src[b, ya, xa, 2]) + w01 * np.float32(src[b, ya, xb, 2])
          + w10 * np.float32(src[b, yb, xa, 2]) + w11 * np.float32(src[b, yb, xb, 2])) * INV255

    brightness = params[b, 9]
    contrast = params[b, 10]
    saturation = params[b, 11]
    r = (r * brightness - HALF) * contrast + HALF
    g = (g * brightness - HALF) * contrast + HALF
    bl = (bl * brightness - HALF) * contrast + HALF
    gray = LUMA_R * r + LUMA_G * g + LUMA_B * bl
    r = min(max(gray + (r - gray) * saturation, F0), F1)
    g = min(max(gray + (g - gray) * saturation, F0), F1)
    bl = min(max(gray + (bl - gray) * saturation, F0), F1)

    fxi = np.float32(x)
    fyi = np.float32(y)
    if params[b, 12] <= fxi and fxi < params[b, 14] and params[b, 13] <= fyi and fyi < params[b, 15]:
        out[b, 0, y, x] = F0
        out[b, 1, y, x] = F0
        out[b, 2, y, x] = F0
    else:
        out[b, 0, y, x] = (r - mean[0]) / std[0]
        out[b, 1, y, x] = (g - mean[1]) / std[1]
        out[b, 2, y, x] = (bl - mean[2]) / std[2]


# ---------------------------------------------------------------- backends

_pixel_cpu = njit(fastmath=True)(_pixel)


cuda = None          # numba.cuda, se asigna en cuda_kernel()
_pixel_dev = None    # _pixel compilada como función de dispositivo


def _warp_kernel(src, params, mean, std, out):
    """Un hilo por píxel de salida. Grid 3D: (columnas, filas, imagen del lote)."""
    x, y, b = cuda.grid(3)
    if b < out.shape[0] and y < out.shape[2] and x < out.shape[3]:
        _pixel_dev(src, b, x, y, params, mean, std, out)


@cache
def cuda_kernel():
    """Compila (una sola vez) el kernel CUDA.

    numba.cuda se importa aquí y no al inicio del módulo para que el camino CPU
    funcione en equipos sin GPU NVIDIA.
    """
    global cuda, _pixel_dev
    from numba import cuda as numba_cuda

    cuda = numba_cuda
    _pixel_dev = numba_cuda.jit(device=True)(_pixel)
    return numba_cuda.jit(fastmath=True)(_warp_kernel)


@njit(parallel=True, fastmath=True)
def _warp_cpu(src, params, mean, std, out):
    batch = out.shape[0]
    height = out.shape[2]
    width = out.shape[3]
    for row in prange(batch * height):
        b = row // height
        y = row - b * height
        for x in range(width):
            _pixel_cpu(src, b, x, y, params, mean, std, out)


def warp_cpu(src: np.ndarray, params: np.ndarray, out: np.ndarray,
             mean: np.ndarray = IMAGENET_MEAN, std: np.ndarray = IMAGENET_STD) -> np.ndarray:
    """Versión CPU multinúcleo (Numba prange). `out` puede ser una vista de un tensor de torch."""
    _warp_cpu(src, params, mean, std, out)
    return out


def grid_for(batch: int, height: int, width: int, block: tuple[int, int]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    bx, by = block
    return (math.ceil(width / bx), math.ceil(height / by), batch), (bx, by, 1)


def launch_warp(d_src, d_params, d_mean, d_std, d_out, block: tuple[int, int] = (16, 16), stream=0) -> None:
    grid, threads = grid_for(d_out.shape[0], d_out.shape[2], d_out.shape[3], block)
    cuda_kernel()[grid, threads, stream](d_src, d_params, d_mean, d_std, d_out)


def warp_numpy(src: np.ndarray, params: np.ndarray, out_size: int,
               mean: np.ndarray = IMAGENET_MEAN, std: np.ndarray = IMAGENET_STD) -> np.ndarray:
    """Referencia vectorizada con NumPy (sin Numba). Devuelve (B, 3, S, S) float32."""
    batch, hs, ws, _ = src.shape
    ys, xs = np.mgrid[0:out_size, 0:out_size].astype(np.float32)
    fx = xs[None] + HALF
    fy = ys[None] + HALF
    p = params.astype(np.float32)
    h = [p[:, i, None, None] for i in range(16)]
    w = h[6] * fx + h[7] * fy + h[8]
    sx = (h[0] * fx + h[1] * fy + h[2]) / w - HALF
    sy = (h[3] * fx + h[4] * fy + h[5]) / w - HALF
    x0 = np.floor(sx)
    y0 = np.floor(sy)
    ax = (sx - x0)[..., None]
    ay = (sy - y0)[..., None]
    x0 = x0.astype(np.int64)
    y0 = y0.astype(np.int64)
    xa, xb = np.clip(x0, 0, ws - 1), np.clip(x0 + 1, 0, ws - 1)
    ya, yb = np.clip(y0, 0, hs - 1), np.clip(y0 + 1, 0, hs - 1)
    bi = np.arange(batch)[:, None, None]
    s = src.astype(np.float32)
    rgb = ((1 - ax) * (1 - ay) * s[bi, ya, xa] + ax * (1 - ay) * s[bi, ya, xb]
           + (1 - ax) * ay * s[bi, yb, xa] + ax * ay * s[bi, yb, xb]) * INV255
    bright, contrast, sat = (p[:, 9, None, None, None], p[:, 10, None, None, None], p[:, 11, None, None, None])
    rgb = (rgb * bright - HALF) * contrast + HALF
    gray = (rgb * np.array([LUMA_R, LUMA_G, LUMA_B], dtype=np.float32)).sum(-1, keepdims=True)
    rgb = np.clip(gray + (rgb - gray) * sat, 0, 1)
    rgb = (rgb - mean) / std
    erased = (h[12] <= xs) & (xs < h[14]) & (h[13] <= ys) & (ys < h[15])
    rgb[erased] = 0.0
    return np.ascontiguousarray(rgb.transpose(0, 3, 1, 2), dtype=np.float32)


# ------------------------------------------------- parámetros por imagen (host)

def _uniform(rng: np.random.Generator, low: float, high: float, n: int) -> np.ndarray:
    return rng.uniform(low, high, n).astype(np.float64)


def sample_params(rng: np.random.Generator | None, batch: int, src_size: int, out_size: int,
                  cfg: AugmentConfig, train: bool) -> AugmentParams:
    """Genera la homografía y el color de cada imagen.

    Composición (de derecha a izquierda, aplicada al centro de cada píxel de salida):
        H = S · A · P · N
    N: píxel de salida -> coordenadas normalizadas [-1, 1]
    P: perspectiva leve (solo entrenamiento, con probabilidad perspective_p)
    A: rotación y zoom
    S: coordenadas normalizadas -> píxel de la fuente (recorte aleatorio o central)
    """
    n = batch
    norm = np.array([[2.0 / out_size, 0, -1], [0, 2.0 / out_size, -1], [0, 0, 1]])
    persp = np.repeat(np.eye(3)[None], n, axis=0)
    rot_zoom = np.repeat(np.eye(3)[None], n, axis=0)
    crop = np.repeat(np.eye(3)[None], n, axis=0)
    color = np.ones((n, 3), dtype=np.float32)
    erase = np.zeros((n, 4), dtype=np.int32)

    if not train or rng is None:
        half = cfg.eval_crop * src_size / 2
        crop[:, 0, 0] = crop[:, 1, 1] = half
        crop[:, 0, 2] = crop[:, 1, 2] = src_size / 2
    else:
        area = _uniform(rng, *cfg.scale, n)
        log_ratio = _uniform(rng, math.log(cfg.ratio[0]), math.log(cfg.ratio[1]), n)
        ratio = np.exp(log_ratio)
        half_w = np.minimum(np.sqrt(area * ratio), 1.0) * src_size / 2
        half_h = np.minimum(np.sqrt(area / ratio), 1.0) * src_size / 2
        cx = half_w + rng.uniform(0, 1, n) * (src_size - 2 * half_w)
        cy = half_h + rng.uniform(0, 1, n) * (src_size - 2 * half_h)
        cx += _uniform(rng, -cfg.translate, cfg.translate, n) * src_size
        cy += _uniform(rng, -cfg.translate, cfg.translate, n) * src_size
        crop[:, 0, 0], crop[:, 1, 1], crop[:, 0, 2], crop[:, 1, 2] = half_w, half_h, cx, cy

        theta = np.deg2rad(_uniform(rng, -cfg.rotation_deg, cfg.rotation_deg, n))
        zoom = 1.0 / _uniform(rng, *cfg.zoom, n)  # zoom > 1 en la imagen = ventana fuente más pequeña
        cos, sin = np.cos(theta) * zoom, np.sin(theta) * zoom
        rot_zoom[:, 0, 0], rot_zoom[:, 0, 1], rot_zoom[:, 1, 0], rot_zoom[:, 1, 1] = cos, -sin, sin, cos

        apply = rng.uniform(0, 1, n) < cfg.perspective_p
        persp[:, 2, 0] = np.where(apply, _uniform(rng, -cfg.perspective, cfg.perspective, n), 0.0)
        persp[:, 2, 1] = np.where(apply, _uniform(rng, -cfg.perspective, cfg.perspective, n), 0.0)

        color[:, 0] = _uniform(rng, 1 - cfg.brightness, 1 + cfg.brightness, n)
        color[:, 1] = _uniform(rng, 1 - cfg.contrast, 1 + cfg.contrast, n)
        color[:, 2] = _uniform(rng, 1 - cfg.saturation, 1 + cfg.saturation, n)

        do_erase = rng.uniform(0, 1, n) < cfg.erase_p
        e_area = _uniform(rng, *cfg.erase_area, n) * out_size * out_size
        e_ratio = np.exp(_uniform(rng, math.log(0.5), math.log(2.0), n))
        e_h = np.clip(np.sqrt(e_area * e_ratio), 1, out_size).astype(np.int32)
        e_w = np.clip(np.sqrt(e_area / e_ratio), 1, out_size).astype(np.int32)
        e_x = (rng.uniform(0, 1, n) * (out_size - e_w + 1)).astype(np.int32)
        e_y = (rng.uniform(0, 1, n) * (out_size - e_h + 1)).astype(np.int32)
        erase[do_erase] = np.stack([e_x, e_y, e_x + e_w, e_y + e_h], axis=1)[do_erase]

    hom = crop @ rot_zoom @ persp @ norm
    hom /= hom[:, 2:3, 2:3]
    return AugmentParams(hom.reshape(n, 9).astype(np.float32), color, erase)
