import os

# Por defecto las pruebas del kernel CUDA corren en el simulador de Numba y no
# requieren GPU. En la laptop con NVIDIA se corre con ASL_REAL_GPU=1 para usar
# el hardware real (y habilitar tests/test_gpu.py).
# Debe definirse antes de que cualquier módulo importe numba.cuda.
if not os.environ.get("ASL_REAL_GPU"):
    os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")
