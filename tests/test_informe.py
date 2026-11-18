"""Resumen por fase y copia de tablas al informe."""

import json

import pandas as pd

from asl_cuda import informe, summarize


def _bench(tmp_path):
    rows = {
        "augment": [("torchvision-PIL (Lab 3)", "cpu", 1, 2.0), ("Numba CPU 8 hilos", "cpu", 8, 0.5),
                    ("CUDA kernel 16x16", "cuda", 256, 0.02)],
        "train": [("CPU 1 hilo (secuencial)", "cpu", 1, 20.0), ("CPU 8 hilos", "cpu", 8, 10.0),
                  ("GPU AMP fp16 + channels_last (propuesta)", "cuda", 8, 0.5)],
    }
    for phase, variants in rows.items():
        frame = pd.DataFrame([{"phase": phase, "variant": v, "repeat": r, "device": d, "threads": t, "images": 1000,
                               "seconds": s} for v, d, t, s in variants for r in (1, 2)])
        frame.to_csv(tmp_path / f"bench_{phase}.csv", index=False)


def test_phase_summary_speedups(tmp_path):
    _bench(tmp_path)
    md = summarize.phase_summary(tmp_path, tmp_path)
    lines = md.splitlines()
    augment = next(line for line in lines if line.startswith("| Transformación"))
    train = next(line for line in lines if line.startswith("| Paso de entrenamiento"))
    assert "Numba CPU 8 hilos" in augment and "| 4.000 |" in augment and "| 100.0 |" in augment
    assert "| 2.000 |" in train and "| 40.0 |" in train
    assert (tmp_path / "speedup_fases.png").exists()


def test_training_and_evaluation_tables(tmp_path):
    run = tmp_path / "train" / "base-cuda"
    run.mkdir(parents=True)
    pd.DataFrame({"train_seconds": [10.0, 10.0], "train_images_per_second": [100.0, 120.0],
                  "data_wait_seconds": [0.5, 0.5], "gpu_peak_memory_mb": [900.0, 950.0]}).to_csv(run / "history.csv", index=False)
    (run / "summary.json").write_text(json.dumps({"device": "cuda:0", "amp": True, "augment_device": "gpu", "epochs_run": 2,
                                                  "total_seconds": 60.0, "validation": {"macro_f1": 0.9}}))
    md = summarize.training_table(tmp_path / "train")
    assert "GPU, AMP, aumentación en GPU" in md and "| 5.000 |" in md and "| 950.0 |" in md

    ev = tmp_path / "eval"
    ev.mkdir()
    (ev / "test_metrics.json").write_text(json.dumps({"name": "test", "images": 10, "accuracy": 0.9, "macro_f1": 0.8,
                                                      "top3_accuracy": 1.0}))
    assert "| test | 10 | 0.900 | 0.800 | 1.000 |" in summarize.evaluation_table(ev)


def test_anchos_respeta_contenido_y_alineacion():
    texto = "| Variante | Tiempo |\n|---|---:|\n| Lab 3 secuencial (0 workers) | 1.5 |\n\nNota al pie.\n"
    salida = informe._anchos(texto)
    separador = salida.splitlines()[1]
    assert separador.startswith("|" + "-" * len("Lab 3 secuencial (0 workers)") + "|")
    assert separador.endswith(":|")
    assert salida.splitlines()[2] == "| Lab 3 secuencial (0 workers) | 1.5 |"
    assert "Nota al pie." in salida


def test_preparar_numera_las_computadoras_de_referencia(tmp_path, monkeypatch):
    root = tmp_path
    for tag in ("pc-a", "pc-b"):
        (root / "results" / tag / "tablas").mkdir(parents=True)
        (root / "results" / tag / "tablas" / "prep.md").write_text(f"| Variante |\n|---|\n| {tag} |\n")
    monkeypatch.setattr(informe, "ROOT", root)
    monkeypatch.setattr(informe, "INFORME", root / "informe")
    faltan = informe.preparar("gpu-x", ["pc-a", "pc-b"])
    assert "pc-a" in (root / "informe" / "tablas" / "cpu2_prep.md").read_text()
    assert "pc-b" in (root / "informe" / "tablas" / "cpu3_prep.md").read_text()
    assert "Tabla pendiente" in (root / "informe" / "tablas" / "train.md").read_text()
    assert any(f.startswith("gpu-x/") for f in faltan)


def test_nombres_de_fotos_se_leen_con_su_clase(tmp_path):
    from asl_cuda.constants import CLASSES
    from asl_cuda.demo import nombre_foto
    from asl_cuda.manifest import label_for_file

    for clase in CLASSES:
        assert label_for_file(tmp_path / "hansel" / nombre_foto(clase)) == clase
