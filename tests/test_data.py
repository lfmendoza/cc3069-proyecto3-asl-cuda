from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from asl_cuda.config import load_data_config
from asl_cuda.constants import CLASSES
from asl_cuda.manifest import label_for_file, scan_any, subsample, validate_manifest
from asl_cuda.prep import decode_to_npy


def _write_images(folder: Path, n: int, size=(200, 200)) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        path = folder / f"img{i}.jpg"
        Image.fromarray(rng.integers(0, 256, (*size[::-1], 3), dtype=np.uint8)).save(path, quality=90)
        paths.append(str(path))
    return paths


def test_split_windows_do_not_overlap():
    cfg = load_data_config()
    sets = [set(cfg.train_windows), set(cfg.validation_windows), set(cfg.test_windows)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    # entre particiones distintas siempre queda al menos una ventana de separación
    owner = {w: i for i, s in enumerate(sets) for w in s}
    for w, i in owner.items():
        assert owner.get(w + 1, i) == i, f"ventanas {w} y {w + 1} pertenecen a particiones distintas sin separador"


def test_sequential_and_parallel_decode_are_identical(tmp_path):
    paths = _write_images(tmp_path / "imgs", 24)
    seq, _ = decode_to_npy(paths, tmp_path / "seq.npy", 200, workers=1)
    par, _ = decode_to_npy(paths, tmp_path / "par.npy", 200, workers=3, chunk=5)
    assert seq.images == par.images == 24
    np.testing.assert_array_equal(np.load(tmp_path / "seq.npy"), np.load(tmp_path / "par.npy"))


def test_decode_resizes_non_square_images(tmp_path):
    paths = _write_images(tmp_path / "odd", 2, size=(320, 240))
    decode_to_npy(paths, tmp_path / "odd.npy", 200, workers=1)
    assert np.load(tmp_path / "odd.npy").shape == (2, 200, 200, 3)


def test_labels_from_folder_or_prefix(tmp_path):
    assert label_for_file(tmp_path / "del" / "x.jpg") == "DELETE"
    assert label_for_file(tmp_path / "hansel" / "B_2.jpg") == "B"
    assert label_for_file(tmp_path / "otros" / "foto.jpg") is None


def test_scan_any_finds_member_photos(tmp_path):
    _write_images(tmp_path / "A", 2)
    member = tmp_path / "adrian"
    member.mkdir()
    Image.new("RGB", (50, 50)).save(member / "C_1.jpg")
    frame = scan_any(tmp_path)
    assert sorted(frame["class"]) == ["A", "A", "C"]


def test_validate_manifest_rejects_split_window():
    frame = pd.DataFrame({"path": ["a", "b"], "class": ["A", "A"], "window": [3, 3], "split": ["train", "test"]})
    with pytest.raises(ValueError):
        validate_manifest(frame)


def test_subsample_is_balanced_and_deterministic():
    rows = [{"label": i % 3, "index": i, "path": str(i)} for i in range(300)]
    frame = pd.DataFrame(rows)
    a, b = subsample(frame, 10), subsample(frame, 10)
    assert a.equals(b) and a["label"].value_counts().tolist() == [10, 10, 10]


def test_class_list():
    assert len(CLASSES) == 29 and CLASSES[-3:] == ("DELETE", "NOTHING", "SPACE")
