from pathlib import Path

SEED = 19644
CLASSES: tuple[str, ...] = tuple([chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["DELETE", "NOTHING", "SPACE"])
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}
FOLDER_ALIASES = {"del": "DELETE", "nothing": "NOTHING", "space": "SPACE"}

TRAIN_DATASET = "grassknoted/asl-alphabet"
EXTERNAL_DATASET = "danrasband/asl-alphabet-test"

ROOT = Path(__file__).resolve().parents[2]
