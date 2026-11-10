
from pathlib import Path
import random

PROJECT_DIR = Path(__file__).resolve().parent.parent
LABEL_FILE = PROJECT_DIR / "data" / "raw" / "det_train_labels.txt"
OUTPUT_DIR = PROJECT_DIR / "data" / "raw"

SEED = 42
TRAIN_COUNT = 800
VALIDATION_COUNT = 200

lines = LABEL_FILE.read_text(encoding="utf-8").splitlines()

if len(lines) != 1000:
    raise ValueError(
        f"Expected 1000 labelled images, but found {len(lines)}. "
        "Check det_train_labels.txt before splitting."
    )

random.Random(SEED).shuffle(lines)

train_lines = lines[:TRAIN_COUNT]
validation_lines = lines[TRAIN_COUNT:TRAIN_COUNT + VALIDATION_COUNT]

(OUTPUT_DIR / "det_train_800_labels.txt").write_text(
    "\n".join(train_lines) + "\n", encoding="utf-8"
)
(OUTPUT_DIR / "det_val_200_labels.txt").write_text(
    "\n".join(validation_lines) + "\n", encoding="utf-8"
)

train_paths = {line.split("\t", 1)[0] for line in train_lines}
val_paths = {line.split("\t", 1)[0] for line in validation_lines}

print(f"Training images: {len(train_lines)}")
print(f"Validation images: {len(validation_lines)}")
print(f"Overlap between splits: {len(train_paths & val_paths)}")
print(f"Random seed: {SEED}")
print("Training labels:", OUTPUT_DIR / "det_train_800_labels.txt")
print("Validation labels:", OUTPUT_DIR / "det_val_200_labels.txt")
