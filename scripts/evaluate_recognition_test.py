
"""
EXP-REC-ICDAR15-001
Evaluate exported PP-OCRv6-Medium recognition model on ICDAR2015 test crops.

Run from the PaddleOCR project directory:
    python ..\scripts\evaluate_recognition_test.py
"""

import csv
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------
ROOT = Path.cwd().resolve()  # Run from the PaddleOCR directory
PROJECT_ROOT = ROOT.parent

MODEL_DIR = PROJECT_ROOT / "models" / "rec_inference" / "PP-OCRv6_medium_rec_icdar15_best"
DICT_PATH = ROOT / "ppocr" / "utils" / "dict" / "ppocrv6_dict.txt"
TEST_IMAGE_DIR = PROJECT_ROOT / "recognition_data" / "test_crops"
TEST_LABEL_FILE = PROJECT_ROOT / "recognition_data" / "test_list.txt"
OUTPUT_DIR = PROJECT_ROOT / "results" / "recognition"
CONFIG_PATH = PROJECT_ROOT / "configs" / "PP-OCRv6_medium_rec_icdar2015.yml"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "EXP-REC-ICDAR15-001_test_predictions.csv"
SUMMARY_PATH = OUTPUT_DIR / "EXP-REC-ICDAR15-001_test_metrics.txt"
CONFUSION_PATH = OUTPUT_DIR / "EXP-REC-ICDAR15-001_character_confusions.csv"


# ------------------------------------------------------------
# 2. Text metrics
# ------------------------------------------------------------
def levenshtein_details(reference, hypothesis):
    """Return edit distance, substitutions, deletions, insertions and alignment."""
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "D"

    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            choices = [
                (dp[i - 1][j - 1] + cost, "M" if cost == 0 else "S"),
                (dp[i - 1][j] + 1, "D"),
                (dp[i][j - 1] + 1, "I"),
            ]
            dp[i][j], back[i][j] = min(choices, key=lambda x: x[0])

    substitutions = deletions = insertions = 0
    aligned_pairs = []
    i, j = n, m

    while i > 0 or j > 0:
        op = back[i][j]

        if op in ("M", "S"):
            a, b = reference[i - 1], hypothesis[j - 1]
            aligned_pairs.append((a, b))
            if op == "S":
                substitutions += 1
            i -= 1
            j -= 1
        elif op == "D":
            aligned_pairs.append((reference[i - 1], None))
            deletions += 1
            i -= 1
        elif op == "I":
            aligned_pairs.append((None, hypothesis[j - 1]))
            insertions += 1
            j -= 1

    aligned_pairs.reverse()
    return dp[n][m], substitutions, deletions, insertions, aligned_pairs


def safe_div(a, b):
    return a / b if b else 0.0


def load_labels(label_path):
    """Read each line as image filename followed by ground-truth text."""
    records = []

    with open(label_path, "r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Cannot parse label line {line_number}: {line!r}"
                )

            records.append((parts[0], parts[1]))

    return records


# ------------------------------------------------------------
# 3. Check inputs
# ------------------------------------------------------------
for path in (MODEL_DIR, DICT_PATH, TEST_IMAGE_DIR, TEST_LABEL_FILE):
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")

if not CONFIG_PATH.is_file():
    print(f"WARNING: configuration file not found: {CONFIG_PATH}")

records = load_labels(TEST_LABEL_FILE)
if not records:
    raise ValueError("No labels were found in test_list.txt")

print(f"Test label records: {len(records)}")
print(f"Model directory: {MODEL_DIR}")
print(f"Output CSV: {CSV_PATH}")


# ------------------------------------------------------------
# 4. Load exported PaddleOCR recognition model
# ------------------------------------------------------------
sys.path.insert(0, str(ROOT))

from tools.infer.predict_rec import TextRecognizer, utility

sys.argv = [
    "predict_rec.py",
    "--image_dir", str(TEST_IMAGE_DIR),
    "--rec_model_dir", str(MODEL_DIR),
    "--rec_char_dict_path", str(DICT_PATH),
    "--rec_algorithm", "SVTR_LCNet",
    "--use_space_char", "True",
    "--rec_image_shape", "3,48,320",
    "--use_gpu", "True",
    "--gpu_id", "0",
    "--rec_batch_num", "1",
    "--show_log", "False",
]

args = utility.parse_args()
recognizer = TextRecognizer(args)

# Warm-up runs are excluded from timing.
dummy = np.zeros((48, 320, 3), dtype=np.uint8)
for _ in range(3):
    recognizer([dummy])


# ------------------------------------------------------------
# 5. GPU timing helper
# ------------------------------------------------------------
try:
    import paddle
    use_cuda = paddle.is_compiled_with_cuda()
except Exception:
    paddle = None
    use_cuda = False


def synchronize_gpu():
    if use_cuda:
        paddle.device.synchronize()


# ------------------------------------------------------------
# 6. Run inference and calculate per-image metrics
# ------------------------------------------------------------
rows = []
all_times = []
total_chars = 0
correct_chars = 0
total_edit_distance = 0
total_substitutions = 0
total_deletions = 0
total_insertions = 0
exact_matches = 0

confusions = Counter()
class_counts = Counter()
class_correct = Counter()

missing_images = 0
unreadable_images = 0
empty_predictions = 0

for index, (filename, ground_truth) in enumerate(records, 1):
    image_path = TEST_IMAGE_DIR / filename

    if not image_path.is_file():
        print(f"WARNING: missing image, skipping: {image_path}")
        missing_images += 1
        continue

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"WARNING: unreadable image, skipping: {image_path}")
        unreadable_images += 1
        continue

    synchronize_gpu()
    start = time.perf_counter()
    result = recognizer([image])
    synchronize_gpu()
    elapsed = time.perf_counter() - start

    # TextRecognizer returns:
    # ([('recognised text', confidence)], elapsed_time)
    # Extract the first recognition record from the outer tuple.
    recognition_results = (
        result[0]
        if isinstance(result, tuple) and len(result) >= 1
        else None
    )

    if recognition_results and len(recognition_results[0]) >= 2:
        prediction = str(recognition_results[0][0])
        confidence = float(recognition_results[0][1])
    else:
        prediction = ""
        confidence = 0.0
        empty_predictions += 1

    distance, subs, dels, ins, aligned = levenshtein_details(
        ground_truth, prediction
    )

    exact = ground_truth == prediction
    cer = safe_div(distance, len(ground_truth))

    rows.append({
        "image": filename,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "confidence": confidence,
        "exact_match": str(exact).upper(),
        "edit_distance": distance,
        "cer": cer,
        "time_seconds": elapsed,
        "ground_truth_length": len(ground_truth),
        "prediction_length": len(prediction),
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
    })

    all_times.append(elapsed)
    total_chars += len(ground_truth)
    total_edit_distance += distance
    total_substitutions += subs
    total_deletions += dels
    total_insertions += ins
    exact_matches += int(exact)

    for ref_char, pred_char in aligned:
        if ref_char is not None:
            class_counts[ref_char] += 1
            if ref_char == pred_char:
                correct_chars += 1
                class_correct[ref_char] += 1
            else:
                confusions[(ref_char, pred_char or "<DELETED>")] += 1
        elif pred_char is not None:
            confusions[("<INSERTED>", pred_char)] += 1

    if index % 100 == 0 or index == len(records):
        print(f"Processed {index}/{len(records)} label records")

if not rows:
    raise RuntimeError("No test images were successfully evaluated.")


# ------------------------------------------------------------
# 7. Save per-image prediction CSV
# ------------------------------------------------------------
fieldnames = [
    "image", "ground_truth", "prediction", "confidence", "exact_match",
    "edit_distance", "cer", "time_seconds", "ground_truth_length",
    "prediction_length", "substitutions", "deletions", "insertions",
]

with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


# ------------------------------------------------------------
# 8. Calculate overall metrics
# ------------------------------------------------------------
n = len(rows)
exact_accuracy = safe_div(exact_matches, n)
character_accuracy = safe_div(correct_chars, total_chars)
cer_overall = safe_div(total_edit_distance, total_chars)
string_error_rate = 1.0 - exact_accuracy

mean_time = statistics.mean(all_times)
median_time = statistics.median(all_times)
throughput = safe_div(1.0, mean_time)

# Edit-alignment-based character precision, recall and F1.
tp = correct_chars
fp = total_substitutions + total_insertions
fn = total_substitutions + total_deletions

precision_micro = safe_div(tp, tp + fp)
recall_micro = safe_div(tp, tp + fn)
f1_micro = safe_div(
    2 * precision_micro * recall_micro,
    precision_micro + recall_micro,
)

class_recalls = [
    safe_div(class_correct[c], count)
    for c, count in class_counts.items()
]
macro_character_recall = (
    statistics.mean(class_recalls) if class_recalls else 0.0
)
weighted_character_recall = character_accuracy

parameter_file = MODEL_DIR / "inference.pdiparams"
model_size_bytes = (
    parameter_file.stat().st_size if parameter_file.is_file() else None
)
model_size_text = (
    f"{model_size_bytes / (1024 * 1024):.2f} MiB"
    if model_size_bytes is not None
    else "Not available"
)


# ------------------------------------------------------------
# 9. Save character confusion table
# ------------------------------------------------------------
with open(CONFUSION_PATH, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow([
        "ground_truth_character", "predicted_character", "count"
    ])
    for (actual, predicted), count in confusions.most_common():
        writer.writerow([actual, predicted, count])


# ------------------------------------------------------------
# 10. Write experiment report
# ------------------------------------------------------------
report = f"""EXP-REC-ICDAR15-001 — PP-OCRv6-Medium Recognition Test Evaluation
=====================================================================

DATA AND CHECKPOINT
-------------------
Dataset: ICDAR2015
Task: Recognition on ground-truth-derived text crops
Test label records: {len(records)}
Successfully evaluated images: {n}
Missing images: {missing_images}
Unreadable images: {unreadable_images}
Empty recognizer results: {empty_predictions}
Selected checkpoint: best_accuracy (best validation epoch 96)
Exported model directory: {MODEL_DIR}
Model parameter file size: {model_size_text}

TRAINING SETTINGS
-----------------
Configured epochs: 100
Configured training batch size: 8 (dynamic multi-scale sampler may vary)
Optimizer: Adam
Initial learning rate: 0.0005
Loss: MultiLoss combining CTC and NRTR
Architecture: SVTR_LCNet with PPLCNetV4 Medium backbone
Framework: PaddlePaddle 3.3.0
Device: GPU enabled for inference

TEST RECOGNITION METRICS
------------------------
Exact-string accuracy: {exact_accuracy * 100:.4f}%
Exact strings correct: {exact_matches}/{n}
String error rate: {string_error_rate * 100:.4f}%
Character accuracy: {character_accuracy * 100:.4f}%
Total ground-truth characters: {total_chars}
Total Levenshtein edit distance: {total_edit_distance}
Overall character error rate (CER): {cer_overall * 100:.4f}%
Substitutions: {total_substitutions}
Deletions: {total_deletions}
Insertions: {total_insertions}

CHARACTER METRICS (EDIT-ALIGNMENT BASED)
----------------------------------------
Micro precision: {precision_micro * 100:.4f}%
Micro recall: {recall_micro * 100:.4f}%
Micro F1: {f1_micro * 100:.4f}%
Macro character recall: {macro_character_recall * 100:.4f}%
Weighted character recall: {weighted_character_recall * 100:.4f}%

TIMING (SINGLE-IMAGE INFERENCE)
-------------------------------
Mean time per image: {mean_time:.6f} seconds
Median time per image: {median_time:.6f} seconds
Approximate serial throughput: {throughput:.3f} images/second
Timing includes the recognizer call on an already-loaded image.
Image loading is excluded. Warm-up is excluded. Batch size is one.
These timings are not directly comparable with batched evaluation FPS.

OUTPUT FILES
------------
Per-image predictions: {CSV_PATH}
Character confusion counts: {CONFUSION_PATH}
Summary report: {SUMMARY_PATH}

IMPORTANT LIMITATIONS
---------------------
1. These results measure recognition on ground-truth-derived crops,
   not end-to-end text detection plus recognition.
2. ICDAR2015 is a scene-text dataset, not a VIN-specific dataset.
3. AUC-ROC is not calculated because OCR sequence recognition has no
   single standard ROC definition without a separately defined task.
4. Precision, recall and F1 here are edit-alignment-based character
   metrics, not conventional multiclass classification metrics.
5. Weighted character recall equals character accuracy in this implementation.
6. Detection precision, recall and H-mean belong to the separate detector
   experiment and must be reported separately.
7. The cuDNN 9.9 versus installed 9.5 compatibility warning should be
   investigated before treating timing results as definitive.
"""

SUMMARY_PATH.write_text(report, encoding="utf-8")

print("\nEvaluation complete.")
print(f"Successfully evaluated: {n}/{len(records)}")
print(f"Empty predictions: {empty_predictions}")
print(f"CSV: {CSV_PATH}")
print(f"Summary: {SUMMARY_PATH}")
print(f"Character confusions: {CONFUSION_PATH}")
print(f"Exact-string accuracy: {exact_accuracy * 100:.2f}%")
print(f"Character accuracy: {character_accuracy * 100:.2f}%")
print(f"CER: {cer_overall * 100:.2f}%")
print(f"Mean single-image inference time: {mean_time:.4f} seconds")
