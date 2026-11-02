
import json
import csv
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


# --------------------------------------------------
# EXPERIMENT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

GT_FILE = ROOT / (
    "data/processed/paddleocr_vin/"
    "EXP-YOLO-PPOCR-CONVERT-001_20261028_132836_267609/"
    "detection/det_gt_test.txt"
)

PRED_FILE = ROOT / (
    "results/experiments/EXP-VIN-DET-FT-002/"
    "test/full_test_predictions_expanded.txt"
)

OUT_DIR = ROOT / "results/experiments/EXP-VIN-DET-FT-002/evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_CSV = OUT_DIR / "detection_test_image_metrics.csv"
SUMMARY_FILE = OUT_DIR / "detection_test_metrics_summary.txt"

IOU_THRESHOLDS = [0.50, 0.75]


# --------------------------------------------------
# READ ANNOTATION / PREDICTION FILES
# --------------------------------------------------

def read_entries(file_path, is_prediction=False):
    entries = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            json_start = line.find("[")
            if json_start == -1:
                continue

            image_path = line[:json_start].strip()
            annotations = json.loads(line[json_start:])

            image_name = Path(
                image_path.replace("\\", "/")
            ).name

            polygons = []

            for item in annotations:
                if is_prediction:
                    # Use original detected box, not expanded crop box.
                    points = item.get("original_points")

                    if points is None:
                        continue
                else:
                    points = item.get("points")

                if points:
                    polygons.append(
                        np.asarray(points, dtype=np.float32)
                    )

            entries[image_name] = polygons

    return entries


# --------------------------------------------------
# POLYGON IoU
# --------------------------------------------------

def polygon_iou(poly1, poly2):
    area1 = abs(cv2.contourArea(poly1))
    area2 = abs(cv2.contourArea(poly2))

    if area1 <= 0 or area2 <= 0:
        return 0.0

    intersection, _ = cv2.intersectConvexConvex(
        poly1.astype(np.float32),
        poly2.astype(np.float32)
    )

    union = area1 + area2 - intersection

    return float(intersection / union) if union > 0 else 0.0


# --------------------------------------------------
# MATCH PREDICTIONS TO GROUND TRUTH
# --------------------------------------------------

def match_boxes(gt_boxes, pred_boxes, threshold):
    pairs = []

    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            pairs.append((polygon_iou(gt, pred), gi, pi))

    pairs.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches = []

    for iou, gi, pi in pairs:
        if iou < threshold:
            break

        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)
            matches.append(iou)

    tp = len(matches)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    return tp, fp, fn, matches


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

ground_truth = read_entries(GT_FILE)
predictions = read_entries(PRED_FILE, is_prediction=True)

all_images = sorted(set(ground_truth) | set(predictions))

if not all_images:
    raise RuntimeError("No matching image entries were found.")

rows = []
totals = {
    threshold: {"tp": 0, "fp": 0, "fn": 0}
    for threshold in IOU_THRESHOLDS
}

all_ious = []

for image_name in all_images:
    gt_boxes = ground_truth.get(image_name, [])
    pred_boxes = predictions.get(image_name, [])

    row = {
        "image": image_name,
        "ground_truth_boxes": len(gt_boxes),
        "predicted_boxes": len(pred_boxes),
    }

    image_ious = []

    for threshold in IOU_THRESHOLDS:
        tp, fp, fn, matches = match_boxes(
            gt_boxes, pred_boxes, threshold
        )

        key = f"iou_{threshold:.2f}"

        row[f"{key}_TP"] = tp
        row[f"{key}_FP"] = fp
        row[f"{key}_FN"] = fn

        totals[threshold]["tp"] += tp
        totals[threshold]["fp"] += fp
        totals[threshold]["fn"] += fn

        image_ious.extend(matches)

    # Best IoU for each ground-truth box, for localisation analysis.
    if gt_boxes and pred_boxes:
        best_ious = [
            max(polygon_iou(gt, pred) for pred in pred_boxes)
            for gt in gt_boxes
        ]
        row["best_IoU"] = round(max(best_ious), 6)
        all_ious.extend(best_ious)
    else:
        row["best_IoU"] = 0.0

    rows.append(row)


# --------------------------------------------------
# SAVE PER-IMAGE CSV
# --------------------------------------------------

with open(IMAGE_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


# --------------------------------------------------
# SAVE SUMMARY
# --------------------------------------------------

summary = [
    "EXP-VIN-DET-FT-002",
    "HELD-OUT TEST SET DETECTION EVALUATION",
    "=" * 45,
    f"Ground-truth images: {len(ground_truth)}",
    f"Prediction images: {len(predictions)}",
    f"Images evaluated: {len(all_images)}",
    f"Total ground-truth boxes: {sum(len(v) for v in ground_truth.values())}",
    f"Total predicted boxes: {sum(len(v) for v in predictions.values())}",
    "",
]

for threshold in IOU_THRESHOLDS:
    tp = totals[threshold]["tp"]
    fp = totals[threshold]["fp"]
    fn = totals[threshold]["fn"]

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    summary.extend([
        f"IOU THRESHOLD: {threshold:.2f}",
        f"True Positives: {tp}",
        f"False Positives: {fp}",
        f"False Negatives: {fn}",
        f"Precision: {precision:.6f} ({precision * 100:.2f}%)",
        f"Recall: {recall:.6f} ({recall * 100:.2f}%)",
        f"F1 Score: {f1:.6f} ({f1 * 100:.2f}%)",
        "",
    ])

summary.extend([
    f"Mean best-box IoU: "
    f"{np.mean(all_ious):.6f}" if all_ious else "Mean best-box IoU: N/A",
    "",
    "Matching uses polygon IoU.",
    "Predicted original_points are evaluated.",
    "Expanded crop points are not used for detection scoring.",
    "No confidence scores were available in the prediction file.",
])

SUMMARY_FILE.write_text("\n".join(summary), encoding="utf-8")

print("\n".join(summary))
print(f"\nPer-image results: {IMAGE_CSV}")
print(f"Summary: {SUMMARY_FILE}")

