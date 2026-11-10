
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

GT_FILE = ROOT / "data" / "raw" / "det_test_labels.txt"
PRED_FILE = (
    ROOT / "results" / "system" /
    "EXP-SYS-ICDAR15-001" / "system_results.txt"
)

OUT_DIR = ROOT / "results" / "evaluation"
FULL_CSV = OUT_DIR / "system_per_image_error_analysis.csv"
WORST_CSV = OUT_DIR / "worst_errors_top50.csv"

IOU_THRESHOLD = 0.5
TOP_N = 50


def read_records(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    records = {}

    with file_path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(None, 1)
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid input at line {line_number}: {file_path}"
                )

            image_path, json_text = parts
            image_name = Path(
                image_path.replace("\\", "/")
            ).name

            if image_name in records:
                raise ValueError(
                    f"Duplicate image filename: {image_name}"
                )

            records[image_name] = json.loads(json_text)

    return records


def get_text(item):
    return str(item.get("transcription", "")).strip()


def get_polygon(item):
    points = item.get("points", [])
    if len(points) < 3:
        return None

    polygon = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    if not np.isfinite(polygon).all():
        return None

    return polygon


def polygon_area(polygon):
    return float(cv2.contourArea(polygon))


def intersection_area(poly_a, poly_b):
    try:
        area, _ = cv2.intersectConvexConvex(
            np.asarray(poly_a, dtype=np.float32),
            np.asarray(poly_b, dtype=np.float32)
        )
        return max(0.0, float(area))
    except cv2.error:
        return 0.0


def polygon_iou(poly_a, poly_b):
    area_a = polygon_area(poly_a)
    area_b = polygon_area(poly_b)

    if area_a <= 0 or area_b <= 0:
        return 0.0

    intersection = intersection_area(poly_a, poly_b)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def edit_distance(a, b):
    """Levenshtein distance: minimum character insertions,
    deletions and substitutions needed to transform a into b.
    """
    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, 1):
        current = [i]

        for j, char_b in enumerate(b, 1):
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_a != char_b)
            ))

        previous = current

    return previous[-1]


def analyse_image(image_name, gt_items, pred_items):
    ignored = [
        item for item in gt_items
        if get_text(item) == "###"
    ]

    gt = [
        item for item in gt_items
        if get_text(item) != "###"
    ]

    # Only valid polygons can participate in matching.
    valid_gt = [
        (item, get_polygon(item)) for item in gt
    ]
    valid_pred = [
        (item, get_polygon(item)) for item in pred_items
    ]

    invalid_gt = sum(poly is None for _, poly in valid_gt)
    invalid_pred = sum(poly is None for _, poly in valid_pred)

    valid_gt = [(item, poly) for item, poly in valid_gt
                if poly is not None]
    valid_pred = [(item, poly) for item, poly in valid_pred
                  if poly is not None]

    # Find candidate GT/prediction pairs with sufficient overlap.
    candidates = []

    for gi, (gt_item, gt_poly) in enumerate(valid_gt):
        for pi, (pred_item, pred_poly) in enumerate(valid_pred):
            iou = polygon_iou(gt_poly, pred_poly)

            if iou > IOU_THRESHOLD:
                candidates.append((iou, gi, pi))

    # Greedy one-to-one matching: highest IoU first.
    candidates.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()
    pairs = []

    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue

        matched_gt.add(gi)
        matched_pred.add(pi)
        pairs.append((gi, pi, iou))

    missed = len(valid_gt) - len(matched_gt)

    # Ignore unmatched predictions substantially overlapping ### regions.
    ignored_polygons = [
        get_polygon(item) for item in ignored
    ]
    ignored_polygons = [
        poly for poly in ignored_polygons if poly is not None
    ]

    false_positive = 0

    for pi, (_, pred_poly) in enumerate(valid_pred):
        if pi in matched_pred:
            continue

        pred_area = polygon_area(pred_poly)
        ignored_overlap = False

        if pred_area > 0:
            for ignored_poly in ignored_polygons:
                overlap = intersection_area(pred_poly, ignored_poly)

                if overlap / pred_area > 0.5:
                    ignored_overlap = True
                    break

        if not ignored_overlap:
            false_positive += 1

    wrong_recognition = 0
    total_char_edits = 0
    total_gt_characters = 0
    matched_gt_characters = 0

    for gi, pi, _ in pairs:
        true_text = get_text(valid_gt[gi][0])
        predicted_text = get_text(valid_pred[pi][0])

        distance = edit_distance(true_text, predicted_text)
        total_char_edits += distance
        total_gt_characters += len(true_text)
        matched_gt_characters += len(true_text)

        if true_text != predicted_text:
            wrong_recognition += 1

    # Recognition CER for matched regions only; misses are reported separately.
    matched_cer = (
        total_char_edits / total_gt_characters
        if total_gt_characters else 0.0
    )

    # Higher score means more region-level errors.
    error_score = missed + false_positive + wrong_recognition

    if missed and false_positive:
        category = "Missed detections and false positives"
    elif missed:
        category = "Missed detections"
    elif false_positive:
        category = "False positives"
    elif wrong_recognition:
        category = "Recognition errors"
    else:
        category = "No counted errors"

    return {
        "image": image_name,
        "ground_truth_regions": len(valid_gt),
        "predicted_regions": len(valid_pred),
        "matched_regions": len(pairs),
        "missed_detections": missed,
        "false_positive_detections": false_positive,
        "incorrect_recognitions": wrong_recognition,
        "character_edit_distance_matched": total_char_edits,
        "matched_region_CER": round(matched_cer, 4),
        "invalid_ground_truth_polygons": invalid_gt,
        "invalid_prediction_polygons": invalid_pred,
        "error_score": error_score,
        "primary_failure_category": category,
    }


def main():
    gt_records = read_records(GT_FILE)
    pred_records = read_records(PRED_FILE)

    common_names = set(gt_records) & set(pred_records)

    if not common_names:
        raise ValueError("No matching image filenames found.")

    missing_predictions = set(gt_records) - set(pred_records)
    missing_ground_truth = set(pred_records) - set(gt_records)

    rows = []

    for image_name in sorted(common_names):
        rows.append(
            analyse_image(
                image_name,
                gt_records[image_name],
                pred_records[image_name]
            )
        )

    # Worst first: region-level errors, then character edit distance.
    rows.sort(
        key=lambda row: (
            row["error_score"],
            row["character_edit_distance_matched"],
            row["matched_region_CER"],
            row["missed_detections"],
            row["false_positive_detections"]
        ),
        reverse=True
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    columns = list(rows[0].keys())

    with FULL_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    with WORST_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows[:TOP_N])

    print("\nERROR ANALYSIS COMPLETE")
    print("Images analysed:", len(rows))
    print("Ground-truth images without predictions:",
          len(missing_predictions))
    print("Prediction images without ground truth:",
          len(missing_ground_truth))
    print("\nFull report:", FULL_CSV)
    print("Worst 50 errors:", WORST_CSV)
    print("\nTop 10 worst images:")

    for row in rows[:10]:
        print(
            f'{row["image"]}: '
            f'score={row["error_score"]}, '
            f'missed={row["missed_detections"]}, '
            f'false positives={row["false_positive_detections"]}, '
            f'incorrect recognition={row["incorrect_recognitions"]}'
        )


if __name__ == "__main__":
    main()
