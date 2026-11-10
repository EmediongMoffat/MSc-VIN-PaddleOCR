
"""
End-to-end OCR evaluation for ICDAR2015-style annotations.

Save as:
    scripts/evaluate_system.py

Inputs:
    data/raw/det_test_labels.txt
    results/system/EXP-SYS-ICDAR15-001/system_results.txt

Outputs:
    results/evaluation/system_evaluation_summary.json
    results/evaluation/system_evaluation_summary.txt
    results/evaluation/system_evaluation_per_image.csv

IMPORTANT:
This is an ICDAR-style approximation, not the official ICDAR evaluator.
"""

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np


# =========================================================
# 1. PATHS AND EXPERIMENT SETTINGS
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

GT_FILE = ROOT / "data" / "raw" / "det_test_labels.txt"
PRED_FILE = (
    ROOT / "results" / "system" /
    "EXP-SYS-ICDAR15-001" / "system_results.txt"
)
OUT_DIR = ROOT / "results" / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Official ICDAR matching is generally described as IoU > 0.5.
IOU_THRESHOLD = 0.5

# Measured from the completed inference run, not evaluation time.
INFERENCE_SECONDS = 59.96812725067139
INFERENCE_IMAGE_COUNT = 500

# Case-insensitive word comparison is used for the
# ICDAR-style end-to-end word match.
# Raw recognition metrics are also reported case-sensitively.


# =========================================================
# 2. LOAD ANNOTATIONS
# =========================================================

def load_records(path):
    records = {}

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(None, 1)
            if len(parts) != 2:
                print(f"WARNING: malformed line {line_number} in {path}")
                continue

            filename = Path(parts[0].replace("\\", "/")).name

            try:
                regions = json.loads(parts[1])
            except json.JSONDecodeError as exc:
                print(f"WARNING: bad JSON at line {line_number}: {exc}")
                continue

            if isinstance(regions, list):
                records[filename] = regions

    return records


def get_text(region):
    return str(region.get("transcription", ""))


def get_points(region):
    points = region.get("points")
    if not points:
        return None

    array = np.asarray(points, dtype=np.float32)

    if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] != 2:
        return None

    return array


# =========================================================
# 3. POLYGON GEOMETRY
# =========================================================

def area(points):
    if points is None:
        return 0.0
    return abs(float(cv2.contourArea(points)))


def intersection_area(points_a, points_b):
    if points_a is None or points_b is None:
        return 0.0

    try:
        intersection, _ = cv2.intersectConvexConvex(
            points_a.astype(np.float32),
            points_b.astype(np.float32)
        )
        return max(0.0, float(intersection))
    except cv2.error:
        return 0.0


def polygon_iou(points_a, points_b):
    area_a = area(points_a)
    area_b = area(points_b)

    if area_a <= 0 or area_b <= 0:
        return 0.0

    intersection = intersection_area(points_a, points_b)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


# =========================================================
# 4. EDIT DISTANCE AND TEXT METRICS
# =========================================================

def edit_distance(a, b):
    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_a != char_b)
            ))

        previous = current

    return previous[-1]


def divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


# =========================================================
# 5. REGION MATCHING
# =========================================================

def greedy_match(gt_regions, pred_regions):
    """
    Greedy one-to-one matching by descending polygon IoU.

    Returns matched pairs, unmatched GT indices, and
    unmatched prediction indices.
    """
    candidates = []

    for gi, gt in enumerate(gt_regions):
        gt_points = get_points(gt)
        if gt_points is None:
            continue

        for pi, pred in enumerate(pred_regions):
            pred_points = get_points(pred)
            if pred_points is None:
                continue

            overlap = polygon_iou(gt_points, pred_points)

            if overlap > IOU_THRESHOLD:
                candidates.append((overlap, gi, pi))

    candidates.sort(reverse=True)

    used_gt = set()
    used_pred = set()
    matches = []

    for overlap, gi, pi in candidates:
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi)
            used_pred.add(pi)
            matches.append((gi, pi, overlap))

    unmatched_gt = set(range(len(gt_regions))) - used_gt
    unmatched_pred = set(range(len(pred_regions))) - used_pred

    return matches, unmatched_gt, unmatched_pred


def overlaps_do_not_care(prediction, ignored_regions):
    """
    ICDAR-style suppression: ignore an unmatched prediction if
    more than half its area overlaps a do-not-care GT region.
    """
    pred_points = get_points(prediction)
    pred_area = area(pred_points)

    if pred_area <= 0:
        return False

    for ignored in ignored_regions:
        ignored_points = get_points(ignored)
        overlap = intersection_area(pred_points, ignored_points)

        if overlap / pred_area > 0.5:
            return True

    return False


# =========================================================
# 6. MAIN EVALUATION
# =========================================================

def main():
    eval_start = time.perf_counter()

    gt_data = load_records(GT_FILE)
    pred_data = load_records(PRED_FILE)

    all_names = sorted(set(gt_data) | set(pred_data))

    print("Ground-truth images:", len(gt_data))
    print("Prediction images:  ", len(pred_data))
    print("Images evaluated:   ", len(all_names))

    # Detection totals
    det_tp = det_fp = det_fn = 0

    # Recognition among geometrically matched word regions
    rec_matched = 0
    rec_exact_sensitive = 0
    rec_exact_insensitive = 0
    rec_edit_errors = 0
    rec_gt_chars = 0

    # End-to-end totals
    e2e_correct = 0
    e2e_gt_words = 0
    e2e_gt_chars = 0
    e2e_char_errors = 0

    per_image = []

    for filename in all_names:
        gt_all = gt_data.get(filename, [])
        pred_all = pred_data.get(filename, [])

        # Ignore malformed GT polygons.
        gt_all = [g for g in gt_all if get_points(g) is not None]
        pred_all = [p for p in pred_all if get_points(p) is not None]

        gt_ignore = [g for g in gt_all if get_text(g) == "###"]
        gt_care = [g for g in gt_all if get_text(g) != "###"]

        matches, unmatched_gt, unmatched_pred = greedy_match(
            gt_care, pred_all
        )

        # Exclude predictions sufficiently covered by ignored GT regions
        # from the detection false-positive count.
        ignored_predictions = {
            pi for pi in unmatched_pred
            if overlaps_do_not_care(pred_all[pi], gt_ignore)
        }

        image_tp = len(matches)
        image_fn = len(unmatched_gt)
        image_fp = len(unmatched_pred - ignored_predictions)

        det_tp += image_tp
        det_fn += image_fn
        det_fp += image_fp

        # Recognition and end-to-end metrics.
        image_rec_exact = 0
        image_e2e_correct = 0

        e2e_gt_words += len(gt_care)

        for gi, pi, overlap in matches:
            gt_text = get_text(gt_care[gi])
            pred_text = get_text(pred_all[pi])

            rec_matched += 1
            rec_gt_chars += len(gt_text)
            rec_edit_errors += edit_distance(gt_text, pred_text)

            if gt_text == pred_text:
                rec_exact_sensitive += 1
                image_rec_exact += 1

            if gt_text.casefold() == pred_text.casefold():
                rec_exact_insensitive += 1
                image_e2e_correct += 1
                e2e_correct += 1

            # End-to-end CER uses case-sensitive edit distance.
            e2e_gt_chars += len(gt_text)
            e2e_char_errors += edit_distance(gt_text, pred_text)

        # Missed GT regions count as deletions in end-to-end CER.
        for gi in unmatched_gt:
            gt_text = get_text(gt_care[gi])
            e2e_gt_chars += len(gt_text)
            e2e_char_errors += len(gt_text)

        per_image.append({
            "filename": filename,
            "gt_care_words": len(gt_care),
            "gt_do_not_care_regions": len(gt_ignore),
            "predicted_regions": len(pred_all),
            "detection_TP": image_tp,
            "detection_FP": image_fp,
            "detection_FN": image_fn,
            "matched_regions": len(matches),
            "recognition_exact_case_sensitive": image_rec_exact,
            "end_to_end_correct_case_insensitive": image_e2e_correct,
        })

    # Detection
    precision = divide(det_tp, det_tp + det_fp)
    recall = divide(det_tp, det_tp + det_fn)
    detection_f1 = divide(2 * precision * recall, precision + recall)

    # Recognition, conditioned on a geometrically matched region
    rec_exact_cs = divide(rec_exact_sensitive, rec_matched)
    rec_exact_ci = divide(rec_exact_insensitive, rec_matched)
    rec_cer = divide(rec_edit_errors, rec_gt_chars)
    rec_char_accuracy = max(0.0, 1.0 - rec_cer)

    # End-to-end: a correct word must be localized and transcribed correctly
    e2e_exact = divide(e2e_correct, e2e_gt_words)
    e2e_cer = divide(e2e_char_errors, e2e_gt_chars)
    e2e_char_accuracy = max(0.0, 1.0 - e2e_cer)
    e2e_string_error = 1.0 - e2e_exact

    evaluation_seconds = time.perf_counter() - eval_start

    # Recorded inference timing
    inference_ms_per_image = (
        1000 * INFERENCE_SECONDS / INFERENCE_IMAGE_COUNT
        if INFERENCE_IMAGE_COUNT else 0.0
    )
    inference_images_per_second = (
        INFERENCE_IMAGE_COUNT / INFERENCE_SECONDS
        if INFERENCE_SECONDS > 0 else 0.0
    )

    summary = {
        "evaluation_method": (
            "ICDAR-style approximation; not official ICDAR evaluation"
        ),
        "iou_rule": "IoU > 0.5",
        "do_not_care_rule": (
            "Unmatched prediction ignored when >50% of its area "
            "overlaps a do-not-care GT region"
        ),
        "test_images_ground_truth": len(gt_data),
        "test_images_predictions": len(pred_data),
        "test_images_evaluated": len(all_names),

        "DETECTION": {
            "true_positives": det_tp,
            "false_positives": det_fp,
            "false_negatives": det_fn,
            "precision": precision,
            "recall": recall,
            "f1_score": detection_f1,
        },

        "RECOGNITION_MATCHED_REGIONS": {
            "matched_regions": rec_matched,
            "exact_word_accuracy_case_sensitive": rec_exact_cs,
            "exact_word_accuracy_case_insensitive": rec_exact_ci,
            "character_edit_errors": rec_edit_errors,
            "ground_truth_characters": rec_gt_chars,
            "character_error_rate": rec_cer,
            "character_accuracy": rec_char_accuracy,
            "scope": "Only geometrically matched text regions",
        },

        "END_TO_END": {
            "ground_truth_care_words": e2e_gt_words,
            "correct_localised_and_recognised_words": e2e_correct,
            "exact_word_accuracy_case_insensitive": e2e_exact,
            "string_error_rate": e2e_string_error,
            "character_edit_errors": e2e_char_errors,
            "ground_truth_characters": e2e_gt_chars,
            "character_error_rate": e2e_cer,
            "character_accuracy": e2e_char_accuracy,
        },

        "INFERENCE_TIMING_PRIOR_RUN": {
            "images": INFERENCE_IMAGE_COUNT,
            "total_seconds": INFERENCE_SECONDS,
            "milliseconds_per_image": inference_ms_per_image,
            "images_per_second": inference_images_per_second,
            "note": (
                "Timing from the previous inference run; not measured "
                "by this evaluator and not a repeated warmed benchmark."
            ),
        },

        "EVALUATION_TIMING": {
            "seconds": evaluation_seconds,
            "note": "Time spent running this evaluation script only.",
        },

        "AUC_ROC": (
            "Not calculated: this output does not provide suitable "
            "binary labels and per-detection confidence scores."
        ),
    }

    # Save JSON summary
    json_path = OUT_DIR / "system_evaluation_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Save readable text report
    txt_path = OUT_DIR / "system_evaluation_summary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("ICDAR2015-STYLE OCR EVALUATION\n")
        f.write("=" * 50 + "\n")
        f.write("NOT OFFICIAL ICDAR SCORING\n\n")

        for section, values in summary.items():
            f.write(f"\n{section}\n")
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.6f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  {values}\n")

    # Save per-image CSV
    csv_path = OUT_DIR / "system_evaluation_per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_image[0].keys())
        writer.writeheader()
        writer.writerows(per_image)

    # Display key metrics
    print("\nDETECTION")
    print(f"TP={det_tp}, FP={det_fp}, FN={det_fn}")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-score:  {detection_f1:.4f} ({detection_f1 * 100:.2f}%)")

    print("\nRECOGNITION - MATCHED REGIONS")
    print(f"Matched regions: {rec_matched}")
    print(f"Exact word accuracy, case-sensitive: {rec_exact_cs:.4f}")
    print(f"Exact word accuracy, case-insensitive: {rec_exact_ci:.4f}")
    print(f"Character accuracy: {rec_char_accuracy:.4f}")
    print(f"CER: {rec_cer:.4f}")

    print("\nEND-TO-END")
    print(f"Ground-truth care words: {e2e_gt_words}")
    print(f"Correct words: {e2e_correct}")
    print(f"Exact word accuracy: {e2e_exact:.4f}")
    print(f"Character accuracy: {e2e_char_accuracy:.4f}")
    print(f"CER: {e2e_cer:.4f}")
    print(f"String error rate: {e2e_string_error:.4f}")

    print("\nINFERENCE TIMING - PREVIOUS RUN")
    print(f"Total: {INFERENCE_SECONDS:.3f} seconds")
    print(f"Average: {inference_ms_per_image:.3f} ms/image")
    print(f"Throughput: {inference_images_per_second:.3f} images/second")
    print(f"Evaluation calculation time: {evaluation_seconds:.3f} seconds")

    print("\nSaved files:")
    print(json_path)
    print(txt_path)
    print(csv_path)


if __name__ == "__main__":
    main()
