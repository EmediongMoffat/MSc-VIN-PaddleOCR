
from pathlib import Path
import argparse
import csv
import json
import statistics

import cv2
import numpy as np


# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GT_FILE = (
    PROJECT_ROOT / "data" / "raw" / "det_test_labels.txt"
)

PRED_FILE = (
    PROJECT_ROOT / "results" / "system" / "EXP-SYS-ICDAR15-001"
    / "system_results.txt"
)

MANIFEST_FILE = (
    PROJECT_ROOT / "results" / "evaluation" / "EXP-E2E-ICDAR15-002"
    / "eligible_gt_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "evaluation" / "EXP-E2E-ICDAR15-002"
    / "complete_metrics"
)

VISUAL_DIR = OUTPUT_DIR / "failed_image_examples"

IOU_THRESHOLD = 0.50

# Colours are BGR for OpenCV.
GREEN = (0, 180, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
BLUE = (255, 100, 0)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)


# ============================================================
# 2. READ PADDLEOCR LABEL FILES
# ============================================================

def read_label_file(path):
    """
    Expected format:
    image.jpg<TAB>[{"transcription": "...", "points": [...]}]
    """

    records = {}

    with open(path, "r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                image_name, json_text = line.split("\t", 1)
                annotations = json.loads(json_text)
                records[Path(image_name).name] = annotations

            except Exception as exc:
                raise ValueError(
                    f"Error reading {path}, line {line_number}: {exc}"
                )

    return records


def read_manifest(path):
    """
    Reads the eligible GT manifest.

    annotation_index is assumed to be 1-based, as in the
    previously verified manifest.
    """

    eligible = {}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "source_image_basename",
            "annotation_index"
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing manifest columns: {sorted(missing)}"
            )

        for row in reader:
            image_name = Path(
                row["source_image_basename"]
            ).name

            index = int(row["annotation_index"]) - 1

            eligible.setdefault(image_name, set()).add(index)

    return eligible


# ============================================================
# 3. TEXT NORMALISATION
# ============================================================

def normalise_text(text):
    """
    Full-width to half-width conversion and space removal.
    """

    result = []

    for char in str(text):
        code = ord(char)

        if code == 0x3000:
            code = 0x20
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0

        result.append(chr(code))

    return "".join(result).replace(" ", "")


# ============================================================
# 4. POLYGON IoU
# ============================================================

def convex_polygon(points):
    points = np.asarray(points, dtype=np.float32)

    if len(points) < 3:
        return None

    return cv2.convexHull(points)


def polygon_iou(points_a, points_b):

    a = convex_polygon(points_a)
    b = convex_polygon(points_b)

    if a is None or b is None:
        return 0.0

    area_a = float(cv2.contourArea(a))
    area_b = float(cv2.contourArea(b))

    if area_a <= 0 or area_b <= 0:
        return 0.0

    intersection, _ = cv2.intersectConvexConvex(a, b)

    union = area_a + area_b - float(intersection)

    if union <= 0:
        return 0.0

    return float(intersection) / union


# ============================================================
# 5. LEVENSHTEIN DISTANCE AND CHARACTER ERROR COUNTS
# ============================================================

def levenshtein_counts(reference, prediction):
    """
    Returns:
        distance, substitutions, deletions, insertions

    Reference = GT text.
    Prediction = OCR output.
    """

    rows = len(reference) + 1
    cols = len(prediction) + 1

    dp = [
        [0] * cols
        for _ in range(rows)
    ]

    for i in range(rows):
        dp[i][0] = i

    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):

            cost = (
                0 if reference[i - 1] == prediction[j - 1]
                else 1
            )

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    substitutions = 0
    deletions = 0
    insertions = 0

    i = len(reference)
    j = len(prediction)

    while i > 0 or j > 0:

        if i > 0 and j > 0:
            cost = (
                0 if reference[i - 1] == prediction[j - 1]
                else 1
            )

            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost:
                    substitutions += 1

                i -= 1
                j -= 1
                continue

        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
            continue

        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1
            continue

        raise RuntimeError("Levenshtein backtracking failed.")

    return (
        dp[len(reference)][len(prediction)],
        substitutions,
        deletions,
        insertions
    )


# ============================================================
# 6. GREEDY ONE-TO-ONE GEOMETRIC MATCHING
# ============================================================

def match_annotations(gt_annotations, pred_annotations,
                      eligible_indices):

    gt_used = set()
    pred_used = set()
    pairs = []

    for gi, gt in enumerate(gt_annotations):

        points_gt = gt.get("points", [])

        if len(points_gt) < 3:
            continue

        for pi, pred in enumerate(pred_annotations):

            points_pred = pred.get("points", [])

            if len(points_pred) < 3:
                continue

            iou = polygon_iou(points_gt, points_pred)

            if iou >= IOU_THRESHOLD:
                pairs.append((iou, gi, pi))

    pairs.sort(key=lambda item: item[0], reverse=True)

    care_matches = []
    ignored_matches = []

    for iou, gi, pi in pairs:

        if gi in gt_used or pi in pred_used:
            continue

        gt_used.add(gi)
        pred_used.add(pi)

        gt_text = str(
            gt_annotations[gi].get("transcription", "")
        )

        if gi in eligible_indices:
            care_matches.append((gi, pi, iou))

        elif gt_text == "###":
            ignored_matches.append((gi, pi, iou))

    unmatched_gt = [
        gi for gi in eligible_indices
        if gi < len(gt_annotations) and gi not in gt_used
    ]

    # Predictions matched to ignored regions are excluded.
    # Other unmatched predictions are false positives.
    unmatched_pred = [
        pi for pi in range(len(pred_annotations))
        if pi not in pred_used
    ]

    return (
        care_matches,
        ignored_matches,
        unmatched_gt,
        unmatched_pred
    )


# ============================================================
# 7. FIND ORIGINAL FULL-SCENE IMAGES
# ============================================================

def build_image_index(image_root):

    index = {}

    if image_root is None:
        return index

    image_root = Path(image_root)

    if not image_root.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {image_root}"
        )

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            index.setdefault(path.name, path)

    return index


# ============================================================
# 8. DRAW FULL-SCENE FAILURE EXAMPLES
# ============================================================

def draw_polygon(image, points, colour, thickness=2):

    polygon = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)

    cv2.polylines(
        image,
        [polygon],
        True,
        colour,
        thickness
    )


def add_label(image, text, x, y, colour):

    x = max(0, int(x))
    y = max(20, int(y))

    cv2.putText(
        image,
        text[:90],
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        colour,
        2,
        cv2.LINE_AA
    )


def create_failed_image(
    image_name,
    image_path,
    gt_annotations,
    pred_annotations,
    eligible_indices,
    care_matches,
    unmatched_gt,
    unmatched_pred,
    output_path
):

    image = cv2.imread(str(image_path))

    if image is None:
        return False

    image = image.copy()

    match_by_gt = {
        gi: (pi, iou)
        for gi, pi, iou in care_matches
    }

    match_by_pred = {
        pi: gi
        for gi, pi, _ in care_matches
    }

    # --------------------------------------------------------
    # DRAW ELIGIBLE GROUND-TRUTH REGIONS
    # --------------------------------------------------------

    for gi in sorted(eligible_indices):

        if gi >= len(gt_annotations):
            continue

        gt = gt_annotations[gi]
        gt_text = normalise_text(
            gt.get("transcription", "")
        )

        points = gt.get("points", [])

        if gi in match_by_gt:

            pi, iou = match_by_gt[gi]

            pred_text = normalise_text(
                pred_annotations[pi].get("transcription", "")
            )

            if gt_text == pred_text:
                colour = GREEN
                status = "CORRECT"
            else:
                colour = ORANGE
                status = "RECOGNITION ERROR"

            label = (
                f"GT: {gt_text} | PRED: {pred_text} "
                f"| {status} | IoU={iou:.2f}"
            )

        else:
            colour = RED
            label = f"GT: {gt_text} | MISSED DETECTION"

        if len(points) >= 3:
            draw_polygon(image, points, colour)

            x = min(point[0] for point in points)
            y = min(point[1] for point in points)

            add_label(image, label, x, y, colour)

    # --------------------------------------------------------
    # DRAW UNMATCHED PREDICTIONS
    # --------------------------------------------------------

    for pi in unmatched_pred:

        pred = pred_annotations[pi]

        pred_text = normalise_text(
            pred.get("transcription", "")
        )

        points = pred.get("points", [])

        if len(points) >= 3:
            draw_polygon(image, points, MAGENTA)

            x = min(point[0] for point in points)
            y = min(point[1] for point in points)

            add_label(
                image,
                f"FALSE POSITIVE: {pred_text}",
                x,
                y,
                MAGENTA
            )

    # --------------------------------------------------------
    # IMAGE TITLE
    # --------------------------------------------------------

    title = f"FAILED IMAGE: {image_name}"

    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1], 42),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        image,
        title[:100],
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        WHITE,
        2,
        cv2.LINE_AA
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    return cv2.imwrite(str(output_path), image)


# ============================================================
# 9. CSV HELPERS
# ============================================================

def save_csv(path, rows, columns=None):

    if not rows and not columns:
        return

    if columns is None:
        columns = list(rows[0].keys())

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 10. MAIN EVALUATION
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Folder containing the original full-scene images."
    )

    parser.add_argument(
        "--total-seconds",
        type=float,
        default=None,
        help="Measured total inference time in seconds."
    )

    parser.add_argument(
        "--timing-csv",
        type=str,
        default=None,
        help="CSV containing per-image inference_ms."
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading GT...")
    gt_data = read_label_file(GT_FILE)

    print("Loading predictions...")
    pred_data = read_label_file(PRED_FILE)

    print("Loading eligible GT manifest...")
    eligible_data = read_manifest(MANIFEST_FILE)

    print("Indexing original images...")
    image_index = build_image_index(args.image_dir)

    image_names = sorted(
        set(gt_data) | set(pred_data)
    )

    # --------------------------------------------------------
    # GLOBAL COUNTERS
    # --------------------------------------------------------

    TP = 0
    FP = 0
    FN = 0

    correct_strings = 0
    failed_strings = 0
    matched_regions = 0

    substitutions = 0
    deletions = 0
    insertions = 0

    matched_substitutions = 0
    matched_deletions = 0
    matched_insertions = 0

    gt_character_count = 0
    matched_gt_character_count = 0

    exact_hits = 0
    dt_count = 0
    gt_count = 0

    total_edit_distance = 0
    matched_edit_distance = 0

    failed_string_rows = []
    false_positive_rows = []
    per_image_rows = []
    failed_image_rows = []

    image_match_cache = {}

    # --------------------------------------------------------
    # PROCESS EACH IMAGE
    # --------------------------------------------------------

    for image_number, image_name in enumerate(image_names, 1):

        gt = gt_data.get(image_name, [])
        pred = pred_data.get(image_name, [])

        eligible = eligible_data.get(image_name, set())

        care_matches, ignored_matches, unmatched_gt, unmatched_pred = (
            match_annotations(gt, pred, eligible)
        )

        image_TP = len(care_matches)
        image_FP = len(unmatched_pred)
        image_FN = len(unmatched_gt)

        TP += image_TP
        FP += image_FP
        FN += image_FN

        image_correct = 0
        image_failed = 0
        image_edit_distance = 0

        # ----------------------------------------------------
        # MATCHED CARE REGIONS
        # ----------------------------------------------------

        for gi, pi, iou in care_matches:

            gt_text = normalise_text(
                gt[gi].get("transcription", "")
            )

            pred_text = normalise_text(
                pred[pi].get("transcription", "")
            )

            distance, sub, dele, ins = levenshtein_counts(
                gt_text,
                pred_text
            )

            matched_regions += 1
            matched_gt_character_count += len(gt_text)
            gt_character_count += len(gt_text)

            matched_edit_distance += distance
            total_edit_distance += distance
            image_edit_distance += distance

            matched_substitutions += sub
            matched_deletions += dele
            matched_insertions += ins

            substitutions += sub
            deletions += dele
            insertions += ins

            if gt_text == pred_text:

                correct_strings += 1
                image_correct += 1
                exact_hits += 1

            else:

                failed_strings += 1
                image_failed += 1

                failed_string_rows.append({
                    "image": image_name,
                    "ground_truth": gt_text,
                    "prediction": pred_text,
                    "failure_type": "Recognition error",
                    "IoU": round(iou, 5),
                    "edit_distance": distance,
                    "substitutions": sub,
                    "deletions": dele,
                    "insertions": ins
                })

            dt_count += 1
            gt_count += 1

        # ----------------------------------------------------
        # MISSED GT REGIONS
        # ----------------------------------------------------

        for gi in unmatched_gt:

            gt_text = normalise_text(
                gt[gi].get("transcription", "")
            )

            distance, sub, dele, ins = levenshtein_counts(
                gt_text,
                ""
            )

            gt_character_count += len(gt_text)

            total_edit_distance += distance
            image_edit_distance += distance

            # A completely missed string contributes deletions.
            deletions += dele

            failed_strings += 1
            image_failed += 1

            failed_string_rows.append({
                "image": image_name,
                "ground_truth": gt_text,
                "prediction": "",
                "failure_type": "Missed detection",
                "IoU": "",
                "edit_distance": distance,
                "substitutions": sub,
                "deletions": dele,
                "insertions": ins
            })

            gt_count += 1

        # ----------------------------------------------------
        # UNMATCHED PREDICTIONS
        # ----------------------------------------------------

        for pi in unmatched_pred:

            pred_text = normalise_text(
                pred[pi].get("transcription", "")
            )

            distance, sub, dele, ins = levenshtein_counts(
                "",
                pred_text
            )

            total_edit_distance += distance
            image_edit_distance += distance

            # False-positive text contributes insertions to
            # the supplementary overall CER calculation.
            insertions += ins

            dt_count += 1

            false_positive_rows.append({
                "image": image_name,
                "false_positive_text": pred_text,
                "reason": "No matched GT region"
            })

        image_match_cache[image_name] = (
            care_matches,
            unmatched_gt,
            unmatched_pred
        )

        per_image_rows.append({
            "image": image_name,
            "GT_care_regions": len(eligible),
            "predicted_regions": len(pred),
            "TP": image_TP,
            "FP": image_FP,
            "FN": image_FN,
            "correct_strings": image_correct,
            "failed_strings": image_failed,
            "edit_distance": image_edit_distance
        })

        if image_FN > 0 or image_FP > 0 or image_failed > 0:

            failed_image_rows.append({
                "image": image_name,
                "TP": image_TP,
                "FP": image_FP,
                "FN": image_FN,
                "correct_strings": image_correct,
                "failed_strings": image_failed,
                "edit_distance": image_edit_distance
            })

        if image_number % 50 == 0:
            print(f"Processed {image_number}/{len(image_names)} images")

    # ========================================================
    # 11. CALCULATE METRICS
    # ========================================================

    def divide(a, b):
        return a / b if b else 0.0

    total_eligible_gt = TP + FN

    detection_precision = divide(TP, TP + FP)
    detection_recall = divide(TP, TP + FN)

    detection_f1 = divide(
        2 * detection_precision * detection_recall,
        detection_precision + detection_recall
    )

    exact_string_accuracy = divide(
        correct_strings,
        total_eligible_gt
    )

    matched_exact_accuracy = divide(
        correct_strings,
        matched_regions
    )

    matched_string_error_rate = divide(
        matched_regions - correct_strings,
        matched_regions
    )

    overall_string_error_rate = divide(
        failed_strings,
        total_eligible_gt
    )

    matched_only_cer = divide(
        matched_edit_distance,
        matched_gt_character_count
    )

    # Overall CER includes:
    # - matched-region S/D/I
    # - missed GT characters as deletions
    # - unmatched predicted characters as insertions
    overall_cer = divide(
        substitutions + deletions + insertions,
        gt_character_count
    )

    overall_character_accuracy = 1.0 - overall_cer

    # Official-style exact text precision/recall reconstruction.
    official_precision = divide(exact_hits, dt_count)
    official_recall = divide(exact_hits, gt_count)

    official_f1 = divide(
        2 * official_precision * official_recall,
        official_precision + official_recall
    )

    # ========================================================
    # 12. INFERENCE TIMING
    # ========================================================

    timing = {}

    if args.timing_csv:

        timing_values = []

        with open(
            args.timing_csv,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            if "inference_ms" not in (reader.fieldnames or []):
                raise ValueError(
                    "Timing CSV requires an inference_ms column."
                )

            for row in reader:
                value = float(row["inference_ms"])

                if value < 0:
                    raise ValueError(
                        "Inference times cannot be negative."
                    )

                timing_values.append(value)

        if not timing_values:
            raise ValueError("Timing CSV contains no timing values.")

        timing = {
            "Timing source": "Per-image timing CSV",
            "Images timed": len(timing_values),
            "Total inference seconds": sum(timing_values) / 1000,
            "Mean ms per image": statistics.mean(timing_values),
            "Median ms per image": statistics.median(timing_values),
            "P95 ms per image": float(np.percentile(timing_values, 95)),
            "Images per second": divide(
                len(timing_values),
                sum(timing_values) / 1000
            )
        }

    elif args.total_seconds is not None:

        if args.total_seconds <= 0:
            raise ValueError("Total inference time must be positive.")

        timing = {
            "Timing source": "User-supplied total inference time",
            "Images timed": len(image_names),
            "Total inference seconds": args.total_seconds,
            "Mean ms per image": divide(
                args.total_seconds * 1000,
                len(image_names)
            ),
            "Median ms per image": "N/A",
            "P95 ms per image": "N/A",
            "Images per second": divide(
                len(image_names),
                args.total_seconds
            )
        }

    else:

        timing = {
            "Timing source": "Not supplied",
            "Images timed": 0,
            "Total inference seconds": "N/A",
            "Mean ms per image": "N/A",
            "Median ms per image": "N/A",
            "P95 ms per image": "N/A",
            "Images per second": "N/A"
        }

    # ========================================================
    # 13. SUMMARY METRICS
    # ========================================================

    metrics = {
        "Number of images": len(image_names),

        "Detection TP": TP,
        "Detection FP": FP,
        "Detection FN": FN,
        "Detection TN": "N/A - not defined for object detection",
        "Detection precision (%)": detection_precision * 100,
        "Detection recall (%)": detection_recall * 100,
        "Detection F1 (%)": detection_f1 * 100,

        "Eligible GT strings": total_eligible_gt,
        "Correct strings": correct_strings,
        "Failed GT strings": failed_strings,
        "Matched regions": matched_regions,

        "Exact string accuracy overall (%)":
            exact_string_accuracy * 100,

        "Exact string accuracy matched-only (%)":
            matched_exact_accuracy * 100,

        "Matched-region string error rate (%)":
            matched_string_error_rate * 100,

        "Overall string error rate (%)":
            overall_string_error_rate * 100,

        "Substitutions": substitutions,
        "Deletions": deletions,
        "Insertions": insertions,

        "Total GT characters": gt_character_count,

        "Matched-only CER (%)": matched_only_cer * 100,
        "Overall CER (%)": overall_cer * 100,
        "Overall character accuracy (%)":
            overall_character_accuracy * 100,

        "Exact text hits": exact_hits,
        "dt_count": dt_count,
        "gt_count": gt_count,

        "Official-style precision (%)": official_precision * 100,
        "Official-style recall (%)": official_recall * 100,
        "Official-style F1 (%)": official_f1 * 100,

        "Total edit distance": total_edit_distance,

        **timing
    }

    # ========================================================
    # 14. GENERATE FAILED FULL-SCENE IMAGE EXAMPLES
    # ========================================================

    visual_rows = []
    visual_count = 0

    if args.image_dir:

        print("\nCreating failed-image visualisations...")

        for row in failed_image_rows:

            image_name = row["image"]

            image_path = image_index.get(image_name)

            if image_path is None:
                visual_rows.append({
                    **row,
                    "visualisation": "Original image not found"
                })
                continue

            gt = gt_data.get(image_name, [])
            pred = pred_data.get(image_name, [])
            eligible = eligible_data.get(image_name, set())

            care_matches, unmatched_gt, unmatched_pred = (
                image_match_cache[image_name]
            )

            output_path = (
                VISUAL_DIR / f"FAILED_{image_name}.jpg"
            )

            success = create_failed_image(
                image_name,
                image_path,
                gt,
                pred,
                eligible,
                care_matches,
                unmatched_gt,
                unmatched_pred,
                output_path
            )

            if success:
                visual_count += 1
                visual_name = str(output_path)
            else:
                visual_name = "Could not read or save image"

            visual_rows.append({
                **row,
                "visualisation": visual_name
            })

        print(f"Saved failed-image examples: {visual_count}")

    else:

        print(
            "\nNo --image-dir supplied. "
            "Full-scene visualisations were not generated."
        )

    # ========================================================
    # 15. SAVE ALL OUTPUTS
    # ========================================================

    save_csv(
        OUTPUT_DIR / "complete_metrics_summary.csv",
        [{"Metric": k, "Value": v} for k, v in metrics.items()],
        ["Metric", "Value"]
    )

    with open(
        OUTPUT_DIR / "complete_metrics_summary.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write("ICDAR2015 COMPLETE OCR EVALUATION\n")
        f.write("=" * 65 + "\n\n")

        for key, value in metrics.items():

            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")

        f.write("\nMetric definitions:\n")
        f.write(
            "Detection TP/FP/FN use polygon IoU >= 0.50 "
            "and greedy one-to-one matching.\n"
        )
        f.write(
            "TN is not defined for this object-detection task.\n"
        )
        f.write(
            "Overall string error rate = failed eligible GT strings "
            "/ all eligible GT strings.\n"
        )
        f.write(
            "Matched-only CER excludes missed GT and unmatched "
            "predictions.\n"
        )
        f.write(
            "Overall CER includes missed GT characters as deletions "
            "and unmatched prediction characters as insertions.\n"
        )
        f.write(
            "Official-style metrics are a reconstruction and must "
            "not replace the exact PaddleOCR evaluator output.\n"
        )

    save_csv(
        OUTPUT_DIR / "failed_strings.csv",
        failed_string_rows,
        [
            "image", "ground_truth", "prediction", "failure_type",
            "IoU", "edit_distance", "substitutions",
            "deletions", "insertions"
        ]
    )

    save_csv(
        OUTPUT_DIR / "false_positive_detections.csv",
        false_positive_rows,
        ["image", "false_positive_text", "reason"]
    )

    save_csv(
        OUTPUT_DIR / "per_image_metrics.csv",
        per_image_rows
    )

    save_csv(
        OUTPUT_DIR / "failed_images_summary.csv",
        failed_image_rows,
        [
            "image", "TP", "FP", "FN",
            "correct_strings", "failed_strings", "edit_distance"
        ]
    )

    save_csv(
        OUTPUT_DIR / "failed_image_visualisations.csv",
        visual_rows,
        [
            "image", "TP", "FP", "FN",
            "correct_strings", "failed_strings",
            "edit_distance", "visualisation"
        ]
    )

    # ========================================================
    # 16. PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("ICDAR2015 COMPLETE OCR EVALUATION")
    print("=" * 70)

    for key, value in metrics.items():

        if isinstance(value, float):
            print(f"{key:<48}: {value:.4f}")
        else:
            print(f"{key:<48}: {value}")

    print("\nFailed images:", len(failed_image_rows))
    print("Visualisations saved:", visual_count)
    print("\nOutput folder:")
    print(OUTPUT_DIR)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
