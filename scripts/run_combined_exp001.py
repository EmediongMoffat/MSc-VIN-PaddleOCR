
import csv
from email.mime import image
import json
import time
from pathlib import Path

import cv2
import numpy as np
import paddle
from paddlex import create_predictor


# ============================================================
# EXP-VIN-COMBINED-001
# Original image -> Detection -> Crop -> Recognition -> Evaluation
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    ROOT / "data" / "processed" / "paddleocr_vin"
    / "EXP-YOLO-PPOCR-CONVERT-001_20261028_132836_267609"
    / "detection"
)

LABEL_FILE = DATA_ROOT / "det_gt_test.txt"

DET_MODEL = (
    ROOT / "results" / "experiments"
    / "EXP-VIN-COMBINED-001" / "detector_inference"
)

REC_MODEL = (
    ROOT / "results" / "experiments"
    / "EXP-VIN-REC-FT-001" / "inference"
)

OUT_DIR = (
    ROOT / "results" / "experiments"
    / "EXP-VIN-COMBINED-001"
)

CROP_DIR = OUT_DIR / "predicted_test_crops"
EVAL_DIR = OUT_DIR / "evaluation"

CROP_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = EVAL_DIR / "combined_test_predictions.csv"
SUMMARY_FILE = EVAL_DIR / "combined_evaluation_summary.txt"

X_PADDING = 0.10
Y_PADDING = 0.30


def sync_gpu():
    """Synchronise GPU operations before timing."""
    if paddle.is_compiled_with_cuda():
        paddle.device.cuda.synchronize()


def result_to_dict(result):
    """Convert PaddleX result objects into dictionaries."""
    if isinstance(result, dict):
        return result

    for attr in ("json", "res", "data"):
        value = getattr(result, attr, None)

        if callable(value):
            try:
                value = value()
            except Exception:
                continue

        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                continue

        if isinstance(value, dict):
            return value

    if hasattr(result, "to_dict"):
        value = result.to_dict()
        if isinstance(value, dict):
            return value

    raise TypeError(
        "Could not convert PaddleX result to a dictionary. "
        f"Result type: {type(result)}"
    )


def find_key(data, key):
    """Find a key recursively inside nested result dictionaries."""
    if isinstance(data, dict):
        if key in data:
            return data[key]

        for value in data.values():
            found = find_key(value, key)
            if found is not None:
                return found

    elif isinstance(data, list):
        for value in data:
            found = find_key(value, key)
            if found is not None:
                return found

    return None


def edit_distance(a, b):
    """Levenshtein edit distance."""
    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1

            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + cost
                )
            )

        previous = current

    return previous[-1]


def load_test_labels():
    """Read original test image paths and ground-truth VINs."""
    samples = []

    with LABEL_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                image_path, annotation_text = line.split("\t", 1)
                annotations = json.loads(annotation_text)

                if not annotations:
                    raise ValueError("No annotations found")

                # This dataset contains one VIN transcription per image.
                gt_vin = annotations[0]["transcription"]

                samples.append({
                    "image": DATA_ROOT / image_path,
                    "filename": Path(image_path).name,
                    "ground_truth": gt_vin
                })

            except Exception as error:
                raise ValueError(
                    f"Could not parse test label line {line_number}: {error}"
                ) from error

    return samples


def get_detection_polygons(result_dict):
    polygons = find_key(result_dict, "dt_polys")

    if polygons is None:
        polygons = find_key(result_dict, "polys")

    if polygons is None:
        raise KeyError(
            "Detection output did not contain dt_polys or polys. "
            f"Available result: {result_dict}"
        )

    polygons = np.asarray(polygons, dtype=np.float32)

    if polygons.size == 0:
        return []

    return polygons.reshape(-1, 4, 2)


def select_vin_polygon(polygons):
    """Select the largest detected text region by bounding-box area."""
    candidates = []

    for polygon in polygons:
        x_min = float(np.min(polygon[:, 0]))
        x_max = float(np.max(polygon[:, 0]))
        y_min = float(np.min(polygon[:, 1]))
        y_max = float(np.max(polygon[:, 1]))

        width = x_max - x_min
        height = y_max - y_min

        if width > 0 and height > 0:
            candidates.append((width * height, polygon))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def crop_detected_vin(image, polygon):
    """Expand the predicted box and crop the original image."""
    image_height, image_width = image.shape[:2]

    x_min = float(np.min(polygon[:, 0]))
    x_max = float(np.max(polygon[:, 0]))
    y_min = float(np.min(polygon[:, 1]))
    y_max = float(np.max(polygon[:, 1]))

    width = x_max - x_min
    height = y_max - y_min

    x_min -= width * X_PADDING
    x_max += width * X_PADDING
    y_min -= height * Y_PADDING
    y_max += height * Y_PADDING

    x_min = max(0, int(np.floor(x_min)))
    y_min = max(0, int(np.floor(y_min)))
    x_max = min(image_width, int(np.ceil(x_max)))
    y_max = min(image_height, int(np.ceil(y_max)))

    if x_max <= x_min or y_max <= y_min:
        return None

    crop = image[y_min:y_max, x_min:x_max]

    if crop.size == 0:
        return None

    return crop


def get_recognition(result_dict):
    text = find_key(result_dict, "rec_text")
    confidence = find_key(result_dict, "rec_score")

    if text is None:
        text = find_key(result_dict, "text")

    if confidence is None:
        confidence = find_key(result_dict, "score")

    # Handle a single prediction or a list containing one prediction.
    if isinstance(text, (list, tuple, np.ndarray)):
        text = text[0] if len(text) else ""

    if isinstance(confidence, (list, tuple, np.ndarray)):
        confidence = confidence[0] if len(confidence) else 0.0

    if text is None:
        text = ""

    if confidence is None:
        confidence = 0.0

    return str(text), float(confidence)


def main():
    print("=" * 65)
    print("EXP-VIN-COMBINED-001")
    print("Combined VIN Detection and Recognition Evaluation")
    print("=" * 65)

    if not LABEL_FILE.is_file():
        raise FileNotFoundError(f"Test labels not found: {LABEL_FILE}")

    samples = load_test_labels()

    print(f"Test samples loaded: {len(samples)}")
    print("Loading detector on GPU...")

    detector = create_predictor(
        model_name="PP-OCRv5_mobile_det",
        model_dir=str(DET_MODEL),
        device="gpu:0"
    )

    print("Loading recogniser on GPU...")

    recogniser = create_predictor(
        model_name="PP-OCRv5_mobile_rec",
        model_dir=str(REC_MODEL),
        device="gpu:0"
    )

    print("Both predictors loaded.")
    print("Starting combined test evaluation...\n")

    rows = []
    total_edits = 0
    total_gt_characters = 0
    exact_matches = 0
    detection_failures = 0
    total_elapsed = 0.0

    for index, sample in enumerate(samples, start=1):
        image_path = sample["image"]
        filename = sample["filename"]
        ground_truth = sample["ground_truth"]

        predicted_text = ""
        confidence = 0.0
        detection_boxes = 0
        detection_time = 0.0
        recognition_time = 0.0
        crop_path = ""
        bbox_x_min = ""
        bbox_y_min = ""
        bbox_x_max = ""
        bbox_y_max = ""
        status = "FAILED"

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Could not read original test image: {image_path}"
            )

        sync_gpu()
        image_start = time.perf_counter()

        try:
            # 1. Detection on the ORIGINAL test image.
            sync_gpu()
            det_start = time.perf_counter()

            det_results = list(detector.apply(image))

            sync_gpu()
            detection_time = time.perf_counter() - det_start

            if not det_results:
                raise RuntimeError("Detector returned no result")

            det_dict = result_to_dict(det_results[0])
            polygons = get_detection_polygons(det_dict)
            detection_boxes = len(polygons)

            if detection_boxes == 0:
                raise RuntimeError("No VIN text region detected")

            # 2. Select the most plausible VIN region.
            polygon = select_vin_polygon(polygons)

            if polygon is None:
                raise RuntimeError("No valid detection polygon")

            # Save the original detected bounding-box coordinates.
            image_height, image_width = image.shape[:2]

            bbox_x_min = max(0, int(np.floor(np.min(polygon[:, 0]))))
            bbox_y_min = max(0, int(np.floor(np.min(polygon[:, 1]))))
            bbox_x_max = min(image_width, int(np.ceil(np.max(polygon[:, 0]))))
            bbox_y_max = min(image_height, int(np.ceil(np.max(polygon[:, 1]))))

            # 3. Crop the ORIGINAL image using predicted coordinates.
            crop = crop_detected_vin(image, polygon)

            if crop is None:
                raise RuntimeError("Could not create detected VIN crop")

            crop_filename = f"{index:04d}_{Path(filename).stem}.jpg"
            crop_file = CROP_DIR / crop_filename

            if not cv2.imwrite(str(crop_file), crop):
                raise RuntimeError("Could not save predicted VIN crop")

            crop_path = str(crop_file.relative_to(ROOT))

            # 4. Recognition on the predicted detection crop.
            sync_gpu()
            rec_start = time.perf_counter()

            rec_results = list(recogniser.apply(crop))

            sync_gpu()
            recognition_time = time.perf_counter() - rec_start

            if not rec_results:
                raise RuntimeError("Recogniser returned no result")

            rec_dict = result_to_dict(rec_results[0])
            predicted_text, confidence = get_recognition(rec_dict)

            status = "SUCCESS"

        except Exception as error:
            detection_failures += 1
            print(f"WARNING: {filename}: {error}")

        sync_gpu()
        elapsed = time.perf_counter() - image_start
        total_elapsed += elapsed

        distance = edit_distance(ground_truth, predicted_text)
        exact = int(ground_truth == predicted_text)

        total_edits += distance
        total_gt_characters += len(ground_truth)
        exact_matches += exact

        rows.append({
            "image": filename,
            "ground_truth": ground_truth,
            "prediction": predicted_text,
            "confidence": confidence,
            "detected_box_count": detection_boxes,
            "detection_time_s": detection_time,
            "recognition_time_s": recognition_time,
            "total_time_s": elapsed,
            "exact_match": exact,
            "edit_distance": distance,
            "status": status,
            "crop_path": crop_path,
            "bbox_x_min": bbox_x_min,
            "bbox_y_min": bbox_y_min,
            "bbox_x_max": bbox_x_max,
            "bbox_y_max": bbox_y_max,
        })

        print(
            f"[{index}/{len(samples)}] "
            f"{filename} | "
            f"GT: {ground_truth} | "
            f"Pred: {predicted_text or 'EMPTY'} | "
            f"Exact: {bool(exact)} | "
            f"Status: {status}"
        )

    # 5. Save per-image results.
    fieldnames = list(rows[0].keys()) if rows else []

    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 6. Calculate combined pipeline metrics.
    total_samples = len(rows)
    exact_accuracy = (
        exact_matches / total_samples * 100
        if total_samples else 0.0
    )

    character_accuracy = (
        (1 - total_edits / total_gt_characters) * 100
        if total_gt_characters else 0.0
    )

    cer = (
        total_edits / total_gt_characters
        if total_gt_characters else 0.0
    )

    string_error_rate = (
        (total_samples - exact_matches) / total_samples * 100
        if total_samples else 0.0
    )

    average_latency = (
        total_elapsed / total_samples
        if total_samples else 0.0
    )

    fps = (
        total_samples / total_elapsed
        if total_elapsed > 0 else 0.0
    )

    summary = f"""
EXP-VIN-COMBINED-001
Combined VIN Detection and Recognition Evaluation
=================================================

Dataset
-------
Test samples: {total_samples}
Original test images: {DATA_ROOT}
Ground-truth labels: {LABEL_FILE}

Model configuration
-------------------
Detector: PP-OCRv5 Mobile, fine-tuned EXP-VIN-DET-FT-002
Recogniser: PP-OCRv5 Mobile, fine-tuned EXP-VIN-REC-FT-001
Detection crop padding: X={X_PADDING * 100:.0f}%, Y={Y_PADDING * 100:.0f}%

Combined pipeline results
-------------------------
Exact VIN matches: {exact_matches}
Incorrect VIN predictions: {total_samples - exact_matches}
Detection/pipeline failures: {detection_failures}

Exact-string accuracy: {exact_accuracy:.4f}%
Character accuracy: {character_accuracy:.4f}%
Character Error Rate (CER): {cer:.6f}
String Error Rate: {string_error_rate:.4f}%

Total character edits: {total_edits}
Total ground-truth characters: {total_gt_characters}

Total evaluation time: {total_elapsed:.4f} seconds
Average end-to-end latency: {average_latency:.4f} seconds/image
End-to-end throughput: {fps:.4f} images/second

Important
---------
Detection failures are counted as unsuccessful VIN predictions.
Recognition uses crops generated from predicted detection coordinates.
Ground-truth coordinates are not used to create test-time crops.
"""

    SUMMARY_FILE.write_text(summary.strip() + "\n", encoding="utf-8")

    print("\n" + summary)
    print(f"Per-image CSV: {CSV_FILE}")
    print(f"Summary: {SUMMARY_FILE}")
    print("\nCombined evaluation completed.")


if __name__ == "__main__":
    main()
