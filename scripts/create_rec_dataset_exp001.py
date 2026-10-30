import json
import cv2
import numpy as np
from pathlib import Path

# ============================================================
# EXP-VIN-REC-001
# Recognition dataset from EXP-VIN-DET-FT-002 predictions
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "paddleocr_vin"
    / "EXP-YOLO-PPOCR-CONVERT-001_20261028_132836_267609"
)

DETECTION_DIR = SOURCE_DIR / "detection"
RECOGNITION_DIR = SOURCE_DIR / "recognition"

DETECTOR_RESULTS = (
    PROJECT_DIR
    / "results"
    / "experiments"
    / "EXP-VIN-DET-FT-002"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "paddleocr_vin"
    / "EXP-VIN-REC-001"
)

SPLITS = {
    "train": {
        "expected": 460,
        "prediction": DETECTOR_RESULTS / "train" / "full_train_predictions_expanded.txt",
        "ground_truth": DETECTION_DIR / "det_gt_train.txt",
        "candidate": RECOGNITION_DIR / "rec_gt_train_CANDIDATES_UNVERIFIED.txt",
        "label_file": "train_rec.txt",
    },
    "valid": {
        "expected": 101,
        "prediction": DETECTOR_RESULTS / "validation" / "full_validation_predictions_expanded.txt",
        "ground_truth": DETECTION_DIR / "det_gt_valid.txt",
        "candidate": RECOGNITION_DIR / "rec_gt_valid_CANDIDATES_UNVERIFIED.txt",
        "label_file": "val_rec.txt",
    },
    "test": {
        "expected": 51,
        "prediction": DETECTOR_RESULTS / "test" / "full_test_predictions_expanded.txt",
        "ground_truth": DETECTION_DIR / "det_gt_test.txt",
        "candidate": RECOGNITION_DIR / "rec_gt_test_CANDIDATES_UNVERIFIED.txt",
        "label_file": "test_rec.txt",
    },
}


def read_ground_truth(file_path):
    """Read original PaddleOCR detection annotations."""
    records = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid ground-truth line {line_number}: {line}"
                )

            image_name = Path(parts[0].replace("\\", "/")).name
            annotations = json.loads(parts[1])

            transcriptions = [
                item["transcription"].strip()
                for item in annotations
                if item.get("transcription", "").strip()
            ]

            if len(transcriptions) != 1:
                raise ValueError(
                    f"Expected exactly one VIN transcription for "
                    f"{image_name}; found {len(transcriptions)}"
                )

            records[image_name] = transcriptions[0]

    return records


def read_candidate_labels(file_path):
    """Read and index the previously reviewed candidate labels."""
    records = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid candidate line {line_number}: {line}"
                )

            candidate_path = Path(parts[0].replace("\\", "/"))
            candidate_stem = candidate_path.stem

            if candidate_stem.endswith("_vin"):
                candidate_stem = candidate_stem[:-4]

            image_name = candidate_stem + candidate_path.suffix
            transcription = parts[1].strip()

            if image_name in records:
                raise ValueError(f"Duplicate candidate: {image_name}")

            records[image_name] = transcription

    return records


def read_predictions(file_path):
    """Read detector predictions, retaining expanded points."""
    records = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line:
                continue

            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid prediction line {line_number}: {line}"
                )

            image_path = Path(parts[0].strip())
            image_name = image_path.name
            detections = json.loads(parts[1])

            if len(detections) != 1:
                raise ValueError(
                    f"Expected one VIN detection for {image_name}; "
                    f"found {len(detections)}"
                )

            points = detections[0].get("points")

            if not points or len(points) != 4:
                raise ValueError(
                    f"Invalid expanded quadrilateral for {image_name}"
                )

            if image_name in records:
                raise ValueError(f"Duplicate prediction: {image_name}")

            records[image_name] = {
                "image_path": image_path,
                "points": points,
            }

    return records


def extract_crop(image_path, points, output_path):
    """Extract a perspective-corrected crop from expanded coordinates."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    source_points = np.asarray(points, dtype=np.float32)

    # Expected point order: top-left, top-right,
    # bottom-right, bottom-left.
    top_width = np.linalg.norm(
        source_points[1] - source_points[0]
    )
    bottom_width = np.linalg.norm(
        source_points[2] - source_points[3]
    )
    left_height = np.linalg.norm(
        source_points[3] - source_points[0]
    )
    right_height = np.linalg.norm(
        source_points[2] - source_points[1]
    )

    width = max(1, int(round(max(top_width, bottom_width))))
    height = max(1, int(round(max(left_height, right_height))))

    destination_points = np.asarray(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(
        source_points,
        destination_points,
    )

    crop = cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), crop):
        raise RuntimeError(f"Could not save crop: {output_path}")


def process_split(split_name, config):
    print(f"\nProcessing {split_name.upper()} split...")

    for file_path in (
        config["prediction"],
        config["ground_truth"],
        config["candidate"],
    ):
        if not file_path.is_file():
            raise FileNotFoundError(f"Required file missing: {file_path}")

    predictions = read_predictions(config["prediction"])
    ground_truth = read_ground_truth(config["ground_truth"])
    candidates = read_candidate_labels(config["candidate"])

    expected = config["expected"]

    print(f"Predictions: {len(predictions)}")
    print(f"Ground-truth labels: {len(ground_truth)}")
    print(f"Candidate labels: {len(candidates)}")

    if len(predictions) != expected:
        raise ValueError(
            f"{split_name}: expected {expected} predictions, "
            f"found {len(predictions)}"
        )

    if set(predictions) != set(ground_truth):
        missing_gt = sorted(set(predictions) - set(ground_truth))
        missing_pred = sorted(set(ground_truth) - set(predictions))
        raise ValueError(
            f"{split_name}: prediction/ground-truth mismatch.\n"
            f"Missing ground truth: {missing_gt[:10]}\n"
            f"Missing predictions: {missing_pred[:10]}"
        )

    if set(ground_truth) != set(candidates):
        raise ValueError(
            f"{split_name}: original ground-truth and candidate "
            f"image filenames do not match."
        )

    for image_name, true_vin in ground_truth.items():
        candidate_vin = candidates[image_name]

        if true_vin != candidate_vin:
            raise ValueError(
                f"VIN transcription mismatch for {image_name}:\n"
                f"Original:  {true_vin}\n"
                f"Candidate: {candidate_vin}"
            )

        if len(true_vin) != 17:
            raise ValueError(
                f"VIN is not 17 characters for {image_name}: {true_vin}"
            )

    split_dir = OUTPUT_DIR / split_name
    crop_dir = split_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    label_path = split_dir / config["label_file"]

    written = 0

    with open(label_path, "w", encoding="utf-8", newline="\n") as labels:
        for image_name in sorted(predictions):
            prediction = predictions[image_name]
            image_path = prediction["image_path"]

            if not image_path.is_absolute():
                image_path = PROJECT_DIR / image_path

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Prediction image does not exist: {image_path}"
                )

            crop_name = Path(image_name).stem + "_detcrop.jpg"
            crop_path = crop_dir / crop_name

            extract_crop(
                image_path,
                prediction["points"],
                crop_path,
            )

            relative_crop_path = (
                Path(split_name) / "crops" / crop_name
            ).as_posix()

            labels.write(
                f"{relative_crop_path}\t{ground_truth[image_name]}\n"
            )

            written += 1

    print(f"Crops generated: {written}")
    print(f"Recognition labels: {label_path}")

    return written


def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty:\n{OUTPUT_DIR}\n"
            "To protect existing data, this script will not overwrite it."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0

    for split_name, config in SPLITS.items():
        total += process_split(split_name, config)

    print("\n" + "=" * 60)
    print("EXP-VIN-REC-001 DATASET GENERATION COMPLETED")
    print("=" * 60)
    print(f"Training crops:   460 expected")
    print(f"Validation crops: 101 expected")
    print(f"Test crops:        51 expected")
    print(f"Total crops:       {total}")
    print(f"Output directory:  {OUTPUT_DIR}")
    print("\nAll original labels were matched against candidate labels.")
    print("All crops were extracted using expanded detector predictions.")


if __name__ == "__main__":
    main()