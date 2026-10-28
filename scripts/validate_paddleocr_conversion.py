
from pathlib import Path
from datetime import datetime
from collections import Counter
import json
import re

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_BASE = ROOT / "data" / "processed" / "paddleocr_vin"
EXPERIMENT_PREFIX = "EXP-YOLO-PPOCR-CONVERT-001_"
SPLITS = ["train", "valid", "test"]

# Standard VIN characters: digits and letters excluding I, O and Q.
VIN_PATTERN = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")


def main():
    experiments = sorted(
        path for path in OUTPUT_BASE.glob(EXPERIMENT_PREFIX + "*")
        if path.is_dir()
    )

    if not experiments:
        raise FileNotFoundError(
            f"No conversion experiment found in {OUTPUT_BASE}"
        )

    experiment = experiments[-1]
    detection_dir = experiment / "detection"
    recognition_dir = experiment / "recognition"

    errors = []
    warnings = []
    results = {}

    # ---------------------------------------------------------
    # DETECTION ANNOTATIONS
    # ---------------------------------------------------------
    for split in SPLITS:
        label_file = detection_dir / f"det_gt_{split}.txt"
        checked = 0
        empty_annotations = 0
        invalid_annotations = 0

        if not label_file.exists():
            errors.append(f"Missing detection label file: {label_file}")
            results[split] = {}
            continue

        for line_number, line in enumerate(
            label_file.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                warnings.append(
                    f"{split} detection line {line_number}: blank line"
                )
                continue

            parts = line.split("\t", 1)
            if len(parts) != 2:
                errors.append(
                    f"{split} detection line {line_number}: "
                    "expected image path and JSON separated by a tab"
                )
                continue

            relative_path, json_text = parts
            image_path = detection_dir / Path(relative_path)

            if not image_path.is_file():
                errors.append(
                    f"{split} detection line {line_number}: "
                    f"image not found: {image_path}"
                )
                continue

            try:
                annotations = json.loads(json_text)
            except json.JSONDecodeError as exc:
                errors.append(
                    f"{split} detection line {line_number}: "
                    f"invalid JSON: {exc}"
                )
                continue

            if not isinstance(annotations, list):
                errors.append(
                    f"{split} detection line {line_number}: "
                    "annotation JSON must be a list"
                )
                continue

            checked += 1

            if len(annotations) == 0:
                empty_annotations += 1
                warnings.append(
                    f"{split} detection line {line_number}: "
                    f"empty annotation for {relative_path}"
                )
                continue

            try:
                with Image.open(image_path) as image:
                    width, height = image.size
            except Exception as exc:
                errors.append(
                    f"{split} detection line {line_number}: "
                    f"cannot read image: {exc}"
                )
                continue

            for annotation in annotations:
                if not isinstance(annotation, dict):
                    errors.append(
                        f"{split} detection line {line_number}: "
                        "annotation is not a JSON object"
                    )
                    invalid_annotations += 1
                    continue

                points = annotation.get("points")
                transcription = annotation.get("transcription")

                if (
                    not isinstance(points, list)
                    or len(points) != 4
                    or not isinstance(transcription, str)
                ):
                    errors.append(
                        f"{split} detection line {line_number}: "
                        "expected four points and string transcription"
                    )
                    invalid_annotations += 1
                    continue

                valid_points = True
                for point in points:
                    if not isinstance(point, list) or len(point) != 2:
                        valid_points = False
                        break

                    x, y = point
                    if (
                        not isinstance(x, (int, float))
                        or not isinstance(y, (int, float))
                        or not (0 <= x <= width)
                        or not (0 <= y <= height)
                    ):
                        valid_points = False
                        break

                if not valid_points:
                    errors.append(
                        f"{split} detection line {line_number}: "
                        "invalid polygon coordinates"
                    )
                    invalid_annotations += 1

        results[split] = {
            "detection_label_rows_with_existing_images": checked,
            "empty_detection_annotations": empty_annotations,
            "invalid_detection_annotations": invalid_annotations,
        }

    # ---------------------------------------------------------
    # RECOGNITION CANDIDATES
    # ---------------------------------------------------------
    recognition_results = {}

    for split in SPLITS:
        label_file = (
            recognition_dir
            / f"rec_gt_{split}_CANDIDATES_UNVERIFIED.txt"
        )

        checked = 0
        invalid_length = 0
        invalid_characters = 0

        if not label_file.exists():
            errors.append(f"Missing recognition label file: {label_file}")
            recognition_results[split] = {}
            continue

        for line_number, line in enumerate(
            label_file.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue

            parts = line.split("\t", 1)
            if len(parts) != 2:
                errors.append(
                    f"{split} recognition line {line_number}: "
                    "expected image path and text separated by a tab"
                )
                continue

            relative_path, text = parts
            crop_path = recognition_dir / Path(relative_path)

            if not crop_path.is_file():
                errors.append(
                    f"{split} recognition line {line_number}: "
                    f"crop not found: {crop_path}"
                )
                continue

            try:
                with Image.open(crop_path) as image:
                    image.verify()
            except Exception as exc:
                errors.append(
                    f"{split} recognition line {line_number}: "
                    f"unreadable crop: {exc}"
                )
                continue

            checked += 1

            if len(text) != 17:
                invalid_length += 1
                warnings.append(
                    f"{split} recognition line {line_number}: "
                    f"label length is {len(text)}, not 17"
                )

            if not VIN_PATTERN.fullmatch(text):
                invalid_characters += 1
                warnings.append(
                    f"{split} recognition line {line_number}: "
                    "label contains characters outside the standard "
                    "17-character VIN alphabet"
                )

        recognition_results[split] = {
            "recognition_crops_found_and_readable": checked,
            "labels_with_length_not_17": invalid_length,
            "labels_with_nonstandard_vin_characters": invalid_characters,
        }

    # ---------------------------------------------------------
    # SAVE VALIDATION RESULTS
    # ---------------------------------------------------------
    summary = {
        "validation_id": "EXP-PPOCR-CONVERSION-VALIDATE-001",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "conversion_experiment": str(experiment),
        "detection_results": results,
        "recognition_results": recognition_results,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "recognition_ground_truth_verified": False,
        "note": (
            "This validates file paths, basic annotation structure, "
            "polygon bounds and candidate label lengths/characters. "
            "It does not verify that labels match the images or that "
            "the dataset loader can train successfully."
        ),
    }

    summary_path = experiment / "validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    report_path = experiment / "validation_report.txt"
    report = [
        "PADDLEOCR CONVERSION VALIDATION",
        "=" * 60,
        f"Experiment: {experiment}",
        f"Validation time: {summary['timestamp']}",
        f"Errors: {len(errors)}",
        f"Warnings: {len(warnings)}",
        "",
        "DETECTION RESULTS",
        json.dumps(results, indent=2),
        "",
        "RECOGNITION RESULTS",
        json.dumps(recognition_results, indent=2),
        "",
        "ERRORS",
        *(errors if errors else ["None detected"]),
        "",
        "WARNINGS",
        *(warnings if warnings else ["None detected"]),
        "",
        "Recognition labels are still unverified candidates.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\nPADDLEOCR CONVERSION VALIDATION")
    print("=" * 60)
    print(f"Experiment: {experiment.name}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    print("\nDetection:")
    print(json.dumps(results, indent=2))
    print("\nRecognition:")
    print(json.dumps(recognition_results, indent=2))
    print(f"\nReport saved: {report_path}")
    print(f"Summary saved: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
