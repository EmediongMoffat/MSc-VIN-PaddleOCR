
"""
Experiment: EXP-YOLO-PPOCR-CONVERT-001

Convert YOLO character-level VIN annotations into PaddleOCR formats.

Outputs:
- PaddleOCR detection labels for train, valid and test.
- Copies of source images for detection.
- Full-VIN recognition crops and UNVERIFIED candidate labels.
- Crops for nonstandard/partially visible VINs, where possible.
- CSV manifests, errors, summary and run metadata.

IMPORTANT:
Recognition strings reconstructed from character labels are candidates,
not verified ground truth. Do not train recognition until they are checked.
Original dataset files are never modified.
"""

from pathlib import Path
from datetime import datetime
from collections import Counter
from PIL import Image
import ast
import csv
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import time


# ============================================================
# 1. PROJECT PATHS AND EXPERIMENT SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data" / "raw" / "YOLOv5PYTORCH_VIN"
YAML_FILE = DATASET / "data.yaml"

OUTPUT_BASE = ROOT / "data" / "processed" / "paddleocr_vin"

SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

EXPECTED_VIN_LENGTH = 17
EXPERIMENT_ID = "EXP-YOLO-PPOCR-CONVERT-001"

DATASET_URL = (
    "https://universe.roboflow.com/"
    "vin-9rops/vin-character-d9q0c/dataset/10"
)


# ============================================================
# 2. LOAD CLASS MAPPING FROM data.yaml
# ============================================================

def load_dataset_config():
    if not YAML_FILE.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {YAML_FILE}"
        )

    text = YAML_FILE.read_text(encoding="utf-8")

    try:
        import yaml
        config = yaml.safe_load(text) or {}
    except ImportError:
        # Fallback for the current data.yaml where names is a list.
        match = re.search(
            r"(?m)^names:\s*(\[.*\])\s*$",
            text
        )
        if not match:
            raise RuntimeError(
                "Cannot parse class names. Install PyYAML in the venv "
                "or check the names field in data.yaml."
            )

        config = {"names": ast.literal_eval(match.group(1))}

    names = config.get("names")

    if isinstance(names, list):
        class_names = {
            class_id: str(name)
            for class_id, name in enumerate(names)
        }
    elif isinstance(names, dict):
        class_names = {
            int(class_id): str(name)
            for class_id, name in names.items()
        }
    else:
        raise ValueError(
            "Expected data.yaml names to be a list or dictionary."
        )

    return config, class_names


# ============================================================
# 3. RECORD GIT VERSION WHEN AVAILABLE
# ============================================================

def get_git_commit(folder):
    if not folder.exists():
        return None

    try:
        result = subprocess.run(
            [
                "git", "-C", str(folder),
                "rev-parse", "--short", "HEAD"
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if result.returncode == 0:
            return result.stdout.strip()

    except (OSError, subprocess.TimeoutExpired):
        pass

    return None


# ============================================================
# 4. PARSE YOLO CHARACTER BOXES
# ============================================================

def parse_yolo_label(label_path, image_width, image_height, class_names):
    """
    YOLO row:
    class_id x_center y_center width height

    Returns valid character boxes and row-level errors.
    Coordinates are converted into image pixel coordinates.
    """

    boxes = []
    errors = []

    if not label_path.exists():
        return boxes, ["Missing label file"]

    lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            errors.append(
                f"Line {line_number}: expected 5 fields, "
                f"found {len(parts)}"
            )
            continue

        try:
            class_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
        except ValueError:
            errors.append(
                f"Line {line_number}: non-numeric YOLO values"
            )
            continue

        if class_id not in class_names:
            errors.append(
                f"Line {line_number}: unknown class ID {class_id}"
            )
            continue

        values = [cx, cy, bw, bh]

        if not all(math.isfinite(value) for value in values):
            errors.append(
                f"Line {line_number}: non-finite coordinate"
            )
            continue

        if not (
            0 <= cx <= 1
            and 0 <= cy <= 1
            and 0 < bw <= 1
            and 0 < bh <= 1
        ):
            errors.append(
                f"Line {line_number}: coordinates outside "
                f"expected normalized ranges"
            )
            continue

        x1 = (cx - bw / 2) * image_width
        y1 = (cy - bh / 2) * image_height
        x2 = (cx + bw / 2) * image_width
        y2 = (cy + bh / 2) * image_height

        # Clip boxes to image boundaries.
        x1 = max(0.0, min(float(image_width), x1))
        y1 = max(0.0, min(float(image_height), y1))
        x2 = max(0.0, min(float(image_width), x2))
        y2 = max(0.0, min(float(image_height), y2))

        if x2 <= x1 or y2 <= y1:
            errors.append(
                f"Line {line_number}: empty box after clipping"
            )
            continue

        boxes.append({
            "class_id": class_id,
            "character": class_names[class_id],
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x": (x1 + x2) / 2,
            "center_y": (y1 + y2) / 2,
        })

    # VIN characters are arranged from left to right.
    boxes.sort(key=lambda box: (box["center_x"], box["center_y"]))

    return boxes, errors


# ============================================================
# 5. GROUP CHARACTER BOXES INTO ONE VIN REGION
# ============================================================

def make_detection_annotation(boxes, image_width, image_height):
    """
    Combine the visible character boxes into one axis-aligned
    rectangular VIN text region.

    This is suitable only when the characters form one VIN line.
    """

    if not boxes:
        return []

    left = min(box["x1"] for box in boxes)
    top = min(box["y1"] for box in boxes)
    right = max(box["x2"] for box in boxes)
    bottom = max(box["y2"] for box in boxes)

    # Small padding around the combined detection region.
    pad_x = max(1, round((right - left) * 0.005))
    pad_y = max(1, round((bottom - top) * 0.10))

    x1 = max(0, int(math.floor(left - pad_x)))
    y1 = max(0, int(math.floor(top - pad_y)))
    x2 = min(image_width, int(math.ceil(right + pad_x)))
    y2 = min(image_height, int(math.ceil(bottom + pad_y)))

    visible_text = "".join(
        box["character"] for box in boxes
    )

    annotation = {
        "transcription": visible_text,
        "points": [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
    }

    return [annotation]


# ============================================================
# 6. CREATE A CROP AROUND THE VISIBLE VIN CHARACTERS
# ============================================================

def save_vin_crop(image_path, boxes, destination):
    if not boxes:
        return False

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_width, image_height = image.size

        left = min(box["x1"] for box in boxes)
        top = min(box["y1"] for box in boxes)
        right = max(box["x2"] for box in boxes)
        bottom = max(box["y2"] for box in boxes)

        box_width = right - left
        box_height = bottom - top

        # Add padding so character edges are less likely to be cut off.
        pad_x = max(3, round(box_width * 0.02))
        pad_y = max(3, round(box_height * 0.30))

        crop_box = (
            max(0, int(math.floor(left - pad_x))),
            max(0, int(math.floor(top - pad_y))),
            min(image_width, int(math.ceil(right + pad_x))),
            min(image_height, int(math.ceil(bottom + pad_y))),
        )

        if (
            crop_box[2] <= crop_box[0]
            or crop_box[3] <= crop_box[1]
        ):
            return False

        crop = image.crop(crop_box)

        destination.parent.mkdir(parents=True, exist_ok=True)
        crop.save(destination, format="JPEG", quality=95)

    return True


# ============================================================
# 7. CSV HELPER
# ============================================================

def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 8. MAIN CONVERSION
# ============================================================

def main():
    started = time.perf_counter()
    start_datetime = datetime.now()

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET}"
        )

    config, class_names = load_dataset_config()

    timestamp = start_datetime.strftime("%Y%m%d_%H%M%S_%f")

    output_dir = (
        OUTPUT_BASE / f"{EXPERIMENT_ID}_{timestamp}"
    )

    # A unique folder is created for every run.
    output_dir.mkdir(parents=True, exist_ok=False)

    detection_dir = output_dir / "detection"
    recognition_dir = output_dir / "recognition"

    for split in SPLITS:
        (detection_dir / "images" / split).mkdir(
            parents=True, exist_ok=True
        )
        (recognition_dir / "crops" / split).mkdir(
            parents=True, exist_ok=True
        )
        (recognition_dir / "review_crops" / split).mkdir(
            parents=True, exist_ok=True
        )

    total_images = 0
    total_labels_found = 0
    total_row_errors = 0
    total_detection_regions = 0
    total_rec_candidates = 0
    total_review_images = 0
    total_review_crops = 0

    split_summary = {}
    class_counts = Counter()
    box_count_distribution = Counter()

    candidate_rows = []
    review_rows = []
    error_rows = []

    detection_label_paths = {}
    recognition_label_paths = {}

    # --------------------------------------------------------
    # Process train, valid and test independently.
    # --------------------------------------------------------

    for split in SPLITS:
        image_dir = DATASET / split / "images"
        label_dir = DATASET / split / "labels"

        if not image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {image_dir}"
            )

        images = sorted(
            path for path in image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not images:
            raise RuntimeError(
                f"No images found in {image_dir}"
            )

        det_label_file = (
            detection_dir / f"det_gt_{split}.txt"
        )
        rec_label_file = (
            recognition_dir
            / f"rec_gt_{split}_CANDIDATES_UNVERIFIED.txt"
        )

        detection_label_paths[split] = str(det_label_file)
        recognition_label_paths[split] = str(rec_label_file)

        detection_lines = []
        recognition_lines = []

        split_boxes = Counter()
        split_missing_labels = 0
        split_errors = 0
        split_full_candidates = 0
        split_review_images = 0
        split_regions = 0

        for image_path in images:
            total_images += 1

            label_path = label_dir / f"{image_path.stem}.txt"
            label_exists = label_path.exists()

            if label_exists:
                total_labels_found += 1
            else:
                split_missing_labels += 1

            try:
                with Image.open(image_path) as image:
                    image_width, image_height = image.size
            except Exception as exc:
                error_rows.append({
                    "split": split,
                    "image": image_path.name,
                    "label": str(label_path),
                    "error": f"Cannot open image: {exc}",
                })
                continue

            boxes, errors = parse_yolo_label(
                label_path,
                image_width,
                image_height,
                class_names,
            )

            for error in errors:
                if error == "Missing label file":
                    continue

                split_errors += 1
                total_row_errors += 1

                error_rows.append({
                    "split": split,
                    "image": image_path.name,
                    "label": str(label_path),
                    "error": error,
                })

            box_count = len(boxes)
            split_boxes[box_count] += 1
            box_count_distribution[box_count] += 1

            for box in boxes:
                class_counts[box["character"]] += 1

            candidate_text = "".join(
                box["character"] for box in boxes
            )

            # Copy the original image into the converted dataset.
            det_image = (
                detection_dir / "images" / split / image_path.name
            )
            shutil.copy2(image_path, det_image)

            det_relative_path = (
                Path("images") / split / image_path.name
            ).as_posix()

            det_annotations = make_detection_annotation(
                boxes, image_width, image_height
            )

            if det_annotations:
                split_regions += 1
                total_detection_regions += 1

            detection_lines.append(
                det_relative_path
                + "\t"
                + json.dumps(
                    det_annotations,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )

            # Only 17 valid boxes with no row errors are eligible
            # for full-VIN recognition candidate crops.
            is_full_candidate = (
                label_exists
                and len(boxes) == EXPECTED_VIN_LENGTH
                and len(candidate_text) == EXPECTED_VIN_LENGTH
                and len(errors) == 0
            )

            if is_full_candidate:
                crop_path = (
                    recognition_dir
                    / "crops"
                    / split
                    / f"{image_path.stem}_vin.jpg"
                )

                crop_saved = save_vin_crop(
                    image_path, boxes, crop_path
                )

                if crop_saved:
                    crop_relative_path = (
                        Path("crops")
                        / split
                        / crop_path.name
                    ).as_posix()

                    recognition_lines.append(
                        crop_relative_path + "\t" + candidate_text
                    )

                    candidate_rows.append({
                        "split": split,
                        "source_image": image_path.name,
                        "crop_path": crop_relative_path,
                        "candidate_text": candidate_text,
                        "character_boxes": len(boxes),
                        "label_errors": len(errors),
                        "ground_truth_verified": "NO",
                        "status": "UNVERIFIED CANDIDATE",
                    })

                    split_full_candidates += 1
                    total_rec_candidates += 1

            else:
                split_review_images += 1
                total_review_images += 1

                review_crop_relative_path = ""

                # Save a partial crop when at least one character
                # box is available. A zero-box image has no crop.
                if boxes:
                    review_crop_path = (
                        recognition_dir
                        / "review_crops"
                        / split
                        / f"{image_path.stem}_review.jpg"
                    )

                    crop_saved = save_vin_crop(
                        image_path, boxes, review_crop_path
                    )

                    if crop_saved:
                        review_crop_relative_path = (
                            Path("review_crops")
                            / split
                            / review_crop_path.name
                        ).as_posix()
                        total_review_crops += 1

                review_rows.append({
                    "split": split,
                    "source_image": image_path.name,
                    "source_label": str(label_path),
                    "label_exists": label_exists,
                    "valid_character_boxes": len(boxes),
                    "visible_label_candidate": candidate_text,
                    "row_errors": len(errors),
                    "review_crop_path": review_crop_relative_path,
                    "ground_truth_verified": "NO",
                    "note": (
                        "Inspect image and annotations before use. "
                        "Do not use as full-VIN recognition ground truth."
                    ),
                })

        # Write the PaddleOCR detection label file.
        det_label_file.write_text(
            "\n".join(detection_lines) + "\n",
            encoding="utf-8",
        )

        # Write standard recognition-format candidate labels,
        # explicitly marked as unverified in the filename.
        rec_label_file.write_text(
            "\n".join(recognition_lines)
            + ("\n" if recognition_lines else ""),
            encoding="utf-8",
        )

        split_summary[split] = {
            "images": len(images),
            "label_files_found": len(images) - split_missing_labels,
            "missing_label_files": split_missing_labels,
            "valid_character_boxes_per_image": {
                str(count): number
                for count, number in sorted(split_boxes.items())
            },
            "detection_regions_created": split_regions,
            "full_17_character_recognition_candidates": (
                split_full_candidates
            ),
            "images_requiring_review": split_review_images,
            "row_errors": split_errors,
            "detection_label_file": str(det_label_file),
            "recognition_candidate_label_file": str(rec_label_file),
        }

    # --------------------------------------------------------
    # Save manifests, errors, summary and run metadata.
    # --------------------------------------------------------

    write_csv(
        recognition_dir / "recognition_candidates.csv",
        [
            "split",
            "source_image",
            "crop_path",
            "candidate_text",
            "character_boxes",
            "label_errors",
            "ground_truth_verified",
            "status",
        ],
        candidate_rows,
    )

    write_csv(
        recognition_dir / "nonstandard_images_for_review.csv",
        [
            "split",
            "source_image",
            "source_label",
            "label_exists",
            "valid_character_boxes",
            "visible_label_candidate",
            "row_errors",
            "review_crop_path",
            "ground_truth_verified",
            "note",
        ],
        review_rows,
    )

    write_csv(
        output_dir / "annotation_errors.csv",
        ["split", "image", "label", "error"],
        error_rows,
    )

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": start_datetime.isoformat(timespec="seconds"),
        "source_dataset": str(DATASET),
        "source_data_yaml": str(YAML_FILE),
        "source_dataset_url": config.get("roboflow", {}).get(
            "url", DATASET_URL
        ),
        "dataset_license": config.get("roboflow", {}).get(
            "license", "Check source data.yaml"
        ),
        "dataset_version": config.get("roboflow", {}).get(
            "version", "Check source data.yaml"
        ),
        "class_mapping": {
            str(class_id): name
            for class_id, name in class_names.items()
        },
        "total_images_found": total_images,
        "total_label_files_found": total_labels_found,
        "total_missing_label_files": total_images - total_labels_found,
        "total_detection_regions_created": total_detection_regions,
        "total_full_vin_recognition_candidates": total_rec_candidates,
        "total_images_requiring_review": total_review_images,
        "total_review_crops_created": total_review_crops,
        "total_row_errors": total_row_errors,
        "character_class_distribution": dict(
            sorted(class_counts.items())
        ),
        "box_count_distribution": {
            str(count): number
            for count, number in sorted(box_count_distribution.items())
        },
        "recognition_ground_truth_verified": False,
        "split_summary": split_summary,
        "output_directory": str(output_dir),
        "important_note": (
            "Recognition labels are candidate strings reconstructed "
            "from YOLO character classes sorted left-to-right. They "
            "are not independently verified ground truth. Detection "
            "regions are axis-aligned bounding boxes formed from the "
            "union of visible character boxes. Review outputs before "
            "training."
        ),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "run_started": start_datetime.isoformat(timespec="seconds"),
        "run_finished": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "python_version": sys.version,
        "platform": platform.platform(),
        "script_path": str(Path(__file__).resolve()),
        "project_git_commit": get_git_commit(ROOT),
        "paddleocr_git_commit": get_git_commit(ROOT / "PaddleOCR"),
        "source_dataset": str(DATASET),
        "source_data_yaml": str(YAML_FILE),
        "output_directory": str(output_dir),
        "splits": SPLITS,
        "expected_vin_length": EXPECTED_VIN_LENGTH,
        "recognition_candidates_verified": False,
    }

    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    report = [
        "YOLO TO PADDLEOCR DATA CONVERSION",
        "=" * 65,
        f"Experiment ID: {EXPERIMENT_ID}",
        f"Date/time: {start_datetime.isoformat(timespec='seconds')}",
        f"Source dataset: {DATASET}",
        f"Output directory: {output_dir}",
        "",
        "RESULTS",
        f"Images found: {total_images}",
        f"Label files found: {total_labels_found}",
        f"Detection regions created: {total_detection_regions}",
        f"Full-VIN recognition candidates: {total_rec_candidates}",
        f"Images requiring review: {total_review_images}",
        f"Review crops created: {total_review_crops}",
        f"Label row errors: {total_row_errors}",
        "",
        "IMPORTANT",
        "Recognition candidates are UNVERIFIED.",
        "Do not train recognition using candidate labels until their",
        "strings have been visually verified against the source images.",
        "",
        "OUTPUTS",
        f"Detection folder: {detection_dir}",
        f"Recognition folder: {recognition_dir}",
        f"Summary JSON: {output_dir / 'summary.json'}",
        f"Run metadata: {output_dir / 'run_metadata.json'}",
        f"Annotation errors: {output_dir / 'annotation_errors.csv'}",
    ]

    (output_dir / "conversion_report.txt").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("YOLO TO PADDLEOCR CONVERSION COMPLETED")
    print("=" * 65)
    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Images processed: {total_images}")
    print(f"Detection regions: {total_detection_regions}")
    print(f"Recognition candidates (UNVERIFIED): {total_rec_candidates}")
    print(f"Images requiring review: {total_review_images}")
    print(f"Annotation row errors: {total_row_errors}")
    print(f"\nResults saved to:\n{output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
