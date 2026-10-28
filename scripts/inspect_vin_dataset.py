import csv
import json

from pathlib import Path
from collections import Counter
from datetime import datetime
import ast

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "YOLOv5PYTORCH_VIN"
YAML_FILE = DATASET / "data.yaml"
SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_class_names():
    """Read the class mapping from data.yaml."""
    try:
        import yaml
        with YAML_FILE.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        names = config["names"]
    except ImportError:
        # Fallback if PyYAML is not installed.
        names_line = next(
            line for line in YAML_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("names:")
        )
        names = ast.literal_eval(names_line.split(":", 1)[1].strip())

    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}
    return {int(i): str(name) for i, name in names.items()}


def inspect_label(label_path, class_names):
    """Validate YOLO rows and reconstruct characters left to right."""
    rows = []
    errors = []
    unknown_classes = set()

    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {line_number}: expected 5 values")
            continue

        try:
            class_value = float(parts[0])
            if not class_value.is_integer():
                raise ValueError("class ID is not an integer")
            class_id = int(class_value)
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"line {line_number}: non-numeric or invalid values")
            continue

        if class_id not in class_names:
            unknown_classes.add(class_id)

        if not all(0 <= v <= 1 for v in (x, y, w, h)):
            errors.append(f"line {line_number}: coordinate outside 0..1")
        if w <= 0 or h <= 0:
            errors.append(f"line {line_number}: non-positive box size")

        rows.append((x, class_id))

    rows.sort(key=lambda item: item[0])
    reconstructed = "".join(
        class_names.get(class_id, f"<{class_id}>")
        for _, class_id in rows
    )

    return rows, errors, unknown_classes, reconstructed


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET}")

    class_names = load_class_names()
    report = []

    def log(message=""):
        print(message)
        report.append(str(message))

    log("VIN DATASET AUDIT")
    log(f"Date/time: {datetime.now().isoformat(timespec='seconds')}")
    log(f"Dataset: {DATASET}")
    log(f"Declared classes: {class_names}")
    log("Note: reconstructed strings are derived from labels, not independently verified truth.")
    log()

    total_images = 0
    total_label_files = 0
    all_unknown_classes = set()
    all_label_errors = 0
    reconstructed_17 = []
    length_counts = Counter()

    for split in SPLITS:
        image_dir = DATASET / split / "images"
        label_dir = DATASET / split / "labels"

        if not image_dir.exists() or not label_dir.exists():
            log(f"[MISSING FOLDER] {split}:")
            log(f"  images folder exists: {image_dir.exists()}")
            log(f"  labels folder exists: {label_dir.exists()}")
            log()
            continue

        images = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        labels = sorted(label_dir.glob("*.txt"))

        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        missing_labels = sorted(image_stems - label_stems)
        labels_without_images = sorted(label_stems - image_stems)

        total_images += len(images)
        total_label_files += len(labels)

        log(f"=== {split.upper()} ===")
        log(f"Images: {len(images)}")
        log(f"Label files: {len(labels)}")
        log(f"Images without labels: {len(missing_labels)}")
        log(f"Label files without images: {len(labels_without_images)}")

        if missing_labels:
            log("  Missing-label examples: " + ", ".join(missing_labels[:5]))
        if labels_without_images:
            log("  Extra-label examples: " + ", ".join(labels_without_images[:5]))

        split_lengths = Counter()
        split_errors = 0
        split_unknown = set()

        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            rows, errors, unknown, text = inspect_label(
                label_path, class_names
            )
            split_lengths[len(rows)] += 1
            length_counts[len(rows)] += 1
            split_errors += len(errors)
            split_unknown.update(unknown)

            if len(rows) == 17 and not errors and not unknown:
                reconstructed_17.append(
                    (split, image_path.name, text, len(rows))
                )

        all_label_errors += split_errors
        all_unknown_classes.update(split_unknown)

        log(f"Valid parsed character boxes by image count: {dict(sorted(split_lengths.items()))}")
        log(f"Label/coordinate errors: {split_errors}")
        log(f"Unknown class IDs: {sorted(split_unknown)}")
        log()

    log("=== OVERALL SUMMARY ===")
    log(f"Total images found: {total_images}")
    log(f"Total label files found: {total_label_files}")
    log(f"Images grouped by number of parsed boxes: {dict(sorted(length_counts.items()))}")
    log(f"Total label/coordinate errors: {all_label_errors}")
    log(f"Unknown class IDs overall: {sorted(all_unknown_classes)}")
    log(f"Images with exactly 17 parsed boxes and no detected row errors: {len(reconstructed_17)}")
    log()
    log("Up to 10 reconstructed 17-character strings for manual visual verification:")

    for split, filename, text, count in reconstructed_17[:10]:
        log(f"{split} | {filename} | {text} | boxes={count}")

    log()
    log("IMPORTANT: This audit does not prove that a reconstructed VIN is correct.")
    log("Visually compare the listed strings with their original images before training.")

   
    # ---------------------------------------------------------
    # SAVE ALL AUDIT RESULTS TO A UNIQUE EXPERIMENT FOLDER
    # ---------------------------------------------------------

    # Give this run a unique ID so previous results are preserved.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    experiment_id = "EXP-DATA-AUDIT-001"

    results_dir = (
        ROOT
        / "results"
        / "dataset_audits"
        / "vin_character_v10"
        / f"{experiment_id}_{run_timestamp}"
    )

    results_dir.mkdir(parents=True, exist_ok=False)

    # 1. Save the full human-readable audit report.
    report_file = results_dir / "report.txt"
    report_file.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8"
    )

    # 2. Save candidate VIN strings in CSV format.
    # These are reconstructed from character labels and are
    # NOT independently verified ground truth.
    candidates_file = results_dir / "reconstructed_candidates.csv"

    with candidates_file.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "split",
            "image_filename",
            "candidate_vin",
            "character_box_count",
            "verified_ground_truth"
        ])

        for split, filename, text, count in reconstructed_17:
            writer.writerow([
                split,
                filename,
                text,
                count,
                "NO - visual verification required"
            ])

    # 3. Save the machine-readable summary as JSON.
    summary = {
        "experiment_id": experiment_id,
        "run_timestamp": run_timestamp,
        "dataset_id": "vin_character_v10",
        "dataset_path": str(DATASET),
        "dataset_config": str(YAML_FILE),
        "class_mapping": {
            str(class_id): name
            for class_id, name in class_names.items()
        },
        "total_images_found": total_images,
        "total_label_files_found": total_label_files,
        "box_count_distribution": {
            str(count): number
            for count, number in sorted(length_counts.items())
        },
        "total_label_errors": all_label_errors,
        "unknown_class_ids": sorted(all_unknown_classes),
        "images_with_17_boxes_and_no_detected_errors": len(
            reconstructed_17
        ),
        "candidate_strings_verified": False,
        "note": (
            "Reconstructed strings are candidates derived from "
            "character class labels sorted by bounding-box x-coordinate. "
            "Verify each against the original image before using it "
            "as recognition ground truth."
        )
    }

    summary_file = results_dir / "summary.json"
    summary_file.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8"
    )

    # 4. Save the run metadata for future reproducibility.
    metadata = {
        "experiment_id": experiment_id,
        "run_timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "python_version": __import__("sys").version,
        "platform": __import__("platform").platform(),
        "script": str(Path(__file__).resolve()),
        "results_directory": str(results_dir),
        "dataset_split_names": SPLITS,
        "result_files": [
            "report.txt",
            "reconstructed_candidates.csv",
            "summary.json",
            "run_metadata.json"
        ]
    }

    metadata_file = results_dir / "run_metadata.json"
    metadata_file.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8"
    )

    # Show the saved paths in CMD.
    print("\n" + "=" * 60)
    print("VIN DATASET AUDIT COMPLETED")
    print("=" * 60)
    print(f"Experiment ID: {experiment_id}")
    print(f"Images found: {total_images}")
    print(f"Label files found: {total_label_files}")
    print(f"Detected label errors: {all_label_errors}")
    print(f"\nResults folder:\n{results_dir}")
    print("\nFiles saved:")
    print(f"1. {report_file}")
    print(f"2. {candidates_file}")
    print(f"3. {summary_file}")
    print(f"4. {metadata_file}")
    print("=" * 60)



if __name__ == "__main__":
    main()

