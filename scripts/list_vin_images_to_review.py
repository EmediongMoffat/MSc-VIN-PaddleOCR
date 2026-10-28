
from pathlib import Path
from datetime import datetime
import csv

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "YOLOv5PYTORCH_VIN"

SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED_BOXES = 17


def count_valid_boxes(label_path):
    """Count YOLO rows containing a class ID and four numeric values."""
    if not label_path.exists():
        return None

    count = 0

    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            continue

        try:
            int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            continue

        if all(value == value for value in values):
            count += 1

    return count


def main():
    review_rows = []
    checked_images = 0

    for split in SPLITS:
        image_dir = DATASET / split / "images"
        label_dir = DATASET / split / "labels"

        if not image_dir.exists():
            print(f"WARNING: Image folder not found: {image_dir}")
            continue

        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            checked_images += 1
            label_path = label_dir / f"{image_path.stem}.txt"
            box_count = count_valid_boxes(label_path)

            if box_count != EXPECTED_BOXES:
                if box_count is None:
                    issue = "Missing label file"
                elif box_count == 0:
                    issue = "Zero parsed character boxes"
                else:
                    issue = f"Expected 17 boxes; found {box_count}"

                review_rows.append({
                    "split": split,
                    "image_filename": image_path.name,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "label_exists": label_path.exists(),
                    "parsed_box_count": (
                        "MISSING LABEL" if box_count is None else box_count
                    ),
                    "issue": issue,
                    "visually_verified": "NO",
                })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        ROOT / "results" / "dataset_audits" / "vin_character_v10"
        / f"EXP-DATA-REVIEW-001_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    csv_path = output_dir / "images_to_review.csv"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "split",
            "image_filename",
            "image_path",
            "label_path",
            "label_exists",
            "parsed_box_count",
            "issue",
            "visually_verified",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)

    report_path = output_dir / "report.txt"
    report_path.write_text(
        f"VIN IMAGE REVIEW LIST\n"
        f"Date/time: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Dataset: {DATASET}\n"
        f"Images checked: {checked_images}\n"
        f"Images requiring review: {len(review_rows)}\n"
        f"Expected character boxes per VIN: {EXPECTED_BOXES}\n"
        f"CSV report: {csv_path}\n"
        f"Note: this identifies files for review; it does not verify "
        f"their visual annotations.\n",
        encoding="utf-8",
    )

    print("\nVIN IMAGE REVIEW LIST")
    print("=" * 60)
    print(f"Images checked: {checked_images}")
    print(f"Images requiring review: {len(review_rows)}")

    for row in review_rows:
        print(
            f"{row['split']} | {row['image_filename']} | "
            f"boxes={row['parsed_box_count']} | {row['issue']}"
        )

    print(f"\nCSV saved to:\n{csv_path}")
    print(f"Report saved to:\n{report_path}")


if __name__ == "__main__":
    main()

