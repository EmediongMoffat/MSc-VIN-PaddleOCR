
from pathlib import Path
import json

PROJECT_DIR = Path(__file__).resolve().parent.parent


def prepare_labels(split):
    image_dir = PROJECT_DIR / "dataset" / split
    annotation_dir = PROJECT_DIR / f"dataset" / f"{split}_gt"
    output_file = PROJECT_DIR / "data" / "raw" / f"det_{split}_labels.txt"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        image_dir.glob("*.jpg"),
        key=lambda p: int(p.stem.replace("img_", ""))
    )

    label_lines = []
    missing_annotations = []
    malformed_lines = 0

    for image_path in image_files:
        annotation_path = annotation_dir / f"gt_{image_path.stem}.txt"

        if not annotation_path.exists():
            missing_annotations.append(image_path.name)
            continue

        annotation_text = annotation_path.read_text(
            encoding="utf-8-sig"
        ).strip()

        annotations = []

        for line in annotation_text.splitlines():
            parts = line.split(",", 8)

            if len(parts) != 9:
                malformed_lines += 1
                continue

            try:
                coordinates = [int(value) for value in parts[:8]]
            except ValueError:
                malformed_lines += 1
                continue

            text = parts[8]

            polygon = [
                [coordinates[0], coordinates[1]],
                [coordinates[2], coordinates[3]],
                [coordinates[4], coordinates[5]],
                [coordinates[6], coordinates[7]],
            ]

            annotations.append({
                "transcription": text,
                "points": polygon,
            })

        relative_image_path = image_path.relative_to(
            PROJECT_DIR
        ).as_posix()

        label_lines.append(
            f"{relative_image_path}\t"
            f"{json.dumps(annotations, ensure_ascii=False)}"
        )

    output_file.write_text(
        "\n".join(label_lines) + "\n" if label_lines else "",
        encoding="utf-8"
    )

    print(f"\n--- {split.upper()} ---")
    print(f"Images found: {len(image_files)}")
    print(f"Label entries written: {len(label_lines)}")
    print(f"Missing annotations: {len(missing_annotations)}")
    print(f"Malformed annotation lines: {malformed_lines}")
    print(f"Output: {output_file}")

    if missing_annotations:
        print("Missing examples:", missing_annotations[:10])


if __name__ == "__main__":
    prepare_labels("train")
    prepare_labels("test")
