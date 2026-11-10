
import json
import cv2
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_LABELS = PROJECT_DIR / "data" / "raw" / "det_train_800_labels.txt"
VAL_LABELS = PROJECT_DIR / "data" / "raw" / "det_val_200_labels.txt"

OUTPUT_DIR = PROJECT_DIR / "recognition_data"
TRAIN_CROPS = OUTPUT_DIR / "train_crops"
VAL_CROPS = OUTPUT_DIR / "val_crops"

# --------------------------------------------------
# Create output folders
# --------------------------------------------------
TRAIN_CROPS.mkdir(parents=True, exist_ok=True)
VAL_CROPS.mkdir(parents=True, exist_ok=True)


def read_label_line(line):
    """Read an image path and JSON annotations from one split line."""
    line = line.lstrip("\ufeff").strip()

    if not line:
        return None, None

    # The image path is followed by whitespace and a JSON list.
    image_path, annotation_text = line.split(None, 1)
    annotations = json.loads(annotation_text)

    return image_path, annotations


def order_points(points):
    """Order four corners as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32)

    sums = points.sum(axis=1)
    differences = points[:, 1] - points[:, 0]

    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(differences)]
    bottom_left = points[np.argmax(differences)]

    return np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32,
    )


def crop_text_region(image, points):
    """Perspective-correct a four-corner text region."""
    rect = order_points(points)
    tl, tr, br, bl = rect

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    width = int(round(max(width_top, width_bottom)))
    height = int(round(max(height_left, height_right)))

    if width < 2 or height < 2:
        return None

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)

    crop = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Recognition models expect text to be horizontal.
    if crop.shape[0] > crop.shape[1]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

    return crop


def process_split(label_file, crop_dir, split_name):
    """Generate crops and a PaddleOCR recognition label file."""
    if not label_file.exists():
        raise FileNotFoundError(f"Split file not found: {label_file}")

    manifest = []
    total_regions = 0
    skipped_unknown = 0
    skipped_invalid = 0
    skipped_missing_images = 0
    crop_number = 0

    lines = label_file.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        try:
            relative_path, annotations = read_label_line(line)
        except Exception as exc:
            raise ValueError(
                f"Could not parse {label_file.name}, line {line_number}: {exc}"
            ) from exc

        image_path = PROJECT_DIR / Path(relative_path.replace("\\", "/"))

        if not image_path.exists():
            print(f"Missing image: {image_path}")
            skipped_missing_images += 1
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read image: {image_path}")
            skipped_missing_images += 1
            continue

        for annotation in annotations:
            text = str(annotation.get("transcription", "")).strip()
            points = annotation.get("points", [])

            if text == "###" or not text:
                skipped_unknown += 1
                continue

            if len(points) != 4:
                skipped_invalid += 1
                continue

            crop = crop_text_region(image, points)

            if crop is None or crop.size == 0:
                skipped_invalid += 1
                continue

            crop_number += 1
            total_regions += 1

            filename = f"{split_name}_{crop_number:06d}.jpg"
            output_path = crop_dir / filename

            if not cv2.imwrite(str(output_path), crop):
                skipped_invalid += 1
                total_regions -= 1
                continue

            # PaddleOCR label format: relative image path + tab + transcription.
            manifest.append(f"{filename}\t{text}")

    manifest_path = OUTPUT_DIR / f"{split_name}_list.txt"
    manifest_path.write_text(
        "\n".join(manifest) + ("\n" if manifest else ""),
        encoding="utf-8",
    )

    print(f"\n{split_name.upper()} RESULTS")
    print(f"Scene images in split file: {len(lines)}")
    print(f"Text crops created: {len(manifest)}")
    print(f"Unknown/empty labels skipped: {skipped_unknown}")
    print(f"Invalid regions or failed writes: {skipped_invalid}")
    print(f"Missing/unreadable images: {skipped_missing_images}")
    print(f"Label list: {manifest_path}")
    print(f"Crop folder: {crop_dir}")


if __name__ == "__main__":
    process_split(TRAIN_LABELS, TRAIN_CROPS, "train")
    process_split(VAL_LABELS, VAL_CROPS, "val")

    print("\nRecognition dataset preparation finished.")
