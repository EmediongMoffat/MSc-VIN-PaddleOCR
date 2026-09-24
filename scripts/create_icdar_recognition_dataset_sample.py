"""Create a small perspective-corrected ICDAR 2015 recognition sample.

This script intentionally processes only one image and its matching annotation
file. It is a small, easy-to-run test before processing the full dataset.
"""

from pathlib import Path

import cv2
import numpy as np


# Find the repository root from this file so the script works from any folder.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPOSITORY_ROOT / "train_data" / "ic15_data"
IMAGE_PATH = REPOSITORY_ROOT / "train_data" / "ic15_data" / "test" / "img_10.jpg"
ANNOTATION_PATH = (
    REPOSITORY_ROOT
    / "train_data"
    / "ic15_data"
    / "test_gt"
    / "gt_img_10.txt"
)
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "train_data" / "ic15_data" / "rec_test" / "sample"
)
LABELS_PATH = REPOSITORY_ROOT / "train_data" / "ic15_data" / "sample_labels.txt"
MAX_CROPS = 5


def perspective_crop(image: np.ndarray, coordinates: list[float]) -> np.ndarray:
    """Rectify one four-point text region and return it as an image crop."""
    points = np.asarray(coordinates, dtype=np.float32).reshape(4, 2)

    # Treat the first edge as the top edge of the text region.
    top_width = np.linalg.norm(points[1] - points[0])
    bottom_width = np.linalg.norm(points[2] - points[3])
    left_height = np.linalg.norm(points[3] - points[0])
    right_height = np.linalg.norm(points[2] - points[1])
    crop_width = max(1, int(round(max(top_width, bottom_width))))
    crop_height = max(1, int(round(max(left_height, right_height))))

    # Map the quadrilateral to an upright rectangle with a perspective matrix.
    destination = np.array(
        [
            [0, 0],
            [crop_width - 1, 0],
            [crop_width - 1, crop_height - 1],
            [0, crop_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points, destination)
    return cv2.warpPerspective(image, transform, (crop_width, crop_height))


def read_valid_annotations() -> list[tuple[list[float], str]]:
    """Read usable annotations, skipping ignored and malformed entries."""
    annotations = []
    with ANNOTATION_PATH.open("r", encoding="utf-8-sig") as annotation_file:
        for line_number, line in enumerate(annotation_file, start=1):
            fields = line.rstrip("\r\n").split(",")
            if len(fields) < 9:
                print(f"Skipping malformed annotation line {line_number}.")
                continue

            text = ",".join(fields[8:]).strip()
            if text == "###":
                continue

            try:
                coordinates = [float(value.strip()) for value in fields[:8]]
            except ValueError:
                print(f"Skipping non-numeric annotation line {line_number}.")
                continue

            annotations.append((coordinates, text))
            if len(annotations) == MAX_CROPS:
                break
    return annotations


def main() -> None:
    """Load the sample, write five crops, and create their label file."""
    # Load exactly the one requested source image.
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Could not read source image: {IMAGE_PATH}")

    # Create the requested output folder and collect the first five valid crops.
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    annotations = read_valid_annotations()
    if len(annotations) < MAX_CROPS:
        raise ValueError(
            f"Expected at least {MAX_CROPS} valid annotations, found {len(annotations)}."
        )

    labels = []
    for crop_number, (coordinates, text) in enumerate(annotations, start=1):
        crop = perspective_crop(image, coordinates)
        output_path = OUTPUT_DIRECTORY / f"word_{crop_number}.png"
        if not cv2.imwrite(str(output_path), crop):
            raise OSError(f"Could not save crop: {output_path}")

        relative_path = output_path.relative_to(DATASET_ROOT).as_posix()
        labels.append(f"{relative_path}\t{text}")
        print(f"Saved {relative_path} | label: {text}")

    # Write portable repository-relative paths and their ground-truth labels.
    LABELS_PATH.write_text("\n".join(labels) + "\n", encoding="utf-8")
    print(f"Wrote labels to {LABELS_PATH.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
