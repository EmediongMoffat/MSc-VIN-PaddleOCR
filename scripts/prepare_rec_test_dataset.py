import json
import cv2
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent

TEST_LABELS = PROJECT_DIR / "data" / "raw" / "det_test_labels.txt"

OUTPUT_DIR = PROJECT_DIR / "recognition_data"
TEST_CROPS = OUTPUT_DIR / "test_crops"
TEST_LIST = OUTPUT_DIR / "test_list.txt"

# Create the test crop folder without changing train/validation data
TEST_CROPS.mkdir(parents=True, exist_ok=True)


def read_label_line(line):
    """Read an image path and its JSON annotations."""
    line = line.lstrip("\ufeff").strip()

    if not line:
        return None, None

    image_path, annotation_text = line.split(None, 1)
    annotations = json.loads(annotation_text)

    return image_path, annotations


def order_points(points):
    """Order corners: top-left, top-right, bottom-right, bottom-left."""
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

    width = int(round(max(
        np.linalg.norm(tr - tl),
        np.linalg.norm(br - bl),
    )))
    height = int(round(max(
        np.linalg.norm(bl - tl),
        np.linalg.norm(br - tr),
    )))

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

    # Rotate vertical text crops into horizontal orientation
    if crop.shape[0] > crop.shape[1]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

    return crop


def prepare_test_dataset():
    """Generate recognition crops and labels for the held-out test split."""

    if not TEST_LABELS.exists():
        raise FileNotFoundError(
            f"Test annotation file not found: {TEST_LABELS}"
        )

    manifest = []
    skipped_unknown = 0
    skipped_invalid = 0
    skipped_missing = 0
    crop_number = 0

    lines = TEST_LABELS.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        try:
            relative_path, annotations = read_label_line(line)
        except Exception as exc:
            raise ValueError(
                f"Could not parse line {line_number}: {exc}"
            ) from exc

        image_path = PROJECT_DIR / Path(
            relative_path.replace("\\", "/")
        )

        if not image_path.exists():
            print(f"Missing image: {image_path}")
            skipped_missing += 1
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read image: {image_path}")
            skipped_missing += 1
            continue

        for annotation in annotations:
            text = str(
                annotation.get("transcription", "")
            ).strip()

            points = annotation.get("points", [])

            # Ignore unknown or empty annotations, as in the
            # existing training/validation preparation script.
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

            filename = f"test_{crop_number + 1:06d}.jpg"
            output_path = TEST_CROPS / filename

            if not cv2.imwrite(str(output_path), crop):
                skipped_invalid += 1
                continue

            crop_number += 1
            manifest.append(f"{filename}\t{text}")

        if line_number % 50 == 0:
            print(
                f"Processed {line_number}/{len(lines)} "
                f"test annotation lines..."
            )

    TEST_LIST.write_text(
        "\n".join(manifest) + ("\n" if manifest else ""),
        encoding="utf-8",
    )

    print("\n========== TEST DATASET RESULTS ==========")
    print(f"Annotation lines: {len(lines)}")
    print(f"Recognition crops created: {len(manifest)}")
    print(f"Unknown/empty labels skipped: {skipped_unknown}")
    print(f"Invalid regions or failed writes: {skipped_invalid}")
    print(f"Missing/unreadable images: {skipped_missing}")
    print(f"Test crop folder: {TEST_CROPS}")
    print(f"Test label list: {TEST_LIST}")
    print("==========================================")

    if not manifest:
        raise RuntimeError(
            "No test crops were created. Check the paths and annotations."
        )


if __name__ == "__main__":
    prepare_test_dataset()
    