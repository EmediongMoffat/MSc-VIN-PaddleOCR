# Create a small ICDAR 2015 recognition dataset sample from a single image.
# This script reads one scene-text image, keeps only valid word annotations,
# crops each text region using a perspective transform, and saves the
# recognition crops along with labels for training or testing.

import os
from pathlib import Path

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def order_points(pts):
    """Arrange 4 polygon points into the order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")

    # The sum of x + y is smallest for the top-left corner.
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # The difference x - y is smallest for the top-right corner,
    # and largest for the bottom-left corner.
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def perspective_crop(image, quad_points):
    """Rectify a four-corner text region using a perspective transform."""
    # Reorder the points so OpenCV can map the polygon into a rectangle.
    rect = order_points(quad_points)

    # Compute the width and height of the destination image.
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    width = max(int(round(width_a)), int(round(width_b)))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    height = max(int(round(height_a)), int(round(height_b)))

    # Create the destination rectangle in the output image.
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype="float32")

    # Compute the perspective transform and warp the region into a rectangle.
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    return warped


def parse_annotation_line(line):
    """Convert one ICDAR annotation line into (points, text)."""
    # ICDAR format: x1,y1,x2,y2,x3,y3,x4,y4,text
    parts = line.strip().split(",")

    if len(parts) < 9:
        return None, None

    # The final part is the text; the first 8 entries are the 4 polygon corners.
    text = parts[-1].strip()
    coords = list(map(float, parts[:8]))
    points = np.array([
        [coords[0], coords[1]],
        [coords[2], coords[3]],
        [coords[4], coords[5]],
        [coords[6], coords[7]],
    ], dtype="float32")

    return points, text


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def main():
    # Locate the project root from this script so the paths work reliably.
    project_root = Path(__file__).resolve().parent.parent
    dataset_root = project_root / "train_data" / "ic15_data"

    # Read only the selected test image.
    image_path = dataset_root / "test" / "img_10.jpg"
    gt_path = dataset_root / "test_gt" / "gt_img_10.txt"

    # Output folder for the cropped recognition samples.
    output_dir = dataset_root / "rec_test" / "sample"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Labels file for each saved crop.
    labels_path = dataset_root / "sample_labels.txt"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {gt_path}")

    # Read the image in color so the crop can be saved as an image later.
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Open the ground-truth annotation file and read each line.
    with gt_path.open("r", encoding="utf-8") as f:
        annotation_lines = f.readlines()

    valid_results = []

    # Process each annotation line in order.
    for line in annotation_lines:
        # Ignore any annotation with the special marker "###".
        if "###" in line:
            continue

        points, text = parse_annotation_line(line)
        if points is None or text is None:
            continue

        # Keep only readable text regions. This removes blank or invalid labels.
        text = text.strip()
        if not text or text == "###":
            continue

        # Crop the quadrilateral region using a perspective transform.
        cropped = perspective_crop(image, points)

        # Store the crop and its label for later saving.
        valid_results.append((cropped, text))

        # Stop after the first 5 valid word images.
        if len(valid_results) >= 5:
            break

    if len(valid_results) < 5:
        raise ValueError(f"Only {len(valid_results)} valid text regions were found in {gt_path}.")

    # Save the five crops and create the label file.
    label_lines = []

    for index, (crop, text) in enumerate(valid_results, start=1):
        filename = f"word_{index}.png"
        save_path = output_dir / filename
        cv2.imwrite(str(save_path), crop)

        # Build the relative path from the ICDAR dataset root to the saved crop.
        relative_image_path = os.path.relpath(save_path, dataset_root)
        label_lines.append(f"{relative_image_path}\t{text}")

        # Print each saved crop and its corresponding label to the terminal.
        print(f"Saved crop {index}: {relative_image_path} -> {text}")

    # Write all labels to sample_labels.txt.
    with labels_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(label_lines) + "\n")

    print(f"\nLabels saved to: {labels_path}")


if __name__ == "__main__":
    main()
