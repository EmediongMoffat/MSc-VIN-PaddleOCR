
import csv
import json
import cv2
import numpy as np

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

TEST_LABELS = PROJECT_DIR / "data" / "raw" / "det_test_labels.txt"

TEST_CROPS = PROJECT_DIR / "recognition_data" / "test_crops"
TEST_LIST = PROJECT_DIR / "recognition_data" / "test_list.txt"

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "evaluation"
    / "EXP-E2E-ICDAR15-002"
)

MANIFEST_CSV = OUTPUT_DIR / "eligible_gt_manifest.csv"
SUMMARY_TXT = OUTPUT_DIR / "manifest_summary.txt"


# ============================================================
# SAME GEOMETRY AND CROP VALIDATION AS ORIGINAL SCRIPT
# ============================================================

def order_points(points):
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

    if crop.shape[0] > crop.shape[1]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

    return crop


# ============================================================
# INPUT PARSING
# ============================================================

def read_label_line(line):
    line = line.lstrip("\ufeff").strip()

    if not line:
        return None, None

    image_path, annotation_text = line.split(None, 1)

    return image_path, json.loads(annotation_text)


def read_test_list():
    entries = []

    for line_number, line in enumerate(
        TEST_LIST.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        if "\t" in line:
            filename, transcription = line.split("\t", 1)
        else:
            filename, transcription = line.split(None, 1)

        entries.append({
            "filename": filename.strip(),
            "transcription": transcription.strip(),
            "line_number": line_number,
        })

    return entries


# ============================================================
# BUILD AND VERIFY SOURCE-AWARE MANIFEST
# ============================================================

def main():

    if not TEST_LABELS.exists():
        raise FileNotFoundError(TEST_LABELS)

    if not TEST_CROPS.exists():
        raise FileNotFoundError(TEST_CROPS)

    if not TEST_LIST.exists():
        raise FileNotFoundError(TEST_LIST)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_entries = read_test_list()

    label_lines = TEST_LABELS.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    manifest = []

    crop_number = 0

    skipped_unknown = 0
    skipped_invalid = 0
    skipped_missing = 0

    total_images = 0
    total_gt_characters = 0

    errors = []

    for line_number, line in enumerate(label_lines, start=1):

        if not line.strip():
            continue

        relative_path, annotations = read_label_line(line)

        image_path = PROJECT_DIR / Path(
            relative_path.replace("\\", "/")
        )

        total_images += 1

        if not image_path.exists():
            skipped_missing += 1
            errors.append(
                f"Missing source image: {image_path}"
            )
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            skipped_missing += 1
            errors.append(
                f"Unreadable source image: {image_path}"
            )
            continue

        for annotation_index, annotation in enumerate(
            annotations, start=1
        ):

            transcription = str(
                annotation.get("transcription", "")
            ).strip()

            points = annotation.get("points", [])

            if transcription == "###" or not transcription:
                skipped_unknown += 1
                continue

            if len(points) != 4:
                skipped_invalid += 1
                continue

            crop = crop_text_region(image, points)

            if crop is None or crop.size == 0:
                skipped_invalid += 1
                continue

            expected_filename = (
                f"test_{crop_number + 1:06d}.jpg"
            )

            crop_path = TEST_CROPS / expected_filename

            if not crop_path.exists():
                errors.append(
                    f"Expected crop is missing: {expected_filename}"
                )
                continue

            if crop_number >= len(test_entries):
                errors.append(
                    f"No test-list entry for {expected_filename}"
                )
                continue

            list_entry = test_entries[crop_number]

            if list_entry["filename"] != expected_filename:
                errors.append(
                    f"Filename mismatch at crop {crop_number + 1}: "
                    f"expected {expected_filename}, "
                    f"found {list_entry['filename']}"
                )
                continue

            if list_entry["transcription"] != transcription:
                errors.append(
                    f"Transcription mismatch for {expected_filename}: "
                    f"original={transcription!r}, "
                    f"test_list={list_entry['transcription']!r}"
                )
                continue

            crop_image = cv2.imread(str(crop_path))

            if crop_image is None:
                errors.append(
                    f"Existing crop cannot be read: {crop_path}"
                )
                continue

            crop_number += 1
            character_count = len(transcription)

            total_gt_characters += character_count

            manifest.append({
                "crop_filename": expected_filename,
                "source_image": Path(
                    relative_path.replace("\\", "/")
                ).as_posix(),
                "source_image_basename": Path(
                    relative_path.replace("\\", "/")
                ).name,
                "source_annotation_line": line_number,
                "annotation_index": annotation_index,
                "transcription": transcription,
                "character_count": character_count,
                "polygon_points": json.dumps(
                    points, separators=(",", ":")
                ),
                "source_image_width": image.shape[1],
                "source_image_height": image.shape[0],
                "crop_width": crop_image.shape[1],
                "crop_height": crop_image.shape[0],
            })

        if total_images % 50 == 0:
            print(
                f"Processed {total_images} source images..."
            )

    # --------------------------------------------------------
    # FINAL CONSISTENCY CHECKS
    # --------------------------------------------------------

    if len(test_entries) != 2065:
        errors.append(
            f"Expected 2065 test-list entries, "
            f"found {len(test_entries)}"
        )

    if crop_number != 2065:
        errors.append(
            f"Expected 2065 mapped crops, found {crop_number}"
        )

    if len(manifest) != 2065:
        errors.append(
            f"Expected 2065 manifest rows, found {len(manifest)}"
        )

    if errors:
        print("\nMANIFEST VALIDATION FAILED.")
        print(f"Errors found: {len(errors)}")

        for error in errors[:50]:
            print("-", error)

        raise RuntimeError(
            "No final manifest written. Resolve the mismatches first."
        )

    # --------------------------------------------------------
    # WRITE CSV
    # --------------------------------------------------------

    fieldnames = list(manifest[0].keys())

    with MANIFEST_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(manifest)

    # --------------------------------------------------------
    # WRITE SUMMARY
    # --------------------------------------------------------

    summary = [
        "ICDAR2015 SOURCE-AWARE GROUND-TRUTH MANIFEST",
        "================================================",
        f"Source annotation images: {total_images}",
        f"Eligible GT regions mapped: {len(manifest)}",
        f"Existing crop-list entries: {len(test_entries)}",
        f"Total eligible GT characters: {total_gt_characters}",
        f"Unknown/empty annotations skipped: {skipped_unknown}",
        f"Invalid regions skipped: {skipped_invalid}",
        f"Missing/unreadable images: {skipped_missing}",
        "",
        "VALIDATION",
        "----------",
        "Crop-to-source mapping: PASSED",
        "Crop filenames: PASSED",
        "Ground-truth transcription matching: PASSED",
        "Existing crop files readable: PASSED",
        f"Expected crop count: 2065",
        f"Actual mapped crop count: {len(manifest)}",
        "",
        f"Manifest CSV: {MANIFEST_CSV}",
    ]

    SUMMARY_TXT.write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print("\n========== MANIFEST RESULTS ==========")
    print(f"Source images: {total_images}")
    print(f"Eligible GT regions: {len(manifest)}")
    print(f"Total GT characters: {total_gt_characters}")
    print(f"Unknown/empty skipped: {skipped_unknown}")
    print(f"Invalid skipped: {skipped_invalid}")
    print(f"Missing/unreadable images: {skipped_missing}")
    print("\nAll mapping checks PASSED.")
    print(f"\nCSV: {MANIFEST_CSV}")
    print(f"Summary: {SUMMARY_TXT}")
    print("======================================")


if __name__ == "__main__":
    main()
