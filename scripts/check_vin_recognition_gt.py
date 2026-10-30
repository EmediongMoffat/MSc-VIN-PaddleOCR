
"""
check_vin_recognition_gt.py

VIN recognition ground-truth verification.

Dataset structure:

recognition_gt_crops/
    crops/
        train/
        valid/
        test/
    rec_gt_train.txt
    rec_gt_valid.txt
    rec_gt_test.txt

Checks:
1. Label-file format.
2. Image existence.
3. Image readability.
4. VIN length and permitted characters.
5. VIN check digit.
6. Duplicate images and VINs.
7. Image dimensions and aspect ratios.
8. Visual ground-truth samples.

Run from PaddleOCR directory:

python ..\scripts\check_vin_recognition_gt.py
"""

import os
import re
import random
import argparse
from collections import defaultdict

import cv2


SPLITS = ["train", "valid", "test"]

LABEL_FILES = {
    "train": "rec_gt_train.txt",
    "valid": "rec_gt_valid.txt",
    "test": "rec_gt_test.txt"
}

VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")

WEIGHTS = [
    8, 7, 6, 5, 4, 3, 2, 10, 0,
    9, 8, 7, 6, 5, 4, 3, 2
]

TRANSLIT = {
    **{str(d): d for d in range(10)},
    "A": 1, "B": 2, "C": 3, "D": 4,
    "E": 5, "F": 6, "G": 7, "H": 8,
    "J": 1, "K": 2, "L": 3, "M": 4,
    "N": 5, "P": 7, "R": 9,
    "S": 2, "T": 3, "U": 4, "V": 5,
    "W": 6, "X": 7, "Y": 8, "Z": 9
}


def check_digit_ok(vin):

    total = sum(
        TRANSLIT[c] * weight
        for c, weight in zip(vin, WEIGHTS)
    )

    expected = total % 11
    expected = "X" if expected == 10 else str(expected)

    return vin[8] == expected, expected


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=(
            r"..\data\processed\paddleocr_vin"
            r"\EXP-YOLO-PPOCR-CONVERT-001_20261028_132836_267609"
            r"\recognition_gt_crops"
        )
    )

    parser.add_argument(
        "--draw",
        type=int,
        default=20,
        help="Number of random visual samples per split"
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    root = os.path.abspath(args.root)

    random.seed(args.seed)

    report = []

    total_problems = 0
    total_images = 0
    total_labels = 0
    total_format_valid = 0
    total_check_digit_valid = 0

    vin_to_splits = defaultdict(set)
    image_to_splits = defaultdict(set)

    report.append("=" * 65)
    report.append("VIN RECOGNITION GROUND-TRUTH VERIFICATION")
    report.append("=" * 65)
    report.append(f"Dataset: {root}")

    if not os.path.isdir(root):
        print(f"ERROR: Dataset folder does not exist:\n{root}")
        return

    vis_root = os.path.join(root, "label_check_vis")
    os.makedirs(vis_root, exist_ok=True)

    for split in SPLITS:

        label_file = os.path.join(
            root,
            LABEL_FILES[split]
        )

        report.append(f"\n{'=' * 20} {split.upper()} {'=' * 20}")

        if not os.path.isfile(label_file):
            report.append(f"ERROR: Label file missing: {label_file}")
            total_problems += 1
            continue

        with open(label_file, "r", encoding="utf-8-sig") as f:
            lines = [
                line.rstrip("\r\n")
                for line in f
                if line.strip()
            ]

        report.append(f"Label entries: {len(lines)}")

        samples = []
        split_images = set()
        split_vins = set()

        split_problems = 0
        readable_count = 0
        format_valid_count = 0
        check_valid_count = 0

        for line_number, line in enumerate(lines, start=1):

            total_labels += 1

            # Expected format:
            # image_path<TAB>VIN

            parts = line.split("\t", 1)

            if len(parts) != 2:
                report.append(
                    f"MALFORMED LINE {line_number}: {line}"
                )
                total_problems += 1
                split_problems += 1
                continue

            img_rel = parts[0].strip().replace("/", os.sep)
            vin = parts[1].strip().upper()

            if not img_rel or not vin:
                report.append(
                    f"EMPTY IMAGE PATH OR VIN: line {line_number}"
                )
                total_problems += 1
                split_problems += 1
                continue

            img_path = os.path.join(root, img_rel)

            # 1. Image existence

            if not os.path.isfile(img_path):
                report.append(
                    f"MISSING IMAGE: {img_rel}"
                )
                total_problems += 1
                split_problems += 1
                continue

            # 2. Image readability

            img = cv2.imread(img_path)

            if img is None:
                report.append(
                    f"UNREADABLE IMAGE: {img_rel}"
                )
                total_problems += 1
                split_problems += 1
                continue

            readable_count += 1
            total_images += 1

            height, width = img.shape[:2]
            ratio = width / height if height else 0

            # 3. VIN format

            format_ok = bool(VIN_RE.fullmatch(vin))

            if not format_ok:
                report.append(
                    f"BAD VIN FORMAT: {img_rel} | "
                    f"VIN='{vin}' | length={len(vin)}"
                )

                total_problems += 1
                split_problems += 1

            else:
                format_valid_count += 1
                total_format_valid += 1

                # 4. VIN check digit
                check_ok, expected = check_digit_ok(vin)

                if not check_ok:
                    report.append(
                        f"CHECK DIGIT FAIL: {img_rel} | "
                        f"VIN='{vin}' | "
                        f"position 9='{vin[8]}' | "
                        f"expected='{expected}'"
                    )

                    total_problems += 1
                    split_problems += 1

                else:
                    check_valid_count += 1
                    total_check_digit_valid += 1

            # 5. Duplicate VIN within split

            if vin in split_vins:
                report.append(
                    f"DUPLICATE VIN WITHIN {split.upper()}: {vin}"
                )

                total_problems += 1
                split_problems += 1

            split_vins.add(vin)
            vin_to_splits[vin].add(split)

            # 6. Duplicate image within split

            if img_rel in split_images:
                report.append(
                    f"DUPLICATE IMAGE WITHIN {split.upper()}: {img_rel}"
                )

                total_problems += 1
                split_problems += 1

            split_images.add(img_rel)
            image_to_splits[img_rel].add(split)

            # 7. Image dimensions

            if width < 100 or height < 20:
                report.append(
                    f"SMALL CROP: {img_rel} | "
                    f"{width}x{height}"
                )

            if ratio < 2:
                report.append(
                    f"UNUSUAL ASPECT RATIO: {img_rel} | "
                    f"ratio={ratio:.2f}"
                )

            # Check that path agrees with split
            expected_folder = f"crops{os.sep}{split}{os.sep}"

            if expected_folder.lower() not in img_rel.lower():
                report.append(
                    f"WARNING: IMAGE PATH DOES NOT MATCH SPLIT: "
                    f"{split} | {img_rel}"
                )

            samples.append(
                (img_rel, vin, img, width, height)
            )

        # 8. Generate visual samples

        vis_dir = os.path.join(vis_root, split)
        os.makedirs(vis_dir, exist_ok=True)

        selected = random.sample(
            samples,
            min(args.draw, len(samples))
        )

        for index, (img_rel, vin, img, width, height) in enumerate(selected):

            display = cv2.resize(
                img,
                None,
                fx=1.5,
                fy=1.5,
                interpolation=cv2.INTER_CUBIC
            )

            # Add ground-truth VIN to the image
            cv2.putText(
                display,
                vin,
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            image_name = f"sample_{index + 1:03d}.jpg"

            cv2.imwrite(
                os.path.join(vis_dir, image_name),
                display
            )

            with open(
                os.path.join(
                    vis_dir,
                    f"sample_{index + 1:03d}.txt"
                ),
                "w",
                encoding="utf-8"
            ) as f:

                f.write(f"Ground truth: {vin}\n")
                f.write(f"Original image: {img_rel}\n")
                f.write(f"Dimensions: {width}x{height}\n")

        report.append(f"Readable images: {readable_count}")
        report.append(f"Valid VIN format: {format_valid_count}")
        report.append(f"Valid VIN check digit: {check_valid_count}")
        report.append(f"Problems: {split_problems}")
        report.append(f"Visual samples: {len(selected)}")

    # 9. Cross-split VIN leakage

    report.append("\n" + "=" * 65)
    report.append("CROSS-SPLIT VIN DUPLICATE CHECK")
    report.append("=" * 65)

    leakage_count = 0

    for vin, splits in sorted(vin_to_splits.items()):

        if len(splits) > 1:
            report.append(
                f"VIN LEAKAGE: {vin} | splits={sorted(splits)}"
            )

            leakage_count += 1
            total_problems += 1

    if leakage_count == 0:
        report.append("No cross-split VIN duplicates found.")

    # 10. Cross-split image leakage

    report.append("\nCROSS-SPLIT IMAGE DUPLICATE CHECK")

    image_leakage_count = 0

    for img_rel, splits in sorted(image_to_splits.items()):

        if len(splits) > 1:
            report.append(
                f"IMAGE LEAKAGE: {img_rel} | splits={sorted(splits)}"
            )

            image_leakage_count += 1
            total_problems += 1

    if image_leakage_count == 0:
        report.append("No cross-split image duplicates found.")

    # 11. Final summary

    report.append("\n" + "=" * 65)
    report.append("FINAL SUMMARY")
    report.append("=" * 65)

    report.append(f"Total label entries: {total_labels}")
    report.append(f"Total readable images: {total_images}")
    report.append(f"Valid VIN formats: {total_format_valid}")
    report.append(f"Valid check digits: {total_check_digit_valid}")
    report.append(f"VIN leakage cases: {leakage_count}")
    report.append(f"Image leakage cases: {image_leakage_count}")
    report.append(f"TOTAL PROBLEMS: {total_problems}")

    report_path = os.path.join(
        root,
        "recognition_gt_check_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\n".join(report))

    print(f"\nReport saved: {report_path}")
    print(f"Visual samples: {vis_root}")


if __name__ == "__main__":
    main()
