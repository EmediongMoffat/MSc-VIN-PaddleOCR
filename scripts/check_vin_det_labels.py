"""
check_vin_det_labels.py
Quick sanity check for the PaddleOCR detection labels made from the Roboflow VIN dataset.

What it checks, for det_gt_train.txt / det_gt_valid.txt / det_gt_test.txt:
  1. Every box lies inside its image (catches a wrong scale in the YOLO -> PaddleOCR conversion)
  2. Every transcription is a valid 17 character VIN (no I, O, Q) with a correct check digit
  3. The same VIN does not appear in two different splits (data leakage)
  4. Draws the ground truth box on a few images per split so you can look at them

Usage (from the folder that contains det_gt_*.txt and images/):
    python check_vin_det_labels.py
    python check_vin_det_labels.py --root "D:/Vin/detection" --draw 5
Output: label_check_report.txt and label_check_vis/<split>/*.jpg next to the labels.
"""
import argparse
import json
import os
import re
from collections import defaultdict

import cv2

SPLITS = ["train", "valid", "test"]
VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")
WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]
TRANSLIT = {**{str(d): d for d in range(10)},
            "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8,
            "J": 1, "K": 2, "L": 3, "M": 4, "N": 5, "P": 7, "R": 9,
            "S": 2, "T": 3, "U": 4, "V": 5, "W": 6, "X": 7, "Y": 8, "Z": 9}


def check_digit_ok(vin):
    total = sum(TRANSLIT[c] * w for c, w in zip(vin, WEIGHTS))
    expected = total % 11
    expected = "X" if expected == 10 else str(expected)
    return vin[8] == expected, expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="folder with det_gt_*.txt and images/")
    ap.add_argument("--draw", type=int, default=5, help="images to draw per split")
    args = ap.parse_args()

    report = []
    vin_to_splits = defaultdict(set)
    n_problems = 0

    for split in SPLITS:
        label_file = os.path.join(args.root, f"det_gt_{split}.txt")
        if not os.path.exists(label_file):
            report.append(f"[{split}] label file not found: {label_file}")
            continue
        vis_dir = os.path.join(args.root, "label_check_vis", split)
        os.makedirs(vis_dir, exist_ok=True)

        with open(label_file, encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f if l.strip()]
        report.append(f"\n===== {split}: {len(lines)} images =====")

        for i, line in enumerate(lines):
            img_rel, ann = line.split("\t", 1)
            boxes = json.loads(ann)
            img_path = os.path.join(args.root, img_rel)
            img = cv2.imread(img_path)
            if img is None:
                report.append(f"MISSING IMAGE  {img_rel}")
                n_problems += 1
                continue
            h, w = img.shape[:2]

            for b in boxes:
                vin = b["transcription"]
                pts = b["points"]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]

                # 1. box inside image
                if min(xs) < 0 or min(ys) < 0 or max(xs) > w or max(ys) > h:
                    report.append(f"BOX OUTSIDE IMAGE  {img_rel}  image={w}x{h}  "
                                  f"box x={min(xs)}..{max(xs)} y={min(ys)}..{max(ys)}")
                    n_problems += 1

                # 2. VIN format and check digit
                if not VIN_RE.match(vin):
                    report.append(f"BAD VIN FORMAT  {img_rel}  '{vin}'")
                    n_problems += 1
                else:
                    ok, exp = check_digit_ok(vin)
                    if not ok:
                        report.append(f"CHECK DIGIT FAIL  {img_rel}  '{vin}'  "
                                      f"(position 9 is '{vin[8]}', expected '{exp}')")
                        n_problems += 1

                vin_to_splits[vin].add(split)

                # 4. draw a few
                if i < args.draw:
                    import numpy as np
                    poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(img, [poly], True, (0, 255, 0), 4)
                    cv2.putText(img, vin, (int(min(xs)), max(int(min(ys)) - 15, 30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            if i < args.draw:
                cv2.imwrite(os.path.join(vis_dir, os.path.basename(img_rel)), img)
                if i == 0:
                    report.append(f"image size example: {w}x{h}")

    # 3. leakage between splits
    report.append("\n===== VINs found in more than one split =====")
    for vin, s in sorted(vin_to_splits.items()):
        if len(s) > 1:
            report.append(f"LEAK  {vin}  in {sorted(s)}")
            n_problems += 1

    report.append(f"\nTOTAL PROBLEMS: {n_problems}")
    out = os.path.join(args.root, "label_check_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print("\n".join(report))
    print(f"\nReport saved to {out}")
    print(f"Drawn images saved to {os.path.join(args.root, 'label_check_vis')}")


if __name__ == "__main__":
    main()
