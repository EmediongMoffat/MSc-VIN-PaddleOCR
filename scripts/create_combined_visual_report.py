
import csv
import cv2
import os
from pathlib import Path
from urllib.parse import quote
from html import escape

ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    ROOT / "data" / "processed" / "paddleocr_vin"
    / "EXP-YOLO-PPOCR-CONVERT-001_20261028_132836_267609"
    / "detection"
)

LABEL_FILE = DATA_ROOT / "det_gt_test.txt"

EXP_DIR = (
    ROOT / "results" / "experiments"
    / "EXP-VIN-COMBINED-001"
)

CSV_FILE = EXP_DIR / "evaluation" / "combined_test_predictions.csv"

REPORT_DIR = EXP_DIR / "visual_report"
PAIR_DIR = REPORT_DIR / "coordinate_aligned_images"

PAIR_DIR.mkdir(parents=True, exist_ok=True)

HTML_FILE = REPORT_DIR / "coordinate_aligned_visual_report.html"


def load_original_paths():
    paths = {}

    with LABEL_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            relative_path = line.split("\t", 1)[0]
            paths[Path(relative_path).name] = DATA_ROOT / relative_path

    return paths


def image_relative_path(path):
    relative = os.path.relpath(path, REPORT_DIR)
    return quote(Path(relative).as_posix(), safe="/:._-")


def make_comparison(row, original_path, output_path):
    original = cv2.imread(str(original_path))

    if original is None:
        raise RuntimeError(f"Cannot read image: {original_path}")

    height, width = original.shape[:2]

    # White canvas has EXACTLY the same dimensions as the original.
    white_canvas = 255 * (
        original.copy() * 0 + 1
    )
    white_canvas = white_canvas.astype("uint8")

    x1 = int(row["bbox_x_min"])
    y1 = int(row["bbox_y_min"])
    x2 = int(row["bbox_x_max"])
    y2 = int(row["bbox_y_max"])

    prediction = str(row.get("prediction", "")).strip()

    # Draw the same detector bounding box at identical coordinates.
    cv2.rectangle(
        white_canvas,
        (x1, y1),
        (x2, y2),
        (0, 150, 0),
        max(2, width // 700)
    )

    if prediction:
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = max(1, width // 1000)
        font_scale = max(0.4, min(width / 1600, 1.5))

        # Fit the VIN text within the detected box width where possible.
        text_size = cv2.getTextSize(
            prediction, font, font_scale, thickness
        )[0]

        available_width = max(1, x2 - x1)

        if text_size[0] > available_width:
            font_scale *= available_width / text_size[0] * 0.95
            font_scale = max(0.25, font_scale)

        # Anchor the recognised VIN at the detected box's top-left
        # coordinate. The right canvas uses the original image geometry.
        baseline_y = y1 + max(
            text_size[1] + 5,
            min(max(5, y2 - y1), 35)
        )

        baseline_y = min(height - 5, baseline_y)

        cv2.putText(
            white_canvas,
            prediction,
            (x1, baseline_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

    # Side-by-side: original left, white coordinate canvas right.
    comparison = cv2.hconcat([original, white_canvas])

    if not cv2.imwrite(str(output_path), comparison):
        raise RuntimeError(f"Could not save: {output_path}")


def main():
    original_paths = load_original_paths()

    with CSV_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError("No evaluation rows found.")

    cards = []

    correct = 0
    incorrect = 0

    for index, row in enumerate(rows, start=1):
        filename = row["image"]
        original_path = original_paths.get(filename)

        if original_path is None or not original_path.is_file():
            print("MISSING ORIGINAL:", filename)
            continue

        coordinates = [
            row.get("bbox_x_min", ""),
            row.get("bbox_y_min", ""),
            row.get("bbox_x_max", ""),
            row.get("bbox_y_max", "")
        ]

        if not all(str(value).strip() for value in coordinates):
            print("MISSING BOUNDING BOX:", filename)
            continue

        output_path = PAIR_DIR / f"{index:04d}_{Path(filename).stem}_comparison.jpg"

        make_comparison(row, original_path, output_path)

        exact = str(row.get("exact_match", "0")) == "1"

        if exact:
            correct += 1
            status = "CORRECT"
            status_class = "correct"
        else:
            incorrect += 1
            status = "INCORRECT"
            status_class = "incorrect"

        prediction = escape(str(row.get("prediction", "")))
        ground_truth = escape(str(row.get("ground_truth", "")))
        confidence = escape(str(row.get("confidence", "")))

        cards.append(f"""
        <article class="card">
            <div class="heading">
                <h2>Test image {index:02d}: {escape(filename)}</h2>
                <span class="{status_class}">{status}</span>
            </div>

            <img class="comparison"
                 src="{image_relative_path(output_path)}"
                 alt="Original image beside coordinate-aligned white canvas">

            <div class="details">
                <p><b>Ground-truth VIN:</b>
                   <span class="vin">{ground_truth}</span></p>

                <p><b>Recognised VIN:</b>
                   <span class="vin">{prediction or "No prediction"}</span></p>

                <p><b>Confidence:</b> {confidence}</p>
                <p><b>Bounding box:</b>
                   ({coordinates[0]}, {coordinates[1]})
                   to ({coordinates[2]}, {coordinates[3]})</p>
            </div>
        </article>
        """)

    total = len(rows)
    accuracy = correct / total * 100 if total else 0

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VIN Coordinate-Aligned Visual Evaluation</title>
<style>
body {{
    margin: 0;
    padding: 25px;
    background: #f2f4f7;
    color: #202938;
    font-family: Arial, sans-serif;
}}
.container {{
    max-width: 1500px;
    margin: auto;
}}
header {{
    background: #172b4d;
    color: white;
    padding: 24px;
    border-radius: 10px;
    margin-bottom: 24px;
}}
.card {{
    background: white;
    padding: 20px;
    border: 1px solid #dfe4ec;
    border-radius: 10px;
    margin-bottom: 25px;
    break-inside: avoid;
}}
.heading {{
    display: flex;
    justify-content: space-between;
    gap: 15px;
    align-items: center;
}}
.heading h2 {{
    font-size: 16px;
    overflow-wrap: anywhere;
}}
.comparison {{
    display: block;
    width: 100%;
    height: auto;
    background: white;
    border: 1px solid #ddd;
}}
.details {{
    margin-top: 15px;
    background: #f8fafc;
    padding: 12px;
    border-radius: 6px;
}}
.vin {{
    font-family: Consolas, monospace;
    font-size: 17px;
    font-weight: bold;
}}
.correct {{
    color: #166534;
    background: #dcfce7;
    padding: 7px 12px;
    border-radius: 20px;
}}
.incorrect {{
    color: #991b1b;
    background: #fee2e2;
    padding: 7px 12px;
    border-radius: 20px;
}}
@media print {{
    body {{ background: white; padding: 8mm; }}
    .card {{ break-inside: avoid; }}
}}
</style>
</head>
<body>
<div class="container">
<header>
<h1>VIN OCR — Coordinate-Aligned Visual Evaluation</h1>
<p>Experiment: EXP-VIN-COMBINED-001</p>
<p>Original image on the left; same-size white coordinate canvas on the right.</p>
<p>Test images: {total} | Correct: {correct} | Incorrect: {incorrect}</p>
<p>Exact VIN accuracy: {accuracy:.4f}%</p>
</header>
{''.join(cards)}
</div>
</body>
</html>
"""

    HTML_FILE.write_text(document, encoding="utf-8")

    print("=" * 60)
    print("COORDINATE-ALIGNED VISUAL REPORT CREATED")
    print("=" * 60)
    print("Total test images:", total)
    print("Correct:", correct)
    print("Incorrect:", incorrect)
    print("Images generated:", len(cards))
    print("Report:", HTML_FILE)
    print("Image folder:", PAIR_DIR)


if __name__ == "__main__":
    main()
