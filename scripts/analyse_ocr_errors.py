import csv
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "train_data"
    / "ic15_data"
    / "icdar_recognition_results.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "train_data"
    / "ic15_data"
    / "worst_ocr_errors.csv"
)


# ---------------------------------------------------------
# Read existing results
# ---------------------------------------------------------
results = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        row["cer"] = float(row["cer"])
        row["edit_distance"] = int(row["edit_distance"])
        row["confidence"] = float(row["confidence"])

        results.append(row)


# ---------------------------------------------------------
# Keep incorrect predictions only
# ---------------------------------------------------------
errors = [
    row
    for row in results
    if row["exact_match"].lower() == "false"
]


# ---------------------------------------------------------
# Sort by CER, highest first
# ---------------------------------------------------------
errors.sort(
    key=lambda row: (
        row["cer"],
        row["edit_distance"]
    ),
    reverse=True
)


# ---------------------------------------------------------
# Display the 20 worst errors
# ---------------------------------------------------------
print()
print("============================================")
print("TOP 20 WORST OCR ERRORS")
print("============================================")

for rank, row in enumerate(errors[:20], start=1):

    print(
        f"{rank:02d}. "
        f"{row['image']} | "
        f"GT: {row['ground_truth']} | "
        f"Prediction: {row['prediction']} | "
        f"Edit distance: {row['edit_distance']} | "
        f"CER: {row['cer'] * 100:.2f}% | "
        f"Confidence: {row['confidence']:.4f}"
    )


# ---------------------------------------------------------
# Save all incorrect predictions sorted by CER
# ---------------------------------------------------------
with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "image",
        "ground_truth",
        "prediction",
        "confidence",
        "exact_match",
        "edit_distance",
        "cer",
        "time_seconds"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in errors:
        writer.writerow(row)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
print()
print("--------------------------------------------")
print(f"Total incorrect predictions: {len(errors)}")
print()
print("All errors saved to:")
print(OUTPUT_FILE)
print("============================================")