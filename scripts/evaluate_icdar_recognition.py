import csv
import time
from pathlib import Path

from paddleocr import TextRecognition


# ---------------------------------------------------------
# Calculate Character Error Rate (CER)
# ---------------------------------------------------------
def calculate_cer(reference, prediction):
    """
    CER = (substitutions + deletions + insertions) / reference_length

    Uses dynamic programming (Levenshtein distance).
    """

    reference = str(reference)
    prediction = str(prediction)

    if len(reference) == 0:
        return 0.0 if len(prediction) == 0 else 1.0

    # Create dynamic-programming table
    rows = len(reference) + 1
    cols = len(prediction) + 1

    dp = [[0] * cols for _ in range(rows)]

    # Initial deletion costs
    for i in range(rows):
        dp[i][0] = i

    # Initial insertion costs
    for j in range(cols):
        dp[0][j] = j

    # Calculate edit distance
    for i in range(1, rows):
        for j in range(1, cols):

            if reference[i - 1] == prediction[j - 1]:
                substitution_cost = 0
            else:
                substitution_cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,              # deletion
                dp[i][j - 1] + 1,              # insertion
                dp[i - 1][j - 1] + substitution_cost  # substitution
            )

    return dp[-1][-1] / len(reference)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    PROJECT_ROOT
    / "train_data"
    / "ic15_data"
    / "rec_test"
    / "full"
)

LABEL_FILE = (
    PROJECT_ROOT
    / "train_data"
    / "ic15_data"
    / "rec_gt_generated.txt"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "train_data"
    / "ic15_data"
    / "icdar_recognition_results.csv"
)


# ---------------------------------------------------------
# Load ground-truth labels
# ---------------------------------------------------------
ground_truth = {}

with open(LABEL_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        image_path, label = line.split("\t", 1)

        image_name = Path(image_path).name

        ground_truth[image_name] = label


print(f"Ground-truth labels loaded: {len(ground_truth)}")


# ---------------------------------------------------------
# Create PaddleOCR recognition model
# ---------------------------------------------------------
model = TextRecognition()


# ---------------------------------------------------------
# Evaluate all images
# ---------------------------------------------------------
results = []

total_time = 0.0
exact_matches = 0

total_edit_distance = 0
total_reference_characters = 0

image_names = sorted(ground_truth.keys())


for index, image_name in enumerate(image_names, start=1):

    image_path = IMAGE_DIR / image_name
    expected = ground_truth[image_name]

    # Start timing
    start_time = time.perf_counter()

    output = model.predict(str(image_path))

    elapsed = time.perf_counter() - start_time

    total_time += elapsed

    # -----------------------------------------------------
    # Extract PaddleOCR result
    # -----------------------------------------------------
    prediction = ""
    confidence = None

    for res in output:

        data = res.json

        if isinstance(data, str):
            import json
            data = json.loads(data)

        res_data = data.get("res", data)

        prediction = res_data.get("rec_text", "")
        confidence = res_data.get("rec_score", None)

    # -----------------------------------------------------
    # Exact-string accuracy
    # -----------------------------------------------------
    is_correct = prediction == expected

    if is_correct:
        exact_matches += 1

    # -----------------------------------------------------
    # Character-level edit distance
    # -----------------------------------------------------
    reference = str(expected)
    predicted = str(prediction)

    reference_length = len(reference)

    # Calculate edit distance separately
    rows = reference_length + 1
    cols = len(predicted) + 1

    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i

    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):

        for j in range(1, cols):

            if reference[i - 1] == predicted[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    edit_distance = dp[-1][-1]

    total_edit_distance += edit_distance
    total_reference_characters += reference_length

    # -----------------------------------------------------
    # Per-image CER
    # -----------------------------------------------------
    cer = calculate_cer(expected, prediction)

    # -----------------------------------------------------
    # Store result
    # -----------------------------------------------------
    results.append({
        "image": image_name,
        "ground_truth": expected,
        "prediction": prediction,
        "confidence": confidence,
        "exact_match": is_correct,
        "edit_distance": edit_distance,
        "cer": cer,
        "time_seconds": elapsed
    })

    # Progress
    if index % 100 == 0:
        print(f"Processed {index}/{len(image_names)} images")


# ---------------------------------------------------------
# Calculate overall metrics
# ---------------------------------------------------------
total_images = len(results)

exact_accuracy = (
    exact_matches / total_images
    if total_images
    else 0
)

overall_cer = (
    total_edit_distance / total_reference_characters
    if total_reference_characters
    else 0
)

average_time = (
    total_time / total_images
    if total_images
    else 0
)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------
with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image",
            "ground_truth",
            "prediction",
            "confidence",
            "exact_match",
            "edit_distance",
            "cer",
            "time_seconds"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


# ---------------------------------------------------------
# Display summary
# ---------------------------------------------------------
print()
print("============================================")
print("ICDAR RECOGNITION BASELINE COMPLETE")
print("============================================")
print(f"Images evaluated:       {total_images}")
print(f"Exact matches:          {exact_matches}")
print(f"Exact-string accuracy:  {exact_accuracy * 100:.2f}%")
print(f"Overall CER:            {overall_cer * 100:.2f}%")
print(f"Average time/image:     {average_time * 1000:.2f} ms")
print(f"Total processing time:  {total_time:.2f} seconds")
print("--------------------------------------------")
print("Results saved to:")
print(OUTPUT_FILE)
print("============================================")