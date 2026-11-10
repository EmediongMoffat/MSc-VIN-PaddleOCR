
import pandas as pd
from collections import defaultdict

# Input: recognition test predictions
CSV_PATH = (
    r"..\results\recognition"
    r"\EXP-REC-ICDAR15-001_test_predictions.csv"
)

df = pd.read_csv(CSV_PATH).fillna("")

# Per-character counts
counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})


def align_and_count(ground_truth, prediction):
    """Align strings with Levenshtein distance and count character errors."""
    gt = str(ground_truth)
    pred = str(prediction)

    n, m = len(gt), len(pred)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "D"

    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if gt[i - 1] == pred[j - 1] else 1

            choices = [
                (dp[i - 1][j - 1] + cost,
                 "M" if cost == 0 else "S"),
                (dp[i - 1][j] + 1, "D"),
                (dp[i][j - 1] + 1, "I"),
            ]

            dp[i][j], back[i][j] = min(
                choices, key=lambda item: item[0]
            )

    # Trace back through the optimal alignment
    i, j = n, m

    while i > 0 or j > 0:
        operation = back[i][j]

        if operation == "M":
            character = gt[i - 1]
            counts[character]["tp"] += 1
            i -= 1
            j -= 1

        elif operation == "S":
            counts[gt[i - 1]]["fn"] += 1
            counts[pred[j - 1]]["fp"] += 1
            i -= 1
            j -= 1

        elif operation == "D":
            counts[gt[i - 1]]["fn"] += 1
            i -= 1

        elif operation == "I":
            counts[pred[j - 1]]["fp"] += 1
            j -= 1


# Calculate counts across the test set
for row in df.itertuples(index=False):
    align_and_count(row.ground_truth, row.prediction)

# Include character classes appearing in ground-truth strings
classes = sorted(
    character
    for character, values in counts.items()
    if values["tp"] + values["fn"] > 0
)

f1_scores = []

for character in classes:
    tp = counts[character]["tp"]
    fp = counts[character]["fp"]
    fn = counts[character]["fn"]

    denominator = 2 * tp + fp + fn
    f1 = (2 * tp / denominator) if denominator else 0.0
    f1_scores.append(f1)

macro_f1 = (
    sum(f1_scores) / len(f1_scores)
    if f1_scores else 0.0
)

print("=" * 50)
print("CHARACTER-LEVEL MACRO-F1 RESULTS")
print("=" * 50)
print(f"Test samples: {len(df)}")
print(f"Character classes included: {len(classes)}")
print(f"Character-level macro-F1: {macro_f1 * 100:.4f}%")
print("Averaging: unweighted mean of per-character F1 scores")
print("Alignment: Levenshtein edit-distance alignment")
print("Substitutions: false negative + false positive")
print("Deletions: false negative")
print("Insertions: false positive")
print("=" * 50)
