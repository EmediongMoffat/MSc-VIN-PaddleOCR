
from pathlib import Path
import pandas as pd

# Locate the existing results file relative to the PaddleOCR folder
project_root = Path(__file__).resolve().parent.parent
csv_path = project_root / "train_data" / "ic15_data" / "icdar_recognition_results.csv"

df = pd.read_csv(csv_path)

required_columns = {"ground_truth", "prediction"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")

# Count character-level matches and errors using Levenshtein alignment
tp = 0  # Matching characters
fp = 0  # Extra or substituted predicted characters
fn = 0  # Missing or substituted ground-truth characters


def count_character_matches(reference, prediction):
    global tp, fp, fn

    reference = str(reference)
    prediction = str(prediction)

    m, n = len(reference), len(prediction)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if reference[i - 1] == prediction[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Missing predicted character
                dp[i][j - 1] + 1,       # Extra predicted character
                dp[i - 1][j - 1] + cost # Match or substitution
            )

    i, j = m, n
    while i > 0 or j > 0:
        if (
            i > 0 and j > 0
            and reference[i - 1] == prediction[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
            tp += 1
            i -= 1
            j -= 1
        elif (
            i > 0 and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + 1
        ):
            # Substitution: one false positive and one false negative
            fp += 1
            fn += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            fn += 1
            i -= 1
        else:
            fp += 1
            j -= 1


for _, row in df.iterrows():
    count_character_matches(row["ground_truth"], row["prediction"])

precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0.0
)

print(f"Images evaluated: {len(df)}")
print(f"True-positive characters: {tp}")
print(f"False-positive characters: {fp}")
print(f"False-negative characters: {fn}")
print(f"Character-level precision: {precision:.4%}")
print(f"Character-level recall: {recall:.4%}")
print(f"Character-level micro F1-score: {f1:.4%}")
