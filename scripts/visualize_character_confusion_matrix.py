
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Paths relative to the PaddleOCR project root
INPUT_CSV = Path(
    r"..\results\recognition"
    r"\EXP-REC-ICDAR15-001_test_predictions.csv"
)

OUTPUT_DIR = Path(r"..\results\recognition")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUTPUT_DIR / "EXP-REC-ICDAR15-001_confusion_matrix.png"
CSV_PATH = OUTPUT_DIR / "EXP-REC-ICDAR15-001_confusion_matrix.csv"

INSERTION = "<INSERTION>"
DELETION = "<DELETION>"

df = pd.read_csv(INPUT_CSV).fillna("")

# Confusion counts: ground-truth character -> predicted character
confusions = Counter()
true_support = Counter()


def align_and_count(gt, pred):
    """Build confusion counts using Levenshtein edit alignment."""
    gt, pred = str(gt), str(pred)
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

            options = [
                (dp[i - 1][j - 1] + cost,
                 "M" if cost == 0 else "S"),
                (dp[i - 1][j] + 1, "D"),
                (dp[i][j - 1] + 1, "I"),
            ]

            dp[i][j], back[i][j] = min(
                options, key=lambda item: item[0]
            )

    i, j = n, m

    while i > 0 or j > 0:
        op = back[i][j]

        if op == "M":
            char = gt[i - 1]
            confusions[(char, char)] += 1
            true_support[char] += 1
            i -= 1
            j -= 1

        elif op == "S":
            true_char = gt[i - 1]
            predicted_char = pred[j - 1]

            confusions[(true_char, predicted_char)] += 1
            true_support[true_char] += 1
            i -= 1
            j -= 1

        elif op == "D":
            true_char = gt[i - 1]

            confusions[(true_char, DELETION)] += 1
            true_support[true_char] += 1
            i -= 1

        elif op == "I":
            predicted_char = pred[j - 1]

            confusions[(INSERTION, predicted_char)] += 1
            j -= 1


for row in df.itertuples(index=False):
    align_and_count(row.ground_truth, row.prediction)

# Build complete matrix, including characters found in predictions
characters = sorted({
    char
    for pair in confusions
    for char in pair
    if char not in (INSERTION, DELETION)
})

labels = characters + [DELETION, INSERTION]

matrix = pd.DataFrame(
    0, index=labels, columns=labels, dtype=int
)

for (true_char, predicted_char), count in confusions.items():
    matrix.loc[true_char, predicted_char] += count

# Save the full numerical matrix
matrix.index.name = "Ground truth"
matrix.to_csv(CSV_PATH, encoding="utf-8-sig")

# Select the most frequent ground-truth characters for a readable plot.
# All counts remain available in the full CSV.
top_chars = [
    char for char, _ in true_support.most_common(25)
    if char in matrix.index
]

plot_labels = top_chars + [DELETION, INSERTION]
plot_labels = list(dict.fromkeys(
    char for char in plot_labels if char in matrix.index
))

plot_matrix = matrix.loc[plot_labels, plot_labels]

# Draw the confusion matrix
fig_size = max(12, len(plot_labels) * 0.55)
fig, ax = plt.subplots(figsize=(fig_size, fig_size))

image = ax.imshow(plot_matrix.values, interpolation="nearest")
fig.colorbar(image, ax=ax, label="Character count")

ax.set_title(
    "PP-OCRv6-Medium Recognition: Character Confusion Matrix\n"
    "ICDAR2015 Test Set — Levenshtein Alignment"
)
ax.set_xlabel("Predicted character")
ax.set_ylabel("Ground-truth character")

ax.set_xticks(np.arange(len(plot_labels)))
ax.set_yticks(np.arange(len(plot_labels)))
ax.set_xticklabels(plot_labels, rotation=90)
ax.set_yticklabels(plot_labels)

# Annotate non-zero cells
for i in range(plot_matrix.shape[0]):
    for j in range(plot_matrix.shape[1]):
        value = plot_matrix.iloc[i, j]
        if value > 0:
            ax.text(
                j, i, str(value),
                ha="center", va="center", fontsize=7
            )

fig.tight_layout()
fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Confusion matrix created successfully.")
print(f"Test samples: {len(df)}")
print(f"Character classes: {len(characters)}")
print(f"Visual matrix: {PNG_PATH.resolve()}")
print(f"Full confusion counts: {CSV_PATH.resolve()}")
print("The plot displays the 25 most frequent ground-truth characters")
print("plus insertion/deletion categories where applicable.")

