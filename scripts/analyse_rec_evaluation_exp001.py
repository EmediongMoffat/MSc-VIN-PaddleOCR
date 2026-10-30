
import csv
from pathlib import Path
from collections import Counter, defaultdict

# ---------------------------------------------------------
# EXP-VIN-REC-FT-001: Independent recognition evaluation
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "EXP-VIN-REC-FT-001"
    / "evaluation"
)

INPUT_CSV = EVAL_DIR / "test_evaluation.csv"

METRICS_CSV = EVAL_DIR / "recognition_metrics.csv"
INCORRECT_CSV = EVAL_DIR / "incorrect_predictions.csv"
CONFUSIONS_CSV = EVAL_DIR / "character_confusions.csv"
SUMMARY_TXT = EVAL_DIR / "independent_evaluation_summary.txt"


def levenshtein_alignment(reference, prediction):
    """Return edit counts and aligned character operations."""

    n = len(reference)
    m = len(prediction)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == prediction[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # match/substitution
            )

    # Backtrack to identify individual errors.
    i, j = n, m
    operations = []

    while i > 0 or j > 0:

        if (
            i > 0 and j > 0
            and reference[i - 1] == prediction[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
            operations.append(
                ("match", reference[i - 1], prediction[j - 1])
            )
            i -= 1
            j -= 1

        elif (
            i > 0 and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + 1
        ):
            operations.append(
                ("substitution", reference[i - 1], prediction[j - 1])
            )
            i -= 1
            j -= 1

        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            operations.append(
                ("deletion", reference[i - 1], "")
            )
            i -= 1

        else:
            operations.append(
                ("insertion", "", prediction[j - 1])
            )
            j -= 1

    operations.reverse()
    return dp[n][m], operations


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Evaluation CSV not found: {INPUT_CSV}"
        )

    with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = {
        "image",
        "ground_truth",
        "prediction",
        "confidence",
        "exact_match",
        "edit_distance",
    }

    if not rows:
        raise ValueError("The evaluation CSV contains no records.")

    missing_columns = required - set(rows[0].keys())

    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {sorted(missing_columns)}"
        )

    total = len(rows)
    exact_matches = 0
    total_edits = 0
    total_reference_chars = 0
    confidence_values = []

    incorrect_rows = []
    operation_counts = Counter()
    character_errors = defaultdict(list)

    for row in rows:

        image = row["image"]
        gt = row["ground_truth"].strip()
        pred = row["prediction"].strip()

        if not gt:
            raise ValueError(f"Empty ground truth for {image}")

        distance, operations = levenshtein_alignment(gt, pred)

        total_reference_chars += len(gt)
        total_edits += distance

        if gt == pred:
            exact_matches += 1
        else:
            incorrect_rows.append({
                "image": image,
                "ground_truth": gt,
                "prediction": pred,
                "confidence": row["confidence"],
                "edit_distance": distance,
                "error_operations": "; ".join(
                    f"{op}: {ref_char or '-'} -> {pred_char or '-'}"
                    for op, ref_char, pred_char in operations
                    if op != "match"
                )
            })

        for operation, ref_char, pred_char in operations:

            if operation == "match":
                continue

            operation_counts[operation] += 1

            key = (operation, ref_char, pred_char)
            character_errors[key].append(image)

        try:
            confidence_values.append(float(row["confidence"]))
        except (ValueError, TypeError):
            pass

    # Metrics
    exact_accuracy = exact_matches / total
    character_accuracy = (
        1 - total_edits / total_reference_chars
    )
    cer = total_edits / total_reference_chars
    string_error_rate = (total - exact_matches) / total

    average_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values else 0
    )

    # Metrics CSV
    metrics = [
        {"metric": "Experiment", "value": "EXP-VIN-REC-FT-001"},
        {"metric": "Test samples", "value": total},
        {"metric": "Exact matches", "value": exact_matches},
        {"metric": "Incorrect strings", "value": total - exact_matches},
        {"metric": "Exact-string accuracy (%)", "value": round(exact_accuracy * 100, 6)},
        {"metric": "Character accuracy (%)", "value": round(character_accuracy * 100, 6)},
        {"metric": "Character error rate", "value": round(cer, 8)},
        {"metric": "Character error rate (%)", "value": round(cer * 100, 6)},
        {"metric": "String error rate (%)", "value": round(string_error_rate * 100, 6)},
        {"metric": "Total character edits", "value": total_edits},
        {"metric": "Ground-truth characters", "value": total_reference_chars},
        {"metric": "Average confidence", "value": round(average_confidence, 6)},
        {"metric": "Substitutions", "value": operation_counts["substitution"]},
        {"metric": "Deletions", "value": operation_counts["deletion"]},
        {"metric": "Insertions", "value": operation_counts["insertion"]},
    ]

    write_csv(
        METRICS_CSV,
        ["metric", "value"],
        metrics
    )

    # Incorrect predictions CSV
    write_csv(
        INCORRECT_CSV,
        [
            "image",
            "ground_truth",
            "prediction",
            "confidence",
            "edit_distance",
            "error_operations",
        ],
        incorrect_rows
    )

    # Character error analysis CSV
    confusion_rows = []

    for (operation, ref_char, pred_char), images in sorted(
        character_errors.items()
    ):
        confusion_rows.append({
            "operation": operation,
            "ground_truth_character": ref_char,
            "predicted_character": pred_char,
            "count": len(images),
            "example_images": "; ".join(images),
        })

    write_csv(
        CONFUSIONS_CSV,
        [
            "operation",
            "ground_truth_character",
            "predicted_character",
            "count",
            "example_images",
        ],
        confusion_rows
    )

    # Independent summary
    summary = [
        "INDEPENDENT VIN OCR RECOGNITION EVALUATION",
        "===========================================",
        "Experiment: EXP-VIN-REC-FT-001",
        f"Test samples: {total}",
        f"Exact matches: {exact_matches}",
        f"Incorrect strings: {total - exact_matches}",
        f"Exact-string accuracy: {exact_accuracy * 100:.4f}%",
        f"Character accuracy: {character_accuracy * 100:.4f}%",
        f"CER: {cer:.6f}",
        f"CER (%): {cer * 100:.4f}%",
        f"String error rate: {string_error_rate * 100:.4f}%",
        f"Total character edits: {total_edits}",
        f"Ground-truth characters: {total_reference_chars}",
        f"Average confidence: {average_confidence:.6f}",
        "",
        "ERROR BREAKDOWN",
        "---------------",
        f"Substitutions: {operation_counts['substitution']}",
        f"Deletions: {operation_counts['deletion']}",
        f"Insertions: {operation_counts['insertion']}",
        "",
        "OUTPUT FILES",
        "------------",
        METRICS_CSV.name,
        INCORRECT_CSV.name,
        CONFUSIONS_CSV.name,
    ]

    SUMMARY_TXT.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8"
    )

    print("\nEvaluation analysis completed successfully.")
    print(f"Samples: {total}")
    print(f"Exact accuracy: {exact_accuracy * 100:.4f}%")
    print(f"Character accuracy: {character_accuracy * 100:.4f}%")
    print(f"CER: {cer:.6f}")
    print(f"Substitutions: {operation_counts['substitution']}")
    print(f"Deletions: {operation_counts['deletion']}")
    print(f"Insertions: {operation_counts['insertion']}")
    print(f"\nResults saved in: {EVAL_DIR}")


if __name__ == "__main__":
    main()

