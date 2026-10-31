
"""ocr_metrics.py - OVST Section 4.2 VIN OCR evaluation."""

import csv
from pathlib import Path
from itertools import zip_longest

from rapidfuzz.distance import Levenshtein
from sklearn.metrics import f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REC_CSV = (
    PROJECT_ROOT / "results" / "experiments"
    / "EXP-VIN-REC-FT-001" / "evaluation"
    / "test_evaluation.csv"
)

COMBINED_CSV = (
    PROJECT_ROOT / "results" / "experiments"
    / "EXP-VIN-COMBINED-001" / "evaluation"
    / "combined_test_predictions.csv"
)

GAP = "<gap>"
VIN_ALPHABET = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"


def normalise(text):
    """VIN normalisation: uppercase, remove spaces, strip."""
    return text.upper().replace(" ", "").strip()


def align_chars(gt, pred):
    """Align characters using Levenshtein edit operations."""
    y_true, y_pred = [], []

    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(gt, pred):
        for true_char, pred_char in zip_longest(
            gt[i1:i2],
            pred[j1:j2],
            fillvalue=GAP
        ):
            y_true.append(true_char)
            y_pred.append(pred_char)

    return y_true, y_pred


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No records found in {path}")

    required = {"image", "ground_truth", "prediction"}
    missing = required - set(rows[0].keys())

    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    return rows


def evaluate_ocr(
    rows,
    name,
    output_dir,
    times_ms=None,
    hardware=""
):
    gts = [normalise(row["ground_truth"]) for row in rows]
    preds = [normalise(row["prediction"]) for row in rows]

    n = len(gts)

    if n != len(preds):
        raise ValueError("Every image must have exactly one prediction.")

    if any(not gt for gt in gts):
        raise ValueError("An empty ground-truth VIN was found.")

    # Image-level metrics
    correct = sum(g == p for g, p in zip(gts, preds))
    fail = n - correct
    accuracy = correct / n

    # Character-level metrics
    total_chars = sum(len(g) for g in gts)
    edits = sum(
        Levenshtein.distance(g, p)
        for g, p in zip(gts, preds)
    )
    cer = edits / total_chars

    y_true, y_pred = [], []

    for gt, pred in zip(gts, preds):
        true_chars, pred_chars = align_chars(gt, pred)
        y_true.extend(true_chars)
        y_pred.extend(pred_chars)

    char_correct = sum(
        t == p for t, p in zip(y_true, y_pred)
        if t != GAP
    )

    char_accuracy = char_correct / total_chars

    labels = sorted(set(y_true) - {GAP})

    f1_micro = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="micro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    # Optional timing
    mean_ms = None
    throughput = None

    if times_ms:
        mean_ms = sum(times_ms) / len(times_ms)
        throughput = 1000 / mean_ms

    # Failure records
    failures = []

    for index, (gt, pred, row) in enumerate(zip(gts, preds, rows), 1):
        if gt != pred:
            failures.append({
                "failure_number": index,
                "image": row["image"],
                "ground_truth": gt,
                "prediction": pred,
                "edit_distance": Levenshtein.distance(gt, pred)
            })

    output_dir.mkdir(parents=True, exist_ok=True)

    failure_path = output_dir / "ocr_failures.csv"

    with open(
        failure_path, "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "failure_number",
                "image",
                "ground_truth",
                "prediction",
                "edit_distance"
            ]
        )
        writer.writeheader()
        writer.writerows(failures)

    # Report
    report = [
        f"==== {name} ({n} images) ====",
        "",
        "IMAGE-LEVEL METRICS",
        f"Correct VINs: {correct}",
        f"Failed VINs: {fail}",
        f"Exact-string accuracy: {accuracy * 100:.4f}%",
        "",
        "CHARACTER-LEVEL METRICS",
        f"Total ground-truth characters: {total_chars}",
        f"Correct aligned characters: {char_correct}",
        f"Character accuracy: {char_accuracy * 100:.4f}%",
        f"Total edit operations: {edits}",
        f"CER: {cer:.6f}",
        f"CER (%): {cer * 100:.4f}%",
        "",
        "CHARACTER F1",
        f"F1 (micro): {f1_micro:.4f}",
        f"F1 (macro): {f1_macro:.4f}",
        "",
        "NORMALISATION",
        "Uppercase; spaces removed; leading/trailing whitespace stripped.",
        "",
        "FAILURE ANALYSIS",
        f"Number of failed VINs: {len(failures)}",
        f"Failure CSV: {failure_path.name}",
    ]

    if mean_ms is not None:
        report.extend([
            "",
            "INFERENCE TIMING",
            f"Mean latency: {mean_ms:.3f} ms/image",
            f"Throughput: {throughput:.3f} images/second",
            f"Hardware: {hardware}",
        ])
    else:
        report.extend([
            "",
            "INFERENCE TIMING",
            "Not available in the source CSV."
        ])

    report.extend(["", "INDIVIDUAL FAILURES"])

    for failure in failures:
        report.append(
            f"FAIL: GT={failure['ground_truth']} "
            f"PRED={failure['prediction']} "
            f"edits={failure['edit_distance']} "
            f"image={failure['image']}"
        )

    report_path = output_dir / "ocr_metrics_report.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print(f"\nReport saved: {report_path}")
    print(f"Failures saved: {failure_path}")

    return {
        "samples": n,
        "correct": correct,
        "accuracy": accuracy,
        "character_accuracy": char_accuracy,
        "cer": cer,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "failures": failures,
        "mean_ms": mean_ms,
        "throughput": throughput,
    }


def main():

    # Recognition-only evaluation
    rec_rows = load_csv(REC_CSV)

    evaluate_ocr(
        rows=rec_rows,
        name="EXP-VIN-REC-FT-001 - Recognition only",
        output_dir=REC_CSV.parent,
    )

    # Combined detection + recognition evaluation
    combined_rows = load_csv(COMBINED_CSV)

    times_ms = [
        float(row["total_time_s"]) * 1000
        for row in combined_rows
        if row.get("total_time_s")
    ]

    if len(times_ms) != len(combined_rows):
        raise ValueError(
            "Some combined pipeline records have missing total_time_s."
        )

    evaluate_ocr(
        rows=combined_rows,
        name="EXP-VIN-COMBINED-001 - Full pipeline",
        output_dir=COMBINED_CSV.parent,
        times_ms=times_ms,
        hardware="NVIDIA GeForce RTX 4050, batch 1",
    )


if __name__ == "__main__":
    main()
