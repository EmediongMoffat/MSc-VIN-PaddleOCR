
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GT_SOURCE = ROOT / "data" / "raw" / "det_test_labels.txt"
PRED_SOURCE = (
    ROOT / "results" / "system" /
    "EXP-SYS-ICDAR15-001" / "system_results.txt"
)

OUTPUT = ROOT / "results" / "evaluation"
GT_OUT = OUTPUT / "icdar_e2e_gt"
PRED_OUT = OUTPUT / "icdar_e2e_pred"


def read_records(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records = {}

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()

            if not line:
                continue

            parts = line.split(None, 1)

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid line {line_number} in {path}"
                )

            image_path, json_text = parts
            items = json.loads(json_text)

            stem = Path(
                image_path.replace("\\", "/")
            ).stem

            if stem in records:
                raise ValueError(f"Duplicate image: {stem}")

            records[stem] = items

    return records


def convert(records, output_folder, is_ground_truth):
    output_folder.mkdir(parents=True, exist_ok=True)

    for old_file in output_folder.glob("*.txt"):
        old_file.unlink()

    for stem, items in records.items():
        lines = []

        for item in items:
            points = item.get("points", [])
            text = str(item.get("transcription", ""))

            if len(points) != 4:
                raise ValueError(
                    f"{stem}: expected 4 polygon points"
                )

            coordinates = [
                str(value)
                for point in points
                for value in point
            ]

            text = text.replace("\t", " ")
            text = text.replace("\n", " ").replace("\r", " ")

            if is_ground_truth:
                ignore_flag = "1" if text.strip() == "###" else "0"
                fields = coordinates + [ignore_flag, text]
            else:
                fields = coordinates + [text]

            lines.append("\t".join(fields))

        destination = output_folder / f"{stem}.txt"
        destination.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8"
        )


def main():
    gt = read_records(GT_SOURCE)
    predictions = read_records(PRED_SOURCE)

    gt_names = set(gt)
    pred_names = set(predictions)

    if gt_names != pred_names:
        print("ERROR: Ground-truth and prediction images differ.")
        print("Ground truth without predictions:",
              sorted(gt_names - pred_names)[:10])
        print("Predictions without ground truth:",
              sorted(pred_names - gt_names)[:10])
        return

    convert(gt, GT_OUT, True)
    convert(predictions, PRED_OUT, False)

    print("\nCONVERSION SUCCESSFUL")
    print("Ground-truth images:", len(gt))
    print("Prediction images:", len(predictions))
    print("Ground-truth folder:", GT_OUT)
    print("Prediction folder:", PRED_OUT)
    print("\nOriginal input files were not modified.")


if __name__ == "__main__":
    main()
