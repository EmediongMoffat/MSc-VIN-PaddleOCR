
import json
import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

SOURCE_GT = PROJECT_DIR / "data" / "raw" / "det_test_labels.txt"

MANIFEST = (
    PROJECT_DIR / "results" / "evaluation"
    / "EXP-E2E-ICDAR15-002" / "eligible_gt_manifest.csv"
)

OUTPUT = (
    PROJECT_DIR / "results" / "evaluation"
    / "EXP-E2E-ICDAR15-002" / "gt_eligible_2065.txt"
)


def main():

    eligible = set()

    with MANIFEST.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        for row in csv.DictReader(file):
            eligible.add((
                row["source_image_basename"],
                int(row["annotation_index"])
            ))

    output_lines = []
    care_count = 0
    ignored_count = 0
    image_count = 0

    for line_number, line in enumerate(
        SOURCE_GT.read_text(encoding="utf-8-sig").splitlines(),
        start=1
    ):

        if not line.strip():
            continue

        image_path, annotation_text = line.split(None, 1)
        annotations = json.loads(annotation_text)

        basename = Path(
            image_path.replace("\\", "/")
        ).name

        filtered = []

        for index, annotation in enumerate(annotations, start=1):

            text = str(
                annotation.get("transcription", "")
            ).strip()

            # Preserve PaddleOCR do-not-care regions.
            if text == "###":
                filtered.append(annotation)
                ignored_count += 1
                continue

            # Exclude empty labels and invalid care regions.
            if not text:
                continue

            if (basename, index) in eligible:
                filtered.append(annotation)
                care_count += 1

        output_lines.append(
            image_path + "\t" + json.dumps(
                filtered, ensure_ascii=False
            )
        )

        image_count += 1

    if image_count != 500:
        raise RuntimeError(
            f"Expected 500 images, found {image_count}"
        )

    if care_count != 2065:
        raise RuntimeError(
            f"Expected 2065 care regions, found {care_count}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        "\n".join(output_lines) + "\n",
        encoding="utf-8"
    )

    print("PaddleOCR ground-truth input prepared.")
    print("Images:", image_count)
    print("Eligible care regions:", care_count)
    print("Do-not-care regions retained:", ignored_count)
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
