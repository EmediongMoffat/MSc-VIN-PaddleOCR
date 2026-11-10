
from pathlib import Path
from collections import Counter

# Dataset annotation folder

#ANNOTATION_DIR = Path("dataset/train_gt")
ANNOTATION_DIR = Path("dataset/test_gt")
def validate_annotations():
    files = sorted(ANNOTATION_DIR.glob("gt_*.txt"))

    total_lines = 0
    malformed_lines = []
    annotation_counts = Counter()

    if not ANNOTATION_DIR.exists():
        print(f"ERROR: Folder not found: {ANNOTATION_DIR}")
        return

    for file_path in files:
        try:
            lines = file_path.read_text(
                encoding="utf-8-sig"
            ).splitlines()
        except OSError as error:
            print(f"Could not read {file_path.name}: {error}")
            continue

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            total_lines += 1
            parts = line.split(",", 8)
            valid = len(parts) == 9

            if valid:
                try:
                    coordinates = [int(value) for value in parts[:8]]
                except ValueError:
                    valid = False

            if not valid:
                malformed_lines.append(
                    (file_path.name, line_number, line)
                )
            else:
                annotation_counts["valid"] += 1

    print("\n===== ICDAR2015 ANNOTATION VALIDATION =====")
    print(f"Annotation folder: {ANNOTATION_DIR}")
    print(f"Annotation files: {len(files)}")
    print(f"Total non-empty lines: {total_lines}")
    print(f"Valid-format lines: {annotation_counts['valid']}")
    print(f"Malformed lines: {len(malformed_lines)}")

    if malformed_lines:
        print("\nFirst 10 malformed lines:")
        for filename, line_number, line in malformed_lines[:10]:
            print(f"{filename}, line {line_number}: {line}")
    else:
        print("\nAll non-empty lines have the expected format.")

    print("\nNote: This checks annotation format, not coordinate")
    print("bounds, transcription correctness, or model accuracy.")


if __name__ == "__main__":
    validate_annotations()
