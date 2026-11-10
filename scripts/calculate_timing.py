from pathlib import Path
import csv
import statistics
import re
from datetime import datetime

# Main project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
PADDLEOCR_DIR = PROJECT_DIR / "PaddleOCR"
LOG_DIR = PROJECT_DIR / "logs" / "EXP-DET-ICDAR15-001"
RESULT_DIR = PROJECT_DIR / "results" / "timing"

# Find a log containing per-image prediction times
candidates = [
    LOG_DIR / "inference_timing.log",
    LOG_DIR / "benchmark_detection.log",
    PADDLEOCR_DIR / "inference_results" / "inference_timing.log",
]

log_file = next((p for p in candidates if p.exists()), None)

if log_file is None:
    print("No saved per-image timing log was found.")
    print("This script needs a log from an inference run before it can calculate timing.")
    print("Expected format: The predict time of <image path>: <seconds>")
    raise SystemExit(1)

pattern = re.compile(
    r"The predict time of (.*?):\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
)

rows = []
with log_file.open("r", encoding="utf-8", errors="replace") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            rows.append({
                "image": Path(match.group(1)).name,
                "inference_time_seconds": float(match.group(2)),
            })

if not rows:
    print(f"No per-image prediction times found in: {log_file}")
    print("A fresh inference run must save its console output to a timing log.")
    raise SystemExit(1)

times = [row["inference_time_seconds"] for row in rows]
mean_time = statistics.mean(times)
median_time = statistics.median(times)
fps = 1.0 / mean_time if mean_time > 0 else 0.0

RESULT_DIR.mkdir(parents=True, exist_ok=True)

csv_path = RESULT_DIR / "per_image_timing.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["image", "inference_time_seconds"]
    )
    writer.writeheader()
    writer.writerows(rows)

summary_path = RESULT_DIR / "timing_summary.txt"
with summary_path.open("w", encoding="utf-8") as f:
    f.write("ICDAR2015 detector inference timing\n")
    f.write(f"Date: {datetime.now().isoformat(timespec='seconds')}\n")
    f.write(f"Source log: {log_file}\n")
    f.write(f"Images timed: {len(times)}\n")
    f.write(f"Mean time (seconds/image): {mean_time:.6f}\n")
    f.write(f"Mean time (milliseconds/image): {mean_time * 1000:.3f}\n")
    f.write(f"Median time (milliseconds/image): {median_time * 1000:.3f}\n")
    f.write(f"Throughput (images/second): {fps:.3f}\n")
    f.write("\nTiming includes the prediction() function's preprocessing, ")
    f.write("model inference and post-processing, but excludes image loading ")
    f.write("and drawing/saving visualisations.\n")
    f.write("Check the GPU/cuDNN compatibility warning before treating GPU ")
    f.write("performance figures as final.\n")

print(f"Images timed: {len(times)}")
print(f"Mean time: {mean_time * 1000:.3f} ms/image")
print(f"Throughput: {fps:.3f} images/second")
print(f"Per-image CSV: {csv_path}")
print(f"Summary: {summary_path}")