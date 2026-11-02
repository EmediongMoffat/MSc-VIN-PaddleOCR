
import re
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LOG_FILE = Path(r"checkpoints\EXP-VIN-DET-FT-001\train.log")
OUT_DIR = ROOT / "results/experiments/EXP-VIN-DET-FT-001/evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not LOG_FILE.exists():
    raise FileNotFoundError(
        f"Training log not found: {LOG_FILE}\n"
        "Copy your EXP001 training log into the logs folder "
        "with the filename EXP-VIN-DET-FT-001_training.log"
    )

text = LOG_FILE.read_text(encoding="utf-8", errors="replace")

# Training history
training_rows = []

pattern = re.compile(
    r"\[(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\].*?"
    r"epoch: \[(\d+)/(\d+)\], global_step: (\d+), "
    r"lr: ([\d.eE+-]+), loss: ([\d.eE+-]+), "
    r"loss_shrink_maps: ([\d.eE+-]+), "
    r"loss_threshold_maps: ([\d.eE+-]+), "
    r"loss_binary_maps: ([\d.eE+-]+), "
    r"loss_cbn: ([\d.eE+-]+).*?"
    r"max_mem_reserved: (\d+) MB, "
    r"max_mem_allocated: (\d+) MB"
)

for match in pattern.finditer(text):
    values = match.groups()

    training_rows.append({
        "timestamp": values[0],
        "epoch": int(values[1]),
        "total_epochs": int(values[2]),
        "global_step": int(values[3]),
        "learning_rate": float(values[4]),
        "loss": float(values[5]),
        "loss_shrink_maps": float(values[6]),
        "loss_threshold_maps": float(values[7]),
        "loss_binary_maps": float(values[8]),
        "loss_cbn": float(values[9]),
        "max_mem_reserved_MB": int(values[10]),
        "max_mem_allocated_MB": int(values[11]),
    })

# Validation metrics
metric_rows = []

metric_pattern = re.compile(
    r"\[(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\].*?"
    r"cur metric, precision: ([\d.eE+-]+), "
    r"recall: ([\d.eE+-]+), "
    r"hmean: ([\d.eE+-]+), "
    r"fps: ([\d.eE+-]+)"
)

for match in metric_pattern.finditer(text):
    values = match.groups()
    metric_rows.append({
        "timestamp": values[0],
        "precision": float(values[1]),
        "recall": float(values[2]),
        "hmean": float(values[3]),
        "fps": float(values[4]),
    })

# Best metric records
best_pattern = re.compile(
    r"best metric, hmean: ([\d.eE+-]+), "
    r"is_float16: (\w+), precision: ([\d.eE+-]+), "
    r"recall: ([\d.eE+-]+), fps: ([\d.eE+-]+), "
    r"best_epoch: (\d+)"
)

best_rows = [
    {
        "hmean": float(m.group(1)),
        "is_float16": m.group(2),
        "precision": float(m.group(3)),
        "recall": float(m.group(4)),
        "fps": float(m.group(5)),
        "best_epoch": int(m.group(6)),
    }
    for m in best_pattern.finditer(text)
]

def save_csv(path, rows):
    if not rows:
        print(f"No records found for {path.name}")
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

save_csv(OUT_DIR / "det001_training_history.csv", training_rows)
save_csv(OUT_DIR / "det001_validation_metrics.csv", metric_rows)
save_csv(OUT_DIR / "det001_best_metrics.csv", best_rows)

errors = text.count(
    "TypeError: only 0-dimensional arrays can be converted to Python scalars"
)

summary = [
    "EXP-VIN-DET-FT-001 LOG ANALYSIS",
    "=" * 45,
    f"Training log: {LOG_FILE.name}",
    f"Training records extracted: {len(training_rows)}",
    f"Validation evaluations extracted: {len(metric_rows)}",
    f"Best metric records: {len(best_rows)}",
    f"EastRandomCrop TypeErrors recorded: {errors}",
]

if training_rows:
    summary += [
        f"First recorded epoch: {training_rows[0]['epoch']}",
        f"Last recorded epoch: {training_rows[-1]['epoch']}",
        f"Last recorded training loss: {training_rows[-1]['loss']:.6f}",
        f"Maximum reserved GPU memory: "
        f"{max(r['max_mem_reserved_MB'] for r in training_rows)} MB",
        f"Maximum allocated GPU memory: "
        f"{max(r['max_mem_allocated_MB'] for r in training_rows)} MB",
    ]

if best_rows:
    best = max(best_rows, key=lambda r: r["hmean"])
    summary += [
        "",
        "BEST RECORDED VALIDATION RESULT",
        f"Best epoch: {best['best_epoch']}",
        f"Precision: {best['precision']:.6f}",
        f"Recall: {best['recall']:.6f}",
        f"H-mean: {best['hmean']:.6f}",
        f"FPS: {best['fps']:.6f}",
    ]

summary += [
    "",
    "Note: Training loss values are recorded log steps,",
    "not calculated epoch-average losses.",
    "Validation metrics are not held-out test metrics.",
]

summary_path = OUT_DIR / "det001_log_analysis_summary.txt"
summary_path.write_text("\n".join(summary), encoding="utf-8")

print("\n".join(summary))
print("\nReports saved in:", OUT_DIR)
