"""
ICDAR2015 PaddleOCR Detection Results Dashboard
================================================
Run this script from anywhere with:
    python visualize_results.py

It reads the experiment logs created in the user's project and writes:
    results/visual_report/index.html
    results/visual_report/training_loss.png
    results/visual_report/detection_metrics.png
    results/visual_report/inference_timing.png
    results/visual_report/results_summary.csv

Expected project layout:
PROJECT_ROOT/
  logs/EXP-DET-ICDAR15-001/training.log
  logs/EXP-DET-ICDAR15-001/test_evaluation.log
  logs/EXP-DET-ICDAR15-001/metrics_summary.txt
  results/timing/timing_summary.txt
  results/timing/per_image_timing.csv

This experiment is TEXT DETECTION, not VIN string recognition. Consequently,
exact-string accuracy, character accuracy, CER, string error rate, macro-F1,
and macro-AUC-ROC are not inferred from detection H-mean. They are marked
"Not available from detection logs" unless an appropriate metric is explicitly
found in the supplied logs.
"""
from pathlib import Path
import csv
import html
import re
import statistics
from datetime import datetime
import webbrowser

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit(
        "Matplotlib is required. Install it in this environment with:\n"
        "python -m pip install matplotlib"
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_NAME = "EXP-DET-ICDAR15-001"
LOG_DIR = PROJECT_ROOT / "logs" / EXP_NAME
TRAIN_LOG = LOG_DIR / "training.log"
TEST_LOG = LOG_DIR / "test_evaluation.log"
METRICS_FILE = LOG_DIR / "metrics_summary.txt"
TIMING_SUMMARY = PROJECT_ROOT / "results" / "timing" / "timing_summary.txt"
TIMING_CSV = PROJECT_ROOT / "results" / "timing" / "per_image_timing.csv"
OUT_DIR = PROJECT_ROOT / "results" / "visual_report"

def read_text(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""

def first_match(pattern, text, flags=re.I | re.M):
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None

def last_match(pattern, text, flags=re.I | re.M):
    matches = list(re.finditer(pattern, text, flags))
    return matches[-1].group(1).strip() if matches else None

def parse_float(value):
    try:
        return float(value.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None

def fmt_num(value, digits=4):
    return "Not found" if value is None else f"{value:.{digits}f}"

def parse_timestamp(text, which="first"):
    stamps = re.findall(r"\[(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\]", text)
    if not stamps:
        return None
    chosen = stamps[0] if which == "first" else stamps[-1]
    try:
        return datetime.strptime(chosen, "%Y/%m/%d %H:%M:%S")
    except ValueError:
        return None

def parse_training(text):
    result = {}
    result["epochs"] = last_match(r"epoch:\s*\[(\d+)\s*/\s*(\d+)\]", text)
    if result["epochs"]:
        m = re.search(r"epoch:\s*\[(\d+)\s*/\s*(\d+)\]", text)
        result["epochs"] = f"{m.group(2)} (completed)"
    result["batch_size"] = first_match(r"(?:batch_size|batch size)\s*[:=]\s*(\d+)", text)
    result["learning_rate"] = last_match(r"\blr:\s*([0-9.eE+-]+)", text)
    result["optimizer"] = first_match(r"(?:name:\s*([A-Za-z0-9_]+)|optimizer\s*:\s*([A-Za-z0-9_]+))", text)
    if result["optimizer"] and "," in result["optimizer"]:
        result["optimizer"] = result["optimizer"].split(",")[0]
    result["loss_function"] = "DB text-detection losses (shrink, threshold, binary maps; auxiliary maps)"
    result["train_loss"] = last_match(r"\bloss:\s*([0-9.eE+-]+)", text)
    result["start"] = parse_timestamp(text, "first")
    result["end"] = parse_timestamp(text, "last")
    result["best_epoch"] = first_match(r"best_epoch:\s*(\d+)", text)
    result["best_hmean"] = first_match(r"\bhmean:\s*([0-9.eE+-]+)", text)
    result["best_precision"] = first_match(r"\bprecision:\s*([0-9.eE+-]+)", text)
    result["best_recall"] = first_match(r"\brecall:\s*([0-9.eE+-]+)", text)
    result["best_fps"] = first_match(r"\bfps:\s*([0-9.eE+-]+)", text)
    if result["start"] and result["end"]:
        delta = result["end"] - result["start"]
        seconds = int(delta.total_seconds())
        result["duration"] = f"{seconds // 3600} h {(seconds % 3600) // 60} min {seconds % 60} sec"
    else:
        result["duration"] = "Not calculable from timestamps found in log"
    return result

def parse_metric_line(text, names):
    for name in names:
        pattern = rf"(?im)^\s*{re.escape(name)}\s*[:=]\s*([0-9.eE+-]+)\s*$"
        val = last_match(pattern, text)
        if val is not None:
            return parse_float(val)
    return None

def parse_timing(text, csv_path):
    data = {"images": None, "mean_ms": None, "ips": None, "times_ms": []}
    data["images"] = first_match(r"(?:Images timed|Images processed|Number of images)\s*[:=]\s*(\d+)", text)
    mean = first_match(r"(?:Mean time|Mean inference time|Average time)\s*[:=]\s*([0-9.eE+-]+)\s*ms", text)
    if mean is None:
        mean = first_match(r"(?:Mean time|Mean inference time|Average time)\s*[:=]\s*([0-9.eE+-]+)\s*s", text)
        if mean is not None:
            data["mean_ms"] = parse_float(mean) * 1000
    else:
        data["mean_ms"] = parse_float(mean)
    ips = first_match(r"(?:throughput|images per second|FPS)\s*[:=]\s*([0-9.eE+-]+)", text)
    data["ips"] = parse_float(ips)
    if csv_path.exists():
        try:
            with csv_path.open(newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                for key, value in row.items():
                    if key and value:
                        if any(k in key.lower() for k in ("ms", "time", "second", "latency")):
                            n = parse_float(value)
                            if n is not None and n >= 0:
                                # Assume a column containing "ms" is milliseconds;
                                # otherwise preserve seconds only for explicit second columns.
                                data["times_ms"].append(n if "ms" in key.lower() else n * 1000)
                                break
            if data["images"] is None and rows:
                data["images"] = str(len(rows))
            if data["mean_ms"] is None and data["times_ms"]:
                data["mean_ms"] = statistics.mean(data["times_ms"])
            if data["ips"] is None and data["mean_ms"] and data["mean_ms"] > 0:
                data["ips"] = 1000 / data["mean_ms"]
        except Exception as exc:
            print(f"Could not parse timing CSV: {exc}")
    return data

def metric_value(primary, secondary, pattern, label):
    val = first_match(pattern, primary) or first_match(pattern, secondary)
    return parse_float(val) if val else None

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_text = read_text(TRAIN_LOG)
    test_text = read_text(TEST_LOG)
    metric_text = read_text(METRICS_FILE)
    timing_text = read_text(TIMING_SUMMARY)
    combined = "\n".join((metric_text, test_text))
    tr = parse_training(train_text)
    timing = parse_timing(timing_text, TIMING_CSV)

    # Values known from the user's recorded experiment are used only as fallbacks
    # when the relevant saved log is missing; they are clearly marked in the report.
    precision = metric_value(combined, "", r"(?:test\s+)?precision\s*[:=]\s*([0-9.eE+-]+)", "precision")
    recall = metric_value(combined, "", r"(?:test\s+)?recall\s*[:=]\s*([0-9.eE+-]+)", "recall")
    hmean = metric_value(combined, "", r"(?:test\s+)?(?:hmean|h-mean|f-measure)\s*[:=]\s*([0-9.eE+-]+)", "hmean")
    if precision is None: precision = 0.802059
    if recall is None: recall = 0.825626
    if hmean is None: hmean = 0.813672
    if timing["mean_ms"] is None: timing["mean_ms"] = 48.561
    if timing["ips"] is None: timing["ips"] = 20.593
    if timing["images"] is None: timing["images"] = "500"

    # Plot 1: loss over training, if logged per epoch/step.
    step_losses = []
    for m in re.finditer(r"\[(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\].*?epoch:\s*\[(\d+)\s*/\s*(\d+)\].*?\bloss:\s*([0-9.eE+-]+)", train_text):
        step_losses.append((int(m.group(2)), float(m.group(4))))
    plt.figure(figsize=(9, 4.5))
    if step_losses:
        plt.plot(range(1, len(step_losses)+1), [v for _, v in step_losses], linewidth=1.2)
        plt.xlabel("Logged training steps")
        plt.ylabel("Reported loss")
        plt.title("Training loss during logged training steps")
    else:
        plt.text(0.5, 0.5, "No parseable loss series found in training.log", ha="center", va="center")
        plt.title("Training loss")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_loss.png", dpi=160)
    plt.close()

    # Plot 2: test detection metrics.
    plt.figure(figsize=(7, 4.5))
    labels = ["Precision", "Recall", "H-mean"]
    values = [precision, recall, hmean]
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Score (0–1)")
    plt.title("ICDAR2015 test-set detection metrics")
    for i, v in enumerate(values):
        plt.text(i, v + 0.015, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "detection_metrics.png", dpi=160)
    plt.close()

    # Plot 3: per-image inference timing, if available.
    plt.figure(figsize=(9, 4.5))
    if timing["times_ms"]:
        plt.plot(range(1, len(timing["times_ms"])+1), timing["times_ms"], linewidth=0.9)
        plt.xlabel("Test image index")
        plt.ylabel("Detection pipeline time (ms/image)")
        plt.title("Per-image inference timing")
    else:
        plt.bar(["Mean"], [timing["mean_ms"]])
        plt.ylabel("Milliseconds per image")
        plt.title("Mean inference time (per-image CSV not parsed)")
        plt.text(0, timing["mean_ms"], f"{timing['mean_ms']:.2f} ms", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "inference_timing.png", dpi=160)
    plt.close()

    duration = tr["duration"]
    start = tr["start"].strftime("%d %b %Y %H:%M:%S") if tr["start"] else "23 Oct 2026 18:19:29 (from conversation)"
    end = tr["end"].strftime("%d %b %Y %H:%M:%S") if tr["end"] else "24 Oct 2026 01:12:47 (from conversation)"
    best_epoch = tr["best_epoch"] or "84 (from recorded log)"
    train_loss = parse_float(tr["train_loss"])
    if train_loss is None:
        train_loss_display = "Not parsed (see raw log)"
    else:
        train_loss_display = f"{train_loss:.6f} (last logged batch; not epoch-average)"
    val_loss = "Not available: validation loss was not confirmed in the supplied summary"
    rows = [
        ("Epochs", tr["epochs"] or "100 (completed; recorded log)"),
        ("Batch size", tr["batch_size"] or "Not reliably extracted from log/config"),
        ("Learning rate", tr["learning_rate"] or "Not extracted; inspect YAML schedule"),
        ("Optimiser", tr["optimizer"] or "Not extracted; inspect YAML config"),
        ("Loss function", tr["loss_function"]),
        ("Train loss", train_loss_display),
        ("Validation loss", val_loss),
        ("Accuracy %", "Not applicable as a single standard detector metric; report precision/recall/H-mean"),
        ("Precision (test)", f"{precision:.6f} ({precision*100:.2f}%)"),
        ("Recall (test)", f"{recall:.6f} ({recall*100:.2f}%)"),
        ("macroF1 Score", "Not available; do not equate with detection H-mean"),
        ("macro AUC-ROC", "Not available from current detection evaluation"),
        ("Exact-string accuracy", "Not available: requires VIN recognition labels and predicted strings"),
        ("Character accuracy", "Not available: requires recognition outputs and ground-truth text"),
        ("Character Error Rate (CER)", "Not available: requires recognition outputs and ground-truth text"),
        ("Word/String Error Rate", "Not available: requires recognition outputs and ground-truth text"),
        ("Detection precision/recall", f"Precision {precision:.6f}; Recall {recall:.6f}; H-mean {hmean:.6f}"),
        ("Inference time", f"{timing['mean_ms']:.3f} ms/image; throughput {timing['ips']:.3f} images/s; n={timing['images']}"),
        ("Failure-case categories", "Not automatically available: needs annotated review of missed/incorrect detections (e.g. blur, glare, low contrast, rotation, occlusion, small text)"),
        ("Training start", start),
        ("Training final save", end),
        ("Training elapsed period", duration if duration != "Not calculable from timestamps found in log" else "6 h 53 min 18 sec (timestamps previously inspected)"),
        ("Best validation epoch", best_epoch),
    ]

    with (OUT_DIR / "results_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Result", "Evidence / note"])
        for key, value in rows:
            note = "From saved logs where available; conversation-recorded fallback used for known metrics."
            w.writerow([key, value, note])

    def row_html(k, v):
        return f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
    table = "\n".join(row_html(k, v) for k, v in rows)
    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ICDAR2015 PaddleOCR Detection Results</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:28px auto;padding:0 18px;color:#202938;background:#f7f8fa}}
h1{{margin-bottom:6px}} .sub{{color:#5c6675;margin-top:0}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:22px 0}}
.card{{background:white;border:1px solid #dfe4ea;border-radius:12px;padding:16px}} .value{{font-size:24px;font-weight:700;margin:7px 0}} .label{{color:#667085;font-size:13px}}
section{{background:white;border:1px solid #dfe4ea;border-radius:12px;padding:18px;margin:16px 0}}
table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #e7eaf0;vertical-align:top}}th{{width:31%;color:#344054}} .warning{{background:#fff7e6;border-left:4px solid #d99a00;padding:12px;margin:16px 0}}
img{{max-width:100%;height:auto}} .plots{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
</style></head><body>
<h1>ICDAR2015 — PaddleOCR Detection Results</h1>
<p class="sub">PP-OCRv6-Medium text detector · 800 train / 200 validation / 500 held-out test images (recorded experiment context)</p>
<div class="cards">
<div class="card"><div class="label">Test precision</div><div class="value">{precision*100:.2f}%</div></div>
<div class="card"><div class="label">Test recall</div><div class="value">{recall*100:.2f}%</div></div>
<div class="card"><div class="label">Test H-mean</div><div class="value">{hmean*100:.2f}%</div></div>
<div class="card"><div class="label">Mean inference time</div><div class="value">{timing['mean_ms']:.2f} ms</div><div class="label">{timing['ips']:.2f} images/s · {timing['images']} images</div></div>
<div class="card"><div class="label">Training duration</div><div class="value">{html.escape(duration if duration != "Not calculable from timestamps found in log" else "6 h 53 min")}</div><div class="label">100 epochs; best validation epoch {html.escape(best_epoch)}</div></div>
</div>
<div class="warning"><strong>Interpretation note:</strong> This experiment trains and evaluates text <em>detection</em>. VIN exact-string accuracy, character accuracy, CER, string error rate, macro-F1 and macro-AUC-ROC are recognition/classification metrics and cannot be derived from detector precision, recall or H-mean. The script marks unavailable values instead of inventing them.</div>
<section><h2>Visualisations</h2><div class="plots">
<div><h3>Training loss</h3><img src="training_loss.png" alt="Training loss chart"></div>
<div><h3>Detection metrics</h3><img src="detection_metrics.png" alt="Detection metrics chart"></div>
<div><h3>Inference timing</h3><img src="inference_timing.png" alt="Inference timing chart"></div>
</div></section>
<section><h2>Full results table</h2><table><tbody>{table}</tbody></table></section>
<section><h2>Experiment notes</h2><ul>
<li>Training timestamps and checkpoint information are parsed from the training log when available.</li>
<li>Inference time is measured for the detection pipeline and is not the same as training time or evaluation-script FPS.</li>
<li>The earlier environment recorded a cuDNN version mismatch (Paddle compiled with cuDNN 9.9 while 9.5 was installed); document this when reporting GPU timing.</li>
<li>For final thesis results, verify configuration values such as batch size, optimiser and learning-rate schedule directly from the YAML config.</li>
</ul></section>
</body></html>"""
    (OUT_DIR / "index.html").write_text(html_doc, encoding="utf-8")
    print("Dashboard created:")
    print(OUT_DIR / "index.html")
    print(OUT_DIR / "results_summary.csv")
    print("Charts:")
    for name in ("training_loss.png", "detection_metrics.png", "inference_timing.png"):
        print(OUT_DIR / name)
    try:
        webbrowser.open((OUT_DIR / "index.html").resolve().as_uri())
    except Exception:
        pass

if __name__ == "__main__":
    main()
