
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np
import paddle
from paddlex import create_predictor


# ============================================================
# EXP-VIN-COMBINED-001
# Controlled inference speed benchmark
# Detector-only, recogniser-only and full pipeline
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    ROOT / "data" / "processed" / "paddleocr_vin"
    / "EXP-YOLO-PPOCR-CONVERT-001_20261028_132836_267609"
    / "detection"
)

LABEL_FILE = DATA_ROOT / "det_gt_test.txt"

DET_MODEL = (
    ROOT / "results" / "experiments"
    / "EXP-VIN-COMBINED-001" / "detector_inference"
)

REC_MODEL = (
    ROOT / "results" / "experiments"
    / "EXP-VIN-REC-FT-001" / "inference"
)

EXISTING_CSV = (
    ROOT / "results" / "experiments"
    / "EXP-VIN-COMBINED-001" / "evaluation"
    / "combined_test_predictions.csv"
)

OUT_DIR = (
    ROOT / "results" / "experiments"
    / "EXP-VIN-COMBINED-001" / "speed_benchmark"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_RUNS = 10
X_PADDING = 0.10
Y_PADDING = 0.30


def sync_gpu():
    if paddle.is_compiled_with_cuda():
        paddle.device.cuda.synchronize()


def result_to_dict(result):
    if isinstance(result, dict):
        return result

    for attr in ("json", "res", "data"):
        value = getattr(result, attr, None)

        if callable(value):
            try:
                value = value()
            except Exception:
                continue

        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                continue

        if isinstance(value, dict):
            return value

    if hasattr(result, "to_dict"):
        value = result.to_dict()
        if isinstance(value, dict):
            return value

    raise TypeError(f"Cannot convert result type: {type(result)}")


def find_key(data, key):
    if isinstance(data, dict):
        if key in data:
            return data[key]

        for value in data.values():
            found = find_key(value, key)
            if found is not None:
                return found

    elif isinstance(data, list):
        for value in data:
            found = find_key(value, key)
            if found is not None:
                return found

    return None


def load_samples():
    samples = []

    with LABEL_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            image_path, annotation_text = line.split("\t", 1)
            annotations = json.loads(annotation_text)

            if not annotations:
                raise ValueError(
                    f"No annotation on test label line {line_number}"
                )

            samples.append({
                "filename": Path(image_path).name,
                "image_path": DATA_ROOT / image_path,
                "ground_truth": annotations[0]["transcription"]
            })

    return samples


def load_existing_crops(samples):
    if not EXISTING_CSV.is_file():
        raise FileNotFoundError(
            f"Existing combined evaluation CSV not found: {EXISTING_CSV}"
        )

    crop_lookup = {}

    with EXISTING_CSV.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        for row in csv.DictReader(file):
            crop_lookup[row["image"]] = row.get("crop_path", "")

    for sample in samples:
        relative_crop = crop_lookup.get(sample["filename"], "")

        if not relative_crop:
            raise FileNotFoundError(
                "No previously generated predicted crop for "
                f"{sample['filename']}"
            )

        crop_path = Path(relative_crop)

        if not crop_path.is_absolute():
            crop_path = ROOT / crop_path

        crop = cv2.imread(str(crop_path))

        if crop is None:
            raise FileNotFoundError(
                f"Could not load existing predicted crop: {crop_path}"
            )

        sample["crop"] = crop
        sample["crop_path"] = str(crop_path)

    return samples


def get_polygons(result):
    polygons = find_key(result, "dt_polys")

    if polygons is None:
        polygons = find_key(result, "polys")

    if polygons is None:
        raise KeyError("No dt_polys or polys in detector output")

    polygons = np.asarray(polygons, dtype=np.float32)

    if polygons.size == 0:
        return []

    return polygons.reshape(-1, 4, 2)


def select_vin_polygon(polygons):
    candidates = []

    for polygon in polygons:
        x_min = float(np.min(polygon[:, 0]))
        x_max = float(np.max(polygon[:, 0]))
        y_min = float(np.min(polygon[:, 1]))
        y_max = float(np.max(polygon[:, 1]))

        width = x_max - x_min
        height = y_max - y_min

        if width > 0 and height > 0:
            candidates.append((width * height, polygon))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def crop_detected_vin(image, polygon):
    image_height, image_width = image.shape[:2]

    x_min = float(np.min(polygon[:, 0]))
    x_max = float(np.max(polygon[:, 0]))
    y_min = float(np.min(polygon[:, 1]))
    y_max = float(np.max(polygon[:, 1]))

    width = x_max - x_min
    height = y_max - y_min

    x_min -= width * X_PADDING
    x_max += width * X_PADDING
    y_min -= height * Y_PADDING
    y_max += height * Y_PADDING

    x_min = max(0, int(np.floor(x_min)))
    y_min = max(0, int(np.floor(y_min)))
    x_max = min(image_width, int(np.ceil(x_max)))
    y_max = min(image_height, int(np.ceil(y_max)))

    if x_max <= x_min or y_max <= y_min:
        return None

    crop = image[y_min:y_max, x_min:x_max]

    return crop if crop.size else None


def run_detector(detector, image):
    results = list(detector.apply(image))

    if not results:
        raise RuntimeError("Detector returned no result")

    result_dict = result_to_dict(results[0])
    polygons = get_polygons(result_dict)

    if len(polygons) == 0:
        raise RuntimeError("No detection polygons")

    polygon = select_vin_polygon(polygons)

    if polygon is None:
        raise RuntimeError("No valid VIN polygon")

    return polygon


def run_recogniser(recogniser, crop):
    results = list(recogniser.apply(crop))

    if not results:
        raise RuntimeError("Recogniser returned no result")

    return results


def percentile95(values):
    return float(np.percentile(values, 95)) if values else 0.0


def calculate_statistics(rows, timing_key):
    successful = [
        row[timing_key]
        for row in rows
        if row["status"] == "SUCCESS" and row[timing_key] > 0
    ]

    if not successful:
        return {
            "successful": 0,
            "failed": len(rows),
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "throughput": 0.0,
            "total_seconds": 0.0
        }

    total_seconds = sum(successful)

    return {
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "mean_ms": mean(successful) * 1000,
        "median_ms": median(successful) * 1000,
        "p95_ms": percentile95(successful) * 1000,
        "throughput": len(successful) / total_seconds,
        "total_seconds": total_seconds
    }


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_gpu_information():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader"
            ],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return "GPU details unavailable through nvidia-smi"


def timed_detector(detector, image):
    sync_gpu()
    start = time.perf_counter()

    polygon = run_detector(detector, image)

    sync_gpu()
    elapsed = time.perf_counter() - start

    return polygon, elapsed


def timed_recogniser(recogniser, crop):
    sync_gpu()
    start = time.perf_counter()

    results = run_recogniser(recogniser, crop)

    sync_gpu()
    elapsed = time.perf_counter() - start

    return results, elapsed


def main():
    print("=" * 70)
    print("EXP-VIN-COMBINED-001: CONTROLLED SPEED BENCHMARK")
    print("=" * 70)

    if not paddle.is_compiled_with_cuda():
        raise RuntimeError("This benchmark requires the CUDA-enabled PaddlePaddle build.")

    if not DET_MODEL.is_dir():
        raise FileNotFoundError(f"Detector model not found: {DET_MODEL}")

    if not REC_MODEL.is_dir():
        raise FileNotFoundError(f"Recogniser model not found: {REC_MODEL}")

    samples = load_samples()
    samples = load_existing_crops(samples)

    images = []

    for sample in samples:
        image = cv2.imread(str(sample["image_path"]))

        if image is None:
            raise FileNotFoundError(
                f"Could not read test image: {sample['image_path']}"
            )

        sample["image"] = image
        images.append(image)

    print(f"Test images: {len(samples)}")
    print(f"Warm-up runs per model: {WARMUP_RUNS}")
    print("Loading detector...")
    detector = create_predictor(
        model_name="PP-OCRv5_mobile_det",
        model_dir=str(DET_MODEL),
        device="gpu:0"
    )

    print("Loading recogniser...")
    recogniser = create_predictor(
        model_name="PP-OCRv5_mobile_rec",
        model_dir=str(REC_MODEL),
        device="gpu:0"
    )

    # --------------------------------------------------------
    # WARM-UP: detector
    # --------------------------------------------------------
    print("\n1/6 Warming up detector...")

    for index in range(WARMUP_RUNS):
        run_detector(detector, images[index % len(images)])

    sync_gpu()

    # --------------------------------------------------------
    # WARM-UP: recogniser
    # --------------------------------------------------------
    print("2/6 Warming up recogniser...")

    for index in range(WARMUP_RUNS):
        run_recogniser(
            recogniser,
            samples[index % len(samples)]["crop"]
        )

    sync_gpu()

    # --------------------------------------------------------
    # WARM-UP: complete pipeline
    # --------------------------------------------------------
    print("3/6 Warming up full pipeline...")

    for index in range(WARMUP_RUNS):
        image = images[index % len(images)]
        polygon = run_detector(detector, image)
        crop = crop_detected_vin(image, polygon)

        if crop is None:
            raise RuntimeError("Warm-up pipeline could not create crop")

        run_recogniser(recogniser, crop)

    sync_gpu()

    # --------------------------------------------------------
    # BENCHMARK A: detector only
    # --------------------------------------------------------
    print("\n4/6 Benchmarking detector-only inference...")

    detector_rows = []

    for index, sample in enumerate(samples, start=1):
        row = {
            "image": sample["filename"],
            "status": "FAILED",
            "detection_time_s": 0.0,
            "error": ""
        }

        try:
            _, elapsed = timed_detector(detector, sample["image"])
            row["detection_time_s"] = elapsed
            row["status"] = "SUCCESS"
        except Exception as error:
            row["error"] = str(error)

        detector_rows.append(row)
        print(
            f"Detector [{index}/{len(samples)}]: "
            f"{row['status']}"
        )

    # --------------------------------------------------------
    # BENCHMARK B: recogniser only
    # --------------------------------------------------------
    print("\n5/6 Benchmarking recogniser-only inference...")

    recogniser_rows = []

    for index, sample in enumerate(samples, start=1):
        row = {
            "image": sample["filename"],
            "crop_path": sample["crop_path"],
            "status": "FAILED",
            "recognition_time_s": 0.0,
            "error": ""
        }

        try:
            _, elapsed = timed_recogniser(
                recogniser,
                sample["crop"]
            )
            row["recognition_time_s"] = elapsed
            row["status"] = "SUCCESS"
        except Exception as error:
            row["error"] = str(error)

        recogniser_rows.append(row)
        print(
            f"Recogniser [{index}/{len(samples)}]: "
            f"{row['status']}"
        )

    # --------------------------------------------------------
    # BENCHMARK C: complete pipeline
    # --------------------------------------------------------
    print("\n6/6 Benchmarking complete detection-to-recognition pipeline...")

    pipeline_rows = []

    for index, sample in enumerate(samples, start=1):
        row = {
            "image": sample["filename"],
            "status": "FAILED",
            "full_pipeline_time_s": 0.0,
            "error": ""
        }

        try:
            sync_gpu()
            start = time.perf_counter()

            polygon = run_detector(detector, sample["image"])
            crop = crop_detected_vin(sample["image"], polygon)

            if crop is None:
                raise RuntimeError("Could not create predicted VIN crop")

            run_recogniser(recogniser, crop)

            sync_gpu()
            elapsed = time.perf_counter() - start

            row["full_pipeline_time_s"] = elapsed
            row["status"] = "SUCCESS"

        except Exception as error:
            sync_gpu()
            row["error"] = str(error)

        pipeline_rows.append(row)

        print(
            f"Pipeline [{index}/{len(samples)}]: "
            f"{row['status']}"
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------
    write_csv(OUT_DIR / "detector_timings.csv", detector_rows)
    write_csv(OUT_DIR / "recogniser_timings.csv", recogniser_rows)
    write_csv(OUT_DIR / "full_pipeline_timings.csv", pipeline_rows)

    detector_stats = calculate_statistics(
        detector_rows, "detection_time_s"
    )

    recogniser_stats = calculate_statistics(
        recogniser_rows, "recognition_time_s"
    )

    pipeline_stats = calculate_statistics(
        pipeline_rows, "full_pipeline_time_s"
    )

    gpu_info = get_gpu_information()

    summary = f"""
EXP-VIN-COMBINED-001
CONTROLLED VIN OCR SPEED BENCHMARK
==================================

BENCHMARK SETUP
---------------
Test images: {len(samples)}
Warm-up runs per model: {WARMUP_RUNS}
Device: GPU (gpu:0)
GPU: {gpu_info}
PaddlePaddle: {paddle.__version__}
Python: {platform.python_version()}
Operating system: {platform.platform()}
Batch size: 1
Detector: PP-OCRv5_mobile_det
Recogniser: PP-OCRv5_mobile_rec
Detector precision: FP32
Recogniser precision: FP32
Inference backend: PaddleX predictor; TensorRT activation not confirmed

TIMING METHOD
-------------
GPU synchronisation performed before and after timed operations.
Warm-up runs excluded from measured results.
Images and predicted crops preloaded into memory.
Model loading excluded from timings.
Disk image loading excluded from timings.
Detector-only measures detector inference and result processing.
Recogniser-only measures recognition on previously generated predicted crops.
Full pipeline measures detection, predicted-box selection, crop creation
and recognition on the original test images.
Full pipeline excludes disk image loading and crop-file writing.

STATISTICS
----------
Successful timings only are used for latency and throughput statistics.
Mean, median and P95 are reported in milliseconds.
Throughput = successful samples / sum of successful inference times.

DETECTOR-ONLY
-------------
Successful: {detector_stats['successful']}
Failed: {detector_stats['failed']}
Mean latency: {detector_stats['mean_ms']:.4f} ms
Median latency: {detector_stats['median_ms']:.4f} ms
P95 latency: {detector_stats['p95_ms']:.4f} ms
Throughput: {detector_stats['throughput']:.4f} images/second

RECOGNISER-ONLY
---------------
Successful: {recogniser_stats['successful']}
Failed: {recogniser_stats['failed']}
Mean latency: {recogniser_stats['mean_ms']:.4f} ms
Median latency: {recogniser_stats['median_ms']:.4f} ms
P95 latency: {recogniser_stats['p95_ms']:.4f} ms
Throughput: {recogniser_stats['throughput']:.4f} images/second

FULL DETECTION-TO-RECOGNITION PIPELINE
--------------------------------------
Successful: {pipeline_stats['successful']}
Failed: {pipeline_stats['failed']}
Mean latency: {pipeline_stats['mean_ms']:.4f} ms
Median latency: {pipeline_stats['median_ms']:.4f} ms
P95 latency: {pipeline_stats['p95_ms']:.4f} ms
Throughput: {pipeline_stats['throughput']:.4f} images/second

OUTPUT FILES
------------
detector_timings.csv
recogniser_timings.csv
full_pipeline_timings.csv
benchmark_summary.txt

IMPORTANT
---------
These are new controlled speed benchmark measurements.
They are separate from the original combined evaluation timings.
FP32 refers to the exported model tensor types.
TensorRT activation has not been independently verified.
"""

    summary_path = OUT_DIR / "benchmark_summary.txt"
    summary_path.write_text(summary.strip() + "\n", encoding="utf-8")

    metadata = f"""
Experiment: EXP-VIN-COMBINED-001
Benchmark: Controlled speed benchmark
Date/time: {time.strftime("%Y-%m-%d %H:%M:%S")}
Warm-up runs: {WARMUP_RUNS}
Measured test images: {len(samples)}
Device: gpu:0
GPU: {gpu_info}
PaddlePaddle: {paddle.__version__}
Python: {platform.python_version()}
Batch size: 1
Precision: FP32 exported model tensors
TensorRT active: Not confirmed
X crop padding: {X_PADDING}
Y crop padding: {Y_PADDING}
"""

    (OUT_DIR / "benchmark_metadata.txt").write_text(
        metadata.strip() + "\n",
        encoding="utf-8"
    )

    print("\n" + summary)
    print(f"\nAll benchmark results saved to:\n{OUT_DIR}")
    print("\nBENCHMARK COMPLETED.")


if __name__ == "__main__":
    main()
