
import os
import sys
import yaml
import logging
import paddle
import paddle.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PADDLEOCR = os.path.join(ROOT, "PaddleOCR")
sys.path.insert(0, PADDLEOCR)

from ppocr.data import build_dataloader
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.utils.utility import set_seed

CONFIG = os.path.join(ROOT, "configs", "EXP-VIN-DET-FT-002.yml")
CHECKPOINT = os.path.join(
    ROOT, "checkpoints", "EXP-VIN-DET-FT-002", "best_accuracy.pdparams"
)
OUTPUT = os.path.join(
    ROOT, "results", "experiments", "EXP-VIN-DET-FT-002",
    "evaluation", "detection_validation_loss.txt"
)

with open(CONFIG, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
# Create evaluation-only label maps in memory.
# The original experiment YAML is not modified.
config["Eval"]["dataset"]["transforms"] = [
    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
    {"DetLabelEncode": None},
    {"MakeBorderMap": {
        "shrink_ratio": 0.4,
        "thresh_min": 0.3,
        "thresh_max": 0.7,
        "total_epoch": 200
    }},
    {"MakeShrinkMap": {
        "shrink_ratio": 0.4,
        "min_text_size": 8,
        "total_epoch": 200
    }},
    {"NormalizeImage": {
        "scale": "1./255.",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "order": "hwc"
    }},
    {"ToCHWImage": None},
    {"KeepKeys": {
        "keep_keys": [
            "image", "threshold_map", "threshold_mask",
            "shrink_map", "shrink_mask"
        ]
    }}
]
set_seed(config["Global"].get("seed", 1024))

use_gpu = config["Global"].get("use_gpu", False)
device = "gpu:0" if use_gpu and paddle.is_compiled_with_cuda() else "cpu"
paddle.set_device(device)

model = build_model(config["Architecture"])

state = paddle.load(CHECKPOINT)
model.set_state_dict(state)

# Enable DBHead training outputs, but keep backbone and neck in evaluation mode.
model.eval()
model.head.train()

loss_class = build_loss(config["Loss"])

logger = logging.getLogger("detection_validation_loss")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

valid_loader = build_dataloader(config, "Eval", device, logger)

loss_totals = {}
sample_count = 0
batch_count = 0

def pad_to_multiple(tensor, multiple=128):
    height, width = tensor.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
        tensor = F.pad(tensor, [0, pad_w, 0, pad_h], value=0)
        tensor = tensor.squeeze(1)
    else:
        tensor = F.pad(tensor, [0, pad_w, 0, pad_h], value=0)

    return tensor


with paddle.no_grad():
    for batch in valid_loader:
        images = pad_to_multiple(batch[0])

        labels = [
            images,
            pad_to_multiple(batch[1]),
            pad_to_multiple(batch[2]),
            pad_to_multiple(batch[3]),
            pad_to_multiple(batch[4]),
        ]

        predictions = model(images)
        losses = loss_class(predictions, labels)

        batch_size = len(images)
        sample_count += batch_size
        batch_count += 1

        for name, value in losses.items():
            number = float(value.numpy().mean())
            loss_totals[name] = (
                loss_totals.get(name, 0.0) + number * batch_size
            )

results = {
    name: total / sample_count
    for name, total in loss_totals.items()
}

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write("EXP-VIN-DET-FT-002 Validation Loss\n")
    f.write(f"Checkpoint: {CHECKPOINT}\n")
    f.write(f"Validation samples: {sample_count}\n")
    f.write(f"Validation batches: {batch_count}\n")
    f.write(f"Device: {device}\n\n")

    for name, value in results.items():
        f.write(f"{name}: {value:.8f}\n")
        print(f"{name}: {value:.8f}")

print("\nDetection validation loss calculation completed.")
print("Saved to:", OUTPUT)
