import os
import sys
import yaml
import paddle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PADDLEOCR = os.path.join(ROOT, "PaddleOCR")
sys.path.insert(0, PADDLEOCR)

from ppocr.data import build_dataloader
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.utils.utility import set_seed

CONFIG = os.path.join(ROOT, "configs", "EXP-VIN-REC-FT-001.yml")
CHECKPOINT = os.path.join(
    ROOT, "checkpoints", "EXP-VIN-REC-FT-001", "best_accuracy.pdparams"
)
OUTPUT = os.path.join(
    ROOT, "results", "experiments", "EXP-VIN-REC-FT-001",
    "evaluation", "recognition_validation_loss.txt"
)

# Load the original training configuration
with open(CONFIG, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

set_seed(config["Global"].get("seed", 1024))

# Use the same device configuration
use_gpu = config["Global"].get("use_gpu", False)
device = "gpu:0" if use_gpu and paddle.is_compiled_with_cuda() else "cpu"
paddle.set_device(device)

# Build the recognition model exactly as configured
from ppocr.postprocess import build_post_process

post_process = build_post_process(
    config["PostProcess"], config["Global"]
)
char_num = len(post_process.character)

if config["Architecture"]["Head"]["name"] == "MultiHead":
    out_channels_list = {"CTCLabelDecode": char_num}
    if config["PostProcess"]["name"] == "SARLabelDecode":
        char_num -= 2
    elif config["PostProcess"]["name"] == "NRTRLabelDecode":
        char_num -= 3

    out_channels_list["CTCLabelDecode"] = char_num
    out_channels_list["NRTRLabelDecode"] = char_num + 3
    config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
else:
    config["Architecture"]["Head"]["out_channels"] = char_num

config["Architecture"]["return_all_feats"] = True
model = build_model(config["Architecture"])

# Load the trained best checkpoint
state = paddle.load(CHECKPOINT)
model.set_state_dict(state)
model.train()

# Build the original configured recognition loss
loss_class = build_loss(config["Loss"])

# Load the original validation dataset
import logging
logger = logging.getLogger("validation_loss")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

valid_loader = build_dataloader(config, "Eval", device, logger)

loss_totals = {}
batch_count = 0
sample_count = 0

with paddle.no_grad():
    for batch in valid_loader:
        images = batch[0]
        predictions = model(images, data=batch[1:])
        print("Prediction keys:", predictions.keys())
        print("Head output keys:", predictions["head_out"].keys())
        losses = loss_class(predictions["head_out"], batch)


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
    f.write("EXP-VIN-REC-FT-001 Validation Loss\n")
    f.write(f"Checkpoint: {CHECKPOINT}\n")
    f.write(f"Validation samples: {sample_count}\n")
    f.write(f"Validation batches: {batch_count}\n")
    f.write(f"Device: {device}\n\n")

    for name, value in results.items():
        f.write(f"{name}: {value:.8f}\n")
        print(f"{name}: {value:.8f}")

print("\nValidation loss calculation completed.")
print("Saved to:", OUTPUT)
