# ==============================================================
# PP-OCRv6 MEDIUM DETECTION
# POST-TRAINING VALIDATION LOSS CALCULATION
#
# Experiment: EXP-DET-ICDAR15-001
# Checkpoint: best_accuracy (best validation checkpoint)
# Expected checkpoint epoch: 84
# Validation dataset: det_val_200_labels.txt
#
# This script:
#   1. Loads the trained best_accuracy checkpoint.
#   2. Loads the original 200-image validation dataset.
#   3. Generates DB shrink and threshold target maps.
#   4. Calculates the configured DBLoss (DiceFocalLoss).
#   5. Calculates the average validation loss.
#   6. Saves the result to the results folder.
#
# It does NOT retrain the model or update its weights.
# ==============================================================

import copy
import os
import sys

# --------------------------------------------------------------
# 1. Set up the PaddleOCR repository path
# --------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Main project directory
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# PaddleOCR repository
REPO_DIR = os.path.join(PROJECT_DIR, "PaddleOCR")

# Add PaddleOCR to Python's module search path
sys.path.insert(0, REPO_DIR)

import paddle

from ppocr.data import build_dataloader
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.utils.save_load import load_model

import tools.program as program


# --------------------------------------------------------------
# 2. Helper function to retrieve transforms from YAML
# --------------------------------------------------------------

def get_transform(transforms, transform_name):

    for operation in transforms:

        if isinstance(operation, dict) and transform_name in operation:

            return copy.deepcopy(operation)

    raise KeyError(
        f"Transform not found in configuration: {transform_name}"
    )


# --------------------------------------------------------------
# 3. Main validation-loss calculation
# --------------------------------------------------------------

def main():

    print("\n" + "=" * 65)
    print("PP-OCRv6 MEDIUM DETECTION - VALIDATION LOSS")
    print("=" * 65)

    # ----------------------------------------------------------
    # 4. Load the original YAML configuration
    # ----------------------------------------------------------

    config, device, logger, _ = program.preprocess()

    # This calculation is performed on a single CPU process.
    config["Global"]["distributed"] = False

    print("\nConfiguration loaded successfully.")
    print("Device:", device)

    # ----------------------------------------------------------
    # 5. Locate the best saved checkpoint
    # ----------------------------------------------------------

    checkpoint_prefix = os.path.abspath(
        os.path.join(
            config["Global"]["save_model_dir"],
            "best_accuracy"
        )
    )

    checkpoint_file = checkpoint_prefix + ".pdparams"

    if not os.path.isfile(checkpoint_file):

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n{checkpoint_file}"
        )

    print("\nCheckpoint found:")
    print(checkpoint_file)

    # Tell PaddleOCR to load this checkpoint.
    config["Global"]["checkpoints"] = checkpoint_prefix

    # ----------------------------------------------------------
    # 6. Build and load the trained detector
    # ----------------------------------------------------------

    model = build_model(config["Architecture"])

    load_model(
        config,
        model,
        model_type=config["Architecture"].get(
            "model_type",
            "det"
        )
    )

    print("\nTrained detector loaded successfully.")

    # ----------------------------------------------------------
    # 7. Build the configured DBLoss
    # ----------------------------------------------------------

    loss_function = build_loss(config["Loss"])

    print("\nLoss configuration:")
    print("Loss function:", config["Loss"]["name"])
    print("Main loss:", config["Loss"]["main_loss_type"])

    # ----------------------------------------------------------
    # 8. Prepare the validation dataset
    # ----------------------------------------------------------

    val_dataset = copy.deepcopy(config["Eval"]["dataset"])

    validation_label_file = val_dataset["label_file_list"][0]

    if not os.path.isfile(validation_label_file):

        raise FileNotFoundError(
            "\nValidation label file not found:\n"
            f"{validation_label_file}"
        )

    print("\nValidation label file found:")
    print(validation_label_file)

    # Training transforms required to generate DB target maps.
    train_transforms = config["Train"]["dataset"]["transforms"]

    border_map_transform = get_transform(
        train_transforms,
        "MakeBorderMap"
    )

    shrink_map_transform = get_transform(
        train_transforms,
        "MakeShrinkMap"
    )

    keep_keys_transform = get_transform(
        train_transforms,
        "KeepKeys"
    )

    training_keep_keys = (
        keep_keys_transform["KeepKeys"]["keep_keys"]
    )

    # ----------------------------------------------------------
    # 9. Retrieve deterministic validation transforms
    # ----------------------------------------------------------

    eval_transforms = val_dataset["transforms"]

    decode_transform = get_transform(
        eval_transforms,
        "DecodeImage"
    )

    label_transform = get_transform(
        eval_transforms,
        "DetLabelEncode"
    )

    resize_transform = get_transform(
        eval_transforms,
        "DetResizeForTest"
    )

    normalize_transform = get_transform(
        eval_transforms,
        "NormalizeImage"
    )

    chw_transform = get_transform(
        eval_transforms,
        "ToCHWImage"
    )

    # ----------------------------------------------------------
    # 10. Construct validation transforms
    # ----------------------------------------------------------

    # The validation pipeline:
    #
    # DecodeImage
    # DetLabelEncode
    # DetResizeForTest
    # MakeBorderMap
    # MakeShrinkMap
    # NormalizeImage
    # ToCHWImage
    # KeepKeys
    #
    # No random training augmentation is applied.
    #
    # DetResizeForTest resizes the image and corresponding
    # polygon coordinates before generating the DB target maps.

    val_dataset["transforms"] = [

        decode_transform,

        label_transform,

        resize_transform,

        border_map_transform,

        shrink_map_transform,

        normalize_transform,

        chw_transform,

        {
            "KeepKeys": {
                "keep_keys": training_keep_keys
            }
        }
    ]

    val_dataset.setdefault("ratio_list", [1.0])

    # ----------------------------------------------------------
    # 11. Build the validation data loader
    # ----------------------------------------------------------

    val_config = copy.deepcopy(config)

    # Replace the training dataset with the validation dataset.
    val_config["Train"]["dataset"] = val_dataset

    # Use the validation loader configuration.
    val_config["Train"]["loader"] = copy.deepcopy(
        config["Eval"]["loader"]
    )

    # Ensure deterministic validation order.
    val_config["Train"]["loader"]["shuffle"] = False

    # Keep all validation images.
    val_config["Train"]["loader"]["drop_last"] = False

    # Batch size 1 supports different image dimensions.
    val_config["Train"]["loader"]["batch_size_per_card"] = 1

    # Avoid multiprocessing complications on Windows.
    val_config["Train"]["loader"]["num_workers"] = 0

    val_config["Global"]["distributed"] = False

    val_loader = build_dataloader(
        val_config,
        "Train",
        device,
        logger
    )

    if len(val_loader) == 0:

        raise RuntimeError(
            "The validation data loader contains no batches."
        )

    print("\nValidation data loader created.")
    print("Validation batches:", len(val_loader))

    # ----------------------------------------------------------
    # 12. Set the correct model mode for DBLoss
    # ----------------------------------------------------------

    # IMPORTANT:
    # The outer model must be in training mode because the
    # PP-OCRv6 DBHead uses this state to return:
    #
    #   Channel 0: Shrink maps
    #   Channel 1: Threshold maps
    #   Channel 2: Binary maps
    #
    # DBLoss requires these three output channels.
    #
    # We use no_grad() later, so gradients are not calculated.
    # We also freeze BatchNorm and Dropout behaviour below.

    model.train()

    # Keep BatchNorm and Dropout layers in evaluation mode.
    # This prevents running-statistic updates and stochastic
    # dropout during validation.

    frozen_layer_count = 0

    for layer in model.sublayers():

        layer_name = layer.__class__.__name__.lower()

        if (
            "batchnorm" in layer_name
            or "dropout" in layer_name
        ):

            layer.eval()

            frozen_layer_count += 1

    print("\nModel mode configured for DBLoss.")
    print(
        "BatchNorm/Dropout layers held in evaluation mode:",
        frozen_layer_count
    )

    # ----------------------------------------------------------
    # 13. Calculate validation loss
    # ----------------------------------------------------------

    total_weighted_loss = 0.0
    total_images = 0

    print("\nStarting validation-loss calculation...\n")

    with paddle.no_grad():

        for batch_number, batch in enumerate(
            val_loader,
            start=1
        ):

            # Input image tensor.
            images = batch[0]

            # Forward pass through the trained detector.
            predictions = model(images)

            # --------------------------------------------------
            # PP-OCRv6 OUTPUT HANDLING
            # --------------------------------------------------
            #
            # PP-OCRv6 may return predictions as a dictionary:
            #
            #     {"maps": PaddleTensor}
            #
            # DBLoss expects the original prediction dictionary.
            # Therefore, we inspect the "maps" tensor but retain
            # the dictionary for the loss calculation.
            # --------------------------------------------------

            if isinstance(predictions, dict):

                if "maps" not in predictions:

                    raise KeyError(
                        "The model output dictionary does not "
                        "contain the required 'maps' key.\n"
                        f"Available keys: {list(predictions.keys())}"
                    )

                prediction_maps = predictions["maps"]

            elif isinstance(predictions, paddle.Tensor):

                # Support architectures that return a tensor
                # directly rather than a dictionary.
                prediction_maps = predictions

                # DBLoss expects the prediction dictionary.
                predictions = {
                    "maps": prediction_maps
                }

            else:

                raise TypeError(
                    "Unexpected model output type: "
                    f"{type(predictions)}"
                )

            # Confirm that the maps are a Paddle tensor.
            if not isinstance(prediction_maps, paddle.Tensor):

                raise TypeError(
                    "The 'maps' output is not a Paddle tensor.\n"
                    f"Received: {type(prediction_maps)}"
                )

            # Confirm that the DBHead returned three maps.
            if (
                len(prediction_maps.shape) != 4
                or prediction_maps.shape[1] < 3
            ):

                raise RuntimeError(
                    "DBHead did not return the three required "
                    "training output maps.\n"
                    f"Actual prediction shape: "
                    f"{prediction_maps.shape}\n"
                    "Expected shape: [batch, 3, height, width]."
                )

            # Display prediction details for the first batch.
            if batch_number == 1:

                print("Model output type:", type(predictions))
                print("Prediction dictionary keys:",
                      list(predictions.keys()))
                print("Prediction maps shape:",
                      prediction_maps.shape)

            # --------------------------------------------------
            # Calculate DBLoss using the original predictions
            # dictionary and validation target maps.
            # --------------------------------------------------

            loss_dictionary = loss_function(
                predictions,
                batch
            )

            # Confirm that DBLoss returned a loss dictionary.
            if (
                not isinstance(loss_dictionary, dict)
                or "loss" not in loss_dictionary
            ):

                raise RuntimeError(
                    "DBLoss did not return a dictionary "
                    "containing the 'loss' key.\n"
                    f"Received: {type(loss_dictionary)}"
                )

            # Extract the total configured loss.
            loss_tensor = loss_dictionary["loss"]

            batch_loss = float(
                loss_tensor.mean().numpy()
            )

            batch_size = int(images.shape[0])

            # Accumulate loss weighted by the number of images.
            total_weighted_loss += batch_loss * batch_size

            total_images += batch_size

            # Display progress.
            if (
                batch_number == 1
                or batch_number % 20 == 0
                or batch_number == len(val_loader)
            ):

                print(
                    f"Batch {batch_number}/{len(val_loader)} "
                    f"| Loss: {batch_loss:.6f} "
                    f"| Images processed: {total_images}"
                )

    # ----------------------------------------------------------
    # 14. Calculate average validation loss
    # ----------------------------------------------------------

    if total_images == 0:

        raise RuntimeError(
            "No validation images were processed."
        )

    validation_loss = total_weighted_loss / total_images

    # ----------------------------------------------------------
    # 15. Save validation-loss result
    # ----------------------------------------------------------

    output_directory = os.path.join(
        PROJECT_DIR,
        "results",
        "validation"
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    output_file = os.path.join(
        output_directory,
        "PP-OCRv6_medium_det_validation_loss.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("PP-OCRv6 MEDIUM TEXT DETECTION\n")
        file.write("POST-TRAINING VALIDATION LOSS\n")
        file.write("=" * 60 + "\n")

        file.write("Experiment: EXP-DET-ICDAR15-001\n")
        file.write("Checkpoint: best_accuracy\n")
        file.write("Checkpoint epoch: 84 (best validation epoch)\n")

        file.write(
            f"Checkpoint file: {checkpoint_file}\n"
        )

        file.write(
            f"Validation labels: {validation_label_file}\n"
        )

        file.write(
            f"Images evaluated: {total_images}\n"
        )

        file.write(
            f"Loss function: {config['Loss']['name']}\n"
        )

        file.write(
            f"Main loss: {config['Loss']['main_loss_type']}\n"
        )

        file.write(
            f"Average validation loss: {validation_loss:.8f}\n"
        )

        file.write(
            "Validation preprocessing: DetResizeForTest, "
            "MakeBorderMap and MakeShrinkMap; "
            "no random training augmentation.\n"
        )

        file.write(
            "Model weights updated: No\n"
        )

        file.write(
            "Gradient calculation: Disabled\n"
        )

    # ----------------------------------------------------------
    # 16. Display final result
    # ----------------------------------------------------------

    print("\n" + "=" * 65)
    print("VALIDATION LOSS CALCULATION COMPLETED")
    print("=" * 65)

    print("Experiment: EXP-DET-ICDAR15-001")
    print("Checkpoint: best_accuracy")
    print("Checkpoint epoch: 84")
    print("Images evaluated:", total_images)

    print(
        f"\nAVERAGE VALIDATION LOSS: {validation_loss:.8f}"
    )

    print("\nResult saved to:")
    print(output_file)

    print("\nModel weights were not modified.")
    print("No retraining was performed.")
    print("=" * 65)


# --------------------------------------------------------------
# 17. Run script
# --------------------------------------------------------------

if __name__ == "__main__":

    main()