# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

# MSc project root
PROJECT_DIR = os.path.abspath(os.path.join(__dir__, ".."))

# PaddleOCR repository
PADDLEOCR_DIR = os.path.join(PROJECT_DIR, "PaddleOCR")

# Allow imports from PaddleOCR
sys.path.insert(0, PADDLEOCR_DIR)
sys.path.insert(1, PROJECT_DIR)

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import cv2
import json
import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program

# ============================================================
# EXPERIMENT: EXP-VIN-DET-FT-002
# INFERENCE-TIME BOUNDING BOX EXPANSION
# ============================================================

# Additional padding on EACH side of the original box.
# 0.10 = 10% of original box width per horizontal side.
# 0.30 = 30% of original box height per vertical side.

PAD_X_RATIO = 0.10
PAD_Y_RATIO = 0.30


def expand_detection_box(box, image_width, image_height):
    """
    Expand a predicted detection polygon around its centre.

    Original coordinates are preserved separately.
    Expanded coordinates are clipped to image boundaries.

    Returns:
        original_points: original detector prediction
        expanded_points: padded prediction
    """

    original_points = np.asarray(
        box, dtype=np.float32
    ).reshape(-1, 2)

    # Centre of the original polygon
    centre_x = np.mean(original_points[:, 0])
    centre_y = np.mean(original_points[:, 1])

    # Expand horizontally and vertically
    expanded_points = original_points.copy()

    expanded_points[:, 0] = (
        centre_x
        + (original_points[:, 0] - centre_x)
        * (1 + 2 * PAD_X_RATIO)
    )

    expanded_points[:, 1] = (
        centre_y
        + (original_points[:, 1] - centre_y)
        * (1 + 2 * PAD_Y_RATIO)
    )

    # Keep coordinates inside the image
    expanded_points[:, 0] = np.clip(
        expanded_points[:, 0], 0, image_width - 1
    )

    expanded_points[:, 1] = np.clip(
        expanded_points[:, 1], 0, image_height - 1
    )

    return (
        original_points.astype(int).tolist(),
        expanded_points.astype(int).tolist()
    )

def draw_det_res(dt_boxes, config, img, img_name, save_path):
    import cv2

    src_im = img
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, os.path.basename(img_name))
    cv2.imwrite(save_path, src_im)
    logger.info("The detected Image saved in {}".format(save_path))


@paddle.no_grad()
def main():
    global_config = config["Global"]

    # build model
    model = build_model(config["Architecture"])

    load_model(config, model)
    # build post process
    post_process_class = build_post_process(config["PostProcess"])

    # create data ops
    transforms = []
    for op in config["Eval"]["dataset"]["transforms"]:
        op_name = list(op)[0]
        if "Label" in op_name:
            continue
        elif op_name == "KeepKeys":
            op[op_name]["keep_keys"] = ["image", "shape"]
        transforms.append(op)

    ops = create_operators(transforms, global_config)

    save_res_path = config["Global"]["save_res_path"]
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    model.eval()
    infer_img = config["Global"]["infer_img"]
    # Support txt file containing one image path per line
    if infer_img.endswith(".txt") and os.path.isfile(infer_img):
        with open(infer_img, "r") as f:
            img_file_list = [line.strip() for line in f if line.strip()]
    else:
        img_file_list = get_image_file_list(infer_img)
    with open(save_res_path, "wb") as fout:
        for file in img_file_list:
            logger.info("infer_img: {}".format(file))
            with open(file, "rb") as f:
                img = f.read()
                data = {"image": img}
            batch = transform(data, ops)

            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[1], axis=0)
            images = paddle.to_tensor(images)
            preds = model(images)
            post_result = post_process_class(preds, shape_list)

            src_img = cv2.imread(file)

            dt_boxes_json = []
            # parser boxes if post_result is dict
            if isinstance(post_result, dict):
                det_box_json = {}
                for k in post_result.keys():
                    boxes = post_result[k][0]["points"]
                    dt_boxes_list = []
                    for box in boxes:
                        tmp_json = {"transcription": ""}
                        tmp_json["points"] = np.array(box).tolist()
                        dt_boxes_list.append(tmp_json)
                    det_box_json[k] = dt_boxes_list
                    save_det_path = os.path.dirname(
                        config["Global"]["save_res_path"]
                    ) + "/det_results_{}/".format(k)
                    draw_det_res(boxes, config, src_img, file, save_det_path)
            else:
                boxes = post_result[0]["points"]
                
                dt_boxes_json = []
                expanded_boxes = []
                
                image_height, image_width = src_img.shape[:2]

                # Expand every detector prediction
                for box in boxes:

                    original_points, expanded_points = expand_detection_box(
                        box,
                        image_width,
                        image_height
                )

                # Keep both coordinate sets in the prediction output
                tmp_json = {
                    "transcription": "",
                    "original_points": original_points,
                    "points": expanded_points
                }

                dt_boxes_json.append(tmp_json)

                # Visualisation uses expanded coordinates
                expanded_boxes.append(expanded_points)

            # Save expanded visualisations
            save_det_path = (
                os.path.dirname(config["Global"]["save_res_path"])
                + "/det_results/"
            )

            draw_det_res(
                expanded_boxes,
                config,
                src_img,
                file,
                save_det_path
            )
            otstr = file + "\t" + json.dumps(dt_boxes_json) + "\n"
            fout.write(otstr.encode())

    logger.info("success!")

    logger.info("========================================")
    logger.info("VIN DETECTION INFERENCE COMPLETED")
    logger.info("Experiment: EXP-VIN-DET-FT-002")
    logger.info("Horizontal padding per side: {}%".format(
        PAD_X_RATIO * 100
    ))
    logger.info("Vertical padding per side: {}%".format(
        PAD_Y_RATIO * 100
    ))
    logger.info("Original and expanded coordinates saved.")
    logger.info("========================================")


if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess()
    main()
