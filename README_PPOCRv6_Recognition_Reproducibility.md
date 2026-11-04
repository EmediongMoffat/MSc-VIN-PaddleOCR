# PP-OCRv6 Medium Recognition – Experiment Reproducibility

## 1. Project overview

This experiment evaluates the text-recognition performance of PaddleOCR's PP-OCRv6 Medium recognition model using the ICDAR 2015 scene-text dataset.

The purpose is to establish a general OCR recognition baseline before progressing towards the MSc project's main objective: developing an accurate and efficient OCR system for Vehicle Identification Number (VIN) recognition.

The experiment focuses on:
- Preparing word-level recognition images from the ICDAR 2015 dataset.
- Running text recognition using PaddleOCR.
- Measuring exact-string accuracy and Character Error Rate (CER).
- Measuring character-level precision, recall and micro F1-score.
- Recording inference time and confidence scores.
- Analysing incorrect predictions to identify recognition weaknesses.

**Experiment type:** General OCR recognition baseline  
**Model:** PP-OCRv6 Medium Recognition (`PP-OCRv6_medium_rec`)  
**Dataset:** ICDAR 2015 scene-text recognition samples  
**Framework:** PaddleOCR 3.x  
**Evaluation:** Recognition-only evaluation on generated word crops

This experiment is separate from the later VIN-specific PP-OCRv5 Mobile detection and recognition experiments.

---

## 2. Project structure

The existing project folder arrangement is retained. The following is a summary of the main files and folders relevant to this experiment.

```text
MSc-VIN-PaddleOCR/
│
├── configs/
│   └── PP-OCRv6_medium_rec.yml
│
├── scripts/
│   ├── create_icdar_recognition_dataset.py
│   ├── create_icdar_recognition_dataset_full.py
│   ├── evaluate_icdar_recognition.py
│   ├── analyse_ocr_errors.py
│   └── calculate_character_f1.py
│
├── train_data/
│   └── ic15_data/
│       ├── test/
│       ├── test_gt/
│       ├── rec_test/
│       │   ├── sample/
│       │   └── full/
│       ├── sample_labels.txt
│       ├── rec_gt_generated.txt
│       ├── rec_gt_test.txt
│       ├── icdar_recognition_results.csv
│       └── worst_ocr_errors.csv
│
├── BASELINE RESULTS/
│
├── PaddleOCR/
│
├── requirements-original-lock.txt
│
└── README_PPOCRv6_Recognition_Reproducibility.md
```

The `PaddleOCR/` directory contains the PaddleOCR repository. The project scripts, experiment configuration, dataset files and results are maintained in the existing project arrangement.

The structure above is a summary of the experiment-related files. It does not replace the complete project inventory.

---

## 3. Dataset description

The experiment uses the ICDAR 2015 scene-text dataset.

The dataset contains scene images with text annotations. Each annotation identifies a text region using four polygon coordinates and provides the corresponding text transcription.

The original annotation format is:

```text
x1,y1,x2,y2,x3,y3,x4,y4,text
```

The preparation scripts convert the annotated text regions into individual cropped images suitable for text recognition.

### Dataset preparation

Two scripts were developed for dataset preparation.

#### 3.1 Sample dataset preparation

**Script:** `create_icdar_recognition_dataset.py`

This script was used to test the crop-generation process on one selected test image.

It performs the following operations:

1. Loads `img_10.jpg`.
2. Loads the corresponding annotation file, `gt_img_10.txt`.
3. Reads the annotated text-region coordinates.
4. Excludes regions marked `###`.
5. Selects five valid text regions.
6. Applies perspective transformation to rectify each text region.
7. Saves the five cropped images under `rec_test/sample/`.
8. Creates `sample_labels.txt` containing the crop paths and corresponding text labels.

This script provides a small-scale check that the annotation parsing, cropping and label-generation process works before processing the full test set.

#### 3.2 Full dataset preparation

**Script:** `create_icdar_recognition_dataset_full.py`

This script processes the available ICDAR test images and their corresponding annotations.

The workflow is:

1. Reads the scene images from the `test/` directory.
2. Sorts the image filenames naturally to maintain numerical order.
3. Locates the corresponding annotation files in `test_gt/`.
4. Reads the four polygon coordinates and associated text transcription.
5. Excludes `###` regions.
6. Applies perspective transformation to crop and rectify valid text regions.
7. Saves the generated crops under `rec_test/full/`.
8. Creates `rec_gt_generated.txt` containing the relative crop paths and ground-truth text.

The generated label format is:

```text
image_path<TAB>ground_truth_text
```

The script also reports the number of scene images processed, valid crops generated, do-not-care regions skipped and invalid regions skipped.

**Important:** The full dataset-generation script removes previously generated `word_*.png` files in `rec_test/full/` before creating new crops. Existing generated crops should therefore be backed up before rerunning the script if they need to be retained.

---

## 4. Treatment of do-not-care regions

The ICDAR annotation marker `###` identifies text regions that should not be evaluated.

These are designated do-not-care regions rather than ordinary recognition labels.

The dataset-generation scripts deliberately exclude these regions from the recognition crops and labels.

Consequently:
- They are not treated as failed predictions.
- They are not included as normal recognition targets.
- Their exclusion is part of the dataset preparation procedure.

The number of excluded regions is recorded by the full dataset-generation script.

---

## 5. Recognition model

The experiment uses PaddleOCR's text-recognition interface:

```python
from paddleocr import TextRecognition

model = TextRecognition()
```

The evaluation script initialises the model using `TextRecognition()` without explicitly specifying a model name.

The experiment is associated with PP-OCRv6 Medium Recognition. However, the exact model identity used during the historical evaluation should be confirmed from the original model-loading output or saved model files.

The official PP-OCRv6 documentation describes the Medium recognition model as using a PPLCNetV4 backbone, LightSVTR and a multi-head CTC/NRTR decoder.

Official references:

- https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html
- https://github.com/PaddlePaddle/PaddleOCR

The published model benchmark figures are reference information only. They are not results obtained from this experiment.

---

## 6. Recognition evaluation

**Script:** `evaluate_icdar_recognition.py`

This script evaluates the model on the generated recognition crops.

### Evaluation procedure

1. Loads the ground-truth labels from `rec_gt_generated.txt`.
2. Creates the PaddleOCR text-recognition model.
3. Loads each generated crop.
4. Runs the model's `predict()` function.
5. Extracts the predicted text and recognition confidence.
6. Compares the prediction against the ground-truth transcription.
7. Calculates exact-string accuracy and edit distance.
8. Calculates the Character Error Rate (CER).
9. Records the processing time for each crop.
10. Exports the results to `icdar_recognition_results.csv`.

### Recorded output fields

| Field | Description |
|---|---|
| `image` | Crop filename |
| `ground_truth` | Correct annotated text |
| `prediction` | Model-generated text |
| `confidence` | Recognition confidence score |
| `exact_match` | Whether the prediction exactly matches the ground truth |
| `edit_distance` | Minimum character edits required to correct the prediction |
| `cer` | Character Error Rate for the prediction |
| `time_seconds` | Processing time for the prediction |

### Evaluation metrics

**Exact-string accuracy**

Measures the proportion of crops for which the predicted text matches the complete ground-truth transcription exactly.

**Character Error Rate (CER)**

Measures the number of character-level insertions, deletions and substitutions relative to the number of characters in the ground truth.

**Inference time**

The script records the elapsed time around each individual `model.predict()` call.

This is a per-crop recognition timing measurement. It should not automatically be interpreted as a complete end-to-end application benchmark.

---

## 7. Error analysis

**Script:** `analyse_ocr_errors.py`

This script analyses the recognition results saved in `icdar_recognition_results.csv`.

It:

1. Loads the saved recognition results.
2. Identifies predictions where `exact_match` is false.
3. Sorts incorrect predictions by CER, with edit distance used as a secondary sorting criterion.
4. Displays the 20 worst recognition errors.
5. Saves all incorrect predictions to `worst_ocr_errors.csv`.

The error-analysis output provides a way to investigate which text regions were difficult for the model to recognise.

It also helps identify possible patterns in recognition failures and areas that may require further investigation.

---

## 8. Character-level precision, recall and F1-score

**Script:** `calculate_character_f1.py`

This script calculates character-level precision, recall and micro F1-score using Levenshtein alignment.

It compares each ground-truth transcription with its corresponding prediction.

The alignment classifies characters into:

- **True positives (TP):** Characters correctly recognised.
- **False positives (FP):** Extra or substituted predicted characters.
- **False negatives (FN):** Missing or substituted ground-truth characters.

A substitution contributes one false positive and one false negative.

The script aggregates these counts across the evaluated dataset before calculating the metrics.

### Metrics

**Character-level precision**

Measures the proportion of predicted characters that are correct.

**Character-level recall**

Measures the proportion of ground-truth characters that were correctly recognised.

**Character-level micro F1-score**

Combines character-level precision and recall into one overall measure, calculated from the aggregated character counts.

These metrics evaluate character-level recognition performance. They are not image-level or word-level F1 scores.

---

## 9. Experiment outputs

The experiment produces the following main outputs.

| File or folder | Purpose |
|---|---|
| `sample_labels.txt` | Ground-truth labels for the five sample crops |
| `rec_gt_generated.txt` | Ground-truth labels for the full generated crop set |
| `rec_test/sample/` | Sample recognition crops |
| `rec_test/full/` | Full generated recognition crop set |
| `icdar_recognition_results.csv` | Individual recognition predictions and evaluation results |
| `worst_ocr_errors.csv` | Incorrect predictions sorted by CER |
| `BASELINE RESULTS/` | Saved baseline result images |

These outputs should be retained as experimental evidence.

The generated crops, labels, scripts and evaluation results should remain associated with the same experiment version.

---

## 10. Environment and software

The following environment information was recorded during inspection of the project environment:

| Component | Recorded version or status |
|---|---|
| Operating system | Windows |
| Python | 3.12.10 |
| PaddleOCR | 3.7.0 |
| PaddleX | 3.7.2 |
| PaddlePaddle | 3.2.0 |
| PaddlePaddle CUDA support | Not available in the inspected environment |
| Execution device | CPU |

**Environment qualification:** These versions describe the environment inspected for this project. They should only be treated as the exact historical experiment environment if confirmed to be the same environment used during the original PP-OCRv6 evaluation.

A package snapshot was saved as:

```text
requirements-original-lock.txt
```

This file should be retained as the record of the inspected package environment.

---

## 11. Reproducibility considerations

The experiment includes the main scripts required to prepare the recognition crops, evaluate the model and analyse the results.

However, some information was not recorded during the original experiment.

| Reproducibility item | Status |
|---|---|
| Dataset preparation scripts | Available |
| Recognition evaluation script | Available |
| Error-analysis script | Available |
| Character-level F1 script | Available |
| Project configuration file | Available |
| Package version snapshot | Available |
| Original generated recognition results | Available in project inventory |
| Random seed | Not recorded |
| Git commit identifier | Not recorded |
| Git remote | Not recorded |
| Exact historical execution commands | Require confirmation |
| Exact historical model-loading details | Require confirmation |
| Confirmation that inspected environment matches original run | Required |

The absence of a recorded random seed is documented rather than replaced with an assumed value.

Similarly, no Git commit identifier should be assigned retrospectively to the original experiment.

### Reproducibility improvements

For future experiments:

- Record the exact model name and model directory.
- Save the complete command used to run each script.
- Record the Python and package versions.
- Preserve the original dataset and annotations.
- Preserve the generated crops and label files.
- Save the evaluation CSV and error-analysis output.
- Record any changes to the scripts or configuration.
- Use Git commits to identify future experiment versions.
- Record random seeds when randomness is involved.

---

## 12. Running the experiment

The following commands represent the intended workflow based on the available scripts.

Run them from the project root with the appropriate Python environment activated.

```cmd
python scripts\create_icdar_recognition_dataset.py
```

This generates the five sample crops.

```cmd
python scripts\create_icdar_recognition_dataset_full.py
```

This generates the full recognition crop set and its labels.

```cmd
python scripts\evaluate_icdar_recognition.py
```

This runs recognition and saves the evaluation results.

```cmd
python scripts\analyse_ocr_errors.py
```

This identifies and saves the incorrect predictions.

```cmd
python scripts\calculate_character_f1.py
```

This calculates character-level precision, recall and micro F1-score.

These commands are reconstructed from the script paths and should be checked against the actual historical commands before being described as the commands used in the original experiment.

---

## 13. Limitations and interpretation

The experiment provides a general scene-text recognition baseline. It should not be interpreted as a direct measure of VIN recognition performance.

The ICDAR 2015 dataset contains general scene text rather than a dedicated VIN dataset. Its text content, image characteristics and recognition challenges differ from those encountered when reading VINs on vehicle surfaces.

Other limitations include:

- The experiment evaluates generated word crops rather than a complete text-detection and recognition pipeline.
- The evaluation timing measures individual recognition calls and does not represent full-system latency.
- Exact-string accuracy treats a prediction with even one incorrect character as incorrect.
- Character-level metrics provide additional information about partial recognition correctness.
- The historical random seed and exact model-loading details were not recorded.

The results should therefore be used as an initial general OCR recognition baseline, not as evidence that the model is already suitable for production VIN recognition.

---

## 14. Relationship to the MSc VIN OCR project

This experiment forms part of the preliminary investigation into OCR technologies for the MSc project, **AI-Based OCR for Vehicle Identification Number (VIN) Reading**.

It provides practical experience with:

- OCR model inference.
- Recognition dataset preparation.
- Ground-truth annotation handling.
- Text recognition evaluation.
- Character-level performance measurement.
- Recognition error analysis.
- Experiment documentation and reproducibility.

The findings provide a foundation for the subsequent VIN-specific experiments, where the focus shifts towards recognising 17-character VINs under challenging industrial conditions.

The later VIN experiments are maintained separately and should not be merged with the results of this general OCR baseline.

---

## 15. Experiment conclusion

The PP-OCRv6 Medium recognition experiment established a general OCR evaluation workflow using the ICDAR 2015 dataset.

The workflow includes dataset preparation, perspective-based text cropping, recognition inference, exact-string accuracy, CER, inference-time recording, character-level evaluation and error analysis.

The experiment provides a useful starting point for understanding PaddleOCR recognition behaviour and establishing an evaluation process that can be adapted to the main VIN recognition research.

Further work is required to complete the historical reproducibility record, particularly confirmation of the exact model loaded, the original execution commands and the environment used during the original evaluation.