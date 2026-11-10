# ICDAR2015 Text Detection and Recognition Using PaddleOCR

## MSc Research Project

**Project:** ICDAR2015 Text Detection and Recognition Using PaddleOCR  
**Project directory:** `MSC-ICDAR2015-PaddleOCR-CPU-`  
**Framework:** PaddleOCR  
**Dataset:** ICDAR2015  
**Primary programming language:** Python  
**Operating system:** Windows  
**Project focus:** Scene text detection, text recognition, OCR evaluation and model performance analysis.

---

## 1. Project Overview

This project investigates text detection and text recognition using PaddleOCR, with the ICDAR2015 scene text dataset as the principal benchmark.

The work is structured around three main stages:

1. **Text detection:** Identifying and localising text regions in scene images.
2. **Text recognition:** Recognising the characters contained within cropped text regions.
3. **OCR evaluation:** Assessing detection and recognition performance using appropriate quantitative metrics.

The project uses the PP-OCRv6-Medium architecture for the detection and recognition investigations. It also evaluates the performance of a pretrained recognition model against a fine-tuned model to determine whether task-specific training improves recognition accuracy.

The broader objective is to investigate the performance, limitations and practical suitability of PaddleOCR for text-reading applications.

## 2. Research Objectives

The main objectives are:

- Prepare and organise the ICDAR2015 dataset for PaddleOCR.
- Configure a reproducible Python and PaddleOCR environment.
- Train and evaluate a text detection model.
- Fine-tune a pretrained text recognition model.
- Compare pretrained and fine-tuned recognition performance.
- Evaluate text detection and recognition independently.
- Investigate recognition errors, including substitutions, deletions and insertions.
- Measure inference performance and model size.
- Establish a reproducible experimental workflow.
- Investigate the performance of a combined OCR pipeline.

## 3. Dataset

### 3.1 ICDAR2015

The project uses the ICDAR2015 scene text dataset.

The dataset contains images with text appearing in natural scenes, including variations in:

- Text orientation and rotation.
- Image resolution.
- Lighting and illumination.
- Background complexity.
- Text size and appearance.
- Image quality and visual interference.

### 3.2 Original dataset structure

The original dataset was organised into:

```text
dataset/
├── train/
├── train_gt/
├── test/
└── test_gt/
```

The dataset contains:

| Split | Images | Annotation files |
|---|---:|---:|
| Training | 1,000 | 1,000 |
| Testing | 500 | 500 |
| **Total** | **1,500** | **1,500** |

The original annotations contain polygon coordinates and associated text transcriptions.

Each annotation represents a text region using eight polygon coordinates followed by its transcription.

The transcription `###` identifies an ignored or do-not-care text region. These regions must be handled appropriately during detection evaluation.

### 3.3 Prepared recognition dataset

The recognition experiment uses ground-truth-derived text crops rather than detected regions.

A total of **2,065 recognition samples** were evaluated in the recognition test experiment.

This distinction is important:

- Detection evaluates whether text regions are correctly located.
- Recognition evaluates whether characters within supplied text crops are correctly identified.
- Ground-truth-derived crops do not measure the errors introduced by the detection stage.

## 4. Project Structure

The current project directory is organised as follows:

```text
MSC-ICDAR2015-PaddleOCR-CPU-/
│
├── .gitignore
├── .vscode/
│
├── configs/
│
├── data/
│
├── dataset/
│
├── logs/
│
├── models/
│
├── PaddleOCR/
│
├── recognition_data/
│
├── results/
│
├── scripts/
│
├── environment.txt
├── gpu_info.txt
├── requirements.txt
├── requirements-lock.txt
│
└── venv/
```

### Directory descriptions

| Directory/file | Purpose |
|---|---|
| `configs/` | Experiment configuration files. |
| `data/` | Dataset preparation and intermediate data. |
| `dataset/` | Original ICDAR2015 images and annotations. |
| `logs/` | Training logs and experiment records. |
| `models/` | Trained models, checkpoints and exported inference models. |
| `PaddleOCR/` | Original cloned PaddleOCR source repository. |
| `recognition_data/` | Prepared text recognition data and crops. |
| `results/` | Evaluation reports, predictions, metrics and visualisations. |
| `scripts/` | Custom project scripts for preparation, training support and evaluation. |
| `environment.txt` | Environment information. |
| `gpu_info.txt` | Hardware information. |
| `requirements.txt` | Project dependencies. |
| `requirements-lock.txt` | Recorded dependency versions. |
| `venv/` | Local Python virtual environment. |

The `PaddleOCR/` directory is maintained as the original cloned framework repository, separate from the project-level Git repository.

## 5. Software and Environment

The project was developed on Windows using a dedicated Python virtual environment.

Recorded environment details include:

| Component | Recorded information |
|---|---|
| Operating system | Windows |
| Python | 3.12.10 |
| PaddlePaddle | 3.3.0 |
| OCR framework | PaddleOCR |
| Virtual environment | `venv/` |
| Dependency records | `requirements.txt`, `requirements-lock.txt` |

The project also contains hardware information in `gpu_info.txt`.

**Hardware and inference note:** Although this project directory is named `MSC-ICDAR2015-PaddleOCR-CPU-`, the recognition experiment report records GPU-enabled inference. The actual hardware used for each experiment must therefore be confirmed from its environment and experiment records before describing all results as CPU-only.

For reproducibility, each experiment should retain:

- Python version.
- Operating system.
- PaddlePaddle version.
- PaddleOCR repository commit.
- Relevant package versions.
- CPU/GPU hardware.
- CUDA and cuDNN versions, where applicable.
- Configuration file.
- Model checkpoint.
- Dataset and split information.

## 6. Experimental Work

### 6.1 EXP-DET-ICDAR15-001 — Text Detection

**Objective:** Train and evaluate a text detection model using ICDAR2015.

The detection experiment investigated the PP-OCRv6-Medium detection architecture.

#### Training configuration

| Parameter | Recorded value |
|---|---|
| Architecture | PP-OCRv6-Medium detection |
| Training epochs | 100 |
| Batch size | 2 |
| Optimiser | Adam |
| Initial learning rate | 0.0001 |
| Loss | DBLoss / DiceFocalLoss |
| EMA | Enabled |
| Best validation epoch | 84 |

#### Validation results

| Metric | Result |
|---|---:|
| Precision | 78.5558% |
| Recall | 83.1019% |
| H-mean | 80.7649% |

#### Test results

| Metric | Result |
|---|---:|
| Precision | 83.51% |
| Recall | 79.49% |
| H-mean | 81.45% |

#### Interpretation

The detector achieved an H-mean of 81.45% on the recorded test evaluation.

Test precision was higher than recall, indicating that the detector's predictions were relatively more precise than its coverage of annotated text regions.

The lower recall suggests that some annotated text regions were missed.

A separate set of detector metrics was also recorded as:

- Precision: 80.2059%.
- Recall: 82.5626%.
- H-mean: 81.3672%.

These figures should be retained as a separate evaluation record until their precise evaluation configuration and relationship to the test results above are confirmed.

### 6.2 EXP-REC-ICDAR15-001 — Text Recognition Fine-tuning

**Objective:** Fine-tune a pretrained recognition model using ICDAR2015 text crops and evaluate its recognition performance.

#### Recognition architecture

The experiment used PP-OCRv6-Medium recognition with:

- SVTR_LCNet / PPLCNetV4 Medium.
- CTC recognition head.
- NRTR recognition head.
- Combined MultiLoss training objective.

#### Training configuration

| Parameter | Recorded value |
|---|---|
| Architecture | PP-OCRv6-Medium recognition |
| Training epochs | 100 |
| Batch size | 8 |
| Sampling | Dynamic multi-scale |
| Optimiser | Adam |
| Initial learning rate | 0.0005 |
| Loss | MultiLoss (CTC + NRTR) |
| Best checkpoint | `best_accuracy` |
| Best checkpoint epoch | 96 |
| PaddlePaddle | 3.3.0 |

#### Validation results

| Metric | Result |
|---|---:|
| Validation accuracy | 70.9635% |
| Normalised edit distance | 87.6034% |
| Reported FPS | 283.68 |

#### Test results

The model was evaluated on 2,065 ground-truth-derived recognition crops.

| Metric | Result |
|---|---:|
| Total samples | 2,065 |
| Correct complete strings | 1,392 |
| Incorrect complete strings | 673 |
| Exact-string accuracy | **67.4092%** |
| Character accuracy | 87.0128% |
| Character error rate (CER) | 14.2444% |
| Character precision | 89.2983% |
| Character recall | 87.0128% |
| Character F1 | 88.1407% |

#### Character error analysis

| Error type | Count |
|---|---:|
| Substitutions | 1,014 |
| Deletions | 422 |
| Insertions | 139 |
| **Total edits** | **1,575** |

Substitutions were the largest source of recognition errors.

#### Inference performance

| Metric | Result |
|---|---:|
| Mean inference time | 7.495 ms per crop |
| Median inference time | 7.317 ms per crop |
| Throughput | Approximately 133.43 crops/second |

#### Model artefacts

The trained model and checkpoints were recorded under:

```text
models/
└── rec_trained/
    └── PP-OCRv6_medium_rec_icdar2015/
        ├── config.yml
        ├── train.log
        ├── best_accuracy.pdparams
        ├── latest.pdparams
        ├── best_model/
        └── iter_epoch_10 ... iter_epoch_100/
```

The exported inference model was recorded under:

```text
models/
└── rec_inference/
    └── PP-OCRv6_medium_rec_icdar15_best/
```

The reported model parameter file size was approximately 72.92 MiB.

#### Interpretation

The recognition model achieved 87.01% character accuracy but only 67.41% exact-string accuracy.

This difference demonstrates that recognising most characters correctly does not guarantee that an entire text string is correct.

A single incorrect character can cause a complete string to be counted as incorrect.

### 6.3 Pretrained recognition baseline comparison

The pretrained recognition baseline achieved an exact-string accuracy of **80.05%**.

The fine-tuned recognition model achieved **67.4092%**.

| Model | Exact-string accuracy |
|---|---:|
| Pretrained baseline | 80.05% |
| Fine-tuned model | 67.4092% |
| Difference | **−12.6408 percentage points** |

#### Interpretation

Fine-tuning reduced the measured exact-string accuracy by approximately 12.64 percentage points.

Potential explanations include:

1. **Limited training data:** The fine-tuning dataset may not have provided enough variation to improve generalisation.
2. **Overfitting:** The model may have adapted to the training samples rather than learning features that generalise to unseen images.
3. **Dictionary compatibility:** Differences in the character dictionary, character ordering or supported symbols may have affected recognition.
4. **Case handling:** The baseline and fine-tuned model may have been evaluated using different case-normalisation rules.
5. **Training configuration:** The learning rate, sampling strategy or checkpoint selection may have affected generalisation.

These are hypotheses requiring further investigation, not confirmed causes.

A fair comparison requires both models to use the same test samples, preprocessing, character dictionary, decoding procedure and case-sensitivity setting.

### 6.4 EXP-EVAL-ICDAR15-001 — Combined OCR Evaluation

**Objective:** Evaluate the OCR system beyond isolated detection and recognition experiments.

The intended combined evaluation considers the relationship between text localisation and text recognition.

A complete OCR pipeline should be evaluated using:

- Detection precision.
- Detection recall.
- Detection H-mean.
- Recognition exact-string accuracy.
- Character accuracy.
- Character error rate.
- End-to-end text-reading accuracy.
- Inference time.

The final end-to-end results for `EXP-EVAL-ICDAR15-001` have not yet been confirmed in the available experiment records.

The recognition experiment's 67.4092% exact-string accuracy must not be presented as end-to-end OCR accuracy because it uses ground-truth-derived text crops.

## 7. Evaluation Metrics

### 7.1 Detection metrics

**Precision**

Measures the proportion of predicted text regions that are correctly detected.

\[
Precision = \frac{TP}{TP+FP}
\]

**Recall**

Measures the proportion of ground-truth text regions that are correctly detected.

\[
Recall = \frac{TP}{TP+FN}
\]

**H-mean**

The harmonic mean of precision and recall.

\[
H = \frac{2PR}{P+R}
\]

Where:

- TP = true positives.
- FP = false positives.
- FN = false negatives.

### 7.2 Recognition metrics

**Exact-string accuracy**

Measures the proportion of complete text strings recognised without any character errors.

\[
Accuracy = \frac{\text{Correct strings}}{\text{Total strings}}
\]

**Character accuracy**

Measures recognition performance at character level.

**Character error rate**

\[
CER = \frac{S+D+I}{N}
\]

Where:

- S = substitutions.
- D = deletions.
- I = insertions.
- N = number of ground-truth characters.

**Character-level precision, recall and F1**

These metrics are calculated using character alignment and should be interpreted according to the evaluation implementation.

### 7.3 Case sensitivity

The case-sensitivity setting of the recognition evaluation has not yet been confirmed.

Both case-sensitive and case-insensitive evaluation can produce different exact-string accuracy values.

For reproducibility, the evaluation procedure must explicitly document whether predictions and ground-truth strings are converted to a common case before comparison.

## 8. Experiment Outputs and Records

The project maintains separate directories for training artefacts and evaluation outputs.

Known recognition evaluation outputs include:

```text
results/
└── recognition/
    ├── EXP-REC-ICDAR15-001_test_predictions.csv
    ├── EXP-REC-ICDAR15-001_character_confusions.csv
    └── EXP-REC-ICDAR15-001_test_metrics.txt
```

The project also maintains:

- Training logs.
- Experiment configuration files.
- Validation records.
- Model checkpoints.
- Evaluation predictions.
- Character confusion analysis.
- Hardware and environment records.

The objective is to make each experiment traceable and reproducible.

## 9. Current Findings

The experiments establish several important findings.

1. The detection model achieved a recorded test H-mean of 81.45%.
2. The fine-tuned recognition model achieved 67.4092% exact-string accuracy.
3. Character accuracy was higher than exact-string accuracy, demonstrating the effect of individual character errors on complete-string recognition.
4. The pretrained recognition baseline achieved 80.05%, outperforming the fine-tuned model.
5. Recognition substitutions were the largest error category.
6. The isolated detection and recognition experiments do not establish complete end-to-end OCR performance.

The most important research finding so far is that fine-tuning did not improve recognition performance over the pretrained baseline under the recorded evaluation results.

## 10. Reproducibility and Version Control

The project uses Git to track source code, configuration files, scripts and selected experiment documentation.

The project-level Git repository is separate from the original cloned PaddleOCR repository.

The original PaddleOCR clone is retained under:

```text
PaddleOCR/
```

The project-level repository is intended to exclude:

- Python virtual environments.
- The cloned PaddleOCR source repository.
- Original and prepared datasets.
- Recognition crops.
- Model weights and checkpoints.
- Temporary and cached files.

The `.gitignore` file defines these exclusions.

Large datasets and model checkpoints should be retained locally or in appropriate storage rather than committed to the source-code repository.

### Repository information

| Item | Status |
|---|---|
| Project Git repository | Being initialised |
| Working branch | `icdar2015-paddleocr-cpu` |
| Original PaddleOCR repository | Existing clone under `PaddleOCR/` |
| Original PaddleOCR commit hash | To be recorded |
| Project initial commit hash | To be recorded |
| GitHub remote | To be configured and verified |

## 11. Known Limitations and Outstanding Work

The following items remain to be verified or completed:

- Confirm the exact PaddleOCR source commit and repository URL.
- Reconcile the two recorded detection metric sets.
- Verify the recognition baseline evaluation procedure.
- Confirm case-sensitive or case-insensitive recognition comparison.
- Investigate why fine-tuning reduced recognition accuracy.
- Verify the exact hardware used for each experiment.
- Confirm the complete results of `EXP-EVAL-ICDAR15-001`.
- Establish a reproducible end-to-end evaluation procedure.
- Record environment information alongside each experiment.
- Complete the project-level Git commit and remote configuration.

## 12. Conclusion

This project investigates scene text detection and recognition using PaddleOCR and the ICDAR2015 dataset.

The work has progressed through dataset organisation, environment preparation, model training, checkpoint management, recognition evaluation and comparative performance analysis.

The detection experiment achieved a recorded test H-mean of 81.45%, while the recognition fine-tuning experiment achieved 67.41% exact-string accuracy.

Comparison with the 80.05% pretrained recognition baseline showed that fine-tuning reduced performance rather than improving it. This highlights the importance of dataset quality, training strategy, character dictionary compatibility and consistent evaluation procedures.

Further work will focus on investigating recognition performance, validating the evaluation methodology and establishing a reliable combined OCR pipeline.

---

## 13. Citation and Acknowledgement

This project uses the PaddleOCR open-source OCR framework and the ICDAR2015 scene text benchmark dataset.

PaddleOCR repository: https://github.com/PaddlePaddle/PaddleOCR

ICDAR2015 benchmark: Please retain the dataset's original source and citation details in the final dissertation and project documentation.

