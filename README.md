# AI-Based OCR for Vehicle Identification Numbers (VINs)

## 1. Project objective

This project investigates optical character recognition for extracting Vehicle Identification Numbers (VINs) from images. It evaluates recognition accuracy, robustness, inference speed, model size and computational requirements.

The intended output is a complete VIN string. Evaluation will distinguish character-level performance from exact-string accuracy.

## 2. Research approach

1. Verify the development environment.
2. Identify and document a suitable VIN dataset.
3. Inspect labels and establish reproducible train, validation and test splits.
4. Reproduce an established OCR baseline.
5. Evaluate detection and recognition separately where applicable.
6. Record errors and investigate difficult images.
7. Conduct controlled experiments, changing one major variable at a time.
8. Compare recognition performance and computational cost.
9. Prepare reproducible inference and evaluation scripts.

## 3. Development environment

- Operating system: Windows
- Python: 3.12.10
- PaddlePaddle GPU: 3.3.0
- GPU: NVIDIA GeForce RTX 4050, approximately 6 GB VRAM
- NVIDIA driver: 617.14
- OCR framework repository: PaddleOCR
- Recorded PaddleOCR commit: `dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf`
- Virtual environment: `venv`

See `environment.txt` and `requirements-lock.txt` for the recorded environment.

## 4. Repository layout

- `PaddleOCR/`: cloned upstream PaddleOCR repository
- `data/raw/`: original datasets
- `data/processed/`: derived images and annotations
- `data/splits/`: dataset split manifests
- `configs/`: experiment configurations
- `src/`: project-specific code
- `scripts/`: setup and reusable commands
- `results/`: predictions, evaluation outputs and plots
- `logs/`: progress and experiment records
- `models/`: checkpoints and exported models

## 5. Dataset governance

For each dataset, record:

- Name, source URL and version
- Licence and usage permissions
- Image and annotation formats
- Number of samples and split counts
- Label quality and known limitations
- Preprocessing and duplicate checks

Keep training, validation and test data separate. Do not use the test set to tune model parameters.

## 6. Evaluation

Subject to the task and available annotations, record:

- Exact VIN accuracy
- Character accuracy
- Character Error Rate (CER)
- String Error Rate
- Detection precision, recall and F1 where detection annotations are available
- Training and inference time
- Model size
- Hardware and memory usage
- Failure cases and image conditions

Metrics that cannot be measured or do not apply to the experiment will be marked accordingly, not recorded as zero.

## 7. Experiment reproducibility

Every experiment should have a unique ID and a record of:

- Research question and hypothesis
- Dataset version and split
- Model architecture and pretrained weights
- Configuration and random seed
- Software versions and hardware
- Exact training and evaluation commands
- Results, runtime and errors
- Git commit and output-file locations
- Interpretation and limitations

Raw predictions and evaluation outputs should be retained so reported metrics can be reproduced.

## 8. Current status

- [x] Python virtual environment created
- [x] PaddlePaddle GPU installed
- [x] GPU verification passed
- [x] Environment report created
- [x] Package snapshot created
- [ ] Dataset selected and verified
- [ ] Dataset tracking record completed
- [ ] Baseline configuration prepared
- [ ] First OCR training experiment started
- [ ] Baseline evaluation completed

## 9. Setup notes

The upstream PaddleOCR training dependencies are managed in `PaddleOCR/requirements.txt`. PaddlePaddle GPU was installed separately.

The project-level environment record and package snapshot are `environment.txt` and `requirements-lock.txt`. Update the records whenever the environment changes materially.
