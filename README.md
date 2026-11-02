# MSc VIN OCR Project

**Project:** AI-Based OCR for Vehicle Identification Number (VIN) Reading  
**Research partner:** Oxford Vision and Sensor Technology (OVST) Ltd  
**Framework:** PaddleOCR / PaddlePaddle  
**Python:** 3.12.10  
**GPU:** NVIDIA GeForce RTX 4050 Laptop GPU

## Project objective
Develop and evaluate an accurate, efficient VIN detection and recognition pipeline using deep learning.

## Main experiments
- `EXP-VIN-DET-FT-001` – Initial VIN detection fine-tuning.
- `EXP-VIN-DET-FT-002` – Improved VIN detection fine-tuning.
- `EXP-VIN-REC-FT-001` – VIN recognition fine-tuning.
- `EXP-VIN-COMBINED-001` – Combined detection and recognition evaluation.

## Project structure
- `configs/` – Experiment configurations.
- `scripts/` – Training, inference and evaluation scripts.
- `data/` – Dataset files and processed data.
- `checkpoints/` – Model checkpoints.
- `results/experiments/` – Experiment results and reports.
- `logs/` – Training and execution logs.
- `PaddleOCR/` – Cloned PaddleOCR repository.

## Reproducibility
Experiment configurations, dataset splits, software versions, model checkpoints and evaluation outputs are recorded separately. The held-out test set is reserved for final evaluation.

**Important:** Do not overwrite original experiment configurations or use test results to tune models.