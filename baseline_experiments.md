# OCR Baseline Experiments

## Purpose
Establish unmodified OCR baseline performance before fine-tuning,
preprocessing, augmentation, or VIN-specific optimisation.

| Experiment | OCR Model | Dataset | Exact String Accuracy | Character Accuracy | CER | Avg Time/Image | Model Size | Hardware |
|---|---|---|---|---|---|---|---|---|
| EXP-OCR-001 | PaddleOCR PP-OCRv6 Medium | Public dataset | TBD | TBD | TBD | TBD | TBD | CPU |
| EXP-OCR-002 | TrOCR | OVST/VIN dataset | TBD | TBD | TBD | TBD | TBD | TBD |
| EXP-OCR-003 | GLM-OCR | OVST/VIN dataset | TBD | TBD | TBD | TBD | TBD | TBD |
| EXP-OCR-004 | Proposed model | VIN dataset | TBD | TBD | TBD | TBD | TBD | TBD |



## EXP-OCR-001 Notes

Model:
PP-OCRv6 Medium

Current status:
- PaddlePaddle installed successfully
- PaddleOCR installed successfully
- Standard recognition example completed
- Full detection + recognition OCR pipeline completed
- No fine-tuning performed
- No custom preprocessing performed
- No VIN-specific post-processing performed

Purpose:
This experiment will provide the initial lightweight OCR baseline
before any modification or optimisation.