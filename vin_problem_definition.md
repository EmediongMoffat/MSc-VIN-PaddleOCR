# VIN OCR Problem Definition

## General OCR
General OCR may recognise:
- words
- sentences
- documents
- signs
- mixed text layouts

## VIN OCR
VIN OCR is a constrained industrial text-recognition problem.

Important characteristics:
- fixed/known character length
- alphanumeric output
- restricted character set
- no spaces
- visually similar characters
- etched/stamped/printed text
- metallic or reflective surfaces
- low contrast
- blur
- variable lighting
- perspective distortion
- manufacturing variation

## Main project concern
The system must not only recognise characters accurately,
but also process VIN images quickly enough for possible industrial use.

Therefore the project must evaluate:
- Exact VIN accuracy
- Character accuracy
- CER
- Inference time
- Model size
- Memory/computational requirements