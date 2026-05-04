# ImageNet Data Setup

## Overview

This project uses preprocessed ImageNet validation data for evaluating Adaptive AutoEval under synthetic covariate shift.

The dataset consists of:

- Model-derived loss / correctness signals (`phi`)
- Synthetic annotator outputs (`synthetic`)
- Outputs for 5 ResNet models:
  - ResNet-18
  - ResNet-34
  - ResNet-50
  - ResNet-101
  - ResNet-152

These are stored as NumPy arrays and are required to run:

```bash
python scripts/run_extension1_imagenet.py
```
Important
Raw ImageNet data is not included due to licensing restrictions.
Preprocessed .npy files are also not included due to size.
You must generate or obtain these files separately.
Required File Structure

After setup, your repository must contain:

```bash
results/
├── phi_imagenet/
│   ├── resnet18.npy
│   ├── resnet34.npy
│   ├── resnet50.npy
│   ├── resnet101.npy
│   └── resnet152.npy
│
├── synthetic_imagenet/
│   ├── resnet18.npy
│   ├── resnet34.npy
│   ├── resnet50.npy
│   ├── resnet101.npy
│   └── resnet152.npy
```
These paths are hardcoded in the experiment script.

How to Generate the Data:

You will need to:

1. Download ImageNet Validation Data

Register and download:

Dataset: ILSVRC2012 validation set
Source: http://www.image-net.org/


2. Run ResNet Models

For each model:

ResNet-18, 34, 50, 101, 152
Use pretrained weights (e.g., PyTorch)

For every validation image:

Compute:

(a) Ground-truth signal (phi)

(b) Synthetic annotator (synthetic)

3. Save Outputs

Save arrays as:

np.save("results/phi_imagenet/resnet50.npy", phi_array)

np.save("results/synthetic_imagenet/resnet50.npy", synthetic_array)
