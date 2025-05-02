# Medical Image Segmentation for MRI-Guided Radiation Therapy

This repository contains the codebase and results for a deep learning pipeline developed for **TDT4265: Computer Vision and Deep Learning** at NTNU. The project addresses the **automatic segmentation of head and neck tumors** in pre-treatment 3D MRI scans as part of the HNTS-MRG challenge, using PyTorch and MONAI.

## Project Overview

- **Goal**: Segment tumors from 3D MRI volumes to support radiation therapy planning.
- **Dataset**: HNTS-MRG preRT (pre-radiation therapy) scans, with provided masks.
- **Architecture**: [DynUNet](https://arxiv.org/abs/2011.12304) (no deep supervision), chosen via a structured model preselection phase.
- **Frameworks**: [MONAI](https://monai.io/), PyTorch.
- **Performance**: Final test-time average Dice score: **0.7024**

## Project Structure
.
├── dataset.py # MONAI-based dataset loader
├── transforms.py # Training/validation transforms
├── model.py # Model architecture config
├── loss.py # Loss functions (Dice, Tversky, BCE+Dice)
├── metrics.py # Dice metric logic (uses MONAI)
├── train.py # Training loop with early stopping and logging
├── utils.py # Learning rate scheduling, reproducibility helpers
├── config.py # Centralized parameters and paths
├── data_analysis.ipynb # EDA and tumor volume/intensity visualization
└── inference/ # Scripts and tools for test-time evaluation


## Key Features

- Modular, reproducible training setup with configuration-driven architecture.
- Patch-based 3D training with class-balanced sampling and sliding window inference.
- Dynamic LR scheduling: **linear warm-up + cosine annealing**.
- Supports visualization in **3D Slicer** (automated export to NIfTI format).
- Sustainability metrics included: **~22.4h compute time, 10.09 kWh energy**.

## Sample Predictions

Visual comparison of ground truth (red) and predicted tumor masks (green overlay) for best, mid-range, and worst-performing test cases. Rendered using 3D Slicer.

**Best Prediction**
![best](https://github.com/user-attachments/assets/f469aba2-9f47-4b8f-88f1-9c15d0fc23f5)


**Mid-range Prediction**
![mid](https://github.com/user-attachments/assets/2fe3509f-a153-4061-b722-789c5e7e8e4b)



**Worst Prediction**
![wosrt](https://github.com/user-attachments/assets/6fc8939c-4793-4c2f-addb-9f65ebc39938)



## Performance Summary

| Model        | Loss Function   | Validation Dice | Test Dice |
|--------------|------------------|------------------|------------|
| DynUNet (no DS) | BCE + Dice      | 0.308 (preselect) | **0.7024** |

## Environment

- Python 3.10
- PyTorch 2.1+
- MONAI 1.2+
- NVIDIA RTX 4090
- 3D Slicer (for visualization)

## Reproducibility

Training and validation use fixed seeds, mixed precision, and deterministic patch sampling. You can retrain using:

```bash
python train.py --config config.py
