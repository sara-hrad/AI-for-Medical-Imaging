# Evaluating Google CT Foundation Model As A Generalist To Detect Central Pulmonary Embolism

This repository contains the code for our study evaluating a classification pipeline that integrates the [Google CT Foundation model](https://github.com/Google-Health/imaging-research/tree/master/ct-foundation) with various classifiers (MLP, Random Forest, and Ensemble) to detect central pulmonary embolism (PE) from CTPA scans.

> **Paper:** *Evaluating Google CT Foundation Model As A Generalist To Detect Central Pulmonary Embolism*
> Sara Hosseinirad, Mobina Mobaraki, Dhanav Handa, Ryan Arman, Philip Edgcumbe
> Preprint submitted to European Journal of Radiology, 2026

## Overview

The pipeline generates 1408-dimensional embedding vectors from CT scans using the frozen Google CT Foundation model, then trains lightweight classifiers on these embeddings to predict the presence of central PE. The study uses the public [RSNA CTPA dataset](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pe-detection-challenge-2020) for model development and the [INSPECT cohort](https://stanfordaimi.azurewebsites.net/datasets/3a7548a4-8f65-4ab7-85fa-3d68c9efc1bd) for external validation.

### Key Findings

- **Best test AUC of 0.80** using a Random Forest classifier trained on a balanced dataset of 408 CT studies (Table 2 in paper)
- **Vascular segmentation does not improve performance** -- removing surrounding anatomy via [VesselFM](https://github.com/bwittmann/vesselFM) may remove contextual information the foundation model relies on
- **Significant generalization gap** -- a 0.17 AUC drop on the external INSPECT dataset, with statistically confirmed distributional shift (MMD p=0.001, C2ST accuracy 83.9%)
- Performance scales with training data size, consistent with CT Foundation benchmarks across other medical tasks

### Datasets

| Name | Description | Positive | Negative |
|------|-------------|----------|----------|
| D1 (Small Balanced) | 64 studies | 32 | 32 |
| D2 (Medium Balanced) | 128 studies | 64 | 64 |
| D3 (Large Balanced) | 408 studies | 204 | 204 |
| D4 (Imbalanced) | 838 studies | 204 | 634 |
| RSNA Test Set | 144 studies | 36 | 108 |
| INSPECT Test Set | 241 studies | 59 | 182 |

## Project Structure

```
AI-for-Medical-Imaging/
├── training.py                     # Training & validation pipeline (MLP, RF, Ensemble)
├── distributional_analysis.py      # RSNA vs. INSPECT embedding comparison (t-SNE, MMD, C2ST)
├── utils.py                        # Shared utilities (embedding parsing, metrics, thresholds)
├── savebestmodel.py                # Custom Keras callback for best weight tracking
├── datasets/                       # Embedding CSVs for all training/test splits
├── data preprocessing/             # INSPECT test data cleaning pipeline (see its own README)
├── CT_Foundation_Demo.ipynb        # Embedding generation notebook (DICOM input)
├── CT_Foundation_NIfTI_Demo.ipynb  # Embedding generation notebook (NIfTI input)
├── requirements.txt
└── projects/                       # Side projects
    ├── cxr-foundation/             # CXR Foundation model experiments (NIH Chest X-ray)
    └── VesselFM/                   # Vessel segmentation utilities (DICOM/NIfTI conversion)
```

## Setup

```bash
conda create -n med-imaging python=3.11
conda activate med-imaging
pip install -r requirements.txt
```

## Training

Three classifiers are trained on each dataset: MLP (with 5-fold CV hyperparameter tuning via Keras Tuner), Random Forest (with GridSearchCV), and an Ensemble (MLP features + RF head).

```bash
# Train a single experiment (e.g., MLP on the large balanced dataset)
python training.py single --dataset "data_no_mask_3" --model "mlp" --sensitivity 0.85

# Train with manual hyperparameters (includes TensorBoard logging)
python training.py manual --dataset "data_no_mask_3" --lr 0.001 --alpha 0.5 \
  --dropout 0.3 --steps 100 --wd 1e-7 --layers "64,64" --sensitivity 0.85 --noise 0.01

# Evaluate a trained model on a test set
python training.py evaluate --model_path results/<run>/best_rf_model.pkl \
  --model_type rf --test_data_path datasets/dataset_test_inspect.csv
```

Available datasets: `data_no_mask_1` (D1), `data_no_mask_2` (D2), `data_no_mask_3` (D3), `data_no_mask_4` (D4), `data_mask_1`, `data_mask_2`, `smoke_test_data`.

## Distributional Analysis

Compares RSNA and INSPECT embedding distributions to quantify domain shift using three methods: t-SNE visualization, MMD with permutation testing, and a classifier two-sample test (C2ST).

```bash
python distributional_analysis.py \
  --rsna_csv datasets/dataset_validation.csv \
  --inspect_csv datasets/dataset_test_inspect.csv \
  --n_pos 36 --n_neg 108
```

## Data Preprocessing

The INSPECT dataset (NIfTI format) requires harmonization before use. See [`data preprocessing/README.md`](data%20preprocessing/README.md) for the four-stage pipeline:

1. **Harmonize** -- Apply RescaleSlope/Intercept, correct voxel spacing, clamp HU values
2. **Slice-check** -- Detect disordered slices via inter-slice correlation
3. **Body-check** -- Filter non-chest scans using lung volume ratio
4. **Filter** -- Remove flagged series from the embeddings CSV

## Embedding Generation

To generate embeddings, use `CT_Foundation_Demo.ipynb` (DICOM) or `CT_Foundation_NIfTI_Demo.ipynb` (NIfTI). Access to the Google CT Foundation model and a Google Cloud Healthcare API DICOM store is required.

## Side Projects

- **[CXR Foundation](projects/cxr-foundation/)** -- Experiments using the [CXR Foundation model](https://github.com/Google-Health/imaging-research/blob/master/cxr-foundation/README.md) on the [NIH Chest X-ray dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- **[VesselFM](projects/VesselFM/)** -- DICOM/NIfTI conversion utilities for the [VesselFM](https://github.com/bwittmann/vesselFM) segmentor, used for the vascular segmentation experiments
