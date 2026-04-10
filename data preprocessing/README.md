# Test Data Cleaning Pipeline

A four-stage pipeline for preparing CT test data by harmonizing NIfTI physics, detecting corrupted slices, filtering non-chest scans, and removing all flagged series from embeddings.

## Directory Setup

Before running the pipeline, organize your files under the `data preprocessing/` folder as follows:

```
AI-for-Medical-Imaging/          (repo root)
├── datasets/
│   └── dataset_test_inspect.csv   <-- Final output (auto-copied by the pipeline)
├── training.py
└── data preprocessing/
    ├── test_data_cleaning.py
    ├── README.md
    ├── input/
    │   ├── nifti/              <-- Place your raw .nii.gz CT series here
    │   ├── metadata.tsv        <-- Series metadata exported from DICOM headers
    │   └── embeddings.csv      <-- CT-Foundation embeddings for the test set
    └── output/                    (created automatically by the pipeline)
        ├── harmonized/          <-- Corrected NIfTI files (Stage 1)
        ├── corrupted_slices.csv <-- Disordered-slice flags (Stage 2)
        ├── non_chest_scans.csv  <-- Non-chest scan flags (Stage 3)
        └── dataset_test_inspect.csv <-- Filtered embeddings (Stage 4)
```

When running the `all` command, the final `dataset_test_inspect.csv` is automatically copied into `datasets/` so it is ready for use by `training.py`.

### What to put where

| File / Folder | What goes here |
|---|---|
| `input/nifti/` | All raw `.nii.gz` CT series files you want to clean. Copy both positive and negative cases into this folder. |
| `input/metadata.tsv` | A tab-separated file with columns: `image_id`, `RescaleSlope`, `RescaleIntercept`, `PixelSpacing_0`, `PixelSpacing_1`, `SliceThickness`. Each row corresponds to one CT series. The `image_id` must match the NIfTI filename (without the `.nii.gz` extension). |
| `input/embeddings.csv` | The embeddings CSV produced by CT-Foundation. Must contain a `file_name` column matching the NIfTI filenames (without extension) and a `labels` column. |

## Stages

| Stage | Command | Description |
|-------|---------|-------------|
| 1 | `harmonize` | Apply RescaleSlope/Intercept from metadata, clamp HU values to -1024, and correct voxel spacing |
| 2 | `slice-check` | Detect disordered or corrupted slices via inter-slice correlation analysis |
| 3 | `body-check` | Classify scans as chest vs. abdomen using lung tissue volume ratio and flag non-chest series |
| 4 | `filter` | Remove all flagged series (from stages 2 and 3) from the embeddings CSV |

## Requirements

- Python 3.8+
- `nibabel`, `numpy`, `pandas`, `scipy`, `tqdm`

## Usage

### Run all stages at once

```bash
python test_data_cleaning.py all \
  --nifti_dir ./input/nifti \
  --tsv ./input/metadata.tsv \
  --output_dir ./output/harmonized \
  --embed_csv ./input/embeddings.csv \
  --clean_embed_csv ./output/dataset_test_inspect.csv
```

### Run stages individually

**Stage 1 -- Harmonize NIfTI physics**

```bash
python test_data_cleaning.py harmonize \
  --nifti_dir ./input/nifti \
  --tsv ./input/metadata.tsv \
  --output_dir ./output/harmonized
```

Reads raw NIfTI files from `input/nifti/`, applies rescale slope/intercept and voxel spacing corrections using `input/metadata.tsv`, and writes corrected files to `output/harmonized/`.

**Stage 2 -- Slice-order consistency check**

```bash
python test_data_cleaning.py slice-check \
  --nifti_dir ./output/harmonized \
  --flagged_csv ./output/corrupted_slices.csv
```

Scans the harmonized NIfTI files for disordered or corrupted slices using inter-slice correlation. Writes flagged filenames to `output/corrupted_slices.csv`.

**Stage 3 -- Body-part classification (chest vs. abdomen)**

```bash
python test_data_cleaning.py body-check \
  --nifti_dir ./output/harmonized \
  --flagged_csv ./output/non_chest_scans.csv \
  --lung_threshold 0.10
```

Classifies each scan as chest or abdomen based on the ratio of internal air (lung) volume to total body volume. Scans below the `--lung_threshold` (default 10%) are flagged as non-chest. Writes flagged filenames to `output/non_chest_scans.csv`.

**Stage 4 -- Filter flagged series from embeddings**

```bash
python test_data_cleaning.py filter \
  --embed_csv ./input/embeddings.csv \
  --flagged_csvs ./output/corrupted_slices.csv ./output/non_chest_scans.csv \
  --clean_embed_csv ./output/dataset_test_inspect.csv
```

Merges all flagged CSVs and removes matching series from the embeddings file. Accepts one or more `--flagged_csvs` paths so you can combine results from multiple checks in a single filter step.

## Arguments Reference

| Argument | Used by | Description |
|----------|---------|-------------|
| `--nifti_dir` | harmonize, slice-check, body-check, all | Directory containing NIfTI files |
| `--tsv` | harmonize, all | Path to series metadata TSV |
| `--output_dir` | harmonize, all | Directory to save corrected NIfTI files |
| `--flagged_csv` | slice-check, body-check | Output CSV listing flagged file names |
| `--flagged_csvs` | filter | One or more input CSVs with flagged file names |
| `--lung_threshold` | body-check, all | Lung volume ratio cutoff for chest classification (default: 0.10) |
| `--embed_csv` | filter, all | Input embeddings CSV |
| `--clean_embed_csv` | filter, all | Output cleaned embeddings CSV |
