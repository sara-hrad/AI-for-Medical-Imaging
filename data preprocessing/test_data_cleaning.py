"""
Test Data Cleaning Pipeline

A four-stage pipeline for preparing CT test data:
  1. harmonize   — Apply RescaleSlope/Intercept, clamp HU values, and correct voxel spacing.
  2. slice-check — Detect disordered or corrupted slices via inter-slice correlation.
  3. body-check  — Classify chest vs. abdomen scans and flag non-chest series.
  4. filter      — Remove all flagged series from the embeddings CSV.

Expected directory layout (see README.md for details):

    data preprocessing/
    ├── test_data_cleaning.py
    ├── input/
    │   ├── nifti/              # Raw .nii.gz CT series
    │   ├── metadata.tsv        # DICOM-derived series metadata
    │   └── embeddings.csv      # CT-Foundation embeddings
    └── output/                 # Created by the pipeline
        ├── harmonized/
        ├── corrupted_slices.csv
        ├── non_chest_scans.csv
        └── dataset_test_inspect.csv

Usage:
  Run all stages:
    python test_data_cleaning.py all --nifti_dir ./input/nifti --tsv ./input/metadata.tsv \
        --output_dir ./output/harmonized --embed_csv ./input/embeddings.csv \
        --clean_embed_csv ./output/dataset_test_inspect.csv

  Run a single stage:
    python test_data_cleaning.py harmonize --nifti_dir ./input/nifti --tsv ./input/metadata.tsv \
        --output_dir ./output/harmonized
    python test_data_cleaning.py slice-check --nifti_dir ./output/harmonized \
        --flagged_csv ./output/corrupted_slices.csv
    python test_data_cleaning.py body-check --nifti_dir ./output/harmonized \
        --flagged_csv ./output/non_chest_scans.csv
    python test_data_cleaning.py filter --embed_csv ./input/embeddings.csv \
        --flagged_csvs ./output/corrupted_slices.csv ./output/non_chest_scans.csv \
        --clean_embed_csv ./output/dataset_test_inspect.csv
"""
import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import scipy.ndimage as ndi
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Hounsfield Unit floor value
MIN_BOUND_HU = -1024.0


# ---------------------------------------------------------------------------
# Stage 1: Harmonize NIfTI physics
# ---------------------------------------------------------------------------

def load_metadata(tsv_path: Path) -> pd.DataFrame:
    """Load series metadata TSV and return a deduplicated DataFrame indexed by image_id."""
    df = pd.read_csv(tsv_path, sep='\t')

    required_cols = ['image_id', 'RescaleSlope', 'RescaleIntercept',
                     'PixelSpacing_0', 'PixelSpacing_1', 'SliceThickness']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"TSV is missing columns: {missing}")

    df['image_id'] = df['image_id'].astype(str).str.strip()

    n_dup = df['image_id'].duplicated().sum()
    if n_dup > 0:
        logging.warning(f"Found {n_dup} duplicate image_ids in TSV. Keeping the first occurrence.")
        df = df.drop_duplicates(subset=['image_id'], keep='first')

    return df.set_index('image_id')


def _is_already_scaled(raw_data: np.ndarray) -> bool:
    """Heuristic: data is likely already in HU if minimum is below -500."""
    return bool(np.min(raw_data) < -500)


def run_harmonize(nifti_dir: Path, tsv_path: Path, output_dir: Path):
    """Apply RescaleSlope/Intercept, clamp HU, and correct voxel spacing for every NIfTI in *nifti_dir*."""
    logging.info("=== Stage 1: Harmonize NIfTI Physics ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_df = load_metadata(tsv_path)
    nifti_files = list(nifti_dir.glob('*.nii*'))
    logging.info(f"Found {len(nifti_files)} NIfTI files to process.")

    success_count = 0
    for nifti_path in tqdm(nifti_files, desc="Harmonizing"):
        try:
            file_id = nifti_path.name.split('.')[0]
            if file_id not in meta_df.index:
                logging.warning(f"Skipping {nifti_path.name}: ID not found in metadata.")
                continue

            row = meta_df.loc[file_id]
            slope = float(row['RescaleSlope'])
            intercept = float(row['RescaleIntercept'])
            ps_y = float(row['PixelSpacing_0'])
            ps_x = float(row['PixelSpacing_1'])
            thickness = float(row['SliceThickness'])

            img = nib.load(nifti_path)
            raw_data = np.asanyarray(img.dataobj).astype(np.float32)

            if _is_already_scaled(raw_data):
                corrected_data = raw_data
            else:
                corrected_data = (raw_data * slope) + intercept

            corrected_data = np.maximum(corrected_data, MIN_BOUND_HU)

            new_img = nib.Nifti1Image(corrected_data, img.affine, img.header)
            new_img.header.set_slope_inter(slope=1.0, inter=0.0)
            new_img.header.set_zooms((ps_x, ps_y, thickness))

            nib.save(new_img, output_dir / nifti_path.name)
            success_count += 1

        except Exception as e:
            logging.error(f"Error processing {nifti_path.name}: {e}")

    logging.info(f"Harmonization complete. {success_count}/{len(nifti_files)} files saved to {output_dir}")


# ---------------------------------------------------------------------------
# Stage 2: Slice-order consistency check
# ---------------------------------------------------------------------------

def check_slice_order(nifti_path: Path, axis: int = 2, threshold: float = 0.9):
    """
    Compute inter-slice correlations and flag files with disordered slices.

    Returns (correlations, is_suspicious, jaggedness_score).
    """
    try:
        img = nib.load(nifti_path)
        data = img.get_fdata()

        if data.ndim == 4:
            data = data[..., 0]

        data = np.moveaxis(data, axis, 0)
        num_slices = data.shape[0]

        correlations = []
        for i in range(num_slices - 1):
            corr = np.corrcoef(data[i].ravel(), data[i + 1].ravel())[0, 1]
            correlations.append(corr)

        correlations = np.array(correlations)
        min_corr = np.min(correlations)
        diffs = np.abs(np.diff(correlations))
        jaggedness = np.mean(diffs)

        is_suspicious = (min_corr < threshold) or (jaggedness > 0.05)
        return correlations, is_suspicious, jaggedness

    except Exception as e:
        logging.error(f"Error processing {nifti_path}: {e}")
        return np.array([]), False, 0.0


def run_slice_check(nifti_dir: Path, flagged_csv: Path):
    """Scan all NIfTI files in *nifti_dir* and write flagged filenames to *flagged_csv*."""
    logging.info("=== Stage 2: Slice-Order Consistency Check ===")

    nifti_files = sorted(nifti_dir.glob('*.nii*'))
    logging.info(f"Checking {len(nifti_files)} files...")

    flagged = []
    for filepath in tqdm(nifti_files, desc="Checking slices"):
        corrs, is_bad, jagged_score = check_slice_order(filepath)
        if is_bad:
            min_corr = float(np.min(corrs)) if len(corrs) > 0 else 0.0
            logging.info(f"[FLAGGED] {filepath.name} | Min Corr: {min_corr:.4f} | Jaggedness: {jagged_score:.4f}")
            flagged.append({
                'file_name': filepath.name,
                'reason': 'corrupted_slices',
                'min_correlation': round(min_corr, 4),
                'jaggedness': round(jagged_score, 4),
            })

    df_flagged = pd.DataFrame(flagged)
    df_flagged.to_csv(flagged_csv, index=False)

    if flagged:
        logging.info(f"Found {len(flagged)} suspicious scans. Saved to {flagged_csv}")
    else:
        logging.info("No disordered slices detected.")


# ---------------------------------------------------------------------------
# Stage 3: Body-part classification (chest vs. abdomen)
# ---------------------------------------------------------------------------

def classify_body_part(nifti_path: Path, lung_threshold_ratio: float = 0.10):
    """
    Determine if a CT scan is chest or abdomen based on lung tissue volume.

    Chest scans typically have 15-40% internal air (lung) volume relative to
    total body volume. Abdominal scans (bowel gas only) are usually below 5%.

    Returns (classification, lung_ratio).
    """
    try:
        img = nib.load(nifti_path)
        data = img.get_fdata()

        # Downsample for speed (every 4th voxel)
        data_small = data[::4, ::4, ::4]

        # Solid body mask: soft tissue / bone / fluid (> -300 HU)
        solid_body_mask = data_small > -300

        # Fill holes slice-by-slice to close the rib cage / skin contour
        filled_body_mask = np.zeros_like(solid_body_mask)
        for i in range(solid_body_mask.shape[2]):
            filled_body_mask[:, :, i] = ndi.binary_fill_holes(solid_body_mask[:, :, i])

        # Internal air = inside body but not solid (i.e. lungs)
        internal_air_mask = filled_body_mask & (~solid_body_mask)

        total_body_voxels = np.sum(filled_body_mask)
        lung_voxels = np.sum(internal_air_mask)

        if total_body_voxels == 0:
            return "UNKNOWN", 0.0

        ratio = lung_voxels / total_body_voxels
        classification = "CHEST" if ratio > lung_threshold_ratio else "ABDOMEN"
        return classification, ratio

    except Exception as e:
        logging.error(f"Error processing {nifti_path}: {e}")
        return "ERROR", 0.0


def run_body_check(nifti_dir: Path, flagged_csv: Path, lung_threshold: float = 0.10):
    """Classify each NIfTI as chest or abdomen and write non-chest filenames to *flagged_csv*."""
    logging.info("=== Stage 3: Body-Part Classification (Chest vs. Abdomen) ===")

    nifti_files = sorted(nifti_dir.glob('*.nii*'))
    logging.info(f"Classifying {len(nifti_files)} files (lung threshold ratio: {lung_threshold:.0%})...")

    flagged = []
    for filepath in tqdm(nifti_files, desc="Classifying body part"):
        classification, ratio = classify_body_part(filepath, lung_threshold_ratio=lung_threshold)

        if classification != "CHEST":
            logging.info(f"[FLAGGED] {filepath.name} | {classification} | Lung ratio: {ratio:.1%}")
            flagged.append({
                'file_name': filepath.name,
                'reason': 'non_chest',
                'classification': classification,
                'lung_ratio': round(ratio, 4),
            })

    df_flagged = pd.DataFrame(flagged)
    df_flagged.to_csv(flagged_csv, index=False)

    if flagged:
        logging.info(f"Found {len(flagged)} non-chest scans. Saved to {flagged_csv}")
    else:
        logging.info("All scans classified as chest.")


# ---------------------------------------------------------------------------
# Stage 4: Filter flagged series from embeddings
# ---------------------------------------------------------------------------

def run_filter(embed_csv: Path, flagged_csvs: list[Path], clean_embed_csv: Path):
    """Remove rows whose file_name appears in any of *flagged_csvs* from the embeddings CSV."""
    logging.info("=== Stage 4: Filter Flagged Series ===")

    df_embed = pd.read_csv(embed_csv)

    # Merge all flagged CSVs into one set of IDs
    all_flagged_ids = set()
    for csv_path in flagged_csvs:
        if not csv_path.exists():
            logging.warning(f"Flagged CSV not found, skipping: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        if 'file_name' not in df.columns:
            logging.warning(f"No 'file_name' column in {csv_path}, skipping.")
            continue
        # Strip .nii.gz extension so IDs match the embeddings file_name column
        ids = df['file_name'].str.replace('.nii.gz', '', regex=False).str.replace('.nii', '', regex=False)
        all_flagged_ids.update(ids)
        logging.info(f"  Loaded {len(ids)} flagged IDs from {csv_path}")

    before = len(df_embed)
    df_clean = df_embed[~df_embed['file_name'].isin(all_flagged_ids)]
    after = len(df_clean)

    logging.info(f"Embeddings before filtering: {before}")
    logging.info(f"Embeddings after filtering:  {after}  (removed {before - after})")

    if 'labels' in df_clean.columns:
        logging.info(f"Positive samples remaining: {int(df_clean['labels'].sum())}")

    df_clean.to_csv(clean_embed_csv, index=False)
    logging.info(f"Clean embeddings saved to {clean_embed_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test-data cleaning pipeline for CT series.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='stage', required=True)

    # --- harmonize ---
    p_harm = subparsers.add_parser('harmonize', help="Stage 1: harmonize NIfTI physics")
    p_harm.add_argument('--nifti_dir', type=Path, required=True, help="Directory with raw NIfTI files")
    p_harm.add_argument('--tsv', type=Path, required=True, help="Path to series metadata TSV")
    p_harm.add_argument('--output_dir', type=Path, required=True, help="Where to save corrected NIfTIs")

    # --- slice-check ---
    p_slice = subparsers.add_parser('slice-check', help="Stage 2: detect disordered slices")
    p_slice.add_argument('--nifti_dir', type=Path, required=True, help="Directory with NIfTI files to check")
    p_slice.add_argument('--flagged_csv', type=Path, default=Path('output/corrupted_slices.csv'),
                         help="Output CSV listing flagged files (default: output/corrupted_slices.csv)")

    # --- body-check ---
    p_body = subparsers.add_parser('body-check', help="Stage 3: classify chest vs. abdomen")
    p_body.add_argument('--nifti_dir', type=Path, required=True, help="Directory with NIfTI files to classify")
    p_body.add_argument('--flagged_csv', type=Path, default=Path('output/non_chest_scans.csv'),
                        help="Output CSV listing non-chest files (default: output/non_chest_scans.csv)")
    p_body.add_argument('--lung_threshold', type=float, default=0.10,
                        help="Lung volume ratio to classify as chest (default: 0.10 = 10%%)")

    # --- filter ---
    p_filt = subparsers.add_parser('filter', help="Stage 4: remove flagged series from embeddings")
    p_filt.add_argument('--embed_csv', type=Path, required=True, help="Input embeddings CSV")
    p_filt.add_argument('--flagged_csvs', type=Path, nargs='+', required=True,
                        help="One or more CSVs with flagged file names")
    p_filt.add_argument('--clean_embed_csv', type=Path, required=True, help="Output cleaned embeddings CSV")

    # --- all ---
    p_all = subparsers.add_parser('all', help="Run all four stages sequentially")
    p_all.add_argument('--nifti_dir', type=Path, required=True, help="Directory with raw NIfTI files")
    p_all.add_argument('--tsv', type=Path, required=True, help="Path to series metadata TSV")
    p_all.add_argument('--output_dir', type=Path, required=True, help="Where to save corrected NIfTIs")
    p_all.add_argument('--embed_csv', type=Path, required=True, help="Input embeddings CSV")
    p_all.add_argument('--clean_embed_csv', type=Path, required=True, help="Output cleaned embeddings CSV")
    p_all.add_argument('--lung_threshold', type=float, default=0.10,
                       help="Lung volume ratio to classify as chest (default: 0.10 = 10%%)")

    args = parser.parse_args()

    if args.stage == 'harmonize':
        run_harmonize(args.nifti_dir, args.tsv, args.output_dir)

    elif args.stage == 'slice-check':
        args.flagged_csv.parent.mkdir(parents=True, exist_ok=True)
        run_slice_check(args.nifti_dir, args.flagged_csv)

    elif args.stage == 'body-check':
        args.flagged_csv.parent.mkdir(parents=True, exist_ok=True)
        run_body_check(args.nifti_dir, args.flagged_csv, args.lung_threshold)

    elif args.stage == 'filter':
        run_filter(args.embed_csv, args.flagged_csvs, args.clean_embed_csv)

    elif args.stage == 'all':
        output_base = args.output_dir.parent  # e.g. ./output
        output_base.mkdir(parents=True, exist_ok=True)
        corrupted_slices_csv = output_base / 'corrupted_slices.csv'
        non_chest_csv = output_base / 'non_chest_scans.csv'

        run_harmonize(args.nifti_dir, args.tsv, args.output_dir)
        run_slice_check(args.output_dir, corrupted_slices_csv)
        run_body_check(args.output_dir, non_chest_csv, args.lung_threshold)
        run_filter(args.embed_csv, [corrupted_slices_csv, non_chest_csv], args.clean_embed_csv)

        # Copy final output to the datasets/ folder used by training.py
        datasets_dir = Path(__file__).resolve().parent.parent / 'datasets'
        datasets_dir.mkdir(parents=True, exist_ok=True)
        dest = datasets_dir / 'dataset_test_inspect.csv'
        shutil.copy2(args.clean_embed_csv, dest)
        logging.info(f"Copied final output to {dest}")

    logging.info("Pipeline complete.")


if __name__ == "__main__":
    main()
