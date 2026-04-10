"""
Distributional Analysis: RSNA vs. INSPECt Embeddings

Compares the CT-Foundation embedding distributions between the RSNA validation
set and the INSPECt test set using three methods:
  1. t-SNE visualization (single and dual overlay plots)
  2. Classifier two-sample test (Random Forest + permutation p-value)
  3. Maximum Mean Discrepancy (MMD) test with permutation p-value

Because the INSPECt set is larger than RSNA, a size-matched subset is created
first (preserving the positive/negative ratio) before running comparisons.

Usage:
  python distributional_analysis.py \
    --rsna_csv datasets/dataset_validation.csv \
    --inspect_csv datasets/dataset_test_inspect.csv \
    --n_pos 36 --n_neg 108
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, permutation_test_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def embedding_array(embed):
    embedding_numpy = []
    for x in embed:
        embedding_numpy.append(np.fromstring(x[1:-1], dtype=float, sep=','))
    return embedding_numpy


def load_embeddings(csv_path: Path):
    """Load embeddings CSV and return (embeddings_array, labels, file_names)."""
    df = pd.read_csv(csv_path)
    file_names = df['file_name']
    labels = df['labels']
    embeddings = embedding_array(df['embedding'].values)
    return np.array(embeddings), np.array(labels), file_names


def create_subset(csv_path: Path, n_pos: int, n_neg: int) -> pd.DataFrame:
    """Create a size-matched subset of the INSPECt dataset."""
    df = pd.read_csv(csv_path)
    df_pos = df[df['labels'] == 1].head(n_pos)
    df_neg = df[df['labels'] == 0].head(n_neg)
    df_subset = pd.concat([df_neg, df_pos], ignore_index=True)
    logging.info(f"Created INSPECt subset: {len(df_neg)} negative + {len(df_pos)} positive = {len(df_subset)} total")
    return df_subset


# ---------------------------------------------------------------------------
# t-SNE visualization
# ---------------------------------------------------------------------------

def plot_tsne(embeddings, labels, title):
    """Single-dataset t-SNE scatter plot."""
    logging.info(f"Running t-SNE for '{title}'...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(4, 3))

    colors = {0: '#3498db', 1: '#e74c3c'}
    class_labels = {0: 'Negative PE', 1: 'Positive PE'}

    scatter = sns.scatterplot(
        x=embeddings_2d[:, 0], y=embeddings_2d[:, 1],
        hue=labels, palette=colors,
        s=30, alpha=0.8, edgecolor='w', linewidth=0.5
    )

    plt.xlabel("t-SNE Dimension 1", fontsize=10)
    plt.ylabel("t-SNE Dimension 2", fontsize=10)

    handles, _ = scatter.get_legend_handles_labels()
    scatter.legend(handles, [class_labels[0], class_labels[1]],
                   title='Case Type', fontsize=8, loc='lower left')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_dual_tsne(embeddings1, labels1, name1, embeddings2, labels2, name2):
    """Overlay two embedding sets in a single t-SNE plot."""
    logging.info(f"Running combined t-SNE for '{name1}' vs '{name2}'...")

    combined_embeddings = np.vstack([embeddings1, embeddings2])
    source_labels = [name1] * len(embeddings1) + [name2] * len(embeddings2)
    combined_labels = np.concatenate([labels1, labels2])

    label_map = {0: 'Negative PE', 1: 'Positive PE'}
    text_labels = [label_map[l] for l in combined_labels]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(combined_embeddings)

    df_plot = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'Source': source_labels,
        'Class': text_labels
    })

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(6, 5))

    set_colors = {name1: '#1abc9c', name2: '#9b59b6'}
    class_markers = {'Negative PE': 'o', 'Positive PE': 'X'}

    sns.scatterplot(
        data=df_plot, x='x', y='y',
        hue='Source', style='Class',
        palette=set_colors, markers=class_markers,
        s=40, alpha=0.7, edgecolor='w', linewidth=0.5
    )

    plt.xlabel("t-SNE Dimension 1", fontsize=10)
    plt.ylabel("t-SNE Dimension 2", fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Classifier two-sample test
# ---------------------------------------------------------------------------

def run_classifier_test(X_a, X_b, n_permutations=100):
    """
    Train a classifier to distinguish two embedding sets.
    Returns (accuracy, p_value). Accuracy near 0.5 means indistinguishable.
    """
    X = np.vstack([X_a, X_b])
    y = np.hstack([np.zeros(len(X_a)), np.ones(len(X_b))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    _, _, p_value = permutation_test_score(
        clf, X, y, scoring="accuracy", cv=5, n_permutations=n_permutations, n_jobs=-1
    )

    return accuracy, p_value


# ---------------------------------------------------------------------------
# Maximum Mean Discrepancy (MMD) test
# ---------------------------------------------------------------------------

def compute_mmd_statistic(X, Y, gamma):
    """Calculate MMD^2 statistic using RBF kernel."""
    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def run_mmd_test(X_a, X_b, n_permutations=1000):
    """Run MMD test with permutation to get a p-value."""
    n_features = X_a.shape[1]
    gamma = 1.0 / n_features

    observed_mmd = compute_mmd_statistic(X_a, X_b, gamma)

    combined = np.vstack([X_a, X_b])
    n_a = len(X_a)
    count = 0

    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_mmd = compute_mmd_statistic(perm_a, perm_b, gamma)
        if perm_mmd >= observed_mmd:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return observed_mmd, p_value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare RSNA and INSPECt embedding distributions.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--rsna_csv', type=Path, default=Path('datasets/dataset_validation.csv'),
                        help="Path to RSNA validation embeddings CSV")
    parser.add_argument('--inspect_csv', type=Path, default=Path('datasets/dataset_test_inspect.csv'),
                        help="Path to INSPECt test embeddings CSV")
    parser.add_argument('--n_pos', type=int, default=36,
                        help="Number of positive samples in INSPECt subset (default: 36)")
    parser.add_argument('--n_neg', type=int, default=108,
                        help="Number of negative samples in INSPECt subset (default: 108)")

    args = parser.parse_args()

    # 1. Load RSNA embeddings
    logging.info("Loading RSNA embeddings...")
    rsna_embed, rsna_labels, _ = load_embeddings(args.rsna_csv)
    logging.info(f"RSNA: {len(rsna_embed)} samples ({int(rsna_labels.sum())} positive)")

    # 2. Create size-matched INSPECt subset and load
    logging.info("Creating size-matched INSPECt subset...")
    df_subset = create_subset(args.inspect_csv, n_pos=args.n_pos, n_neg=args.n_neg)
    inspect_embed = np.array(embedding_array(df_subset['embedding'].values))
    inspect_labels = np.array(df_subset['labels'])

    # 3. t-SNE: dual overlay plot
    plot_dual_tsne(
        rsna_embed, rsna_labels, 'RSNA',
        inspect_embed, inspect_labels, 'INSPECt',
    )

    # 4. Classifier two-sample test
    logging.info("--- Classifier Two-Sample Test ---")
    acc, p_val_clf = run_classifier_test(rsna_embed, inspect_embed)
    logging.info(f"Classifier Accuracy: {acc:.4f} (random chance is 0.5)")
    logging.info(f"P-value: {p_val_clf:.4f}")
    if acc > 0.6 and p_val_clf < 0.05:
        logging.info(">> Distributions are significantly DIFFERENT (classifier can tell them apart).")
    else:
        logging.info(">> Distributions are NOT significantly different.")

    # 5. MMD test
    logging.info("--- MMD Test ---")
    mmd_stat, p_val_mmd = run_mmd_test(rsna_embed, inspect_embed)
    logging.info(f"MMD Statistic: {mmd_stat:.6f}")
    logging.info(f"P-value: {p_val_mmd:.4f}")
    if p_val_mmd < 0.05:
        logging.info(">> Distributions are significantly DIFFERENT (p < 0.05).")
    else:
        logging.info(">> Distributions are indistinguishable.")


if __name__ == "__main__":
    main()
