import sklearn
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix

matplotlib.use('TkAgg')


def embedding_array(embed):
    embedding_numpy = []
    for x in embed:
        embedding_numpy.append(np.fromstring(x[1:-1], dtype=float, sep=','))
    return embedding_numpy

def input_output(df):
    """
    :param df: Pandas dataset containing both embeddings and labels
    :return: Tuple (embeddings as a numpy array, labels as a list)
    """
    X = np.array(embedding_array(df['embedding'].values))
    y = df['labels'].values
    # directories = df['series_dir'].values
    directories = df['file_name'].values
    return X, y, directories

def create_dataset(filename):
    """
    :param filename: file's name
    :return: dataset
    """
    df_labels = pd.read_csv(filename)
    df_labels = df_labels.reset_index(drop=True)
    return df_labels

def split_dataset(df_labels, label, split_ratio = 0.2):
    """
    Split the dataset
    :param df_labels: The file contains the dataset labels and path
    :param label: The key for labels
    :param split_ratio: The ratio for splitting the dataset
    return: Two separate datasets
    """
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    df_train, df_validate = train_test_split(df_labels, test_size=split_ratio,
                                             stratify=df_labels[[label]])

    return df_train, df_validate

def data_augmentation(original_data,
                      TOKEN_NUM=1,
                      EMBEDDINGS_SIZE = 1408,
                      noise_std=1e-4,
                      ratio=4)->pd.DataFrame:
    """
    :param original_data: dataset using the training one
    :param TOKEN_NUM: 1
    :param EMBEDDINGS_SIZE: 1408
    :param noise_std: std of the noise to be added to the embeddings of original dataset
    :param ratio: The ratio of the augmented dataset size to the original one
    :return: the augmented dataset
    """
    series_original = original_data['series_id'].values
    embeddings_original = embedding_array(original_data['embedding'].values)
    labels_original = original_data['labels'].values

    embeddings_new = []
    series_new = []
    labels_new = []
    for i in range(len(series_original)):
        embedding_datapoint = np.array(embeddings_original[i])
        series_datapoint =series_original[i]
        label_datapoint = labels_original[i]
        embeddings_new.append(embedding_datapoint)
        series_new.append(series_datapoint)
        labels_new.append(label_datapoint)
        for j in range(ratio):
            noise = np.random.normal(0, noise_std, (EMBEDDINGS_SIZE * TOKEN_NUM,))
            embeddings_new.append(embedding_datapoint + noise)
            series_new.append(f'syntethic_{j}_{series_datapoint}')
            labels_new.append(label_datapoint)

    dataset = {'labels': labels_new, 'series_id': series_new, 'embedding':embeddings_new}
    augmented_data = pd.DataFrame(data=dataset)
    return augmented_data


def auc_confidence_interval(y_true, y_pred, num_bootstraps=1000, alpha=0.05):
    """
    Calculates the confidence interval for the AUC using bootstrapping.

    Args:
      y_true: True binary labels.
      y_pred: Predicted probabilities for the positive class.
      num_bootstraps: Number of bootstrap samples.
      alpha: Significance level for the confidence interval.

    Returns:
      Tuple: (lower_bound, upper_bound) of the confidence interval.
    """

    auc_values = []
    for _ in range(num_bootstraps):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        auc = sklearn.metrics.roc_auc_score(y_true[indices], y_pred[indices])
        auc_values.append(auc)

    alpha /= 2  # Two-tailed test
    lower_percentile = int(num_bootstraps * alpha)
    upper_percentile = int(num_bootstraps * (1 - alpha))
    auc_values.sort()

    return auc_values[lower_percentile], auc_values[upper_percentile]

def class_weight_calculator(n_neg, n_pos):
    """
    Calculate the class weight for the imbalanced dataset

    :param n_neg: number of negative cases
    :param n_pos: number of positive cases
    :return: the class weight
    """
    total = n_neg + n_pos
    weight_for_0 = (1 / n_neg) * (total / 2.0)
    weight_for_1 = (1 / n_pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    return class_weight

def oversampling_training(feature, labels):
    """
    Replicates samples of the minority class in the training dataset.
    :param feature: embeddings
    :param labels: the labels corresponds to embeddings
    :return:
    """
    feature_ps = []
    labels_ps = []
    feature_ng = []
    labels_ng = []
    for (ft, lb) in zip(feature, labels):
        if lb:
            feature_ps.append(ft)
            labels_ps.append(lb)
        else:
            feature_ng.append(ft)
            labels_ng.append(lb)

    pos_ds =  tf.data.Dataset.from_tensor_slices((feature_ps, labels_ps))
    neg_ds = tf.data.Dataset.from_tensor_slices((feature_ng, labels_ng))

    pos_ds = pos_ds.shuffle(500).repeat()
    neg_ds = neg_ds.shuffle(500).repeat()
    train_ds = tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
    return train_ds


def plot_curve(x, y, auc, x_label=None, y_label=None, label=None):

    fig = plt.figure(figsize=(10, 10))
    plt.plot(x, y, label=f'{label} (AUC: %.3f)' % auc, color='black')
    plt.legend(loc='lower right', fontsize=18)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    if x_label:
        plt.xlabel(x_label, fontsize=24)
    if y_label:
        plt.ylabel(y_label, fontsize=24)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(visible=True)
    plt.show()


def calculate_optimal_threshold_metrics(y_true, y_pred_prob, target_sensitivity=None):
    """
    Calculates the threshold.

    Args:
        y_true: True labels
        y_pred_prob: Predicted probabilities
        target_sensitivity (float): If set (e.g., 0.90), finds the threshold
                                    that yields at least this sensitivity.
                                    If None, uses Youden's J statistic.
    """
    # 1. Calculate ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)

    if target_sensitivity is not None:
        # STRATEGY 1: FIXED SENSITIVITY (Medical Standard)
        # Find the first index where TPR is >= target_sensitivity
        # We look for the smallest threshold (highest index in standard ROC arrays)
        # that satisfies the condition.

        # Get indices where TPR >= target
        qualifying_indices = np.where(tpr >= target_sensitivity)[0]

        if len(qualifying_indices) > 0:
            # Use the index that maximizes specificity (minimizes FPR) among those that qualify
            # Since FPR and TPR are sorted, usually the first one that hits the target
            # is the one with the lowest FPR.
            best_idx = qualifying_indices[0]

            # Safety check: if the resulting FPR is 1.0, the model might be failing to reach target
            if fpr[best_idx] == 1.0 and tpr[best_idx] < target_sensitivity:
                # Fallback to Youden if target is unreachable
                best_idx = np.argmax(tpr - fpr)
        else:
            # Target unreachable, fallback to Youden
            best_idx = np.argmax(tpr - fpr)

    else:
        # STRATEGY 2: YOUDEN'S INDEX (Balanced)
        J = tpr - fpr
        best_idx = np.argmax(J)

    best_threshold = thresholds[best_idx]

    # 3. Apply Threshold
    y_pred_binary = (y_pred_prob >= best_threshold).astype(int)

    # 4. Calculate Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    # 5. Calculate Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'optimal_threshold': float(best_threshold),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def calculate_metrics_at_threshold(y_true, y_pred_prob, threshold):
    """
    Calculates confusion matrix metrics given a specific threshold.

    Returns:
        A dictionary with sensitivity, specificity, and confusion matrix counts,
        all as native Python types (int/float) for JSON serialization.
    """
    # Apply the threshold to get binary predictions
    y_pred_binary = (y_pred_prob >= threshold).astype(int)

    # Calculate raw confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'threshold_used': float(threshold),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def generate_threshold_performance_report(y_val, y_pred_prob,thresholds):
    results_list = []
    # Loop through each threshold to calculate metrics
    for threshold in thresholds:
        # Convert probabilities to binary predictions based on the threshold
        y_pred_binary = (y_pred_prob >= threshold).astype(int)

        # Calculate the confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred_binary).ravel()

        # Calculate the rates, handling potential division by zero
        total_positives = tp + fn
        total_negatives = tn + fp

        tpr = tp / total_positives if total_positives > 0 else 0  # True Positive Rate (Sensitivity)
        tnr = tn / total_negatives if total_negatives > 0 else 0  # True Negative Rate (Specificity)
        fpr = fp / total_negatives if total_negatives > 0 else 0  # False Positive Rate
        fnr = fn / total_positives if total_positives > 0 else 0  # False Negative Rate

        # Store results in a dictionary
        results_list.append({
            'Threshold': f"{threshold:.1f}",
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'True Positive Rate (TPR)': f"{tpr:.2%}",
            'True Negative Rate (TNR)': f"{tnr:.2%}",
            'False Positive Rate (FPR)': f"{fpr:.2%}",
            'False Negative Rate (FNR)': f"{fnr:.2%}"
        })

    # Create a pandas DataFrame from the results
    confusion_matrix_df = pd.DataFrame(results_list)

    # Display the results table
    print("Confusion Matrix and Rates for Specific Thresholds:")
    print(confusion_matrix_df.to_string(index=False))