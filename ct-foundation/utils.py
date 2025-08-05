import sklearn
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

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
    directories = df['series_dir'].values
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