from enum import Enum
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')




class DatasetType(Enum):
    Consolidation_1000 = 1
    Consolidation_2000 = 2
    CTPE = 3
    Abnormal_1000 = 4
    Normal_1000 = 5



def dataset_directory(dataset_type: DatasetType):
    if dataset_type == DatasetType.Consolidation_1000:
        dataset_dir = 'consolidation_1000'
        filename = 'consolidation_1000/labels_consolidation.csv'
        DIAGNOSIS = 'CONSOLIDATION'
    elif dataset_type == DatasetType.Consolidation_2000:
        dataset_dir = 'consolidation_2000'
        filename = 'consolidation_2000/labels_consolidation.csv'
        DIAGNOSIS = 'CONSOLIDATION'
    elif dataset_type == DatasetType.Normal_1000:
        dataset_dir = 'normal_vs_abnormal'
        filename = 'normal_vs_abnormal/labels_normal.csv'
        DIAGNOSIS = 'Normal'
    elif dataset_type == DatasetType.Abnormal_1000:
        dataset_dir = 'normal_vs_abnormal'
        filename = 'normal_vs_abnormal/labels_abnormal.csv'
        DIAGNOSIS = 'Abnormal'
    elif dataset_type == DatasetType.CTPE:
        dataset_dir = 'ctpe_dataset'
        filename = 'ctpe_dataset/labels_pe.csv'
        DIAGNOSIS = 'labels'
    else:
        raise ValueError(f"dataset: {dataset_type} does not exist." )
    return dataset_dir, filename, DIAGNOSIS


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
