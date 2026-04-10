import sklearn
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix


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
    directories = df['file_name'].values
    return X, y, directories


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