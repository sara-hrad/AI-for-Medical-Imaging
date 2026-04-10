"""
Training & Validation Script

This script provides a modular and configurable pipeline for training
and evaluating multiple classifier models (MLP, Random Forest, Ensemble)
across various datasets.

Usage:
  python training.py single --dataset "data_no_mask_1" --model "mlp" --sensitivity 0.9
  python training.py manual --dataset "data_no_mask_1" --lr 0.001 --alpha 0.5 --dropout 0.3 --steps 100 --wd 1e-7 --layers "64,64" --sensitivity 0.9 --noise 0.01
  python training.py evaluate --model_path path/to/model --model_type mlp --test_data_path path/to/test.csv
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import logging
import datetime
import argparse
from pathlib import Path
import json

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import keras_tuner as kt
from utils import *
from savebestmodel import SaveBestModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# Experiment configuration: dataset paths and their corresponding validation sets.
# 'data_no_mask_1' to 'data_no_mask_4' use 'valid_no_mask'.
# 'data_mask_1' and 'data_mask_2' use 'valid_mask'.
EXPERIMENT_CONFIG = {
    'datasets': {
        # --- No-Mask Datasets ---
        'data_no_mask_1': {
            'train_path': Path('datasets/dataset_training_32.csv'),
            'validation_set': 'valid_no_mask'
        },
        'data_no_mask_2': {
            'train_path': Path('datasets/dataset_training_64.csv'),
            'validation_set': 'valid_no_mask'
        },
        'data_no_mask_3': {
            'train_path': Path('datasets/dataset_training_all_balanced.csv'),
            'validation_set': 'valid_no_mask'
        },
        'data_no_mask_4': {
            'train_path': Path('datasets/dataset_training_imbalanced.csv'),
            'validation_set': 'valid_no_mask'
        },
        # --- Mask Datasets ---
        'data_mask_1': {
            'train_path': Path('datasets/dataset_training_masked_32.csv'),
            'validation_set': 'valid_mask'
        },
        'data_mask_2': {
            'train_path': Path('datasets/dataset_training_masked_64.csv'),
            'validation_set': 'valid_mask'
        },
        'smoke_test_data': {
            'train_path': Path('datasets/dataset_training_32.csv'),
            'validation_set': 'valid_no_mask'
        }
    },
    'validation_sets': {
        'valid_no_mask': Path('datasets/dataset_validation.csv'),
        'valid_mask': Path('datasets/dataset_validation_masked.csv'),
        'smoke_test_valid': Path('datasets/dataset_validation.csv')
    },
    'model_types': ['mlp', 'rf', 'ensemble', 'mlp_manual'],
}

# Base directory for all outputs
RESULTS_BASE_DIR = Path('results')


class CVRandomSearch(kt.RandomSearch):
    """
    Custom Keras Tuner that performs K-Fold Cross-Validation
    instead of a single validation split.
    """
    def run_trial(self, trial, x, y, batch_size=32, epochs=1, callbacks=None, class_weight=None):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        val_aucs = []

        # Iterate through the 5 folds
        for train_idx, test_idx in cv.split(x, y):
            x_train_fold, x_val_fold = x[train_idx], x[test_idx]
            y_train_fold, y_val_fold = y[train_idx], y[test_idx]

            model = self.hypermodel.build(trial.hyperparameters)

            # Shorter patience for speed during HP tuning
            fold_callbacks = [EarlyStopping(monitor='val_auc_roc', patience=50, mode='max')]

            model.fit(x_train_fold,
                      y_train_fold,
                      validation_data=(x_val_fold, y_val_fold),
                      epochs=epochs,
                      batch_size=batch_size,
                      callbacks=fold_callbacks,
                      class_weight=class_weight,
                      verbose=0)  # Silence individual fold logs

            # 'auc_roc' matches the metric name defined in build_model()
            eval_metrics = model.evaluate(x_val_fold, y_val_fold, verbose=0, return_dict=True)
            val_aucs.append(eval_metrics['auc_roc'])

        mean_val_auc = np.mean(val_aucs)

        # Report mean CV AUC as 'val_auc_roc' to match the tuner objective
        self.oracle.update_trial(trial.trial_id, {'val_auc_roc': mean_val_auc})

        # Fold models are discarded; the best HP set is rebuilt in the training function.


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_model(hp):
    """
    Keras Tuner "HyperModel" builder function.
    """

    # Hyperparameters to tune
    hp_learning_rate = hp.Choice('learning_rate', values=[0.002, 0.001, 0.0005])
    hp_alpha = hp.Choice('alpha', values=[0.1, 0.5, 0.8])
    hp_dropout = hp.Choice('dropout', values=[0.2, 0.5])
    hp_first_decay_steps = hp.Choice('first_decay_steps', values=[50, 100])
    hp_weight_decay = hp.Choice('weight_decay', values=[1e-7])
    hp_noise_stddev = hp.Choice('noise_stddev', values=[0.0, 0.1, 0.01])

    # Architecture variants encoded as underscore-separated layer sizes
    hp_layer_config = hp.Choice('hidden_layers', values=['32_32', '64_64'])

    hidden_layer_sizes = [int(size) for size in hp_layer_config.split('_')]

    # Fixed parameters
    token_num = 1
    embeddings_size = 1408
    seed = 42

    # Model architecture
    inputs = tf.keras.Input(shape=(token_num * embeddings_size,))
    noise = tf.keras.layers.GaussianNoise(hp_noise_stddev)(inputs)
    inputs_reshape = tf.keras.layers.Reshape((token_num, embeddings_size))(noise)
    inputs_pooled = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')(inputs_reshape)
    hidden = inputs_pooled

    for size in hidden_layer_sizes:
        hidden = tf.keras.layers.Dense(
            units=size,
            activation='relu',
            kernel_initializer=tf.keras.initializers.HeUniform(seed=seed),
            kernel_regularizer=tf.keras.regularizers.l1(l1=hp_weight_decay),
            bias_regularizer=tf.keras.regularizers.l1(l1=hp_weight_decay))(
            hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.LayerNormalization()(hidden)
        hidden = tf.keras.layers.Dropout(hp_dropout, seed=seed)(hidden)

    output = tf.keras.layers.Dense(
        units=1,
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.HeUniform(seed=seed))(
        hidden)

    model = tf.keras.Model(inputs, output)

    # Compilation with cosine decay restarts LR schedule
    learning_rate_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=hp_learning_rate,
        first_decay_steps=hp_first_decay_steps,
        alpha=hp_alpha)

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate_fn),
        loss='binary_crossentropy',
        weighted_metrics=[
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.TruePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.AUC(curve='ROC', name='auc_roc')  # Primary tuning metric
        ])
    return model

def build_model_fixed(
    input_shape,
    hidden_layer_sizes: list,
    learning_rate: float,
    alpha: float,
    dropout: float,
    first_decay_steps: int,
    weight_decay: float,
    noise_stddev: float,
    seed: int = 42
):
    """
    Constructs the MLP model with fixed, manually specified hyperparameters.
    Used for manual training runs (without Keras Tuner).
    """
    token_num = 1
    embeddings_size = 1408

    # Architecture
    inputs = tf.keras.Input(shape=(input_shape,))
    noise = tf.keras.layers.GaussianNoise(noise_stddev)(inputs)
    inputs_reshape = tf.keras.layers.Reshape((token_num, embeddings_size))(noise)
    inputs_pooled = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')(inputs_reshape)
    hidden = inputs_pooled

    for size in hidden_layer_sizes:
        hidden = tf.keras.layers.Dense(
            units=size,
            activation='relu',
            kernel_initializer=tf.keras.initializers.HeUniform(seed=seed),
            kernel_regularizer=tf.keras.regularizers.l1(l1=weight_decay),
            bias_regularizer=tf.keras.regularizers.l1(l1=weight_decay))(
            hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.LayerNormalization()(hidden)
        hidden = tf.keras.layers.Dropout(dropout, seed=seed)(hidden)

    output = tf.keras.layers.Dense(
        units=1,
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.HeUniform(seed=seed))(
        hidden)

    model = tf.keras.Model(inputs, output)

    # Compilation
    learning_rate_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=learning_rate,
        first_decay_steps=first_decay_steps,
        alpha=alpha)

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate_fn),
        loss='binary_crossentropy',
        weighted_metrics=[
            tf.keras.metrics.AUC(curve='ROC', name='auc_roc')
        ])

    return model

# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_mlp_with_cv(df_train: pd.DataFrame,
                      df_validate: pd.DataFrame,
                      model_save_path: Path,
                      log_dir: Path,
                      sensitivity_threshold: float) -> tf.keras.Model:
    """
    Trains and tunes an MLP model using 5-Fold Cross-Validation via a Custom Keras Tuner.
    """
    logging.info("Starting MLP training with 5-Fold CV (Custom Keras Tuner)...")

    # Data preparation
    X_train, y_train, _ = input_output(df_train)
    X_val, y_val, _ = input_output(df_validate)  # Used only for final evaluation
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)

    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    class_weights = class_weight_calculator(n_neg, n_pos)
    logging.info(f"Calculated class weights: {class_weights}")

    # Custom CV tuner (HP tuning via 5-fold CV on training data only)
    tuner_directory = log_dir / 'keras_tuner_cv'
    project_name = 'mlp_cv_tuning'

    tuner = CVRandomSearch(
        hypermodel=build_model,
        objective=kt.Objective("val_auc_roc", direction="max"),
        max_trials=250,
        executions_per_trial=1,
        directory=str(tuner_directory),
        project_name=project_name,
        overwrite=True
    )

    # Run the CV search (splitting is handled internally by CVRandomSearch)
    logging.info("Starting Custom CV Search (5-Fold)...")
    tuner.search(
        x=X_train,
        y=y_train,
        epochs=500,  # Max epochs per fold
        batch_size=32,
        class_weight=class_weights
    )

    logging.info("CV Search finished.")

    # Retrieve best hyperparameters
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    best_cv_score = best_trial.score
    best_hps = best_trial.hyperparameters

    logging.info(f"Best Mean CV AUC: {best_cv_score:.4f}")
    logging.info(f"Best hyperparameters: {best_hps.values}")

    # Rebuild and retrain using the best HPs.
    # CV fold models are discarded, so we rebuild from scratch.
    # A 90/10 stratified split of the training pool provides an internal
    # validation set for early stopping, keeping the external test set untouched.
    logging.info("Rebuilding best model (90/10 stratified split for early stopping)...")
    best_model = build_model(best_hps)

    es_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train_idx, val_idx = next(es_split.split(X_train, y_train))
    X_fit, X_es = X_train[train_idx], X_train[val_idx]
    y_fit, y_es = y_train[train_idx], y_train[val_idx]

    final_es = EarlyStopping(
        monitor='val_auc_roc',
        patience=50,
        restore_best_weights=True,
        mode='max'
    )

    best_model.fit(
        X_fit,
        y_fit,
        epochs=100,
        batch_size=32,
        callbacks=[final_es],
        validation_data=(X_es, y_es),
        class_weight=class_weights,
        verbose=1
    )

    best_model.save(model_save_path)
    logging.info(f"Best MLP model saved to {model_save_path}")

    # Final evaluation and reporting
    y_pred_train_prob = best_model.predict(X_train).flatten()
    y_pred_val_prob = best_model.predict(X_val).flatten()

    auc_train = roc_auc_score(y_train, y_pred_train_prob)
    auc_val = roc_auc_score(y_val, y_pred_val_prob)

    threshold_metrics = calculate_optimal_threshold_metrics(y_val, y_pred_val_prob, target_sensitivity=sensitivity_threshold)
    logging.info(f"Optimal Threshold: {threshold_metrics['optimal_threshold']:.4f}")
    logging.info(f"Sensitivity: {threshold_metrics['sensitivity']:.4f}, Specificity: {threshold_metrics['specificity']:.4f}")

    # Bootstrap confidence interval for validation AUC
    alpha = 0.05
    try:
        val_lower_ci, val_upper_ci = auc_confidence_interval(
            y_val,
            y_pred_val_prob,
            num_bootstraps=1000,
            alpha=alpha
        )
        logging.info(f"Best Model - Validation AUC CI ({(1 - alpha) * 100}%): ({val_lower_ci:.4f}, {val_upper_ci:.4f})")
    except Exception as e:
        logging.warning(f"Could not calculate CI: {e}. Setting CI to 'None'.")
        val_lower_ci, val_upper_ci = None, None

    logging.info(f"Best Model - Training AUC: {auc_train:.4f}")
    logging.info(f"Best Model - Validation AUC: {auc_val:.4f}")

    # Save results to JSON
    results_data = {
        'model_name': 'mlp',
        'training_auc': auc_train,
        'validation_auc': auc_val,
        'validation_ci_alpha': alpha,
        'validation_ci_lower': val_lower_ci,
        'validation_ci_upper': val_upper_ci,
        'best_hyperparameters': best_hps.values
    }

    results_data.update(threshold_metrics)
    results_json_path = model_save_path.with_name("mlp_validation_stats.json")

    try:
        with open(results_json_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        logging.info(f"Saved validation statistics to {results_json_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON results: {e}")

    return best_model

def train_mlp_manual(df_train: pd.DataFrame,
                     df_validate: pd.DataFrame,
                     model_save_path: Path,
                     log_dir: Path,
                     sensitivity_threshold: float,
                     # --- Hyperparameters ---
                     learning_rate: float,
                     alpha: float,
                     dropout: float,
                     first_decay_steps: int,
                     weight_decay: float,
                     noise_stddev: float,
                     hidden_layer_sizes: list):
    """
    Trains and saves an MLP model using manually specified hyperparameters.
    Includes TensorBoard logging for visualization.
    """
    logging.info("Starting MLP training with manual hyperparameters...")

    # Data preparation
    X_train, y_train, _ = input_output(df_train)
    X_val, y_val, _ = input_output(df_validate)
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)

    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    class_weights = class_weight_calculator(n_neg, n_pos)
    logging.info(f"Calculated class weights: {class_weights}")

    input_shape = X_train.shape[1]

    # Build model with fixed hyperparameters
    model = build_model_fixed(
        input_shape=input_shape,
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate=learning_rate,
        alpha=alpha,
        dropout=dropout,
        first_decay_steps=first_decay_steps,
        weight_decay=weight_decay,
        noise_stddev=noise_stddev
    )

    # Callbacks: TensorBoard, EarlyStopping, and SaveBestModel
    tensorboard_log_path = log_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.info(f"TensorBoard logs path set to: {tensorboard_log_path}")

    tensorboard_cb = TensorBoard(
        log_dir=tensorboard_log_path,
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )

    # EarlyStopping only stops training; SaveBestModel manages weight restoration
    early_stopping_cb = EarlyStopping(
        monitor='val_auc_roc',
        patience=50,
        restore_best_weights=False,
        mode='max'
    )

    best_weights_cb = SaveBestModel(save_best_metric='val_auc_roc', this_max=True)

    callbacks = [tensorboard_cb, early_stopping_cb, best_weights_cb]

    # Train
    logging.info("Starting training for 500 epochs...")
    model.fit(
        X_train,
        y_train,
        epochs=500,
        batch_size=32,
        callbacks=callbacks,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        verbose=1
    )

    # Restore best weights tracked by SaveBestModel callback
    logging.info("Restoring best weights based on val_auc_roc...")
    model.set_weights(best_weights_cb.best_weights)
    best_model_keras = model

    best_model_keras.save(model_save_path)
    logging.info(f"Best MLP model saved to {model_save_path}")

    # Final evaluation and reporting
    y_pred_train_prob = best_model_keras.predict(X_train).flatten()
    y_pred_val_prob = best_model_keras.predict(X_val).flatten()

    auc_train = roc_auc_score(y_train, y_pred_train_prob)
    auc_val = roc_auc_score(y_val, y_pred_val_prob)

    threshold_metrics = calculate_optimal_threshold_metrics(y_val, y_pred_val_prob,
                                                            target_sensitivity=sensitivity_threshold)

    logging.info(f"Best Model - Training AUC: {auc_train:.4f}")
    logging.info(f"Best Model - Validation AUC: {auc_val:.4f}")

    # Save results to JSON
    best_hps_manual = {
        'learning_rate': learning_rate,
        'alpha': alpha,
        'dropout': dropout,
        'first_decay_steps': first_decay_steps,
        'weight_decay': weight_decay,
        'hidden_layers': "_".join(map(str, hidden_layer_sizes))
    }

    results_data = {
        'model_name': 'mlp_manual',
        'training_auc': auc_train,
        'validation_auc': auc_val,
        'best_hyperparameters': best_hps_manual
    }
    results_data.update(threshold_metrics)

    results_json_path = model_save_path.with_name("mlp_manual_validation_stats.json")

    try:
        with open(results_json_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        logging.info(f"Saved manual validation statistics to {results_json_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON results: {e}")

    return best_model_keras


def train_rf(df_train: pd.DataFrame,
             df_validate: pd.DataFrame,
             model_save_path: Path,
             results_path: Path,
             sensitivity_threshold: float):
    """
    Trains, evaluates, and saves a Random Forest model using GridSearchCV
    with 5-fold stratified cross-validation.
    """
    logging.info("Starting Random Forest training with GridSearchCV...")
    X_train, y_train, _ = input_output(df_train)
    X_val, y_val, val_dir = input_output(df_validate)

    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    class_weight = class_weight_calculator(n_neg, n_pos)
    logging.info(f"Calculated class weights: {class_weight}")
    param_grid = {
        'n_estimators': [50, 75, 100, 125, 150],
        'max_depth': [3, 5, 7, 9, 11],
        'min_samples_split': [10, 15, 20],
        'min_samples_leaf': [5, 7, 10, 15],
        'max_features':  [0.3, 0.5, 'sqrt'],
        'class_weight': ['balanced_subsample', {0: 1, 1: 3}, class_weight]
    }

    rf_model = RandomForestClassifier(random_state=42)
    scorer = make_scorer(roc_auc_score)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=cv_strategy, n_jobs=-1, verbose=2, scoring=scorer)
    grid_search.fit(X_train, y_train)

    best_rf_model = grid_search.best_estimator_
    joblib.dump(best_rf_model, model_save_path)
    logging.info(f"Best RF model saved to {model_save_path}")
    logging.info(f"Best hyperparameters: {grid_search.best_params_}")

    # Final evaluation and reporting
    logging.info("Calculating final metrics for RF...")

    # Out-of-fold predictions for threshold calibration (avoids overfitting to training data)
    oof_predictions = cross_val_predict(estimator=best_rf_model, X=X_train, y=y_train, cv=cv_strategy, method='predict_proba')
    y_pred_prob_val = best_rf_model.predict_proba(X_val)[:, 1]
    y_pred_prob_train = best_rf_model.predict_proba(X_train)[:, 1]

    auc_train = roc_auc_score(y_train, y_pred_prob_train)
    auc_val = roc_auc_score(y_val, y_pred_prob_val)

    # Threshold calibrated on OOF predictions (not on validation set)
    threshold_metrics = calculate_optimal_threshold_metrics(y_train, oof_predictions[:,1], target_sensitivity=sensitivity_threshold)
    logging.info(f"Optimal Threshold: {threshold_metrics['optimal_threshold']:.4f}")
    logging.info(f"Sensitivity: {threshold_metrics['sensitivity']:.4f}, Specificity: {threshold_metrics['specificity']:.4f}")

    # Bootstrap confidence interval
    alpha = 0.05
    try:
        val_lower_ci, val_upper_ci = auc_confidence_interval(
            y_val,
            y_pred_prob_val,
            num_bootstraps=1000,
            alpha=alpha
        )
        logging.info(f"RF Model - Validation AUC CI ({(1 - alpha) * 100}%): ({val_lower_ci:.4f}, {val_upper_ci:.4f})")
    except Exception as e:
        logging.warning(f"Could not calculate CI for RF: {e}. Setting CI to 'None'.")
        val_lower_ci, val_upper_ci = None, None

    logging.info(f"RF Model - Training AUC: {auc_train:.4f}")
    logging.info(f"RF Model - Validation AUC: {auc_val:.4f}")

    # Save results to JSON
    results_data = {
        'model_name': 'rf',
        'training_auc': auc_train,
        'validation_auc': auc_val,
        'validation_ci_alpha': alpha,
        'validation_ci_lower': val_lower_ci,
        'validation_ci_upper': val_upper_ci,
        'best_hyperparameters': grid_search.best_params_
    }
    results_data.update(threshold_metrics)
    results_json_path = model_save_path.with_name("rf_validation_stats.json")

    try:
        with open(results_json_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        logging.info(f"Saved RF validation statistics to {results_json_path}")
    except Exception as e:
        logging.error(f"Failed to save RF JSON results: {e}")

    # Save validation predictions
    val_results_df = pd.DataFrame({'series_dir': val_dir, 'prediction': y_pred_prob_val, 'label': y_val})
    val_results_df.to_csv(results_path, index=False)
    logging.info(f"RF validation predictions saved to {results_path}")

    return best_rf_model


def train_ensemble(df_train: pd.DataFrame,
                   df_validate: pd.DataFrame,
                   mlp_model_path: Path,
                   model_save_path: Path,
                   results_path: Path,
                   sensitivity_threshold: float):
    """
    Trains an ensemble model: extracts intermediate features from a pre-trained
    MLP (penultimate layer) and trains a Random Forest on those features.
    """
    logging.info("Starting Ensemble (MLP+RF) training...")
    if not mlp_model_path.exists():
        logging.error(f"MLP model not found at {mlp_model_path}. Cannot train ensemble.")
        return

    # Load base MLP and create feature extractor from the penultimate layer
    logging.info(f"Loading base MLP model from {mlp_model_path}")
    model_mlp = tf.keras.models.load_model(mlp_model_path)
    try:
        feature_layer_index = -2
        model_mlp.get_layer(index=feature_layer_index)
    except ValueError:
        logging.warning("Could not get layer at index -2, falling back to name-based lookup.")
        try:
            dense_layers = [l for l in model_mlp.layers if isinstance(l, tf.keras.layers.Dense)]
            feature_layer_name = dense_layers[-2].name
            feature_layer_index = model_mlp.get_layer(name=feature_layer_name).name
        except Exception as e:
            logging.error(f"Could not find feature layer: {e}")
            return

    logging.info(f"Using MLP layer '{model_mlp.get_layer(index=feature_layer_index).name}' for features.")
    intermediate_layer_model = tf.keras.Model(
        inputs=model_mlp.input,
        outputs=model_mlp.get_layer(index=feature_layer_index).output
    )

    # Extract intermediate features
    X_train_emb, y_train, _ = input_output(df_train)
    X_val_emb, y_val, val_dir = input_output(df_validate)

    train_ds = tf.data.Dataset.from_tensor_slices(X_train_emb).batch(128).cache()
    val_ds = tf.data.Dataset.from_tensor_slices(X_val_emb).batch(128).cache()

    logging.info("Extracting features from MLP...")
    train_features = intermediate_layer_model.predict(train_ds)
    val_features = intermediate_layer_model.predict(val_ds)

    # Train RF on extracted features
    logging.info("Training RF on extracted features...")

    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    class_weight = class_weight_calculator(n_neg, n_pos)
    logging.info(f"Calculated class weights: {class_weight}")

    param_grid = {
        'n_estimators': [50, 75, 100, 125, 150],
        'max_depth': [3, 5, 7, 9, 11],
        'min_samples_split': [10, 15, 20],
        'min_samples_leaf': [5, 7, 10, 15],
        'max_features': [0.3, 0.5, 'sqrt'],
        'class_weight': ['balanced_subsample', {0: 1, 1: 3}, class_weight]
    }

    rf_model = RandomForestClassifier(random_state=42)
    scorer = make_scorer(roc_auc_score)

    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring=scorer)
    grid_search.fit(train_features, y_train)

    best_rf_model = grid_search.best_estimator_
    joblib.dump(best_rf_model, model_save_path)
    logging.info(f"Best Ensemble-RF model saved to {model_save_path}")
    logging.info(f"Best hyperparameters: {grid_search.best_params_}")

    # Final evaluation and reporting
    logging.info("Calculating final metrics for Ensemble...")

    y_pred_prob_val = best_rf_model.predict_proba(val_features)[:, 1]
    y_pred_prob_train = best_rf_model.predict_proba(train_features)[:, 1]

    auc_train = roc_auc_score(y_train, y_pred_prob_train)
    auc_val = roc_auc_score(y_val, y_pred_prob_val)

    threshold_metrics = calculate_optimal_threshold_metrics(y_val, y_pred_prob_val, target_sensitivity=sensitivity_threshold)
    logging.info(f"Optimal Threshold: {threshold_metrics['optimal_threshold']:.4f}")
    logging.info(f"Sensitivity: {threshold_metrics['sensitivity']:.4f}, Specificity: {threshold_metrics['specificity']:.4f}")

    # Bootstrap confidence interval
    alpha = 0.05
    try:
        val_lower_ci, val_upper_ci = auc_confidence_interval(
            y_val,
            y_pred_prob_val,
            num_bootstraps=1000,
            alpha=alpha
        )
        logging.info(f"RF Model - Validation AUC CI ({(1 - alpha) * 100}%): ({val_lower_ci:.4f}, {val_upper_ci:.4f})")
    except Exception as e:
        logging.warning(f"Could not calculate CI for RF: {e}. Setting CI to 'None'.")
        val_lower_ci, val_upper_ci = None, None

    logging.info(f"Ensemble - Training AUC: {auc_train:.4f}")
    logging.info(f"Ensemble - Validation AUC: {auc_val:.4f}")

    # Save results to JSON
    results_data = {
        'model_name': 'ensemble',
        'training_auc': auc_train,
        'validation_auc': auc_val,
        'validation_ci_alpha': alpha,
        'validation_ci_lower': val_lower_ci,
        'validation_ci_upper': val_upper_ci,
        'best_hyperparameters': grid_search.best_params_
    }
    results_data.update(threshold_metrics)
    results_json_path = model_save_path.with_name("rf_validation_stats.json")

    try:
        with open(results_json_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        logging.info(f"Saved RF validation statistics to {results_json_path}")
    except Exception as e:
        logging.error(f"Failed to save RF JSON results: {e}")

    # Save validation predictions
    val_results_df = pd.DataFrame({'series_dir': val_dir, 'prediction': y_pred_prob_val, 'label': y_val})
    val_results_df.to_csv(results_path, index=False)
    logging.info(f"Ensemble validation predictions saved to {results_path}")

    return best_rf_model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model_on_test_set(model_path: Path, model_type: str, test_data_path: Path):
    """
    Loads a trained model and evaluates it on a final, hold-out test set.

    It reads the optimal_threshold from the model's validation JSON file
    and uses it to calculate sensitivity/specificity on the test set.
    """
    logging.info(f"--- Starting Test Set Evaluation for {model_type} ---")
    logging.info(f"Model: {model_path}")
    logging.info(f"Test Data: {test_data_path}")

    # Locate the validation stats JSON (contains the optimal threshold)
    run_dir = model_path.parent

    if model_type == 'mlp':
        val_stats_json_path = run_dir / "mlp_validation_stats.json"
    elif model_type == 'mlp_manual':
        val_stats_json_path = run_dir / "mlp_manual_validation_stats.json"
    elif model_type == 'rf':
        val_stats_json_path = run_dir / "rf_validation_stats.json"
    elif model_type == 'ensemble':
        run_dir = model_path
        # Ensemble saves its stats under 'rf_validation_stats.json'
        val_stats_json_path = model_path / "rf_validation_stats.json"
    else:
        logging.error(f"Unknown model type: {model_type}")
        return

    # Load test data and threshold
    try:
        df_test = pd.read_csv(test_data_path)
        X_test, y_test, test_dir = input_output(df_test)

        with open(val_stats_json_path, 'r') as f:
            val_stats = json.load(f)
        threshold = val_stats['optimal_threshold']

        logging.info(f"Loaded {len(df_test)} test samples.")
        logging.info(f"Using optimal threshold from validation: {threshold:.4f}")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}. Cannot proceed with evaluation.")
        return
    except Exception as e:
        logging.error(f"Error loading data or threshold: {e}")
        return

    # Load model and generate predictions
    try:
        if model_type == 'mlp':
            model = tf.keras.models.load_model(model_path)
            y_pred_prob = model.predict(X_test).flatten()

        elif model_type == 'mlp_manual':
            model = tf.keras.models.load_model(model_path)
            y_pred_prob = model.predict(X_test).flatten()

        elif model_type == 'rf':
            model = joblib.load(model_path)
            y_pred_prob = model.predict_proba(X_test)[:, 1]

        elif model_type == 'ensemble':
            logging.info("Loading ensemble (RF + MLP base)...")
            model_rf = joblib.load(model_path / 'best_ensemble_model.pkl')

            base_mlp_path = model_path / 'base_mlp_for_ensemble.keras'
            model_mlp = tf.keras.models.load_model(base_mlp_path)

            # Build feature extractor from the penultimate MLP layer
            feature_layer_index = -2
            try:
                model_mlp.get_layer(index=feature_layer_index)
            except ValueError:
                dense_layers = [l for l in model_mlp.layers if isinstance(l, tf.keras.layers.Dense)]
                feature_layer_name = dense_layers[-2].name
                feature_layer_index = model_mlp.get_layer(name=feature_layer_name).name

            intermediate_layer_model = tf.keras.Model(
                inputs=model_mlp.input,
                outputs=model_mlp.get_layer(index=feature_layer_index).output
            )

            logging.info("Extracting test features from base MLP...")
            test_ds = tf.data.Dataset.from_tensor_slices(X_test).batch(128).cache()
            X_test_features = intermediate_layer_model.predict(test_ds)
            y_pred_prob = model_rf.predict_proba(X_test_features)[:, 1]

    except Exception as e:
        logging.error(f"Error loading model or making predictions: {e}", exc_info=True)
        return

    # Calculate test metrics
    logging.info("Calculating test set metrics...")
    auc_test = roc_auc_score(y_test, y_pred_prob)
    alpha = 0.05
    try:
        test_lower_ci, test_upper_ci = auc_confidence_interval(
            y_test, y_pred_prob, num_bootstraps=1000, alpha=alpha
        )
    except Exception as e:
        logging.warning(f"Could not calculate CI for test set: {e}")
        test_lower_ci, test_upper_ci = None, None

    threshold_metrics = calculate_metrics_at_threshold(y_test, y_pred_prob, threshold)

    logging.info(f"Test Set AUC: {auc_test:.4f}")
    logging.info(f"Test Set AUC CI (95%): ({test_lower_ci:.4f}, {test_upper_ci:.4f})")
    logging.info(f"Test Set Sensitivity (at {threshold:.4f}): {threshold_metrics['sensitivity']:.4f}")
    logging.info(f"Test Set Specificity (at {threshold:.4f}): {threshold_metrics['specificity']:.4f}")

    # Save results
    test_stats_json_path = run_dir / f"{model_type}_test_set_stats.json"
    test_preds_csv_path = run_dir / f"{model_type}_test_set_predictions.csv"

    test_results_data = {
        'model_name': model_type,
        'model_path': str(model_path),
        'test_data_path': str(test_data_path),
        'test_auc': auc_test,
        'test_ci_alpha': alpha,
        'test_ci_lower': test_lower_ci,
        'test_ci_upper': test_upper_ci
    }
    test_results_data.update(threshold_metrics)

    try:
        with open(test_stats_json_path, 'w') as f:
            json.dump(test_results_data, f, indent=4)
        logging.info(f"Saved test statistics to {test_stats_json_path}")
    except Exception as e:
        logging.error(f"Failed to save test JSON results: {e}")

    try:
        val_results_df = pd.DataFrame({'series_dir': test_dir, 'prediction': y_pred_prob, 'label': y_test})
        val_results_df.to_csv(test_preds_csv_path, index=False)
        logging.info(f"Test predictions saved to {test_preds_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save test predictions CSV: {e}")

    logging.info(f"--- Finished Test Set Evaluation ---")


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(dataset_name: str, model_type: str, threshold_limit: float, manual_hps: dict = None):
    """
    Runs a single, complete experiment for a given dataset and model type.
    """
    logging.info(f"--- Starting Experiment: DATASET={dataset_name}, MODEL={model_type} ---")

    # Load experiment configuration
    try:
        dataset_config = EXPERIMENT_CONFIG['datasets'][dataset_name]
        val_set_name = dataset_config['validation_set']
        val_path = EXPERIMENT_CONFIG['validation_sets'][val_set_name]
        train_path = dataset_config['train_path']
    except KeyError:
        logging.error(f"Configuration error: '{dataset_name}' or its validation set not found in EXPERIMENT_CONFIG.")
        return

    # Create timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}_{dataset_name}_{model_type}"
    run_dir = RESULTS_BASE_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Output file paths
    mlp_model_path = run_dir / 'best_mlp_model.keras'
    mlp_manual_model_path = run_dir / 'best_mlp_manual_model.keras'
    rf_model_path = run_dir / 'best_rf_model.pkl'
    ensemble_model_path = run_dir / 'best_ensemble_model.pkl'
    rf_results_path = run_dir / 'rf_validation_results.csv'
    ensemble_results_path = run_dir / 'ensemble_validation_results.csv'
    log_dir = run_dir / 'logs'
    log_dir.mkdir()

    logging.info(f"Train path: {train_path}")
    logging.info(f"Validate path: {val_path}")
    logging.info(f"Results will be saved to: {run_dir}")

    # Load data
    try:
        df_train = pd.read_csv(train_path)
        df_validate = pd.read_csv(val_path)
        logging.info(f"Loaded training data: {len(df_train)} rows")
        logging.info(f"Loaded validation data: {len(df_validate)} rows")
    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e.filename}")
        return

    # Run model training
    try:
        if model_type == 'mlp':
            train_mlp_with_cv(
                df_train, df_validate,
                model_save_path=mlp_model_path,
                log_dir=log_dir,
                sensitivity_threshold=threshold_limit
            )
        elif model_type == 'mlp_manual':
            if manual_hps is None:
                logging.error("Model type 'mlp_manual' requires manual_hps dictionary.")
                return

            train_mlp_manual(
                df_train, df_validate,
                model_save_path=mlp_manual_model_path,
                log_dir=(log_dir / 'mlp_manual'),
                sensitivity_threshold=threshold_limit,
                **manual_hps
            )

        elif model_type == 'rf':
            train_rf(
                df_train, df_validate,
                model_save_path=rf_model_path,
                results_path=rf_results_path,
                sensitivity_threshold=threshold_limit
            )

        elif model_type == 'ensemble':
            # Ensemble requires a base MLP — train it first, then train RF on its features
            logging.info("Training base MLP for ensemble...")
            base_mlp_path_for_ensemble = run_dir / 'base_mlp_for_ensemble.keras'

            train_mlp_with_cv(
                df_train, df_validate,
                model_save_path=base_mlp_path_for_ensemble,
                log_dir=(log_dir / 'ensemble_base_mlp'),
                sensitivity_threshold=threshold_limit
            )

            if base_mlp_path_for_ensemble.exists():
                train_ensemble(
                    df_train, df_validate,
                    mlp_model_path=base_mlp_path_for_ensemble,
                    model_save_path=ensemble_model_path,
                    results_path=ensemble_results_path,
                    sensitivity_threshold=threshold_limit
                )
            else:
                logging.error("Base MLP tuning/training failed, cannot proceed with ensemble.")

        else:
            logging.warning(f"Model type '{model_type}' is not recognized.")

    except Exception as e:
        logging.error(f"An error occurred during training for {dataset_name}, {model_type}: {e}", exc_info=True)

    logging.info(f"--- Finished Experiment: {dataset_name}, {model_type} ---")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ML Model Training and Validation Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # Command: 'single'
    parser_single = subparsers.add_parser(
        'single',
        help="Run a single, specific experiment for testing."
    )
    parser_single.add_argument(
        '-d', '--dataset',
        type=str,
        required=True,
        choices=EXPERIMENT_CONFIG['datasets'].keys(),
        help="The name of the dataset to use (defined in EXPERIMENT_CONFIG)"
    )
    parser_single.add_argument(
        '-m', '--model',
        type=str,
        required=True,
        choices=EXPERIMENT_CONFIG['model_types'],
        help="The type of model to train"
    )

    parser_single.add_argument(
        '-s', '--sensitivity',
        type=float,
        required=True,
        help="The lowest sensitivity for operating point"
    )
    parser_manual = subparsers.add_parser(
        'manual',
        help="Run a single MLP experiment with manually set hyperparameters and TensorBoard logging."
    )
    parser_manual.add_argument(
        '-d', '--dataset',
        type=str,
        required=True,
        choices=EXPERIMENT_CONFIG['datasets'].keys(),
        help="The name of the dataset to use (defined in EXPERIMENT_CONFIG)"
    )
    parser_manual.add_argument(
        '--lr',
        type=float,
        required=True,
        help="Initial learning rate (e.g., 0.001)"
    )
    parser_manual.add_argument(
        '--alpha',
        type=float,
        required=True,
        help="CosineDecayRestarts alpha (e.g., 0.5)"
    )
    parser_manual.add_argument(
        '--dropout',
        type=float,
        required=True,
        help="Dropout rate (e.g., 0.3)"
    )
    parser_manual.add_argument(
        '--steps',
        type=int,
        required=True,
        help="First decay steps for LR scheduler (e.g., 100)"
    )
    parser_manual.add_argument(
        '--wd',
        type=float,
        required=True,
        help="Weight decay (L1 regularizer) (e.g., 1e-6)"
    )
    parser_manual.add_argument(
        '--layers',
        type=str,
        required=True,
        help="Comma-separated list of hidden layer sizes (e.g., '64,64')"
    )
    parser_manual.add_argument(
        '-s', '--sensitivity',
        type=float,
        required=True,
        help="The lowest sensitivity for operating point"
    )
    parser_manual.add_argument(
        '--noise',
        type=float,
        required=True,
        help="Standard deviation for GaussianNoise layer (e.g., 0.005)"
    )

    parser_evaluate = subparsers.add_parser(
        'evaluate',
        help="Evaluate a trained model on a hold-out test set."
    )
    parser_evaluate.add_argument(
        '--model_path',
        type=Path,
        required=True,
        help="Path to the saved model file (.keras or .pkl)"
    )
    parser_evaluate.add_argument(
        '--model_type',
        type=str,
        required=True,
        choices=EXPERIMENT_CONFIG['model_types'],
        help="The type of model this path points to (mlp, rf, ensemble)"
    )
    parser_evaluate.add_argument(
        '--test_data_path',
        type=Path,
        required=True,
        help="Path to the final test set CSV file (e.g., dataset_validation.csv)"
    )

    args = parser.parse_args()

    if args.command == 'single':
        run_single_experiment(
            dataset_name=args.dataset,
            model_type=args.model,
            threshold_limit=args.sensitivity
        )
    elif args.command == 'manual':
        try:
            hidden_layer_sizes = [int(s.strip()) for s in args.layers.split(',')]
        except ValueError:
            logging.error("Invalid format for --layers. Use comma-separated integers (e.g., '64,64').")
            return

        run_single_experiment(
            dataset_name=args.dataset,
            model_type='mlp_manual',
            threshold_limit=args.sensitivity,
            manual_hps={
                'learning_rate': args.lr,
                'alpha': args.alpha,
                'dropout': args.dropout,
                'first_decay_steps': args.steps,
                'weight_decay': args.wd,
                'noise_stddev': args.noise,
                'hidden_layer_sizes': hidden_layer_sizes
            }
        )
    elif args.command == 'evaluate':
        evaluate_model_on_test_set(
            model_path=args.model_path,
            model_type=args.model_type,
            test_data_path=args.test_data_path
        )


if __name__ == "__main__":
    main()