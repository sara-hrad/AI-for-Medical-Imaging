"""
Professional Training & Validation Script

This script provides a modular and configurable pipeline for training
and evaluating multiple classifier models (MLP, Random Forest, Ensemble)
across various datasets.

It can be run in two modes:
1. 'all': Runs all 18 experiments (6 datasets * 3 model types).
   Usage: python this_script.py all
2. 'single': Runs a single, specified experiment for testing/debugging.
   Usage: python this_script.py single --dataset "data_no_mask_1" --model "mlp"
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

from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score, roc_curve, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasClassifier
import keras_tuner as kt
from utils import *
from savebestmodel import SaveBestModel


# ---
# 1. GLOBAL CONFIGURATION & SETUP
# ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

#


# ---
# !! UPDATE THIS SECTION !!
# Define all datasets and their corresponding validation sets.
#
# 'data_no_mask_1' to 'data_no_mask_4' use 'valid_no_mask'.
# 'data_mask_1' and 'data_mask_2' use 'valid_mask'.
# ---
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
            # Split the TRAINING POOL into internal Train/Val folds
            x_train_fold, x_val_fold = x[train_idx], x[test_idx]
            y_train_fold, y_val_fold = y[train_idx], y[test_idx]

            # Build the model with the current trial's hyperparameters
            model = self.hypermodel.build(trial.hyperparameters)

            # Train on the fold
            # We use a shorter patience here for speed during tuning
            fold_callbacks = [EarlyStopping(monitor='val_auc_roc', patience=50, mode='max')]

            model.fit(x_train_fold,
                      y_train_fold,
                      validation_data=(x_val_fold, y_val_fold),
                      epochs=epochs,
                      batch_size=batch_size,
                      callbacks=fold_callbacks,
                      class_weight=class_weight,
                      verbose=0)  # Silence individual fold logs

            # Evaluate on the fold's validation set
            # Note: The metric name 'auc_roc' comes from your build_model compilation
            eval_metrics = model.evaluate(x_val_fold, y_val_fold, verbose=0, return_dict=True)
            val_aucs.append(eval_metrics['auc_roc'])

        # Calculate Mean AUC across 5 folds
        mean_val_auc = np.mean(val_aucs)

        # Tell the Oracle this trial's score (Mean CV AUC)
        # We map it to "val_auc_roc" so the objective matches your existing setup
        self.oracle.update_trial(trial.trial_id, {'val_auc_roc': mean_val_auc})

        # Note: Keras Tuner usually saves a model here.
        # Since we trained 5 models, we don't save them all.
        # We rely on the "Rebuild" step in the main function.


# ---
# 2. MODEL DEFINITIONS & HELPERS
# ---

def build_model(hp):
    """
    Keras Tuner "HyperModel" builder function.
    """

    # --- 1. Define Hyperparameters to Tune ---

    # # We use hp.Choice to test a specific list of values
    # hp_learning_rate = hp.Choice('learning_rate', values=[0.003, 0.002, 0.001, 0.0005])
    # hp_alpha = hp.Choice('alpha', values=[0.01, 0.1, 0.5, 0.8])
    # hp_dropout = hp.Choice('dropout', values=[0.1, 0.2, 0.3, 0.5])
    # hp_first_decay_steps = hp.Choice('first_decay_steps', values=[50, 100])
    # # hp_weight_decay = hp.Choice('weight_decay', values=[1e-6, 1e-7])
    # hp_weight_decay = hp.Choice('weight_decay', values=[1e-7])
    # hp_noise_stddev = hp.Choice('noise_stddev', values=[0.0, 0.1, 0.01])  # 0.0 means no noise

    hp_learning_rate = hp.Choice('learning_rate', values=[0.002, 0.001, 0.0005])
    hp_alpha = hp.Choice('alpha', values=[0.1, 0.5, 0.8])
    hp_dropout = hp.Choice('dropout', values=[0.2, 0.5])
    hp_first_decay_steps = hp.Choice('first_decay_steps', values=[50, 100])
    hp_weight_decay = hp.Choice('weight_decay', values=[1e-7])
    hp_noise_stddev = hp.Choice('noise_stddev', values=[0.0, 0.1, 0.01])  # 0.0 means no noise


    # A simple way to test different architectures, similar to GridSearchCV
    # We pass a string and then parse it.
    # hp_layer_config = hp.Choice('hidden_layers', values=['32_32', '64_64', '32_16_1'])
    hp_layer_config = hp.Choice('hidden_layers', values=['32_32',  '64_64'])

    hidden_layer_sizes = [int(size) for size in hp_layer_config.split('_')]

    # --- 2. Define Fixed Parameters (from your original model) ---
    token_num = 1
    embeddings_size = 1408
    seed = 42  # Fixed seed for reproducibility across trials

    # --- 3. Model Architecture (using hp values) ---
    inputs = tf.keras.Input(shape=(token_num * embeddings_size,))
    noise = tf.keras.layers.GaussianNoise(hp_noise_stddev)(inputs)
    inputs_reshape = tf.keras.layers.Reshape((token_num, embeddings_size))(noise)
    inputs_pooled = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')(inputs_reshape)
    hidden = inputs_pooled

    # Loop over the layer sizes from the tuned hyperparameter
    for size in hidden_layer_sizes:
        hidden = tf.keras.layers.Dense(
            units=size,  # Use the tuned size
            activation='relu',
            kernel_initializer=tf.keras.initializers.HeUniform(seed=seed),
            kernel_regularizer=tf.keras.regularizers.l1(l1=hp_weight_decay),  # Use tuned WD
            bias_regularizer=tf.keras.regularizers.l1(l1=hp_weight_decay))(  # Use tuned WD
            hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.LayerNormalization()(hidden)
        hidden = tf.keras.layers.Dropout(hp_dropout, seed=seed)(hidden)  # Use tuned dropout

    output = tf.keras.layers.Dense(
        units=1,
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.HeUniform(seed=seed))(
        hidden)

    model = tf.keras.Model(inputs, output)

    # --- 4. Model Compilation (using hp values) ---
    learning_rate_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=hp_learning_rate,  # Use tuned LR
        first_decay_steps=hp_first_decay_steps,  # Use tuned steps
        alpha=hp_alpha)  # Use tuned alpha

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate_fn),
        loss='binary_crossentropy',
        weighted_metrics=[
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.TruePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.AUC(),
            # This is the key: the tuner will look for this metric's name
            tf.keras.metrics.AUC(curve='ROC', name='auc_roc')
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
    This replaces the Keras Tuner build_model(hp) function for manual runs.
    """
    token_num = 1
    embeddings_size = 1408
    # embeddings_size = input_shape[0] // token_num # Reverse-calculate embeddings_size

    # 1. Define Architecture
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

    # 2. Model Compilation
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

# ---
# 3. TRAINING FUNCTIONS
# ---

# Train MLP with hyperparameter tuning

def train_mlp_no_cv(df_train: pd.DataFrame,
                         df_validate: pd.DataFrame,
                         model_save_path: Path,
                         log_dir: Path,
                         sensitivity_threshold: float) -> tf.keras.Model:
    """
    Trains, tunes, and saves an MLP model using Keras Tuner.
    This function replaces the GridSearchCV version.
    """
    logging.info("Starting MLP training with Hyperparameter Tuning (Keras Tuner)...")

    # --- 1. Data Prep ---
    # This step is identical to your original function.
    logging.info("Using 'class_weight' strategy for Keras Tuner.")

    X_train, y_train, _ = input_output(df_train)
    X_val, y_val, _ = input_output(df_validate)
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)

    # Calculate class weight
    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    class_weights = class_weight_calculator(n_neg, n_pos)
    logging.info(f"Calculated class weights: {class_weights}")

    # --- 2. Define Callbacks ---
    # This is also identical. The monitor 'val_auc_roc' must match
    # a metric name defined in build_model's compile() step.
    early_stopping_cb = EarlyStopping(
        monitor='val_auc_roc',  # Monitor validation AUC
        patience=50,
        restore_best_weights=True,
        mode='max'
    )

    # --- 3. Instantiate Keras Tuner ---
    # This replaces the KerasClassifier and GridSearchCV setup.
    # The hyperparameter grid is now defined inside build_model.

    tuner_directory = log_dir / 'keras_tuner'
    project_name = 'mlp_tuning'

    logging.info(f"Keras Tuner logs will be saved to: {tuner_directory / project_name}")

    tuner = kt.RandomSearch(
        build_model,  # Pass the HyperModel build function

        # The objective MUST match a metric name from compile()
        # Keras Tuner automatically adds 'val_' for validation data
        objective=kt.Objective("val_auc_roc", direction="max"),

        max_trials=500,  # Total number of hyperparameter combinations to test
        executions_per_trial=1,  # How many times to train each trial
        directory=str(tuner_directory),
        project_name=project_name,
        overwrite=True  # Set to True to start a fresh search
    )

    # --- 4. Run the Search ---
    # All fit parameters are passed directly to tuner.search()
    logging.info(f"Starting Keras Tuner search... (Max trials: {tuner.oracle.max_trials})")
    tuner.search(
        X_train,
        y_train,
        epochs=500,  # Max epochs *per trial*
        batch_size=32,
        callbacks=[early_stopping_cb],
        validation_data=(X_val, y_val),  # Use the explicit validation set
        class_weight=class_weights
    )

    # --- 5. Save & Report ---
    logging.info("Keras Tuner search finished.")

    # Get the best trial results
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    best_score = best_trial.score
    best_hps = best_trial.hyperparameters

    logging.info(f"Best model validation AUC: {best_score:.4f}")
    logging.info(f"Best hyperparameters found: {best_hps.values}")

    # Get the best Keras model
    # The tuner automatically provides the model with best weights
    # thanks to restore_best_weights=True
    best_model_keras = tuner.get_best_models(num_models=1)[0]

    # Save the best Keras model
    best_model_keras.save(model_save_path)
    logging.info(f"Best MLP model saved to {model_save_path}")

    # Final evaluation, CI calculation, and JSON saving.
    logging.info("Calculating final metrics, optimal threshold, and confidence intervals...")

    # 1. Get model predictions (probabilities)
    y_pred_train_prob = best_model_keras.predict(X_train).flatten()
    y_pred_val_prob = best_model_keras.predict(X_val).flatten()

    # 2. Calculate AUC
    auc_train = roc_auc_score(y_train, y_pred_train_prob)
    auc_val = roc_auc_score(y_val, y_pred_val_prob)

    # 3. Calculate Optimal Threshold and Extended Metrics (Sensitivity/Specificity/CM)
    threshold_metrics = calculate_optimal_threshold_metrics(y_val, y_pred_val_prob, target_sensitivity=sensitivity_threshold)

    # Log the metrics
    logging.info(f"Optimal Threshold found: {threshold_metrics['optimal_threshold']:.4f}")
    logging.info(
        f"Sensitivity: {threshold_metrics['sensitivity']:.4f}, Specificity: {threshold_metrics['specificity']:.4f}")

    # 4. Calculate Confidence Interval for validation set
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

    # 5. Log final results
    logging.info(f"Best Model - Training AUC: {auc_train:.4f}")
    logging.info(f"Best Model - Validation AUC: {auc_val:.4f}")

    # 6. Prepare and save results to JSON
    results_data = {
        'model_name': 'mlp',
        'training_auc': auc_train,
        'validation_auc': auc_val,
        'validation_ci_alpha': alpha,
        'validation_ci_lower': val_lower_ci,
        'validation_ci_upper': val_upper_ci,
        'best_hyperparameters': best_hps.values
    }

    # Merge the threshold metrics into the results dictionary
    results_data.update(threshold_metrics)

    # Save the JSON in the same directory as the model
    results_json_path = model_save_path.with_name("mlp_validation_stats.json")

    try:
        with open(results_json_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        logging.info(f"Saved validation statistics to {results_json_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON results: {e}")


    return best_model_keras


def train_mlp_with_cv(df_train: pd.DataFrame,
                      df_validate: pd.DataFrame,
                      model_save_path: Path,
                      log_dir: Path,
                      sensitivity_threshold: float) -> tf.keras.Model:
    """
    Trains and tunes an MLP model using 5-Fold Cross-Validation via a Custom Keras Tuner.
    """
    logging.info("Starting MLP training with 5-Fold CV (Custom Keras Tuner)...")

    # --- 1. Data Prep ---
    X_train, y_train, _ = input_output(df_train)
    X_val, y_val, _ = input_output(df_validate)  # Loaded ONLY for final evaluation/early stopping
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)

    # Calculate class weight
    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    class_weights = class_weight_calculator(n_neg, n_pos)
    logging.info(f"Calculated class weights: {class_weights}")

    # --- 2. Instantiate Custom CV Tuner ---
    tuner_directory = log_dir / 'keras_tuner_cv'
    project_name = 'mlp_cv_tuning'

    tuner = CVRandomSearch(
        hypermodel=build_model,
        objective=kt.Objective("val_auc_roc", direction="max"),  # Maximizing Mean CV AUC
        max_trials=250,  # Reduced trials because 50 trials * 5 folds = 250 training runs
        executions_per_trial=1,
        directory=str(tuner_directory),
        project_name=project_name,
        overwrite=True
    )

    # --- 3. Run the CV Search ---
    logging.info(f"Starting Custom CV Search (5-Fold)...")

    # The CVRandomSearch class handles the splitting internally on X_train.
    tuner.search(
        x=X_train,
        y=y_train,
        epochs=500,  # Max epochs per fold
        batch_size=32,
        class_weight=class_weights
    )

    logging.info("CV Search finished.")

    # --- 4. Get Best HPs and Rebuild Final Model ---
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    best_cv_score = best_trial.score  # This is the Mean CV AUC
    best_hps = best_trial.hyperparameters

    logging.info(f"Best Mean CV AUC (Training Pool): {best_cv_score:.4f}")
    logging.info(f"Best hyperparameters: {best_hps.values}")

    # --- CRITICAL: Rebuild and Safe Training ---
    # We must rebuild the model because the CV loop discarded the fold models.
    # To avoid "overfitting on the whole dataset" (your concern), we:
    # 1. Train on X_train
    # 2. Use the EXTERNAL X_val strictly for Early Stopping (to stop training at the right time)
    # 3. We do NOT use X_val for tuning (tuning is already done via CV)

    logging.info("Rebuilding best model on full training pool (with Early Stopping on Val Set)...")
    best_model = build_model(best_hps)

    final_es = EarlyStopping(
        monitor='val_auc_roc',
        patience=50,
        restore_best_weights=True,
        mode='max'
    )

    best_model.fit(
        X_train,
        y_train,
        # validation_data=(X_val, y_val),  # Used ONLY for Early Stopping here
        epochs=100,
        batch_size=32,
        callbacks=[final_es],
        class_weight=class_weights,
        verbose=1
    )

    # Save the best Keras model
    best_model.save(model_save_path)
    logging.info(f"Best MLP model saved to {model_save_path}")

    # --- 5. Final Evaluation & JSON Saving ---
    # This generates the results for Table 2

    y_pred_train_prob = best_model.predict(X_train).flatten()
    y_pred_val_prob = best_model.predict(X_val).flatten()

    auc_train = roc_auc_score(y_train, y_pred_train_prob)
    auc_val = roc_auc_score(y_val, y_pred_val_prob)

    # 3. Calculate Optimal Threshold and Extended Metrics (Sensitivity/Specificity/CM)
    threshold_metrics = calculate_optimal_threshold_metrics(y_val, y_pred_val_prob, target_sensitivity=sensitivity_threshold)

    # Log the metrics
    logging.info(f"Optimal Threshold found: {threshold_metrics['optimal_threshold']:.4f}")
    logging.info(
        f"Sensitivity: {threshold_metrics['sensitivity']:.4f}, Specificity: {threshold_metrics['specificity']:.4f}")

    # 4. Calculate Confidence Interval for validation set
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

    # 5. Log final results
    logging.info(f"Best Model - Training AUC: {auc_train:.4f}")
    logging.info(f"Best Model - Validation AUC: {auc_val:.4f}")

    # 6. Prepare and save results to JSON
    results_data = {
        'model_name': 'mlp',
        'training_auc': auc_train,
        'validation_auc': auc_val,
        'validation_ci_alpha': alpha,
        'validation_ci_lower': val_lower_ci,
        'validation_ci_upper': val_upper_ci,
        'best_hyperparameters': best_hps.values
    }

    # Merge the threshold metrics into the results dictionary
    results_data.update(threshold_metrics)

    # Save the JSON in the same directory as the model
    results_json_path = model_save_path.with_name("mlp_validation_stats.json")

    try:
        with open(results_json_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        logging.info(f"Saved validation statistics to {results_json_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON results: {e}")

    # threshold_metrics = calculate_optimal_threshold_metrics(y_val, y_pred_val_prob,
    #                                                         target_sensitivity=sensitivity_threshold)
    #
    # logging.info(f"Final Model - Validation AUC (Reported in Table 2): {auc_val:.4f}")
    #
    # # Prepare and save results to JSON
    # results_data = {
    #     'model_name': 'mlp',
    #     'training_cv_auc': best_cv_score,  # The tuning metric
    #     'final_training_auc': auc_train,
    #     'validation_auc': auc_val,  # The reporting metric
    #     'best_hyperparameters': best_hps.values
    # }
    # results_data.update(threshold_metrics)
    #
    # results_json_path = model_save_path.with_name("mlp_validation_stats.json")
    # try:
    #     with open(results_json_path, 'w') as f:
    #         json.dump(results_data, f, indent=4)
    # except Exception as e:
    #     logging.error(f"Failed to save JSON results: {e}")

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
        Includes TensorBoard for visualization.
        """
    logging.info("Starting MLP training with Manual Hyperparameters and TensorBoard...")

    # --- 1. Data Prep ---
    X_train, y_train, _ = input_output(df_train)
    X_val, y_val, _ = input_output(df_validate)
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)

    # Calculate class weight
    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    class_weights = class_weight_calculator(n_neg, n_pos)
    logging.info(f"Calculated class weights: {class_weights}")

    input_shape = X_train.shape[1]

    # --- 2. Build Model ---
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

    # --- 3. Define Callbacks (including TensorBoard and CUSTOM SaveBestModel) ---
    tensorboard_log_path = log_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.info(f"TensorBoard logs path set to: {tensorboard_log_path}")

    tensorboard_cb = TensorBoard(
        log_dir=tensorboard_log_path,
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )

    # EarlyStopping no longer restores weights, only stops training
    early_stopping_cb = EarlyStopping(
        monitor='val_auc_roc',
        patience=50,
        restore_best_weights=False,  # We let the custom callback manage weights
        mode='max'
    )

    # Instantiate your custom callback
    # We want to maximize 'val_auc_roc'
    best_weights_cb = SaveBestModel(save_best_metric='val_auc_roc', this_max=True)

    callbacks = [tensorboard_cb, early_stopping_cb, best_weights_cb]

    # --- 4. Train the Model ---
    logging.info(f"Starting training for {1000} epochs...")
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

    # Load the best weights from your custom callback
    logging.info("Restoring best weights based on val_auc_roc...")
    model.set_weights(best_weights_cb.best_weights)
    best_model_keras = model  # The model now holds the best weights

    # --- 5. Save & Report ---
    # Save the model in the native Keras format
    best_model_keras.save(model_save_path)
    logging.info(f"Best MLP model saved to {model_save_path}")

    # Final evaluation, CI calculation, and JSON saving (as per train_mlp_with_cv)
    # ... (Metrics calculation block is identical to train_mlp_with_cv and is omitted for brevity)

    # Get model predictions (probabilities)
    y_pred_train_prob = best_model_keras.predict(X_train).flatten()
    y_pred_val_prob = best_model_keras.predict(X_val).flatten()

    # Calculate AUC
    auc_train = roc_auc_score(y_train, y_pred_train_prob)
    auc_val = roc_auc_score(y_val, y_pred_val_prob)

    # Calculate Optimal Threshold and Extended Metrics
    threshold_metrics = calculate_optimal_threshold_metrics(y_val, y_pred_val_prob,
                                                            target_sensitivity=sensitivity_threshold)

    # Log final results
    logging.info(f"Best Model - Training AUC: {auc_train:.4f}")
    logging.info(f"Best Model - Validation AUC: {auc_val:.4f}")

    # Prepare and save results to JSON
    # Note: best_hyperparameters are now the manual ones.
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


def train_rf_no_cv(df_train: pd.DataFrame,
             df_validate: pd.DataFrame,
             model_save_path: Path,
             results_path: Path,
             sensitivity_threshold: float):
    """
        Trains, evaluates, and saves a Random Forest model using GridSearchCV.
        """
    logging.info("Starting Random Forest training with GridSearchCV...")
    X_train, y_train, _ = input_output(df_train)
    X_val, y_val, val_dir = input_output(df_validate)

    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    class_weight = class_weight_calculator(n_neg, n_pos)
    logging.info(f"Calculated class weights: {class_weight}")

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [50, 75, 100, 125, 150],
        'max_depth': [3, 5, 7, 9, 11],
        'min_samples_split': [10, 15, 20],
        'min_samples_leaf': [5, 7, 10, 15],
        'max_features': [0.3, 0.5, 'sqrt'],
        'class_weight': ['balanced_subsample', {0: 1, 1: 3}, class_weight]
    }
    # 3. Manual Grid Search Loop
    best_score = -1
    best_model = None
    best_params = None

    # Create all combinations of parameters
    grid = list(ParameterGrid(param_grid))
    total_combinations = len(grid)
    logging.info(f"Evaluating {total_combinations} hyperparameter combinations on Validation Set...")

    for i, params in enumerate(grid):
        # Train on FULL training set
        rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        # Evaluate on INDEPENDENT validation set
        y_val_pred = rf.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred)

        # Track best model
        if val_auc > best_score:
            best_score = val_auc
            best_model = rf
            best_params = params
            # Optional: Log improvements
            # logging.info(f"New Best AUC: {val_auc:.4f} with params {params}")

    logging.info(f"Search finished. Best Validation AUC: {best_score:.4f}")
    logging.info(f"Best hyperparameters: {best_params}")

    logging.info(f"Search finished. Best Validation AUC: {best_score:.4f}")
    logging.info(f"Best hyperparameters: {best_params}")

    # 4. Save Best Model
    # The model is already trained on X_train, so we just save it.
    joblib.dump(best_model, model_save_path)
    logging.info(f"Best RF model saved to {model_save_path}")

    # --- Final Reporting ---
    logging.info("Calculating final metrics...")

    y_pred_prob_val = best_model.predict_proba(X_val)[:, 1]
    y_pred_prob_train = best_model.predict_proba(X_train)[:, 1]

    auc_train = roc_auc_score(y_train, y_pred_prob_train)
    auc_val = roc_auc_score(y_val, y_pred_prob_val)

    threshold_metrics = calculate_optimal_threshold_metrics(y_val, y_pred_prob_val,
                                                            target_sensitivity=sensitivity_threshold)

    logging.info(f"Optimal Threshold: {threshold_metrics['optimal_threshold']:.4f}")

    # Calculate CI
    alpha = 0.05
    try:
        val_lower_ci, val_upper_ci = auc_confidence_interval(
            y_val, y_pred_prob_val, num_bootstraps=1000, alpha=alpha
        )
        logging.info(f"Validation AUC CI: ({val_lower_ci:.4f}, {val_upper_ci:.4f})")
    except Exception as e:
        val_lower_ci, val_upper_ci = None, None

    # Save stats
    results_data = {
        'model_name': 'rf',
        'training_auc': auc_train,
        'validation_auc': auc_val,
        'validation_ci_alpha': alpha,
        'validation_ci_lower': val_lower_ci,
        'validation_ci_upper': val_upper_ci,
        'best_hyperparameters': best_params
    }
    results_data.update(threshold_metrics)

    results_json_path = model_save_path.with_name("rf_validation_stats.json")
    try:
        with open(results_json_path, 'w') as f:
            json.dump(results_data, f, indent=4)
    except Exception as e:
        logging.error(f"Failed to save RF JSON results: {e}")

    # Save predictions
    val_results_df = pd.DataFrame({'series_dir': val_dir, 'prediction': y_pred_prob_val, 'label': y_val})
    val_results_df.to_csv(results_path, index=False)

    return best_model



def train_rf(df_train: pd.DataFrame,
             df_validate: pd.DataFrame,
             model_save_path: Path,
             results_path: Path,
             sensitivity_threshold: float):
    """
    Trains, evaluates, and saves a Random Forest model using GridSearchCV.
    """
    logging.info("Starting Random Forest training with GridSearchCV...")
    X_train, y_train, _ = input_output(df_train)
    X_val, y_val, val_dir = input_output(df_validate)

    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    class_weight = class_weight_calculator(n_neg, n_pos)
    logging.info(f"Calculated class weights: {class_weight}")

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [50, 75, 100, 125, 150],
        'max_depth': [3, 5, 7, 9, 11],
        'min_samples_split': [10, 15, 20],
        'min_samples_leaf': [5, 7, 10, 15],
        'max_features':  [0.3, 0.5, 'sqrt'],
        'class_weight': ['balanced_subsample', {0: 1, 1: 3}, class_weight]
    }

    # # Hyperparameter grid
    # param_grid = {
    #     'n_estimators': [50, 100],
    #     'max_depth': [5],
    #     'min_samples_split': [10],
    #     'min_samples_leaf': [5, 6],
    #     'max_features': ['sqrt'],
    #     'class_weight': ['balanced_subsample']
    # }
    rf_model = RandomForestClassifier(random_state=42)
    scorer = make_scorer(roc_auc_score)

    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring=scorer)
    grid_search.fit(X_train, y_train)

    best_rf_model = grid_search.best_estimator_
    joblib.dump(best_rf_model, model_save_path)
    logging.info(f"Best RF model saved to {model_save_path}")
    logging.info(f"Best hyperparameters: {grid_search.best_params_}")

    # Final evaluation, CI calculation, and JSON saving.

    logging.info("Calculating final metrics, optimal threshold, and confidence intervals for RF...")

    # 1. Get model predictions (probabilities)
    y_pred_prob_val = best_rf_model.predict_proba(X_val)[:, 1]
    y_pred_prob_train = best_rf_model.predict_proba(X_train)[:, 1]

    # 2. Calculate AUC
    auc_train = roc_auc_score(y_train, y_pred_prob_train)
    auc_val = roc_auc_score(y_val, y_pred_prob_val)

    # 3. Calculate Optimal Threshold and Extended Metrics
    threshold_metrics = calculate_optimal_threshold_metrics(y_val, y_pred_prob_val, target_sensitivity=sensitivity_threshold)

    logging.info(f"Optimal Threshold found: {threshold_metrics['optimal_threshold']:.4f}")
    logging.info(
        f"Sensitivity: {threshold_metrics['sensitivity']:.4f}, Specificity: {threshold_metrics['specificity']:.4f}")

    # 4. Calculate Confidence Interval
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

    # 5. Log final results
    logging.info(f"RF Model - Training AUC: {auc_train:.4f}")
    logging.info(f"RF Model - Validation AUC: {auc_val:.4f}")

    # 6. Prepare and save results to JSON
    results_data = {
        'model_name': 'rf',
        'training_auc': auc_train,
        'validation_auc': auc_val,
        'validation_ci_alpha': alpha,
        'validation_ci_lower': val_lower_ci,
        'validation_ci_upper': val_upper_ci,
        'best_hyperparameters': grid_search.best_params_
    }
    # Merge threshold metrics
    results_data.update(threshold_metrics)

    # Save the JSON in the same directory as the model
    results_json_path = model_save_path.with_name("rf_validation_stats.json")

    try:
        with open(results_json_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        logging.info(f"Saved RF validation statistics to {results_json_path}")
    except Exception as e:
        logging.error(f"Failed to save RF JSON results: {e}")

    # Save validation results
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
    Trains an ensemble RF model using features from a pre-trained MLP.
    """
    logging.info("Starting Ensemble (MLP+RF) training...")
    if not mlp_model_path.exists():
        logging.error(f"MLP model not found at {mlp_model_path}. Cannot train ensemble.")
        return

    # 1. Load MLP and create feature extractor
    logging.info(f"Loading base MLP model from {mlp_model_path}")
    model_mlp = tf.keras.models.load_model(mlp_model_path)
    try:
        # Try to get the second-to-last layer (before sigmoid)
        feature_layer_index = -2
        model_mlp.get_layer(index=feature_layer_index)
    except ValueError:
        logging.warning("Could not get layer -2, trying by name 'dense'")
        try:
            # Fallback: try to get the last dense layer by name
            dense_layers = [l for l in model_mlp.layers if isinstance(l, tf.keras.layers.Dense)]
            feature_layer_name = dense_layers[-2].name  # second to last dense layer
            feature_layer_index = model_mlp.get_layer(name=feature_layer_name).name
        except Exception as e:
            logging.error(f"Could not find feature layer. Error: {e}")
            return

    logging.info(f"Using MLP layer '{model_mlp.get_layer(index=feature_layer_index).name}' for features.")
    intermediate_layer_model = tf.keras.Model(
        inputs=model_mlp.input,
        outputs=model_mlp.get_layer(index=feature_layer_index).output
    )

    # 2. Extract features
    X_train_emb, y_train, _ = input_output(df_train)
    X_val_emb, y_val, val_dir = input_output(df_validate)

    train_ds = tf.data.Dataset.from_tensor_slices(X_train_emb).batch(128).cache()
    val_ds = tf.data.Dataset.from_tensor_slices(X_val_emb).batch(128).cache()

    logging.info("Extracting features from MLP...")
    train_features = intermediate_layer_model.predict(train_ds)
    val_features = intermediate_layer_model.predict(val_ds)

    # 3. Train RF on extracted features
    logging.info("Training RF on extracted features...")

    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    class_weight = class_weight_calculator(n_neg, n_pos)
    logging.info(f"Calculated class weights: {class_weight}")

    # Hyperparameter grid
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


    # Final evaluation, CI calculation, and JSON saving.
    logging.info("Calculating final metrics, optimal threshold, and confidence intervals for Ensemble...")

    # 1. Get model predictions (probabilities)
    y_pred_prob_val = best_rf_model.predict_proba(val_features)[:, 1]
    y_pred_prob_train = best_rf_model.predict_proba(train_features)[:, 1]

    # 2. Calculate AUC
    auc_train = roc_auc_score(y_train, y_pred_prob_train)
    auc_val = roc_auc_score(y_val, y_pred_prob_val)

    # 3. Calculate Optimal Threshold and Extended Metrics
    threshold_metrics = calculate_optimal_threshold_metrics(y_val, y_pred_prob_val, target_sensitivity=sensitivity_threshold)

    logging.info(f"Optimal Threshold found: {threshold_metrics['optimal_threshold']:.4f}")
    logging.info(
        f"Sensitivity: {threshold_metrics['sensitivity']:.4f}, Specificity: {threshold_metrics['specificity']:.4f}")

    # 4. Calculate Confidence Interval
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

    # 5. Log final results
    logging.info(f"RF Model - Training AUC: {auc_train:.4f}")
    logging.info(f"RF Model - Validation AUC: {auc_val:.4f}")

    # 6. Prepare and save results to JSON
    results_data = {
        'model_name': 'rf',
        'training_auc': auc_train,
        'validation_auc': auc_val,
        'validation_ci_alpha': alpha,
        'validation_ci_lower': val_lower_ci,
        'validation_ci_upper': val_upper_ci,
        'best_hyperparameters': grid_search.best_params_
    }
    # Merge threshold metrics
    results_data.update(threshold_metrics)

    # Save the JSON in the same directory as the model
    results_json_path = model_save_path.with_name("rf_validation_stats.json")

    try:
        with open(results_json_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        logging.info(f"Saved RF validation statistics to {results_json_path}")
    except Exception as e:
        logging.error(f"Failed to save RF JSON results: {e}")

    # Save validation results
    val_results_df = pd.DataFrame({'series_dir': val_dir, 'prediction': y_pred_prob_val, 'label': y_val})
    val_results_df.to_csv(results_path, index=False)
    logging.info(f"Ensemble validation predictions saved to {results_path}")

    return best_rf_model


# ---
# 4. EVALUATION FUNCTION
# ---

def evaluate_model_on_test_set(model_path: Path, model_type: str, test_data_path: Path):
    """
    Loads a trained model and evaluates it on a final, hold-out test set.

    It reads the optimal_threshold from the model's validation JSON file
    and uses it to calculate sensitivity/specificity on the test set.
    """
    logging.info(f"--- Starting Test Set Evaluation for {model_type} ---")
    logging.info(f"Model: {model_path}")
    logging.info(f"Test Data: {test_data_path}")

    # --- 1. Define Paths ---
    run_dir = model_path.parent

    # Define path to find the *validation* stats (to get the threshold)
    if model_type == 'mlp':
        val_stats_json_path = run_dir / "mlp_validation_stats.json"
    elif model_type == 'mlp_manual':
        val_stats_json_path = run_dir / "mlp_manual_validation_stats.json"
    elif model_type == 'rf':
        val_stats_json_path = run_dir / "rf_validation_stats.json"
    elif model_type == 'ensemble':
        run_dir = model_path
        # --- WORKAROUND ---
        # Look for the 'rf' JSON name, since we skipped the bug fix
        val_stats_json_path = model_path / "rf_validation_stats.json"
        # print(val_stats_json_path)
        # input()
        logging.warning("Looking for 'rf_validation_stats.json' for ensemble model.")
    else:
        logging.error(f"Unknown model type: {model_type}")
        return

    # --- 2. Load Data and Threshold ---
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

    # --- 3. Load Model(s) and Get Predictions ---
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
            # Load the final RF
            model_rf = joblib.load(model_path/ 'best_ensemble_model.pkl')

            # Infer and load the base MLP
            base_mlp_path = model_path / 'base_mlp_for_ensemble.keras'
            model_mlp = tf.keras.models.load_model(base_mlp_path)

            # Create feature extractor (logic copied from train_ensemble)
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

            # Extract features from test set
            logging.info("Extracting test features from base MLP...")
            test_ds = tf.data.Dataset.from_tensor_slices(X_test).batch(128).cache()
            X_test_features = intermediate_layer_model.predict(test_ds)

            # Get final predictions
            y_pred_prob = model_rf.predict_proba(X_test_features)[:, 1]

    except Exception as e:
        logging.error(f"Error loading model or making predictions: {e}", exc_info=True)
        return

    # --- 4. Calculate Metrics ---
    logging.info("Calculating test set metrics...")

    # AUC and CI
    auc_test = roc_auc_score(y_test, y_pred_prob)
    alpha = 0.05
    try:
        test_lower_ci, test_upper_ci = auc_confidence_interval(
            y_test, y_pred_prob, num_bootstraps=1000, alpha=alpha
        )
    except Exception as e:
        logging.warning(f"Could not calculate CI for test set: {e}")
        test_lower_ci, test_upper_ci = None, None

    # Threshold-based metrics
    threshold_metrics = calculate_metrics_at_threshold(y_test, y_pred_prob, threshold)

    logging.info(f"Test Set AUC: {auc_test:.4f}")
    logging.info(f"Test Set AUC CI (95%): ({test_lower_ci:.4f}, {test_upper_ci:.4f})")
    logging.info(f"Test Set Sensitivity (at {threshold:.4f}): {threshold_metrics['sensitivity']:.4f}")
    logging.info(f"Test Set Specificity (at {threshold:.4f}): {threshold_metrics['specificity']:.4f}")

    # --- 5. Save Results ---

    # Define paths for the test set results
    test_stats_json_path = run_dir / f"{model_type}_test_set_stats.json"
    test_preds_csv_path = run_dir / f"{model_type}_test_set_predictions.csv"

    # Save statistics to JSON
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

    # Save predictions to CSV
    try:
        val_results_df = pd.DataFrame({'series_dir': test_dir, 'prediction': y_pred_prob, 'label': y_test})
        val_results_df.to_csv(test_preds_csv_path, index=False)
        logging.info(f"Test predictions saved to {test_preds_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save test predictions CSV: {e}")

    logging.info(f"--- Finished Test Set Evaluation ---")


# ---
# 5. EXPERIMENT RUNNER
# ---

def run_single_experiment(dataset_name: str, model_type: str, threshold_limit: float, manual_hps: dict = None):
    """
    This is the "test function" you requested.
    It runs a single, complete experiment for a given dataset and model.
    """
    logging.info(f"--- Starting Experiment: DATASET={dataset_name}, MODEL={model_type} ---")

    # 1. Get configuration
    try:
        dataset_config = EXPERIMENT_CONFIG['datasets'][dataset_name]
        val_set_name = dataset_config['validation_set']
        val_path = EXPERIMENT_CONFIG['validation_sets'][val_set_name]
        train_path = dataset_config['train_path']
    except KeyError:
        logging.error(f"Configuration error: '{dataset_name}' or its validation set not found in EXPERIMENT_CONFIG.")
        return

    # 2. Create unique output directory for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}_{dataset_name}_{model_type}"
    run_dir = RESULTS_BASE_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Define standard file paths
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

    # 3. Load data
    try:
        df_train = pd.read_csv(train_path)
        df_validate = pd.read_csv(val_path)
        logging.info(f"Loaded training data: {len(df_train)} rows")
        logging.info(f"Loaded validation data: {len(df_validate)} rows")
    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e.filename}")
        return

    # 4. Run the specified model training
    try:
        if model_type == 'mlp':
            # Call the CV-tuning function.
            # We no longer pass augmentation/oversampling flags.
            train_mlp_with_cv(
                df_train, df_validate,
                model_save_path=mlp_model_path,
                log_dir=log_dir,
                sensitivity_threshold=threshold_limit
            )
        elif model_type == 'mlp_manual':  # ADD THIS BLOCK
            if manual_hps is None:
                logging.error("Model type 'mlp_manual' requires manual_hps dictionary.")
                return

            train_mlp_manual(
                df_train, df_validate,
                model_save_path=mlp_manual_model_path,
                log_dir=(log_dir / 'mlp_manual'),
                sensitivity_threshold=threshold_limit,
                # Pass unpacked manual hyperparameters to the function
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
            # This 'ensemble' depends on an 'mlp' model.
            # It will now train and TUNE the base MLP first.

            logging.info("Ensemble requires an MLP model. Training and tuning base MLP first...")
            base_mlp_path_for_ensemble = run_dir / 'base_mlp_for_ensemble.keras'

            # The base MLP for the ensemble is now also tuned.
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


def run_all_experiments():
    """
    Runs the full automation pipeline: 6 datasets * 3 model types = 18 experiments.
    """
    logging.info("=== STARTING FULL AUTOMATION RUN ===")
    dataset_names = EXPERIMENT_CONFIG['datasets'].keys()
    model_types = EXPERIMENT_CONFIG['model_types']
    sensitivity_limit = EXPERIMENT_CONFIG['sensitivity'].keys()

    run_count = 1
    total_runs = len(dataset_names) * len(model_types)

    for dataset_name in dataset_names:
        for model_type in model_types:
            logging.info(f"=== RUN {run_count}/{total_runs} ===")
            run_single_experiment(dataset_name, model_type, sensitivity_limit)
            run_count += 1

    logging.info("=== FULL AUTOMATION RUN FINISHED ===")


# ---
# 6. MAIN EXECUTION
# ---

def main():
    parser = argparse.ArgumentParser(
        description="ML Model Training and Validation Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Subparsers for 'all' and 'single' commands
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Command: 'all'
    parser_all = subparsers.add_parser(
        'all',
        help="Run all experiments (6 datasets * 3 model types = 18 runs)"
    )

    # Command: 'single' (this is your 'test' function)
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
    # --- Command: 'manual' ---
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

    # --- Command: 'evaluate' ---
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

    # --- End of parser definitions ---

    args = parser.parse_args()

    # Execute the chosen command
    if args.command == 'all':
        run_all_experiments()
    elif args.command == 'single':
        run_single_experiment(
            dataset_name=args.dataset,
            model_type=args.model,
            threshold_limit=args.sensitivity
        )
    elif args.command == 'manual':  # ADD THIS BLOCK
        # Convert '64,64' string to [64, 64] list of integers
        try:
            hidden_layer_sizes = [int(s.strip()) for s in args.layers.split(',')]
        except ValueError:
            logging.error("Invalid format for --layers. Use comma-separated integers (e.g., '64,64').")
            return

        run_single_experiment(
            dataset_name=args.dataset,
            model_type='mlp_manual',
            threshold_limit=args.sensitivity,
            # Pass manual HPs as kwargs
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