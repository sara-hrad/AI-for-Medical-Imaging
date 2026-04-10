import os

import pandas as pd
import tensorflow as tf
import tensorflow_models as tfm
import sklearn

from cxr_foundation import embeddings_data
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from savebestmodel import SaveBestModel
from utils import *

import datetime

log_folder = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks = [TensorBoard(log_dir=log_folder,
                         histogram_freq=1,
                         write_graph=True,
                         write_images=True,
                         update_freq='epoch',
                         profile_batch=2,
                         embeddings_freq=1)]


def create_model(token_num=1,
                 embeddings_size=1376,
                 learning_rate=0.08,
                 alpha=.8,
                 dropout=0.5,
                 first_decay_steps=50,
                 hidden_layer_sizes=[32, 64, 32],
                 weight_decay=1e-7,
                 seed=None) -> tf.keras.Model:
    """

    :param token_num: 1 for original CXR foundation embedding and 32 for ELIXR embedding
    :param embeddings_size: 1376 for 'cxr_foundation', 768 for 'elixr', and 128 for 'elixr_img_contrastive' embeddings
    :param learning_rate: initial leraning rate
    :param alpha:  minimum learning rate value
    :param dropout: rate of dropout
    :param first_decay_steps: first decay step for CosineDecayRestart
    :param hidden_layer_sizes: the size of each layer
    :param weight_decay: weight decay rate for regularization
    :param seed:
    :return:
    """
    inputs = tf.keras.Input(shape=(token_num * embeddings_size,))
    inputs_reshape = tf.keras.layers.Reshape((token_num, embeddings_size))(inputs)
    inputs_pooled = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')(inputs_reshape)
    hidden = inputs_pooled
    # If no hidden_layer_sizes are provided, model will be a linear probe.
    for size in hidden_layer_sizes:
        hidden = tf.keras.layers.Dense(
            size,
            activation='relu',
            kernel_initializer=tf.keras.initializers.HeUniform(seed=seed),
            # kernel_regularizer=tf.keras.regularizers.l1(l1=weight_decay),
            # bias_regularizer=tf.keras.regularizers.l1(l1=weight_decay))(
            # hidden)
            kernel_regularizer = tf.keras.regularizers.l2(l2=weight_decay),
            bias_regularizer = tf.keras.regularizers.l2(l2=weight_decay))(
            hidden)

        hidden = tf.keras.layers.BatchNormalization()(hidden)       # choose your layer of normalization
        # hidden = tf.keras.layers.LayerNormalization()(hidden)
        hidden = tf.keras.layers.Dropout(dropout, seed=seed)(hidden)

    output = tf.keras.layers.Dense(
        units=1,
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.HeUniform(seed=seed))(
        hidden)

    model = tf.keras.Model(inputs, output)
    learning_rate_fn = tf.keras.experimental.CosineDecayRestarts(
        initial_learning_rate=learning_rate,
        first_decay_steps=first_decay_steps,
        alpha=alpha)
    model.compile(
        # optimizer=tfm.optimization.lars.LARS(
        optimizer=tfm.optimization.legacy_adamw.AdamWeightDecay(
            learning_rate=learning_rate_fn),
        loss='binary_crossentropy',
        weighted_metrics=[
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.TruePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.AUC(curve='ROC', name='auc_roc')])
    return model


def train_model(dataset_type: DatasetType,
                token_num=1,
                embeddings_size=1376):
    """
    :param dataset_type: the type of dataset to work on
    :param token_num: 1 for original CXR foundation embedding and 32 for ELIXR embedding
    :param embeddings_size: 1376 for 'cxr_foundation', 768 for 'elixr', and 128 for 'elixr_img_contrastive' embeddings
    :return: train dataset, validation dataset, test dataset, the trained model
    """
    TOKEN_NUM = token_num
    EMBEDDINGS_SIZE = embeddings_size

    dataset_dir, filename, DIAGNOSIS = dataset_directory(dataset_type)


    embedding_dataset_folder = f'./{dataset_dir}/data/outputs'


    df_labels = create_dataset(filename)
    df_labels["embedding_file"] = df_labels["embedding_file"].apply(
    lambda x: os.path.join(embedding_dataset_folder, os.path.basename(x)))
    df_train, df_test = split_dataset(df_labels, DIAGNOSIS)
    df_train, df_validate = split_dataset(df_train, DIAGNOSIS, split_ratio=0.25)

    # Create training and validation Datasets
    training_data = embeddings_data.get_dataset(
        filenames=df_train["embedding_file"].values,
        labels=df_train[DIAGNOSIS].values,
        embeddings_size=TOKEN_NUM * EMBEDDINGS_SIZE)

    validation_data = embeddings_data.get_dataset(
        filenames=df_validate["embedding_file"].values,
        labels=df_validate[DIAGNOSIS].values,
        embeddings_size=TOKEN_NUM * EMBEDDINGS_SIZE)

    # Create and train the model
    model = create_model(token_num=TOKEN_NUM, embeddings_size=EMBEDDINGS_SIZE)
    early_stopping = EarlyStopping(
        monitor='val_auc',
        patience=50,
        restore_best_weights=True
    )
    # Defining the callback to find the best model
    save_best_model = SaveBestModel(save_best_metric='val_auc', this_max=True)

    model.fit(
        x=training_data.batch(16).prefetch(tf.data.AUTOTUNE).cache(),
        validation_data=validation_data.batch(8).cache(),
        epochs=250,
        callbacks=[callbacks, early_stopping, save_best_model]
    )
    # Set the weights to the best model
    model.set_weights(save_best_model.best_weights)
    model.summary()
    model.save('model.keras')

    # Evaluate results (FN, FP, TN, TP, AUC) of training and evaluation

    train_results = model.evaluate(
        training_data.batch(1).prefetch(tf.data.AUTOTUNE).cache(),
        verbose=1
    )
    print(f"Best Model - Train_auc results: {train_results}")

    validation_results = model.evaluate(
        validation_data.batch(1).prefetch(tf.data.AUTOTUNE).cache(),
        verbose=1
    )
    print(f"Best Model - Validation results: {validation_results}")

    return df_train, df_validate, df_test, model


def evaluation(dataset_type: DatasetType,
               df_test: pd.DataFrame,
               trained_model:tf.keras.Model,
               token_num=1,
               embeddings_size=1376):

    TOKEN_NUM = token_num
    EMBEDDINGS_SIZE = embeddings_size

    dataset_dir, filename, DIAGNOSIS = dataset_directory(dataset_type)
    test_results_file = f'./{dataset_dir}/test_results_{DIAGNOSIS}.csv'

    X_test = df_test["embedding_file"].values
    y_test = df_test[DIAGNOSIS].values

    test_data = embeddings_data.get_dataset(
        filenames=X_test,
        labels=y_test,
        embeddings_size=TOKEN_NUM * EMBEDDINGS_SIZE)

    rows = []
    i = 0
    k_iteration = 0
    for embeddings, label in test_data.batch(1):
        row = {
            f'{DIAGNOSIS}_prediction': trained_model(embeddings).numpy().flatten()[0],
            f'{DIAGNOSIS}_value': label.numpy().flatten()[0],
            'embedding file': X_test[i]
        }
        if y_test[i]==label.numpy().flatten()[0]:
            k_iteration+=1
        i += 1
        rows.append(row)


    test_df = pd.DataFrame(rows)
    test_df.to_csv(test_results_file)
    print(test_df.head(20))

    labels = test_df[f'{DIAGNOSIS}_value'].values
    predictions = test_df[f'{DIAGNOSIS}_prediction'].values
    false_positive_rate, true_positive_rate, thresholds = sklearn.metrics.roc_curve(
        labels,
        predictions,
        drop_intermediate=False)
    auc = sklearn.metrics.roc_auc_score(labels, predictions)
    plot_curve(false_positive_rate, true_positive_rate, auc, x_label='False Positive Rate',
               y_label='True Positive Rate', label=DIAGNOSIS)

    test_results = trained_model.evaluate(
        test_data.batch(1).prefetch(tf.data.AUTOTUNE).cache(),
        verbose=1
    )
    print(f"Best Model - Test results: {test_results}")

    alpha = 0.05
    lower_ci, upper_ci = auc_confidence_interval(labels, predictions, num_bootstraps=1000, alpha= alpha)
    print(f"AUC Confidence Interval ({(1-alpha)*100}%): ({lower_ci:.3f}, {upper_ci:.3f})")


def main():
    TOKEN_NUM = 1
    EMBEDDINGS_SIZE = 1376
    dataset_name = DatasetType.Abnormal_1000
    df_train, df_validate, df_test, trained_model = train_model(dataset_name)
    evaluation(dataset_type=dataset_name, df_test=df_test, trained_model=trained_model,
               token_num=TOKEN_NUM, embeddings_size=EMBEDDINGS_SIZE)



if __name__ == "__main__":
    main()