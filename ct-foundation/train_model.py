import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, roc_curve, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from savebestmodel import SaveBestModel
from utils import *
import joblib


import datetime
log_folder = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks = TensorBoard(log_dir=log_folder,
                         histogram_freq=1,
                         write_graph=True,
                         write_images=True,
                         update_freq='epoch',
                         profile_batch=2,
                         embeddings_freq=1)

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

def create_model(token_num=1,
                 embeddings_size=1408,
                 learning_rate=0.002,
                 alpha=0.8,
                 dropout=0.3,
                 first_decay_steps=50,
                 hidden_layer_sizes=[32, 32],
                 weight_decay=1e-7,
                 seed=None) -> tf.keras.Model:
    """

    :param token_num: 1
    :param embeddings_size: 1408 for Google CT Foundation model
    :param learning_rate: initial leraning rate
    :param alpha:  minimum learning rate value
    :param dropout: rate of dropout
    :param first_decay_steps: first decay step for CosineDecayRestart
    :param hidden_layer_sizes: the size of each layer
    :param weight_decay: weight decay rate for regularization
    :param seed:
    :return: an MLP model
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
            kernel_regularizer=tf.keras.regularizers.l1(l1=weight_decay),
            bias_regularizer=tf.keras.regularizers.l1(l1=weight_decay))(
            hidden)

        hidden = tf.keras.layers.BatchNormalization()(hidden)       # choose your layer of normalization
        hidden = tf.keras.layers.LayerNormalization()(hidden)
        hidden = tf.keras.layers.Dropout(dropout, seed=seed)(hidden)

    output = tf.keras.layers.Dense(
        units=1,
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.HeUniform(seed=seed))(
        hidden)

    model = tf.keras.Model(inputs, output)
    learning_rate_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=learning_rate,
        first_decay_steps=first_decay_steps,
        alpha=alpha)
    model.compile(
        # optimizer=tf.keras.optimizers.Adam(
        #     learning_rate=learning_rate_fn),
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate = learning_rate),
        loss='binary_crossentropy',
        weighted_metrics=[
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.TruePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.AUC(curve='ROC', name='auc_roc')])
    return model



def train_model(df_train:pd.DataFrame, df_validate:pd.DataFrame,
                augmentation=False, oversampling=True,
                model_name='best_mlp_model.keras') -> tf.keras.Model:
    """
    :param df_train: training dataset
    :param df_validate: validation dataset
    :param augmentation: embedding augmentation mode: True or False
    :param oversampling: oversampling mode: True or False
    :param model_name:  Name of the model to be saved
    :return: trained keras model
    """
    # extract labels of training and validation dataset
    df_train_labels = df_train['labels'].values
    df_validate_labels = df_validate['labels'].values
    # calculate the number of positive and negative cases to calculate class weight.
    n_pos = sum(df_train_labels) + sum(df_validate_labels)
    n_neg = len(df_train_labels) + len(df_validate_labels) - n_pos
    batch_size = 128
    # update training dataset the augmentation is true
    if augmentation:
        df_train_embedding = list(df_train['embedding'].values)
    else:
        df_train_embedding = embedding_array(df_train['embedding'].values)

    df_validate_embedding = embedding_array(df_validate['embedding'].values)
    # update the training dataset if oversampling is true and create tensorflow training dataset
    if oversampling:
        train_ds = oversampling_training(df_train_embedding, df_train_labels)
        steps_per_epoch = int(np.ceil(2.0 * n_neg / batch_size))
    else:
        train_ds = tf.data.Dataset.from_tensor_slices((df_train_embedding, df_train_labels))
        steps_per_epoch = int(np.ceil((n_pos+n_neg)/ batch_size))

    # create the tensorflow evaluation dataset
    eval_ds = tf.data.Dataset.from_tensor_slices((df_validate_embedding, df_validate_labels))

    # create model and define the callbacks
    model = create_model()
    early_stopping = EarlyStopping(
        monitor='val_auc',
        patience=50,
        restore_best_weights=True
    )
    save_best_model = SaveBestModel(save_best_metric='val_auc', this_max=True)

    model.fit(
        x=train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache(),
        validation_data=eval_ds.batch(8).cache(),
        epochs=500,
        steps_per_epoch=steps_per_epoch,
        callbacks=[callbacks, early_stopping, save_best_model],
        class_weight=class_weight_calculator(n_neg, n_pos)
    )
    model.set_weights(save_best_model.best_weights)
    model.summary()
    model.save(model_name)

    # evaluate the best model on the training dataset when there is no augmentation
    if not augmentation and not oversampling:
        train_results = model.evaluate(
            train_ds.batch(1).prefetch(tf.data.AUTOTUNE).cache(),
            verbose=1
        )
        print(f"Best Model - Training results: {train_results}")


    # evaluate the best model on the validation dataset
    validation_results = model.evaluate(
        eval_ds.batch(1).prefetch(tf.data.AUTOTUNE).cache(),
        verbose=1
    )
    print(f"Best Model - Validation results: {validation_results}")

    return model

def evaluation(df_train:pd.DataFrame,
               df_test: pd.DataFrame,
               trained_model:tf.keras.Model):
    """
    :param df_train: The dataset containing all samples used for training
    :param df_test: The test dataset
    :param trained_model: the trained MLP model
    :return:
    """

    test_results_file = 'test_results.csv'  # save the test results inside this file for further analysis

    # create test and training tensorflow dataset to feed the model
    df_test_labels = df_test['labels'].values
    df_test_embedding = embedding_array(df_test['embedding'].values)
    test_data = tf.data.Dataset.from_tensor_slices((df_test_embedding, df_test_labels))

    df_train_labels = df_train['labels'].values
    df_train_embedding = embedding_array(df_train['embedding'].values)
    train_data = tf.data.Dataset.from_tensor_slices((df_train_embedding, df_train_labels))

    series = df_test['series_dir'].values
    DIAGNOSIS = 'PE'

    # extract embeddings, labels, study number of the test dataset, and the evaluation of the trained model.
    rows = []
    i = 0
    for embeddings, label in test_data.batch(1):
        row = {
            f'{DIAGNOSIS}_prediction': trained_model(embeddings).numpy().flatten()[0],
            f'{DIAGNOSIS}_labels': label.numpy().flatten()[0],
            'series_dir': series[i]
        }
        i += 1
        rows.append(row)


    test_df = pd.DataFrame(rows)
    test_df.to_csv(test_results_file)
    print(test_df.head(20))

    # extract labels and prediction of the test dataset to compute rates and plot roc curve.
    labels = test_df[f'{DIAGNOSIS}_labels'].values
    predictions = test_df[f'{DIAGNOSIS}_prediction'].values
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels,
        predictions,
        drop_intermediate=False)
    auc = roc_auc_score(labels, predictions)
    plot_curve(false_positive_rate, true_positive_rate, auc, x_label='False Positive Rate',
               y_label='True Positive Rate', label=DIAGNOSIS)

    # evaluation of the model on the whole training dataset
    train_results = trained_model.evaluate(
        train_data.batch(1).prefetch(tf.data.AUTOTUNE).cache(),
        verbose=1
    )
    print(f"Best Model - Train results: {train_results}")

    # evaluation of the model on the test dataset
    test_results = trained_model.evaluate(
        test_data.batch(1).prefetch(tf.data.AUTOTUNE).cache(),
        verbose=1
    )
    print(f"Best Model - Test results: {test_results}")

    # calculate the confidence interval of the test dataset auc
    alpha = 0.05
    lower_ci, upper_ci = auc_confidence_interval(labels, predictions, num_bootstraps=1000, alpha= alpha)
    print(f"AUC Confidence Interval ({(1-alpha)*100}%): ({lower_ci:.3f}, {upper_ci:.3f})")

def random_forest_model(dataset_train: pd.DataFrame,
                        dataset_test: pd.DataFrame,
                        best_model_file_name='best_model_RF.pkl'
                        ):
    """
    :param dataset_train: training dataset
    :param dataset_test: test dataset
    :param best_model_file_name: the file name to save the best RF model
    :return:
    """
    # extract the embeddings and labels for training and test
    X_train, y_train, train_dir = input_output(dataset_train)
    X_val, y_val, test_dir = input_output(dataset_test)

    # calculate the number of positive and negative cases to calculate class weight.
    n_pos = sum(y_train)
    n_neg = len(y_train) - sum(y_train)
    class_weight = class_weight_calculator(n_neg, n_pos)
    print(class_weight)
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [25, 50, 75, 100],
        'max_depth': [3, 5, 7, 9, 11],
        'min_samples_split': [10, 15, 20],
        'min_samples_leaf': [5, 10, 15],
        'max_features': [0.5],
        # 'max_features':  [0.3, 0.5, 'sqrt'],
        # 'class_weight': ['balanced_subsample', {0: 1, 1: 3}, class_weight]
        'class_weight': ['balanced_subsample']
        # 'class_weight': [{0: 1, 1: 3}]
    }
    rf_model = RandomForestClassifier()
    # Use AUC as the scoring metric
    scorer = make_scorer(roc_auc_score)
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring=scorer)
    grid_search.fit(X_train, y_train)
    best_rf_model = grid_search.best_estimator_
    joblib.dump(best_rf_model, best_model_file_name)  # Save the model


    # Predict and evaluate
    y_pred_prob = best_rf_model.predict_proba(X_val)[:, 1]
    y_pred_prob_train = best_rf_model.predict_proba(X_train)[:, 1]
    d_train = {'series_dir': train_dir, 'prediction':y_pred_prob_train, 'label': y_train }
    d_test = {'series_dir': test_dir, 'prediction':y_pred_prob, 'label': y_val }

    train_results = pd.DataFrame(data = d_train)
    test_resutls = pd.DataFrame(data = d_test)
    train_results.to_csv('train_results.csv')
    test_resutls.to_csv('test_resutls.csv')

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_val,
        y_pred_prob,
        drop_intermediate=False)
    auc_test = roc_auc_score(y_val, y_pred_prob)
    plot_curve(false_positive_rate, true_positive_rate, auc_test, x_label='False Positive Rate',
               y_label='True Positive Rate', label='Central PE Detection')

    print(f'AUC_training_CV: {roc_auc_score(y_train, y_pred_prob_train)}')
    print(f'AUC_validation_CV: {roc_auc_score(y_val, y_pred_prob)}')
    print('Best hyperparameters:', grid_search.best_params_)

    # calculate the confidence interval of the test dataset auc
    alpha = 0.05
    lower_ci, upper_ci = auc_confidence_interval(y_val, y_pred_prob, num_bootstraps=1000, alpha=alpha)
    print(f"AUC Confidence Interval ({(1 - alpha) * 100}%): ({lower_ci:.3f}, {upper_ci:.3f})")
    return best_rf_model

def validate_rf_model(dataset_train: pd.DataFrame,
                        dataset_test: pd.DataFrame,
                        best_model_file_name='best_model_RF.pkl'):
    """
        :param dataset_train: training dataset
        :param dataset_test: test dataset
        :param best_model_file_name: the file name of the best RF model that is saved.
        :return:
        """
    # extract the embeddings and labels for training and test
    X_train, y_train, train_dir = input_output(dataset_train)
    X_val, y_val, test_dir = input_output(dataset_test)
    loaded_model = joblib.load(best_model_file_name)  # Load the saved model
    # Predict and evaluate
    y_pred_prob = loaded_model.predict_proba(X_val)[:, 1]
    y_pred_prob_train = loaded_model.predict_proba(X_train)[:, 1]
    d_train = {'series_dir': train_dir, 'prediction': y_pred_prob_train, 'label': y_train}
    d_test = {'series_dir': test_dir, 'prediction': y_pred_prob, 'label': y_val}

    train_results = pd.DataFrame(data=d_train)
    test_resutls = pd.DataFrame(data=d_test)
    train_results.to_csv('train_results.csv')
    test_resutls.to_csv('test_resutls.csv')

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_val,
        y_pred_prob,
        drop_intermediate=False)
    auc_test = roc_auc_score(y_val, y_pred_prob)
    plot_curve(false_positive_rate, true_positive_rate, auc_test, x_label='False Positive Rate',
               y_label='True Positive Rate', label='Central PE Detection')

    thresholds = [0.1, 0.4, 0.5, 0.7, 0.9]
    generate_threshold_performance_report(y_val, y_pred_prob, thresholds)

    print(f'AUC_training_CV: {roc_auc_score(y_train, y_pred_prob_train)}')
    print(f'AUC_validation_CV: {roc_auc_score(y_val, y_pred_prob)}')
    print('Best hyperparameters:', loaded_model)

    # calculate the confidence interval of the test dataset auc
    alpha = 0.05
    lower_ci, upper_ci = auc_confidence_interval(y_val, y_pred_prob, num_bootstraps=1000, alpha=alpha)
    print(f"AUC Confidence Interval ({(1 - alpha) * 100}%): ({lower_ci:.3f}, {upper_ci:.3f})")

def ensemble_random_forest_model(dataset_train: pd.DataFrame,
                                 dataset_test: pd.DataFrame,
                                 mlp_model_file='best_mlp_model.keras'):
    """
    :param dataset_train: training dataset
    :param dataset_test: test dataset
    :param mlp_model_file: The name of the trained mlp model
    :return:
    """
    # extract the embeddings and labels for training and test
    X_train, y_train,_ = input_output(dataset_train)
    X_val, y_val,_ = input_output(dataset_test)

    # Set up the mlp model and choose which layers should be used for feature extraction.
    model_mlp = create_model()
    model_mlp.load_weights(mlp_model_file)
    intermediate_layer_model = tf.keras.Model(
        inputs=model_mlp.input,
        outputs=model_mlp.get_layer(index=-2).output  # Modify the index to the layer from which you want to extract features
    )
    # Create tensorflow dataset to predict the features using the best mlp model
    train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(list(X_train)),  y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((tf.constant(list(X_val)),  y_val))

    train_features = intermediate_layer_model.predict(train_ds.batch(128).cache())
    test_features = intermediate_layer_model.predict(test_ds.batch(8).cache())

    # calculate the number of positive and negative cases to calculate class weight.
    n_pos = sum(y_train)
    n_neg = len(y_train) - sum(y_train)
    class_weight = class_weight_calculator(n_neg, n_pos)
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [96, 97, 98, 99, 100, 101, 102, 103, 104],
        # 'n_estimators': [25, 50, 75, 100],
        # 'n_estimators': [25, 50, 10, 15],
        'max_depth': [3, 5, 7, 9, 11],
        'min_samples_split': [10, 15, 20],
        # 'min_samples_split': [5, 10, 12],
        'min_samples_leaf': [5, 10, 15],
        # 'max_features':  [0.3, 0.5],
        'max_features': ['sqrt'],
        'class_weight': [class_weight]
        # 'class_weight': [{0: 1, 1: 3}]
        # 'class_weight': ['balanced_subsample']
    }
    rf_model = RandomForestClassifier()
    # Use AUC as the scoring metric
    scorer = make_scorer(roc_auc_score)
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring=scorer)
    grid_search.fit(train_features, y_train)
    best_rf_model = grid_search.best_estimator_
    # Predict and evaluate
    y_pred = best_rf_model.predict(test_features)
    y_pred_prob = best_rf_model.predict_proba(test_features)[:, 1]

    y_pred_train = best_rf_model.predict(train_features)
    y_pred_prob_train = best_rf_model.predict_proba(train_features)[:, 1]

    print(f'AUC_training_CV: {roc_auc_score(y_train, y_pred_prob_train)}')
    print(f'AUC_validation_CV: {roc_auc_score(y_val, y_pred_prob)}')
    print('Best hyperparameters:', grid_search.best_params_)

    # calculate the confidence interval of the test dataset auc
    alpha = 0.05
    lower_ci, upper_ci = auc_confidence_interval(y_val, y_pred_prob, num_bootstraps=1000, alpha=alpha)
    print(f"AUC Confidence Interval ({(1 - alpha) * 100}%): ({lower_ci:.3f}, {upper_ci:.3f})")
    return best_rf_model

def create_data_subset(data, label, total_size):
    dataset_size = int(total_size/2)
    positive = data[data[label]==1]
    negative = data[data[label]==0]
    positive_subset = positive[0:dataset_size]
    negative_subset = negative[0:dataset_size]
    return pd.concat([positive_subset, negative_subset], ignore_index=True)





def main():
    TOKEN_NUM = 1
    EMBEDDINGS_SIZE = 1408
    file_name = 'dataset_imbalanced.csv'

    # crate the pd dataset and split them into training, validation, and test dataset.
    raw_data = create_dataset(file_name)
    label = 'labels'

    data = raw_data
    # data = create_data_subset(raw_data, label, 80)

    df_whole, df_test = split_dataset(data, label, split_ratio=0.2)
    df_train, df_validate = split_dataset(df_whole, label, split_ratio=0.25)
    mlp_model_file_name = 'best_mlp_model.keras'
    best_model_file_name = 'best_model_RF_1000.pkl'



    # # If training MLP. Please modify the hyperparameters in create_model()
    # augmentation = False    # True for naive augmentation on embeddings using a small noise
    # oversampling = False    # True for oversampling: replicates the existing positive cases.
    # if augmentation:
    #     df_augmented = data_augmentation(original_data=df_train, TOKEN_NUM=TOKEN_NUM,
    #                                      EMBEDDINGS_SIZE=EMBEDDINGS_SIZE, noise_std=1e-4, ratio=4)
    #     df_train = df_augmented
    #
    # trained_model = train_model(df_train, df_validate, augmentation=augmentation,
    #                             oversampling=oversampling, model_name=mlp_model_file_name)
    # evaluation(df_train = df_whole, df_test=df_test, trained_model=trained_model,
    #            token_num=TOKEN_NUM, embeddings_size=EMBEDDINGS_SIZE)

    # If training random forest, Please modify the hyperparameters in random_forest_model()
    # random_forest_model(df_whole, df_test, best_model_file_name)
    validate_rf_model(df_whole, df_test, best_model_file_name)

    # # If training ensemble model, first, train mlp model, then run the following lines, Please modify the hyperparameters in ensemble_random_forest_model()
    # ensemble_random_forest_model(df_train, df_test, label, mlp_model_file=mlp_model_file_name)





if __name__ == "__main__":
    main()



