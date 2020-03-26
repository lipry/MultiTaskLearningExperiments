from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.utils import multi_gpu_model
import tensorflow as tf
from kerastuner import BayesianOptimization
from src.config.config import config
from src.metrics.metrics import auprc, auroc

def cnn_full_params_sharing_model(hp):
    with tf.device("/cpu:0"):
        inputs = Input(shape=(200,5))

        config_cnn = config['cnn_full_params_sharing']
        config_cnn_hp = config['cnn_full_params_sharing']['bayesian_opt']['hyperparameters']

        # hyper-parameters grid preparation
        learning_rate = hp.Float('learning_rate', min_value=config_cnn_hp['learning_rate'][0],
                                 max_value=config_cnn_hp['learning_rate'][1])
        kernel_size1 = hp.Choice('kernel_size1', values=config_cnn_hp['kernel_size1'])
        kernel_size2 = hp.Choice('kernel_size2', values=config_cnn_hp['kernel_size2'])
        units2 = hp.Choice('units2', values=config_cnn_hp['units2'])
        dense1 = hp.Choice('dense1', values=config_cnn_hp['dense1'])
        dense2 = hp.Choice('dense2', values=config_cnn_hp['dense2'])

        x = Conv1D(64, kernel_size=kernel_size1, activation='relu')(inputs)
        x = Conv1D(64, kernel_size=kernel_size1, activation='relu')(x)
        x = Conv1D(64, kernel_size=kernel_size1, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)

        x = Conv1D(units2,
                   kernel_size=kernel_size2,
                   activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)

        x = Dense(dense1, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(dense2, activation='relu')(x)
        x = Dropout(0.1)(x)

        predictions1 = Dense(1, activation='sigmoid', name="pred0")(x)
        predictions2 = Dense(1, activation='sigmoid', name="pred1")(x)
        predictions3 = Dense(1, activation='sigmoid', name="pred2")(x)
        predictions4 = Dense(1, activation='sigmoid', name="pred3")(x)

        cnn_model = Model(inputs, [predictions1, predictions2, predictions3, predictions4])
    if config['execution']['n_gpu'] > 1:
        cnn_model = multi_gpu_model(cnn_model, gpus=config['execution']['n_gpu'])

    nadam_opt = Nadam(lr=learning_rate,
                      beta_1=config_cnn['beta_1'],
                      beta_2=config_cnn['beta_2'])

    cnn_model.compile(loss={'pred0': 'binary_crossentropy', 'pred1': 'binary_crossentropy',
                            'pred2': 'binary_crossentropy', 'pred3': 'binary_crossentropy'},
                       optimizer=nadam_opt,
                       metrics=[auprc, auroc])

    return cnn_model


def get_batch_size():
    n_gpu = config['execution']['n_gpu']
    batch_size = config['cnn_full_params_sharing']['batch_size']
    return n_gpu * batch_size if n_gpu > 0 else batch_size


def check_input_type(admitted_input, message):
    if config['general']['input_type'] not in admitted_input:
        raise ValueError(message)


def hp_tuning_cnn_full_params_sharing(X_train, y_train, X_val, y_val, class_weight, n_best_models=1):
    check_input_type(['seq'], "Cnn full parameter sharing models work just with sequence data, {} found"
                     .format(config['general']['input_type']))

    print(X_train, y_train, X_val, y_val)

    config_cnn_bayesian = config['cnn_full_params_sharing']['bayesian_opt']
    batch_size_total = get_batch_size()

    tuner = BayesianOptimization(
        cnn_full_params_sharing_model,
        objective='val_loss', # TODO: binary_crossentropy, put in config?
        max_trials=config_cnn_bayesian['max_trials'],
        num_initial_points=config_cnn_bayesian['num_initial_points'],
        directory='tuner_results',
        project_name='cnn_full_params_sharing')

    es = EarlyStopping(monitor='val_loss', patience=config_cnn_bayesian['patience'],
                       min_delta=config_cnn_bayesian['min_delta'])

    tuner.search(X_train, y_train,
                 epochs=config['cnn_full_params_sharing']['epochs'],
                 batch_size=batch_size_total,
                 callbacks=[es],
                 class_weight=class_weight,
                 validation_data=(X_val, y_val))

    return tuner, tuner.get_best_models(num_models=n_best_models), tuner.get_best_hyperparameters(num_trials=n_best_models)


def train_cnn_full_params_sharing(X_train, y_train, X_val, y_val, class_weight, training_hp):
    check_input_type(['seq'], "Cnn full parameter sharing models work just with sequence data, {} found"
                     .format(config['general']['input_type']))

    batch_size_total = get_batch_size()
    model = cnn_full_params_sharing_model(training_hp)
    history = model.fit(X_train, y_train,
             epochs=config['cnn_full_params_sharing']['epochs'],
             class_weight=class_weight,
             batch_size=batch_size_total,
             validation_data=(X_val, y_val))

    return model, history
