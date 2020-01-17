from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.utils import multi_gpu_model
import tensorflow as tf
from kerastuner import BayesianOptimization

from src.config.config import config
from src.metrics.metrics import auprc, auroc


# TODO: make hyperparameter search space in config
def cnn_full_params_sharing_model(hp):
    with tf.device("/cpu:0"):
        inputs = Input(shape=(200,5))

        config_cnn = config['cnn_full_params_sharing']
        config_cnn_hp = config['cnn_full_params_sharing']['bayesian_opt']['hyperparameters']
        # hyper-parameters grid preparation
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

        predictions1 = Dense(1, activation='sigmoid')(x)
        predictions2 = Dense(1, activation='sigmoid')(x)
        predictions3 = Dense(1, activation='sigmoid')(x)
        predictions4 = Dense(1, activation='sigmoid')(x)

        cnn_model = Model(inputs, [predictions1, predictions2, predictions3, predictions4])
    if config['execution']['n_gpu'] > 1:
        cnn_model = multi_gpu_model(cnn_model, gpus=config['execution']['n_gpu'])

    nadam_opt = Nadam(lr=config_cnn['learning_rate'],
                      beta_1=config_cnn['beta_1'],
                      beta_2=config_cnn['beta_2'])
    cnn_model.compile(loss=['binary_crossentropy']*4,
                       optimizer=nadam_opt,
                       metrics=[auprc, auroc])

    return cnn_model


# TODO: Do some test
def hp_tuning_cnn_full_params_sharing(X_train, y_train, X_val, y_val):
    config_cnn_bayesian = config['cnn_full_params_sharing']['bayesian_opt']
    n_gpu = config['execution']['n_gpu']
    batch_size = config['cnn_full_params_sharing']['batch_size']
    batch_size_total = n_gpu * batch_size if n_gpu > 0 else batch_size

    tuner = BayesianOptimization(
        cnn_full_params_sharing_model,
        objective='val_loss', # TODO: binary_crossentropy
        max_trials=config_cnn_bayesian['max_trials'],
        directory='tuner_results',
        project_name='cnn_full_params_sharing')

    tuner.search(X_train, y_train,
                 epochs=config['cnn_full_params_sharing']['epochs'],
                 batch_size=batch_size_total,
                 validation_data=(X_val, y_val))

    return tuner.get_best_models(num_models=1)[0]


#def evaluate_model()

#def train_cnn_full_params_sharing(model, X_test, y_test):
#    model.fit(X_train, y_train,
#                 epochs=1000,
#                 callbacks=[es],
#                 validation_data=(X_val, y_val))
