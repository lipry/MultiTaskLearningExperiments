from tensorflow.keras import Input, Model

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Nadam

from src.config.config import config
from src.metrics.metrics import auprc, auroc


def cnn_full_params_sharing_model(hp):
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

    inputs = Input(shape=(1000,5))
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

    predictions = [Dense(1, activation='sigmoid', name="pred_{}".format(c))(x)
                   for c in config['general']['cell_lines']]

    cnn_model = Model(inputs, predictions)

    nadam_opt = Nadam(lr=learning_rate,
                      beta_1=config_cnn['beta_1'],
                      beta_2=config_cnn['beta_2'])

    losses = {"pred_{}".format(c): 'binary_crossentropy' for c in config['general']['cell_lines']}
    cnn_model.compile(loss=losses,
                       optimizer=nadam_opt,
                       metrics=[auprc, auroc])

    return cnn_model
