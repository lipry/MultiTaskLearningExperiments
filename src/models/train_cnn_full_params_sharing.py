from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.utils import multi_gpu_model
from kerastuner import BayesianOptimization

from src.config.config import config
from src.metrics.metrics import auprc, auroc


# TODO: make hyperparameter search space in config
def cnn_full_params_sharing_model(hp):
    inputs = Input(shape=(200, 5)) # TODO: 5? Why?
    kernel_size1 = hp.Choice('kernel_size1', values=[5, 10])
    x = Conv1D(64, kernel_size=kernel_size1, activation='relu')(inputs)
    x = Conv1D(64, kernel_size=kernel_size1, activation='relu')(x)
    x = Conv1D(64, kernel_size=kernel_size1, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(hp.Choice('units2', values=[32, 64]),
               kernel_size=hp.Choice('kernel_size2', values=[5, 10]),
               activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)

    x = Dense(hp.Choice('dense1', values=[32, 64]), activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(hp.Choice('dense2', values=[32, 64]), activation='relu')(x)
    x = Dropout(0.1)(x)

    predictions1 = Dense(1, activation='sigmoid')(x)
    predictions2 = Dense(1, activation='sigmoid')(x)
    predictions3 = Dense(1, activation='sigmoid')(x)
    predictions4 = Dense(1, activation='sigmoid')(x)

    cnn_model = Model(inputs, [predictions1, predictions2, predictions3, predictions4])
    if config['execution']['gpu']:
        cnn_model = multi_gpu_model(cnn_model, gpus=config['execution']['n_gpu'])

    nadam_opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999)
    cnn_model.compile(loss=['binary_crossentropy']*4,
                           optimizer=nadam_opt,
                           metrics=[auprc, auroc])
    print(cnn_model)
    return cnn_model


# TODO: Do some test
def hp_tuning_cnn_full_params_sharing(X_train, y_train, X_val, y_val):
    tuner = BayesianOptimization(
        cnn_full_params_sharing_model,
        objective='val_loss',
        max_trials=10)

    es = EarlyStopping(monitor='val_loss',
                       patience=10,
                       min_delta=0.005,
                       baseline=0.2)
    tuner.search(X_train, y_train,
                 epochs=1000,
                 callbacks=[es],
                 validation_data=(X_val, y_val))

    return tuner.get_best_models(num_models=1)


#def evaluate_model()

#def train_cnn_full_params_sharing(model, X_test, y_test):
#    model.fit(X_train, y_train,
#                 epochs=1000,
#                 callbacks=[es],
#                 validation_data=(X_val, y_val))
