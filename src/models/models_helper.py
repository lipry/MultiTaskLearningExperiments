import time

from src.config.config import config
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner import BayesianOptimization


def hp_tuner(X_train, y_train, X_val, y_val, model_fun, exp_name, class_weight, n_best_models=1):
    config_bayesian = config[exp_name]['bayesian_opt']
    batch_size = config[exp_name]['batch_size']

    tuner = BayesianOptimization(
        model_fun,
        objective='val_loss',
        max_trials=config_bayesian['max_trials'],
        num_initial_points=config_bayesian['num_initial_points'],
        directory='tuner_results',
        project_name="{}_{}".format(exp_name, int(time.time())))

    es = EarlyStopping(monitor='val_loss', patience=config_bayesian['patience'],
                       min_delta=config_bayesian['min_delta'])

    tuner.search(X_train, y_train,
                 epochs=config[exp_name]['epochs'],
                 batch_size=batch_size,
                 callbacks=[es],
                 class_weight=class_weight,
                 validation_data=(X_val, y_val))

    return tuner, tuner.get_best_models(num_models=n_best_models), tuner.get_best_hyperparameters(num_trials=n_best_models)


def model_trainer(X_train, y_train, X_val, y_val, model_fun, exp_name, class_weight, training_hp):
    #batch_size_total = get_batch_size()
    model = model_fun(training_hp)
    history = model.fit(X_train, y_train,
             epochs=config[exp_name]['epochs'],
             class_weight=class_weight,
             batch_size=config[exp_name]['batch_size'],
             validation_data=(X_val, y_val))

    return model, history
