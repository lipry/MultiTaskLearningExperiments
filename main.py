import logging
import time
import os

from src.config.config import config
from src.data.datasets_import import import_intersected_sequences
from src.data.datasets_helper import filter_labels, split_datasets, group_labels
from src.models.train_cnn_full_params_sharing import hp_tuning_cnn_full_params_sharing
from src.visualizations.results_export import save_metrics, copy_experiment_configuration

# get configuration file
files_path = config['data']['data_path']
holdouts = config['general']['n_holdouts']
tasks = config['tasks']
logs_path = config['data']['logs_path']

path_logs = "{}/{}_fps_experiment".format(logs_path, time.strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(path_logs):
    os.makedirs(path_logs)

# configuring stream and file logger
logger = logging.getLogger(__name__)
c_handler = logging.StreamHandler()

f_handler = logging.FileHandler('{}/{}_fps_experiment.log'.format(path_logs, time.strftime("%Y%m%d-%H%M%S")))
c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
logger.addHandler(c_handler)
logger.addHandler(f_handler)
logger.setLevel(logging.DEBUG)

# TODO: RISOLVERE WARNINGS
print("Importing data...")
X, y = import_intersected_sequences(files_path)

for t in tasks:

    # Grouping particular labels for some tasks
    if t[0] == 'A-E+A-P':
        y = group_labels(y, ['A-E', 'A-P'], 'A-E+A-P')

    if t[0] == 'BG':
        y = group_labels(y, ["I-E", "I-P", "UK", "A-X", "I-X"], 'BG')

    if t[1] == 'A-E+A-P':
        y = group_labels(y, ['A-E', 'A-P'], 'A-E+A-P')

    if t[1] == 'BG':
        y = group_labels(y, ["I-E", "I-P", "UK", "A-X", "I-X"], 'BG')

    X_filtered, y_filtered = filter_labels(X, y, t[0], t[1])
    # print(y_filtered[0][0:100], type(y_filtered[0]))
    metrics = {'losses': [], 'auprc': [], 'auroc': []}
    for h in range(holdouts):
        logger.debug("{}/{} holdouts training started".format(h, holdouts))
        # splitting train/test from the data
        X_train, y_train, X_test, y_test = split_datasets(X_filtered, y_filtered, perc=0.3)

        X_train_int, y_train_int, X_val, y_val = split_datasets(X_train, y_train, perc=0.3)

        logger.debug("Tuning hyper-parameters of {}/{} holdouts".format(h, holdouts))
        best_models = hp_tuning_cnn_full_params_sharing(X_train_int, y_train_int, X_val, y_val)

        eval_score = best_models.evaluate(X_test, y_test)

        metrics['losses'].append(eval_score[0])
        metrics['auprc'].append(eval_score[1])
        metrics['auroc'].append(eval_score[2])

    # save metrics at the end of every tasks executions
    save_metrics(path_logs, "fps", "{}_{}".format(t[0], t[1]), metrics)
copy_experiment_configuration(path_logs)


