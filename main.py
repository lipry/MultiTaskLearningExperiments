import logging
import time
import os
from functools import partial

from src.config.config import config
from src.data.datasets_import import input_data
from src.experiments.experiments_helper import holdouts_experiments_executor, check_input_type
from src.experiments.statistics import statistics_executor
from src.experiments.tsne import tsne_executor
from src.models.cnn_full_params_sharing_model import cnn_full_params_sharing_model
from src.models.mlp_model import mlp_model


files_path = config['data']['data_path']
logs_path = config['data']['logs_path']
exp = config['general']['experiment']

path_logs = "{}/{}_{}_experiment".format(logs_path, time.strftime("%Y%m%d-%H%M%S"), exp)
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

logger.debug("Importing data...")


X, y = input_data(files_path, input_type=config['general']['input_type'])

if exp == "fps":
    check_input_type(['seq'], "Cnn full parameter sharing models work just with sequence data, {} found"
                     .format(config['general']['input_type']))

    holdouts_experiments_executor("cnn_full_params_sharing", X, y, logger, path_logs, cnn_full_params_sharing_model, "seq")

if exp == "mlp":
    check_input_type(['epi'], "Multi Layer Perceptron model work just with epigenomic data, {} found"
                     .format(config['general']['input_type']))

    input_dims = [len(x[0]) for x in X]
    mlp_model_red = partial(mlp_model, input_dims)

    holdouts_experiments_executor("multi_layers_perceptron", X, y, logger, path_logs, mlp_model_red, "epi")

if exp == "stats":
    statistics_executor(X, y, logger, path_logs)

if exp == "tsne":
    tsne_executor(X, y, logger, path_logs)


