import logging
import time
import os
from src.config.config import config
from src.data.datasets_import import import_intersected_sequences
from src.experiments.full_parameters_sharing import fps_executor
from src.experiments.statistics import statistics_executor


# get configurations
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
X, y = import_intersected_sequences(files_path)

if exp == "fps":
    fps_executor(X, y, logger, path_logs)

if exp == "stats":
    statistics_executor(X, y, logger, path_logs)


