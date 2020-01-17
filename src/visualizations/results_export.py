import shutil
import time
import numpy as np

def save_metrics(path, exp_name, task, metrics):
    np.save("{}/{}_{}_{}".format(path, time.strftime("%Y%m%d-%H%M%S"), exp_name, task), metrics)

def copy_experiment_configuration(path):
    dest = "{}/{}_fps_config.yaml".format(path, time.strftime("%Y%m%d-%H%M%S"))
    shutil.copy("config/config.yaml", dest)

