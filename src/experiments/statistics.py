import numpy as np
from src.config.config import config
from src.config.config_utils import get_task_labels
from src.data.datasets_helper import group_labels, filter_labels
from src.visualizations.results_export import save_metrics

def bool2int(x: object) -> object:
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y

def statistics_executor(X, y, logger, path_logs):
    logger.debug("STATISTICS: ")
    tasks_dict = config['general']['tasks']
    cell_lines = config['general']['cell_lines']
    #tasks = get_task_list(config['general']['tasks'])
    logger.debug("NUMBER OF FEATURE FOR EVERY CELL LINES:")
    for cl_idx, cl in enumerate(cell_lines):
        logger.debug("{}: {}".format(cl, len(X[cl_idx][0])))

    for t in tasks_dict:
        task_name, _, y_filtered = filter_labels(X, y, t)
        logger.debug("NEW EXPERIMENT: {}".format(task_name))
        stacked = np.column_stack((y_filtered[0], y_filtered[1], y_filtered[2], y_filtered[3]))

        logger.debug("Task {}".format(task_name))
        logger.debug("Number of items for task: {}".format(len(stacked)))

        unique, counts = np.unique(stacked, axis=0, return_counts=True)

        for k, v in zip(unique, counts):
            logger.debug("{} - {}: {}".format(k, bool2int(k[::-1]), v))

        to_save = {"uniques": unique, "counts": counts}
        save_metrics(path_logs, "stats", task_name, to_save)






