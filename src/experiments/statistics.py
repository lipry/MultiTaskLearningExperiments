import numpy as np
from src.config.config import config
from src.data.datasets_helper import group_labels, filter_labels
from src.visualizations.results_export import save_metrics

def bool2int(x: object) -> object:
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y

def statistics_executor(X, y, logger, path_logs):
    logger.debug("STATISTICS: ")
    tasks = config['general']['tasks']

    for t in tasks:
        task_name = "{}vs{}".format(t[0], t[1])
        logger.debug("NEW EXPERIMENT: {}".format(task_name))

        if t[0] == 'A-E+A-P':
            y = group_labels(y, ['A-E', 'A-P'], 'A-E+A-P')

        if t[0] == 'BG':
            y = group_labels(y, ["I-E", "I-P", "UK", "A-X", "I-X"], 'BG')

        if t[1] == 'A-E+A-P':
            y = group_labels(y, ['A-E', 'A-P'], 'A-E+A-P')

        if t[1] == 'BG':
            y = group_labels(y, ["I-E", "I-P", "UK", "A-X", "I-X"], 'BG')

        _, y_filtered = filter_labels(X, y, t[0], t[1], logger)
        stacked = np.column_stack((y_filtered[0], y_filtered[1], y_filtered[2], y_filtered[3]))

        logger.debug("Task {}".format(task_name))
        logger.debug("Number of items for task: {}".format(len(stacked)))

        unique, counts = np.unique(stacked, axis=0, return_counts=True)

        for k, v in zip(unique, counts):
            logger.debug("{} - {}: {}".format(k, bool2int(k[::-1]), v))

        to_save = {"uniques": unique, "counts": counts}
        save_metrics(path_logs, "stats", task_name, to_save)






