from src.config.config import config
from src.config.config_utils import get_task_labels
from src.data.datasets_helper import group_labels, filter_labels
from src.experiments.experiments_helper import check_input_type
from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np

import multiprocessing

from src.visualizations.results_export import save_tsne
from src.visualizations.results_plotting import plot_tsne


def tsne_executor(X, y, logger, path_logs):
    check_input_type(['epi'], "t-SNE experiment work just with epigenetic data, {} found"
                     .format(config['general']['input_type']))

    logger.debug("")

    cell_lines = config['general']['cell_lines']

    tasks_dict = config['general']['tasks']

    results = {}
    for t in tasks_dict:
        t_labels = get_task_labels(t)
        task_name = "{}vs{}".format(t_labels[0], t_labels[1])

        logger.debug("TASK: {}".format(task_name))
        # TODO: put in some general function
        # Grouping particular labels for some tasks
        if t_labels[0] == 'A-E+A-P':
            y = group_labels(y, ['A-E', 'A-P'], 'A-E+A-P')

        if t_labels[0] == 'BG':
            y = group_labels(y, ["I-E", "I-P", "UK", "A-X", "I-X"], 'BG')

        if t_labels[1] == 'A-E+A-P':
            y = group_labels(y, ['A-E', 'A-P'], 'A-E+A-P')

        if t_labels[1] == 'BG':
            y = group_labels(y, ["I-E", "I-P", "UK", "A-X", "I-X"], 'BG')

        logger.debug("X: {}, y: {}".format(X, y))
        X_filtered, y_filtered = filter_labels(X, y, t_labels[0], t_labels[1], t)

        cpus = multiprocessing.cpu_count() // 2  # we use just half of avaible cpus to not overload the machine
        logger.debug("Using {} cpus".format(cpus))

        for cl, data, labels in zip(cell_lines, X_filtered, y_filtered):
            logger.debug("Computing t-SNE for {}".format(cl))

            tsne = TSNE(perplexity = config['tsne']['perplexity'], n_jobs = cpus) # TODO: add parameters
            tsne_results = tsne.fit_transform(data)
            assert len(tsne_results) == len(labels)
            tsne_results = np.c_[tsne_results, labels] # to save the labels with the tsne results
            logger.debug("results {}".format(tsne_results))

            results["{}_{}".format(task_name, cl)] = tsne_results

    save_tsne(path_logs, "tsne_results", results)
    if config['tsne']['save_plots']:
        plot_tsne(results, path_logs, "tsne_plot")

