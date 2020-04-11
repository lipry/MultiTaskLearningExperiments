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

    cell_lines = config['general']['cell_lines']

    tasks_dict = config['general']['tasks']

    results = {}
    for t in tasks_dict:
        task_name, X_filtered, y_filtered = filter_labels(X, y, t)
        logger.debug("TASK: {}".format(task_name))

        cpus = multiprocessing.cpu_count() // 2  # we use just half of avaible cpus to not overload the machine
        logger.debug("Using {} cpus".format(cpus))

        for cl, data, labels in zip(cell_lines, X_filtered, y_filtered):
            logger.debug("Computing t-SNE for {}".format(cl))

            tsne = TSNE(perplexity = config['tsne']['perplexity'], n_jobs = cpus) # TODO: add parameters
            tsne_results = tsne.fit_transform(data)
            assert len(tsne_results) == len(labels)
            tsne_results = np.c_[tsne_results, labels] # to save the labels with the tsne results
            results["{}_{}".format(task_name, cl)] = tsne_results

    save_tsne(path_logs, "tsne_results", results)
    if config['tsne']['save_plots']:
        plot_tsne(results, path_logs, "tsne_plot")

