from src.config.config import config
from src.data.datasets_helper import calculate_class_weights, filter_labels, split_datasets
from src.experiments.experiments_helper import check_input_type


# def mlp_executor(X, y, logger, path_logs):
#     check_input_type(['epi'], "Cnn full parameter sharing models work just with epigenomic data, {} found"
#                      .format(config['general']['input_type']))
#
#     holdouts = config['general']['n_holdouts']
#     tasks_dict = config['general']['tasks']
#
#     for t in tasks_dict:
#         task_name, X_filtered, y_filtered = filter_labels(X, y, t)
#         logger.debug("NEW EXPERIMENT: {}".format(task_name))
#
#         weight_class = calculate_class_weights(y_filtered)
#         for h in range(holdouts):
#             logger.debug("{}/{} holdouts started".format(h + 1, holdouts))
#             # splitting train/test from the data
#             X_train, y_train, X_test, y_test = split_datasets(X_filtered, y_filtered, perc=0.3)
#
#             X_train_int, y_train_int, X_val, y_val = split_datasets(X_train, y_train, perc=0.3)
#
#             if h == 0:
#                 logger.debug("Datasets size - training (internal): {}, validation: {}, test: {}"
#                              .format(len(X_train_int[0]), len(X_val[0]), len(X_test[0])))
