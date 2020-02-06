from src.config.config import config
from src.data.datasets_helper import group_labels, filter_labels, split_datasets
from src.models.train_cnn_full_params_sharing import hp_tuning_cnn_full_params_sharing, train_cnn_full_params_sharing
from src.visualizations.ResultsCollector import ResultsCollector
from src.visualizations.results_export import copy_experiment_configuration
from src.visualizations.results_helper import hyperparametrs2str
from src.visualizations.results_plotting import train_val_loss_plot, au_plot, evaluation_performance_plot

def fps_executor(X, y, logger, path_logs):
    holdouts = config['general']['n_holdouts']
    tasks = config['general']['tasks']

    results = ResultsCollector()
    results.init_eval_metrics()
    for t in tasks:
        task_name = "{}vs{}".format(t[0], t[1])
        logger.debug("NEW EXPERIMENT: {}".format(task_name))
        # TODO: put in some general function
        # Grouping particular labels for some tasks
        if t[0] == 'A-E+A-P':
            y = group_labels(y, ['A-E', 'A-P'], 'A-E+A-P')

        if t[0] == 'BG':
            y = group_labels(y, ["I-E", "I-P", "UK", "A-X", "I-X"], 'BG')

        if t[1] == 'A-E+A-P':
            y = group_labels(y, ['A-E', 'A-P'], 'A-E+A-P')

        if t[1] == 'BG':
            y = group_labels(y, ["I-E", "I-P", "UK", "A-X", "I-X"], 'BG')

        X_filtered, y_filtered = filter_labels(X, y, t[0], t[1], logger)

        results.init_metrics()
        for h in range(holdouts):
            logger.debug("{}/{} holdouts started".format(h + 1, holdouts))
            # splitting train/test from the data
            X_train, y_train, X_test, y_test = split_datasets(X_filtered, y_filtered, perc=0.3)

            X_train_int, y_train_int, X_val, y_val = split_datasets(X_train, y_train, perc=0.3)

            if h == 0:
                logger.debug("Datasets size - training (internal): {}, validation: {}, test: {}"
                             .format(len(X_train_int), len(X_val), len(X_test)))

            logger.debug("Tuning hyper-parameters ({}/{} holdout)".format(h + 1, holdouts))
            tuner, _, best_hyperparams = hp_tuning_cnn_full_params_sharing(X_train_int, y_train_int, X_val, y_val, 1)

            logger.debug("Best hyperparams found: {}".format(hyperparametrs2str(tuner)))

            # Retraining model with best hyperparameters found
            logger.debug("Training model with best hyperparameters ({}/{} holdout)".format(h + 1, holdouts))
            best_model, history = train_cnn_full_params_sharing(X_train, y_train, X_test, y_test, best_hyperparams[0])
            results.add_holdout_results(history)

            # Evaluating best model performances
            eval_score = best_model.evaluate(X_test, y_test)
            eval_score = {k: v for k, v in zip(best_model.metrics_names, eval_score)}

            logger.debug("eval_score: {}".format(eval_score))
            results.add_holdouts_eval(eval_score, task_name)

        # plotting at the end of every execution:
        # - losses
        losses_avg, val_losses_avg = results.get_losses_mean()
        train_val_loss_plot(losses_avg, val_losses_avg, "Training and Validation Loss", path_logs,
                            "fps", task_name)

        # - auprc
        auprc_avg, val_auprc_avg = results.get_auprc_mean()
        au_plot(auprc_avg, val_auprc_avg, "AUPRC", path_logs, "fps", task_name)

        # - auroc
        auroc_avg, val_auroc_avg = results.get_auroc_mean()
        au_plot(auroc_avg, val_auroc_avg, "AUROC", path_logs, "fps", task_name)

        # save metrics at the end of every tasks executions
        results.save_metrics(path_logs, "fps_metrics", task_name)

    auprc_eval_mean, auprc_eval_std = results.get_eval_auprc()
    evaluation_performance_plot(auprc_eval_mean, auprc_eval_std, "Evaluation AUPRC", path_logs, "auprc_eval", "auprc")

    auroc_eval_mean, auroc_eval_std = results.get_eval_auroc()
    evaluation_performance_plot(auroc_eval_mean, auroc_eval_std, "Evaluation AUROC", path_logs, "auroc_eval", "auroc")

    copy_experiment_configuration(path_logs)


