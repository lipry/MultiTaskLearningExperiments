import numpy as np

from src.config.config import config
from src.visualizations.results_export import save_metrics
from src.visualizations.results_helper import get_holdouts_mean, apply_fun_to_items, mean_list

cell_lines = config['general']['cell_lines']

class ResultsCollector:
    def __init__(self):
        # Loss
        self.losses = None
        self.val_losses = None

        # Auprc
        self.auprc = None
        self.val_auprc = None

        # Auroc
        self.auroc = None
        self.val_auroc = None

        # Evaluations
        self.eval_auprc = None
        self.eval_auroc = None


    def init_metrics(self):
        self.losses = []
        self.val_losses = []
        self.auprc = {c: [] for c in cell_lines}
        self.val_auprc = {c: [] for c in cell_lines}
        self.auroc = {c: [] for c in cell_lines}
        self.val_auroc = {c: [] for c in cell_lines}

    def init_eval_metrics(self):
        self.eval_auprc = {}
        self.eval_auroc = {}
        #{c: [] for c in cell_lines}

    def add_holdout_results(self, history):
        self.losses.append(history.history['loss'])
        self.val_losses.append(history.history['val_loss'])

        for pred_idx, cl in enumerate(cell_lines):
            self.auprc[cl].append(history.history['pred{}_auprc'.format(pred_idx)])
            self.val_auprc[cl].append(history.history['val_pred{}_auprc'.format(pred_idx)])
            self.auroc[cl].append(history.history['pred{}_auroc'.format(pred_idx)])
            self.val_auroc[cl].append(history.history['val_pred{}_auroc'.format(pred_idx)])

    def add_holdouts_eval(self, eval_metrics, task):
        try:
            for pred_idx, cl in enumerate(cell_lines):
                self.eval_auprc[task][cl].append(eval_metrics['pred{}_auprc'.format(pred_idx)])
                self.eval_auroc[task][cl].append(eval_metrics['pred{}_auroc'.format(pred_idx)])
        except KeyError:
            self.eval_auprc[task] = {c: [] for c in cell_lines}
            self.eval_auroc[task] = {c: [] for c in cell_lines}

    def get_losses_mean(self):
        return get_holdouts_mean(self.losses, self.val_losses)

    def get_auprc_mean(self):
        auprc_avg = apply_fun_to_items(self.auprc, mean_list)
        auprc_val = apply_fun_to_items(self.val_auprc, mean_list)

        return auprc_avg, auprc_val

    def get_auroc_mean(self):
        auroc_avg = apply_fun_to_items(self.auroc, mean_list)
        auroc_val = apply_fun_to_items(self.val_auroc, mean_list)

        return auroc_avg, auroc_val

    def get_eval_auprc(self):
        #{'A-PvsI-P': {'K562': [0.49944785, 0.50965005], 'GM12878': [0.500813, 0.5131617],
        #              'HepG2': [0.49964345, 0.50716966], 'HelaS3': [0.4938996, 0.5052579]},
        # 'A-EvsA-P': {'K562': [0.50926554, 0.53062683], 'GM12878': [0.50936884, 0.5050544],
        #              'HepG2': [0.50877047, 0.5054021], 'HelaS3': [0.51233226, 0.511159]}}
        auprc_eval_mean = apply_fun_to_items(self.eval_auprc, lambda x: {k: sum(y)/len(y) for k, y in x.items()})
        auprc_eval_std = apply_fun_to_items(self.eval_auprc, lambda x: {k: np.std(y) for k, y in x.items()})

        return auprc_eval_mean, auprc_eval_std

    def get_eval_auroc(self):
        auroc_eval_mean = apply_fun_to_items(self.eval_auroc, lambda x: {k: sum(y)/len(y) for k, y in x.items()})
        auroc_eval_std = apply_fun_to_items(self.eval_auroc, lambda x: {k: np.std(y) for k, y in x.items()})

        return auroc_eval_mean, auroc_eval_std

    def save_metrics(self, path, exp_name, task):
        to_save = {"losses": self.losses,
                   "val_losses": self.val_losses,
                   "auprc": self.auprc,
                   "val_auprc": self.val_auprc,
                   "auroc": self.auroc,
                   "val_auroc": self.val_auroc,
                   "eval_auprc": self.eval_auprc,
                   "eval_auroc": self.eval_auroc}

        save_metrics(path, exp_name, task, to_save)


