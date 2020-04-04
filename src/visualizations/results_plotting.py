import time
import numpy as np
import matplotlib.pyplot as plt


def train_val_loss_plot(training_data, validation_data, title, path, exp_name, task):
    # Draw Plot
    plt.figure(figsize=(16,10), dpi= 80)
    plt.plot(training_data, color='tab:red', label="Training loss")
    plt.plot(validation_data, color='tab:blue', label="Validation loss")

    # Decoration
    plt.yticks(fontsize=12, alpha=.7)
    plt.title(title, fontsize=22)
    plt.grid(axis='both', alpha=.3)
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss Score', fontsize=15)

    # Remove borders
    plt.gca().spines["top"].set_alpha(0.0)
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.0)
    plt.gca().spines["left"].set_alpha(0.3)
    plt.legend(loc='upper left', prop={'size': 15})
    plt.savefig("{}/{}_{}_loss_{}".format(path, time.strftime("%Y%m%d-%H%M%S"), exp_name, task))


def au_plot(training, test, metric, path, exp_name, task, cell_lines):
    fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(15, 13))
    for cell_line, ax in zip(cell_lines, axs.reshape(-1)):
        ax.plot(training[cell_line], color='tab:red', label="Training {}".format(metric))
        ax.plot(test[cell_line], color='tab:blue', label="Validation {}".format(metric))

        ax.set_title("{} of {} ".format(metric, cell_line), fontsize=12)
        ax.grid(axis='both', alpha=.3)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_xlabel("Epochs", fontsize=10)
        ax.spines["top"].set_alpha(0.0)
        ax.spines["bottom"].set_alpha(0.3)
        ax.spines["right"].set_alpha(0.0)
        ax.spines["left"].set_alpha(0.3)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, prop={'size': 13})
    plt.savefig("{}/{}_{}_{}_{}".format(path, time.strftime("%Y%m%d-%H%M%S"), exp_name, metric, task))


def evaluation_performance_plot(results, std_dev, title, path, exp_name, metric):
    bar_width = 0.12
    plt.figure(figsize=(16, 10))

    # set height of bar
    values = [[x for x in v.values()] for v in results.values()]
    stdev = [[x for x in v.values()] for v in std_dev.values()]
    colors = ['#1C6B0A', '#E4D6A7', '#44A1A0', '#F9E04C', '#9B2915']
    legend_labels = results.keys()
    x_labels = list(results.values())[0].keys()

    # Make the plot
    x = np.arange(len(values[0]))
    for w, v, sd, c, l in zip(np.arange(-2, 3), values, stdev, colors, legend_labels):
        plt.bar(x + w * bar_width, v, bar_width, yerr=sd, capsize=6, alpha=0.90, color=c, label=l)

    # Add xticks on the middle of the group bars
    plt.title(title, fontsize=22)
    plt.grid(axis='both', alpha=.3)
    plt.ylim(0.0, 1.01)
    plt.xlabel('Cell Lines', fontsize=15)
    plt.ylabel('Score', fontsize=15)
    plt.xticks(x, x_labels)

    # Create legend & show graphic
    plt.legend(loc='lower left', prop={'size': 10})
    plt.savefig("{}/{}_{}_{}".format(path, time.strftime("%Y%m%d-%H%M%S"), exp_name, metric))
