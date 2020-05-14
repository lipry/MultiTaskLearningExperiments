import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight

from src.config.config import config
from src.config.config_utils import get_task_labels


def sequence2onehot(sequence, mapping):
    return np.eye(5)[[mapping[i] for i in sequence.lower()]]


def pad_sequence(sequence, max_len):
    if sequence.shape[0] < max_len:
        padding = max_len - sequence.shape[0]
        return np.pad(sequence, [(0, padding), (0, 0)], 'constant')
    return sequence


def encoding_labels(y, t):
    encoder = np.vectorize(lambda label: t[label])
    return encoder(y)


def group_labels(y, labels, group_name):
    return [np.array([group_name if elem in labels else elem for elem in l]) for l in y]


def filter_labels(X, y, t):
    t_labels = get_task_labels(t)
    task_name = "{}vs{}".format(t_labels[0], t_labels[1])

    if t_labels[0] == 'A-E+A-P':
        y = group_labels(y, ['A-E', 'A-P'], 'A-E+A-P')

    if t_labels[0] == 'BG':
        y = group_labels(y, ["I-E", "I-P", "UK", "A-X", "I-X"], 'BG')

    if t_labels[1] == 'A-E+A-P':
        y = group_labels(y, ['A-E', 'A-P'], 'A-E+A-P')

    if t_labels[1] == 'BG':
        y = group_labels(y, ["I-E", "I-P", "UK", "A-X", "I-X"], 'BG')

    conditions = [all(label in [t_labels[0], t_labels[1]] for label in elem) for elem in zip(*y)]
    indices = np.where(conditions)[0]
    return task_name, [x[indices] for x in X], [encoding_labels(l[indices], t) for l in y]


def calculate_class_weights(labels):
    get_weights_dict = lambda y: dict(zip(np.unique(y), class_weight.compute_class_weight('balanced', np.unique(y), y)))
    return {"pred_{}".format(c): get_weights_dict(lab) for c, lab in zip(config['general']['cell_lines'], labels)}


def split_datasets(X, y, perc=0.3):
    assert all([len(l) == len(x) for x, l in zip(X, y)])

    indices = range(len(X[0]))
    indices_train, indices_test = train_test_split(indices, test_size=perc, random_state=None) # random_state generate randomly

    # X_train, y_train, X_test, y_test
    return [x[indices_train] for x in X], [l[indices_train] for l in y], \
           [x[indices_test] for x in X], [l[indices_test] for l in y]

def min_max_scaling(X):
    scaler = MinMaxScaler((0, 1))
    return [scaler.fit_transform(x) for x in X]

def downsample_data(X, y, max_size_given, balancing_index = 1):
    y_balance = y[balancing_index]

    if max_size_given < 0:
        raise ValueError("max_size_given must be greater than 0")
    u, indices = np.unique(y_balance, return_inverse=True)
    num_u = len(u)
    sample_sizes = np.bincount(indices)

    size_min = np.amin(sample_sizes)

    if size_min < max_size_given:
        max_size_given = size_min
    sample_sizes[sample_sizes > max_size_given] = max_size_given

    indices_all = get_indices(indices, sample_sizes, num_u)
    X = [x[indices_all, :] for x in X]
    y = [l[indices_all] for l in y]

    return X, y

def get_indices(indices, sample_sizes, n_classes, replace=False):
    indices_range = np.arange(len(indices))
    indices_all = np.concatenate([np.random.choice(indices_range[indices == i],
                                                   size=sample_sizes[i], replace=replace) for i in range(n_classes)])

    return indices_all

