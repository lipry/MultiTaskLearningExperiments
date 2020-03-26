import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


def sequence2onehot(sequence):
    # TODO: speedup
    ltrdict = {'a': np.array([1, 0, 0, 0, 0]),
               'c': np.array([0, 1, 0, 0, 0]),
               'g': np.array([0, 0, 1, 0, 0]),
               't': np.array([0, 0, 0, 1, 0]),
               'n': np.array([0, 0, 0, 0, 1])
               }
    return np.array([ltrdict[x] for x in sequence.lower()])


def encoding_labels(y, t):
    encoder = np.vectorize(lambda label: t[label])
    return encoder(y)


def group_labels(y, labels, group_name):
    return [np.array([group_name if elem in labels else elem for elem in l]) for l in y]


def filter_labels(X, y, label_a, label_b, t):
    conditions = [all(label in [label_a, label_b] for label in elem) for elem in zip(*y)]
    indices = np.where(conditions)[0]
    return [x[indices] for x in X], [encoding_labels(l[indices], t) for l in y]


def calculate_class_weights(labels):
    get_weights = lambda y: class_weight.compute_class_weight('balanced', np.unique(y), y)
    return [get_weights(lab) for lab in labels]


def split_datasets(X, y, perc=0.3):
    assert all([len(l) == len(x) for x, l in zip(X, y)])

    indices = range(len(X[0]))
    indices_train, indices_test = train_test_split(indices, test_size=perc, random_state=None) # random_state generate randomly

    # X_train, y_train, X_test, y_test
    return [x[indices_train] for x in X], [l[indices_train] for l in y], \
           [x[indices_test] for x in X], [l[indices_test] for l in y]

