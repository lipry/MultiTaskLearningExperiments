import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def sequence2onehot(sequence):
    # TODO: speedup
    ltrdict = {'a': np.array([1, 0, 0, 0, 0]),
               'c': np.array([0, 1, 0, 0, 0]),
               'g': np.array([0, 0, 1, 0, 0]),
               't': np.array([0, 0, 0, 1, 0]),
               'n': np.array([0, 0, 0, 0, 1])
               }
    return np.array([ltrdict[x] for x in sequence.lower()])


def encoding_labels(y):
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    return y


def group_labels(y, labels, group_name):
    return [np.array([group_name if elem in labels else elem for elem in l]) for l in y]


def filter_labels(X, y, label_a, label_b):
    conditions = [all(label in [label_a, label_b] for label in elem) for elem in zip(*y)]
    indices = np.where(conditions)[0]
    return X[indices], [encoding_labels(l[indices]) for l in y]


def split_datasets(X, y, perc=0.3):
    assert all([len(l) == len(X) for l in y])

    indices = range(len(X))
    indices_train, indices_test = train_test_split(indices, test_size=perc, random_state=None) # random_state generate randomly

    # X_train, y_train, X_test, y_test
    return X[indices_train], [l[indices_train] for l in y], X[indices_test], [l[indices_test] for l in y]

