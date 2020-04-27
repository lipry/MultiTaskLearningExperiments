import contextlib
import io


def mean_list(X):
    assert all(len(x) == len(X[0]) for x in X)
    return [sum(x)/len(x) for x in zip(*X)]

def get_holdouts_mean(X, X_val):
    return mean_list(X), mean_list(X_val)

def apply_fun_to_items(X, f):
    return {cell_line: f(values) for cell_line, values in X.items()}

# def hyperparametrs2str(tuner, max_trials=4):
#     f = io.StringIO()
#     with contextlib.redirect_stdout(f):
#         tuner.results_summary(max_trials)
#     return f.getvalue()

def hyperparametrs2str(hyperparams):
    str = "BEST {} MODELS FOUND:\n".format(len(hyperparams))
    for hp in hyperparams:
        str+= "{}\n".format(hp.values)
    return str

