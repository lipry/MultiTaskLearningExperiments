from src.datasets_utils import import_intersected_sequences, import_intersected_epigenetic, filter_labels

files_path = "../dati-ngs/data_ngs_all/intersected"

X, y = import_intersected_sequences(files_path)
X_filtered, y_filtered, num_sample = filter_labels(X, y, 'I-P', 'A-P')

