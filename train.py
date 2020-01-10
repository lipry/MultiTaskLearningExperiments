from src.datasets_utils import import_intersected_sequences, import_intersected_epigenetic

files_path = "../dati-ngs/data_ngs_all/intersected"

X, y = import_intersected_epigenetic(files_path)

print(X[0])
print(len(X), len(y[0]))
