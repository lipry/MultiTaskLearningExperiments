from sklearn.model_selection import train_test_split

from src.config.config import config
from src.data.datasets_import import import_intersected_sequences
from src.data.datasets_helper import encoding_labels, filter_labels, split_datasets
from src.models.train_cnn_full_params_sharing import hp_tuning_cnn_full_params_sharing


files_path = config['data']['data_path']
# TODO: dockerize
# TODO: RISOLVERE WARNINGS
print("Importing data...")
X, y = import_intersected_sequences(files_path)
X_filtered, y_filtered = filter_labels(X, y, 'I-P', 'A-P')
y_filtered = [encoding_labels(y) for y in y_filtered] # TODO: temporary fix

X_train, y_train, X_test, y_test = split_datasets(X_filtered, y_filtered, perc=0.3, seed=42)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

print("Training...")
hp_tuning_cnn_full_params_sharing(X_train, y_train, X_test, y_test)


