import os

import numpy as np
from Bio import SeqIO

from src.config.config import config
from src.data.datasets_helper import sequence2onehot

cell_lines = config['general']['cell_lines']

# TODO: needed?
#def check_cell_line(cell_line):
#    if cell_line not in cell_lines:
#        raise ValueError("Illegal cell line.")


def check_file_exist(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("file {} not found".format(file_path))


def import_epigenetic_data(files_path, file_name):
    epigenetic_data_path = "{}/{}".format(files_path, file_name)
    check_file_exist(epigenetic_data_path)

    epigenetic_data = np.loadtxt(epigenetic_data_path)

    return epigenetic_data


def import_sequences(files_path, file_name):
    seqences_path = "{}/{}".format(files_path, file_name)
    check_file_exist(seqences_path)

    with open(seqences_path) as f:
        sequences = [sequence2onehot(str(s.seq)) for s in SeqIO.parse(f, 'fasta')]

    return sequences


def import_labels(files_path, file_name):
    labels_path = "{}/{}".format(files_path, file_name)
    check_file_exist(labels_path)

    with open(labels_path, "r") as f:
        labels = np.array([line.strip() for line in f.readlines()])

    return labels


def import_intersected_labels(files_path):
    return [import_labels(files_path, "{}_labels_intersected.txt".format(line_name)) for line_name in cell_lines]


def import_intersected_sequences(files_path):
    sequences = import_sequences(files_path, "sequences_intersected.fa")
    labels_list = import_intersected_labels(files_path)

    assert all(len(sequences) == len(labels_list[i]) for i in range(len(labels_list)))
    assert len(labels_list) == len(cell_lines)

    return np.array(sequences), labels_list


def import_intersected_epigenetic(files_path):
    epigenetic_list = [import_epigenetic_data(files_path, "{}_epigenetic_intersected.txt".format(line_name))
                       for line_name in cell_lines]
    labels_list = import_intersected_labels(files_path)

    assert len(epigenetic_list) == len(labels_list) == len(cell_lines)
    assert all(len(epigenetic_list[i]) == len(labels_list[i]) for i in range(len(labels_list)))

    return epigenetic_list, labels_list

