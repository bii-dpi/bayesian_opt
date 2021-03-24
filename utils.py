import numpy as np
import torch
from scipy.spatial import distance_matrix
from sklearn.preprocessing import normalize

from os import listdir
from os.path import isfile, join

from operator import itemgetter


element_dict = {
        "H": 0,
        "Br": 1,
        "C": 2,
        "Cl": 3,
        "F": 4,
        "I": 5,
        "N": 6,
        "O": 7,
        "P": 8,
        "S": 9,
        "Si": 10,
    }

conversion_mat = np.eye(len(element_dict))
element_dict = {key: conversion_mat[value, :] for key, value in element_dict.items()}


def slice_list(lst, indices):
    return itemgetter(*indices)(lst)


def load_ligand(fpath, threshold):
    with open(fpath, "r") as f:
        text = f.read()
    text = text.split("@<TRIPOS>ATOM\n")[1]
    text = text.split("@<TRIPOS>BOND\n")[0]
    text = [line.split() for line in text.split("\n")][:-1]
    features = np.array([element_dict[(line[5]).split(".")[0]]for line in text], dtype=np.float32)
    coords = np.array([[line[2], line[3], line[4]] for line in text], dtype=np.float32)
    adj = distance_matrix(coords, coords)
    np.fill_diagonal(adj, 1)
    adj = 1 / adj
    adj[adj <= 1 / threshold] = 0
    
    return torch.FloatTensor(features), \
        torch.FloatTensor(np.array(normalize(adj, axis=1, norm="l1"), dtype=np.float32))


def load_all_ligands(threshold=3):
    all_active_paths = [join("data/actives", f) for f in listdir("data/actives") if isfile(join("data/actives", f))]
    all_decoy_paths = [join("data/decoys", f) for f in listdir("data/decoys") if isfile(join("data/decoys", f))]
    
    features_list = []
    adj_list = []
    labels = []
    for active_path in all_active_paths:
        output = load_ligand(active_path, threshold)
        features_list.append(output[0])
        adj_list.append(output[1])
        labels.append(torch.FloatTensor([1]))
    
    for decoy_path in all_decoy_paths:
        output = load_ligand(active_path, threshold)
        features_list.append(output[0])
        adj_list.append(output[1])
        labels.append(torch.FloatTensor([1]))

    np.random.seed(12345)
    train_indices = np.random.choice(range(len(labels)), int(0.8 * len(labels)), replace=False)
    validation_indices = [i for i in range(len(labels)) if i not in train_indices]
    

    return (slice_list(features_list, train_indices),
            slice_list(features_list, validation_indices),
            slice_list(adj_list, train_indices),
            slice_list(adj_list, validation_indices),
            slice_list(labels, train_indices),
            slice_list(labels, validation_indices))


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
