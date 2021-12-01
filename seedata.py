# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 16:29:20 2021

@author: James
"""

import numpy as np
import scipy.sparse as sp
import torch
from utils import*


path="./data/cora/"
dataset="cora"

"""Load citation network dataset (cora only for now)"""
print('Loading {} dataset...'.format(dataset))

idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
# features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
features = idx_features_labels[:, 1:-1]

#%%
labels = encode_onehot(idx_features_labels[:, -1])

# build graph
idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
idx_map = {j: i for i, j in enumerate(idx)}#建立非标准的node命名和标准的[0,1,...]命名之间的映射关系
edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
#将非标准的node命名映射成标准的[0,1,...]。array的flatten和reshape默认都是对每一行操作，互为可逆操作。
edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
adj = sp.coo_matrix(
                    (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(labels.shape[0], labels.shape[0]),
                    dtype=np.float32)

# build symmetric adjacency matrix
#in order to process directed graph don't need to do this
# adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

features = normalize_features(features)
adj = normalize_adj(adj + sp.eye(adj.shape[0]))

idx_train = range(140)
idx_val = range(200, 500)
idx_test = range(500, 1500)

adj = torch.FloatTensor(np.array(adj.todense()))
features = torch.FloatTensor(np.array(features.todense()))
labels = torch.LongTensor(np.where(labels)[1])

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

