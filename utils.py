import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = np.array(idx_features_labels[:, 1:-1],dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    # adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = adj + sp.eye(adj.shape[0])

    idx_train = range(300)
    idx_val = range(300, 600)
    idx_test = range(600, 1500)

    adj=torch.FloatTensor(np.array(adj.todense()))
    edge_attr =[adj,adj.t(),adj+adj.t()]
    edge_attr=torch.stack(edge_attr,dim=0)
    edge_attr=DSN(edge_attr)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return edge_attr, features, labels, idx_train, idx_val, idx_test



def DSN2(t):
    a=t.sum(dim=1,keepdim=True)
    b=t.sum(dim=0,keepdim=True)
    lamb=torch.cat([a.squeeze(),b.squeeze()],dim=0).max()
    r=t.shape[0]*lamb-t.sum(dim=0).sum(dim=0)
    
    a=a.expand(-1,t.shape[1])
    b=b.expand(t.shape[0],-1)
    tt=t+(lamb**2-lamb*(a+b)+a*b)/r

    ttmatrix=tt/tt.sum(dim=0)[0]
    ttmatrix=torch.where(t>0,ttmatrix,t)
    return ttmatrix


def DSN(x):
    """Doubly stochastic normalization"""
    p=x.shape[0]
    y1=[]
    for i in range(p):
        y1.append(DSN2(x[i]))
    y1=torch.stack(y1,dim=0)
    return y1

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    """input is a numpy array""" 
    rowsum = mx.sum(axis=1)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

