import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DSN


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, edge_attr):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh, edge_attr)
        e=e*edge_attr
        zero_vec = -9e15*torch.ones_like(e)
        e = torch.where(edge_attr > 0, e, zero_vec)
        e=F.softmax(e, dim=1)
        #e=torch.exp(e)
        
        #e=DSN(e)
        attention = F.dropout(e, self.dropout, training=self.training)
        
        h_prime=[]
        for i in range(edge_attr.shape[0]):
            h_prime.append(torch.matmul(attention[i],Wh))

        if self.concat:
            h_prime = torch.cat(h_prime,dim=1)
            return F.elu(h_prime),e
        else:
            h_prime = torch.stack(h_prime,dim=0)
            h_prime=torch.sum(h_prime,dim=0)
            return h_prime

    #compute attention coefficient
    def _prepare_attentional_mechanism_input(self, Wh,edge_attr):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


