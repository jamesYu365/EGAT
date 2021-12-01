# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 08:17:25 2021

@author: James
"""
from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import networkx as nx

from utils import load_data, accuracy
from models import GAT

#%%

edges={
      0:[3],1:[],2:[1,4],3:[0,6,4],4:[5,7,3],5:[4],6:[3,7,12],
      7:[4,6,9,11],8:[9],9:[8,10,7],10:[9,15,11],11:[10,7,14,12],
      12:[11,6,13],13:[12,18,14],14:[11,19,13,15],15:[17,16,14,10],
      16:[15],17:[],18:[13],19:[14],20:[17]
      }

graph=nx.from_dict_of_lists(edges,create_using=nx.DiGraph)
adj=nx.adjacency_matrix(graph)
adj=adj.todense()
t=torch.FloatTensor(adj)
#%%
def DSN22(t):
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

# def DSN(x):
#     """Doubly stochastic normalization"""
#     p=x.shape[0]
#     y1=[]
#     for i in range(p):
#         y1.append(DSN2(x[i]))
#     y1=torch.stack(y1,dim=0)
#     return y1
time1=time.time()
y2=DSN22(t).numpy()
time2=time.time()
print(y2.shape)
print('It takes',time2-time1,'s !')

#%%DSN论文里算法循环实现
"""
t=torch.randn((3000,3000))

time1=time.time()
a=t.sum(dim=1)
b=t.sum(dim=0)
lamb=torch.cat([a,b],dim=0).max()
r=t.shape[0]*lamb-t.sum(dim=0).sum(dim=0)
tt=torch.empty_like(t)
for i in range(t.shape[0]):
    for j in range(t.shape[1]):
        tt[i,j]=t[i,j]+(lamb-a[i])*(lamb-b[j])/r

ttloop=tt/tt.sum(dim=0)[0]
time2=time.time()
print('循环 takes',time2-time1,'s !')
"""
#%%DSN论文里算法矩阵实现
"""
time1=time.time()
a=t.sum(dim=1,keepdim=True)
b=t.sum(dim=0,keepdim=True)
lamb=torch.cat([a.squeeze(),b.squeeze()],dim=0).max()
r=t.shape[0]*lamb-t.sum(dim=0).sum(dim=0)

a=a.expand(-1,t.shape[1])
b=b.expand(t.shape[0],-1)
tt=t+(lamb**2-lamb*(a+b)+a*b)/r

ttmatrix=tt/tt.sum(dim=0)[0]
time2=time.time()
print('矩阵 takes',time2-time1,'s !')
"""

#%%非论文实现
def DSN21(x):
    x_sz=list(x.shape)
    sumr=torch.sum(x,dim=1,keepdim=True)
    sumr=sumr.expand(-1,x_sz[1])
    xr=x/sumr
    y1=torch.zeros_like(x)
    for i in range(x_sz[0]):
        for j in range(x_sz[1]):
            summ1=0
            for k in range(x_sz[1]):
                summ1=summ1+xr[i,k]*xr[j,k]/torch.sum(xr[:,k])
            y1[i,j]=summ1
    return y1

y1=DSN21(t).numpy()

#%%matlab
# function [Xnorm,U,V] = norm_doublemean(X)
# Represents a non-negative matrix X as X = UX'V, where U,V are diagonal
# matrices and X' is a matrix with unity mean value along each column and
# row

def DSN(t):
    MaxNumIterations = 10000
        
    # first check if the table contains zero columns or zero rows
    if(torch.sum(torch.mean(t,dim=0))==0):
        assert('ERROR: The matrix contains zero colums!')
    
    if(torch.sum(torch.mean(t,dim=1))==0):
        assert('ERROR: The matrix contains zero rows!')
        
    def normalizeByColumns(t):
        colmeans = torch.mean(t,dim=0,keepdim=True)
        t1=t/colmeans.expand(t.shape[0],-1)
        return t1,colmeans.squeeze()
    
    def normalizeByRows(t):
        rowmeans = torch.mean(t,dim=1,keepdim=True)
        t1=t/rowmeans.expand(-1,t.shape[1])
        return t1,rowmeans.squeeze()
    
    
    U = torch.diag(torch.ones(t.shape[0]))
    V = torch.diag(torch.ones(t.shape[1]))
    tnorm = t
    
    eps = torch.sum(torch.sum(t))
    
    for i in range(MaxNumIterations):
        told = tnorm
        tnorm,columnmeans = normalizeByColumns(tnorm)
        #disp(tnorm)
        V = torch.diag(columnmeans)@V
        tnorm,rowmeans = normalizeByRows(tnorm)
        U = U@torch.diag(rowmeans)
        eps = torch.sum(torch.sum(abs(tnorm-told)))
        print('step:',i,'| eps:',eps)
        if(eps<0.001):
            break
        
    tnorm=tnorm/(tnorm.sum(dim=0)[0])
    
    return tnorm

time1=time.time()
y=DSN(t)
time2=time.time()
print(y.shape)
print('It takes',time2-time1,'s !')

