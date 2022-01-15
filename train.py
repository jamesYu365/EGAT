from __future__ import division
from __future__ import print_function

import os
import glob
import time
from datetime import datetime
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils import load_data, accuracy,DSN
from models import GAT
import copy

# Training settings
args={
    'no_cuda':True,
    'no_cuda':False,
    'fastmode':False,
    'seed':72,
    'epochs':100,
    'lr':0.005,
    'weight_decay':5e-4,
    'hidden':[64,8],
    'nb_heads':[1,8],
    'dropout':0.6,
    'alpha':0.2,
    'patience':50,
    'batch_size':20
}

args['cuda'] = not args['no_cuda'] and torch.cuda.is_available()


random.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
if args['cuda']:
    torch.cuda.manual_seed(args['seed'])

#%%Load data
edge_attr, features, labels, idx_train, idx_val, idx_test = load_data()
print('Loading Cora Successfully')

#%%Model and optimizer

model = GAT(nfeat=features.shape[1],
            ef_sz=tuple(edge_attr.shape),
            nhid=args['hidden'], 
            nclass=int(labels.max()) + 1, 
            dropout=args['dropout'],
            nheads=args['nb_heads'],
            alpha=args['alpha']
            )

optimizer = optim.Adam(model.parameters(), 
                       lr=args['lr'], 
                       weight_decay=args['weight_decay'])

if args['cuda']:
    model.cuda()
    features = features.cuda()
    edge_attr = edge_attr.cuda()
    labels = labels.cuda()
    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()

features, edge_attr, labels = Variable(features), Variable(edge_attr), Variable(labels)


def train(epoch,writer):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, edge_attr)
    #output已经做了log_softmax所以不需要CrossEntropyLoss
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    writer.add_scalar('loss_train',loss_train,epoch)
    writer.add_scalar('acc_train',acc_train,epoch)
    loss_train.backward()
    optimizer.step()

    if not args['fastmode']:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, edge_attr)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    writer.add_scalar('loss_val',loss_val,epoch)
    writer.add_scalar('acc_val',acc_val,epoch)
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, edge_attr)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))

#%%Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args['epochs'] + 1
best_epoch = 0
writer=SummaryWriter('./log/'+datetime.now().strftime('%Y%m%d-%H%M%S'))

for epoch in range(args['epochs']):
    loss_values.append(train(epoch,writer))
    epoch_model=copy.deepcopy(model.state_dict())
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        best_model=epoch_model
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args['patience']:
        print('Oops, early stopping!')
        break

torch.save(best_model,f'{best_epoch}.pkl')
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()

torch.cuda.empty_cache()
