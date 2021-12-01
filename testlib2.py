# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 16:07:03 2021

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

from utils import load_data, accuracy,DSN
from models import GAT


a=torch.ones((2,3))*2
b=torch.ones((4,2,3))
c=F.dropout(b,0.6)
d=F.dropout(a,0.5)

