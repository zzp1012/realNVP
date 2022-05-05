import os, sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

# import internal libs
sys.path.insert(0, "/data/zzp1012/realNVP/src")
from data import load_data
from data.utils import logit_transform
from model import build_model

DATASET = "mnist"
DEVICE = torch.device("cpu")
BASE_DIM = 64
RES_BLOCKS = 8
BOTTLENECK = 0
SKIP = 1
WEIGHT_NORM = 1
COUPLING_BN = 0
AFFINE = 1
BATCH_SIZE = 64

# load data
_, val_split, data_info = load_data(dataset = DATASET)
val_loader = DataLoader(val_split,
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# load model
flow = build_model(device = DEVICE,
                    data_info = data_info,
                    base_dim = BASE_DIM,
                    res_blocks = RES_BLOCKS,
                    bottleneck = BOTTLENECK,
                    skip = SKIP,
                    weight_norm = WEIGHT_NORM,
                    coupling_bn = COUPLING_BN,
                    affine = AFFINE)
print(flow)
flow = flow.double()


for batch_idx, data_ in enumerate(val_loader, 1):
    x, lbls = data_
    # log-determinant of Jacobian from the logit transform
    x, _ = logit_transform(x)
    x = x.to(DEVICE).double()

    # get the z
    z, _ = flow.f(x)
    
    # get the x_
    x_ = flow.g(z)
    
    assert torch.allclose(x_, x)
    exit(0)