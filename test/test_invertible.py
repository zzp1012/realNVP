import os, sys
import torch
from torch.utils.data import DataLoader

# Changable configs
MODEL_PATH = "NEED TO BE FILLED"

# import internal libs
sys.path.insert(0, os.path.join(os.path.dirname(MODEL_PATH), "../src/"))
from data import load_data
from data.utils import logit_transform
from model import build_model

DATASET = "mnist"
DEVICE = torch.device("cuda:0")
BASE_DIM = 64
RES_BLOCKS = 8
BOTTLENECK = 0
SKIP = 1
WEIGHT_NORM = 1
COUPLING_BN = 1
AFFINE = 1
BATCH_SIZE = 64
BN_TYPE = "ln"

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
                    affine = AFFINE,
                    bn_type = BN_TYPE)
print(flow)
flow.load_state_dict(torch.load(MODEL_PATH))
flow = flow.double()


flow.()
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