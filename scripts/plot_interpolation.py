import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

# Changable configs
MODEL_PATH = "NEED TO BE FILLED"
EPOCH = 6000
SAMPLE_NUM = 980

# import internal libs
sys.path.insert(0, os.path.join(os.path.dirname(MODEL_PATH), "src/"))
from data import load_data
from data.utils import logit_transform
from model import build_model
from utils import set_seed

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
BN_TYPE = "bn"
SEED = 0

# plot the histogram
def plot_hist(save_path, arr, key, labels = None):
    """plot hist for the array.

    Args:
        save_path (str): save_path
        arr (np.array): (length, n) contain numbers. 
        labels (list): the list of label whose size corresponds to n.
        key (str): what the values stands for
    """
    fig, ax = plt.subplots()
    ax.hist(arr, histtype='bar', label=labels, density=False)
    ax.legend(prop={'size': 10})
    ax.set(xlabel = key, title = '{}\'s distribution'.format(key))
    # define the path & save the fig
    path = os.path.join(save_path, "{}-hist.png".format(key))
    fig.savefig(path)
    plt.close()

# set the seed
set_seed(SEED)

# load data
_, val_split, data_info = load_data(dataset = DATASET)
# define the dataloader
val_loader = DataLoader(val_split,
    batch_size=len(val_split), shuffle=False, num_workers=2)
val_images, val_labels = next(iter(val_loader))
# select the images with 0 or 1 as label
indices_0 = np.where(val_labels == 0)[0][:SAMPLE_NUM]
indices_1 = np.where(val_labels == 1)[0][:SAMPLE_NUM]

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
flow.load_state_dict(torch.load(MODEL_PATH + f"/model_epoch{EPOCH}.pt"))
flow = flow.double()
flow.eval()

log_ll_lst = []
for idx_0, idx_1 in tqdm(zip(indices_0, indices_1)):
    assert val_labels[idx_0] == 0
    assert val_labels[idx_1] == 1

    # get the data
    x = 0.5 * val_images[idx_0] + 0.5 * val_images[idx_1]
    x = x.unsqueeze(0)

    # log-determinant of Jacobian from the logit transform
    x, log_det = logit_transform(x)
    x = x.to(DEVICE).double()
    log_det = log_det.to(DEVICE).double()

    # log-likelihood
    log_ll, _ = flow(x)
    log_ll = log_ll + log_det

    # update the log_ll_lst
    log_ll_lst.append(log_ll.item())
log_ll_lst = np.array(log_ll_lst)

plot_hist(save_path=MODEL_PATH, 
          arr=log_ll_lst, 
          key=f"log_ll-epoch{EPOCH}-interplolation")

log_ll_lst = []
for idx in tqdm(indices_0):
    assert val_labels[idx] == 0
    # get the data
    x = val_images[idx].unsqueeze(0)

    # log-determinant of Jacobian from the logit transform
    x, log_det = logit_transform(x)
    x = x.to(DEVICE).double()
    log_det = log_det.to(DEVICE).double()

    # log-likelihood
    log_ll, _ = flow(x)

    log_ll = log_ll + log_det
    log_ll_lst.append(log_ll.item())
log_ll_lst = np.array(log_ll_lst)

plot_hist(save_path=MODEL_PATH, 
          arr=log_ll_lst, 
          key=f"log_ll-epoch{EPOCH}-0")

log_ll_lst = []
for idx in tqdm(indices_1):
    assert val_labels[idx] == 1
    # get the data
    x = val_images[idx].unsqueeze(0)

    # log-determinant of Jacobian from the logit transform
    x, log_det = logit_transform(x)
    x = x.to(DEVICE).double()
    log_det = log_det.to(DEVICE).double()

    # log-likelihood
    log_ll, _ = flow(x)

    log_ll = log_ll + log_det
    log_ll_lst.append(log_ll.item())
log_ll_lst = np.array(log_ll_lst)

plot_hist(save_path=MODEL_PATH, 
          arr=log_ll_lst, 
          key=f"log_ll-epoch{EPOCH}-1")