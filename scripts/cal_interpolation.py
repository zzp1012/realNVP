import os, sys
import torch
import numpy as np
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader

# Changable configs
RES_PATH = "/data2/zzp1012/batch-norm-CVPR2023-rebuttal/supplement2/outs/0128/0and1and2/origin/random/ln/0129-040116-mnist-lbls0+1+2-seed0-blocks8-bottle0-ln-random-lr0.001-bs64-wd0.0/train"
DATA_PATH = "/data2/zzp1012/batch-norm-CVPR2023-rebuttal/supplement2/data"
EPOCH = 100
SELECTED_LBLS = [0, 1, 2]
assert len(SELECTED_LBLS) == 3, "len(SELECTED_LBLS) must be 3"
BN_TYPE = "ln"

# import internal libs
sys.path.insert(0, os.path.join(os.path.dirname(RES_PATH), "src/"))
from data import load_data
from data.utils import logit_transform
from model import build_model
from utils import set_seed

# fix
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
SEED = 0

def save_images(imgs: torch.Tensor, 
                save_path: str,
                filename: str,  
                nrow: int = 1) -> None:
    """save images.
    Args:
        imgs (torch.Tensor): the images to save.
        save_path (str): the path to save images.
        filename (str): the filename of images.
        nrow (int): the number of images to save in a row.
    Returns:
        None
    """
    torchvision.utils.save_image(imgs, 
                                 os.path.join(save_path, filename), 
                                 nrow=nrow)

# set the seed
set_seed(SEED)

# load data
_, val_split, data_info = load_data(dataset = DATASET, root = DATA_PATH)
val_loader = DataLoader(val_split,
    batch_size=len(val_split), shuffle=False, num_workers=2)
val_images, val_labels = next(iter(val_loader))

# select the images by label
indices_dict = dict()
for lbl in SELECTED_LBLS:
    indices = np.where(val_labels == lbl)[0]
    indices_dict[lbl] = indices

# make sure the len(indices) are the same
sample_num = min([len(indices_dict[lbl]) for lbl in SELECTED_LBLS])
for lbl in SELECTED_LBLS:
    indices_dict[lbl] = indices_dict[lbl][:sample_num]

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
flow.load_state_dict(torch.load(RES_PATH + f"/model_epoch{EPOCH}.pt"))
flow.eval()

# make a save path
save_path = os.path.join(RES_PATH, "interpolation", f"epoch{EPOCH}")
os.makedirs(save_path, exist_ok=True)

# copy the script to the save path
os.system(f"cp {os.path.abspath(__file__)} {save_path}")

# create batches
batch_size = 64
np.random.seed(SEED) 
indexes = np.arange(sample_num)
np.random.shuffle(indexes)
batches = np.array_split(indexes, round(sample_num / batch_size))

# calculate the mean_log_ll and std_log_ll by different alpha
mean_log_ll, std_log_ll = [], []
alpha_lst, beta_lst = [], []

interval = .1
for alpha in np.linspace(0.0, 1.0, num=int(1/interval)+1):
    for beta in np.linspace(0.0, 1 - alpha, num=int((1-alpha)/interval+1)):
        alpha_lst.append(alpha)
        beta_lst.append(beta)

        log_ll_lst = []
        for batch_id, batch in enumerate(tqdm(batches)):
            # get the idxes
            idxes_fir = indices_dict[SELECTED_LBLS[0]][batch]
            idxes_sec = indices_dict[SELECTED_LBLS[1]][batch]
            idxes_thir = indices_dict[SELECTED_LBLS[2]][batch]

            # get the data
            x = alpha * val_images[idxes_fir] + beta * val_images[idxes_sec] + (1 - alpha - beta) * val_images[idxes_thir]

            # # plot some images
            # if alpha == 0 or alpha == 1 or alpha == 0.5:
            #     save_image_path = os.path.join(save_path, "images", f"alpha{alpha}")
            #     os.makedirs(save_image_path, exist_ok=True)
            #     save_images(x[0], save_image_path, f"{batch_id}_images.png")

            # log-determinant of Jacobian from the logit transform
            x = x.to(DEVICE)
            x, log_det = logit_transform(x, DEVICE)

            # log-likelihood
            log_ll, _ = flow(x)
            log_ll = log_ll + log_det

            # update the mean_log_ll and std_log_ll
            log_ll_lst.extend(log_ll.tolist())

        log_ll_lst = np.array(log_ll_lst)

        # update
        mean_log_ll.append(np.mean(log_ll_lst))
        std_log_ll.append(np.std(log_ll_lst))

# save the results
np.save(os.path.join(save_path, "alphas.npy"), np.array(alpha_lst))
np.save(os.path.join(save_path, "betas.npy"), np.array(beta_lst))
np.save(os.path.join(save_path, "mean_log_ll.npy"), np.array(mean_log_ll))
np.save(os.path.join(save_path, "std_log_ll.npy"), np.array(std_log_ll))