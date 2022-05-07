import os, sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import torchvision
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from tqdm import tqdm

# Changable configs
RES_PATH = "NEED TO BE FILLED"
EPOCH = 6000
SAMPLE_NUM = 500

# import internal libs
sys.path.insert(0, os.path.join(os.path.dirname(RES_PATH), "src/"))
from data import load_data
from data.utils import logit_transform
from model import build_model
from utils import set_seed

DATASET = "mnist"
DEVICE = torch.device("cuda:1")
BASE_DIM = 64
RES_BLOCKS = 8
BOTTLENECK = 0
SKIP = 1
WEIGHT_NORM = 1
COUPLING_BN = 1
AFFINE = 1
BATCH_SIZE = 64
BN_TYPE = "ln"
SEED = 0

def plot_2D_scatter(save_path: str,
                    df: pd.DataFrame,
                    filename: str,
                    title: str) -> None:
    """plot 2d scatter from dataframe

    Args:
        save_path (str): the path to save fig
        df (pd.DataFrame): the data
        filename (str): the filename
        title (str): the title

    Return:
        None
    """
    assert os.path.exists(save_path), "path {} does not exist".format(save_path)
    assert len(df.columns) == 3, "the dataframe should have 3 columns"
    fig, ax = plt.subplots(figsize=(10, 8)) 
    ax = sns.scatterplot(data=df, hue=df.columns[-1], x=df.columns[0], y=df.columns[1])
    ax.set_title(title)
    fig.tight_layout()
    # save the fig
    path = os.path.join(save_path, "{}.png".format(filename))
    fig.savefig(path)
    plt.close()


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
_, val_split, data_info = load_data(dataset = DATASET)
# define the dataloader
val_loader = DataLoader(val_split,
    batch_size=len(val_split), shuffle=False, num_workers=2)
val_images, val_labels = next(iter(val_loader))
# select the images with 0 or 1 as label
indices_0 = np.where(val_labels == 0)[0][:SAMPLE_NUM]
indices_1 = np.where(val_labels == 1)[0][:SAMPLE_NUM]
indices = np.concatenate([indices_0, indices_1]) # half 0, half 1

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
flow.load_state_dict(torch.load(RES_PATH + f"/model_epoch{EPOCH}.pt"))
flow = flow.double()
flow.eval()

z_lst, log_ll_lst = [], []
for idx in tqdm(indices):
    # get the image and label
    x = val_images[idx].unsqueeze(0)

    # log-determinant of Jacobian from the logit transform
    x, log_det = logit_transform(x)
    x = x.to(DEVICE).double()
    log_det = log_det.to(DEVICE).double()

    # get the z
    z, _ = flow.f(x)
    z = z.detach().cpu().numpy().reshape(1, -1)

    # get log-likelihood
    log_ll, _ = flow(x)
    log_ll = log_ll + log_det

    # update
    z_lst.extend(z)
    log_ll_lst.append(log_ll.item())
    
# get the Zs and log_lls
Zs = np.array(z_lst)
log_lls = np.array(log_ll_lst)

# now t-SNE
tsne = TSNE(n_components=2)
Zs_df = pd.DataFrame(Zs)
Zs_tsne = tsne.fit_transform(Zs_df) # (n, 2)
labels = val_labels.detach().cpu().numpy()[indices]
data_tsne = np.vstack((Zs_tsne.T, labels)).T
tsne_df = pd.DataFrame(data_tsne, columns=['Dim1', 'Dim2', 'class'])
tsne_df['class'] = tsne_df['class'].astype(str)

# plot tsne
save_path = os.path.join(RES_PATH, "tSNE", f"model_{EPOCH}")
os.makedirs(save_path, exist_ok=True)
plot_2D_scatter(save_path = save_path,
                df = tsne_df,
                filename = f"tsne_Zs_testset-sample_num{SAMPLE_NUM}",
                title = "t-SNE on Zs produced from the testset")

# get the center
center_0 = np.mean(Zs_tsne[labels == 0], axis=0)
center_1 = np.mean(Zs_tsne[labels == 1], axis=0)

# calculate the distance to the center
dist_0 = np.linalg.norm(Zs_tsne[labels == 0] - center_0, axis=1)
dist_1 = np.linalg.norm(Zs_tsne[labels == 1] - center_1, axis=1)

# find the outliers
outliers_0 = indices[labels == 0][np.argsort(dist_0)[-5:]]
outliers_1 = indices[labels == 1][np.argsort(dist_1)[-5:]]

# find the inliers
inliers_0 = indices[labels == 0][np.argsort(dist_0)[SAMPLE_NUM//2:SAMPLE_NUM//2+5]]
inliers_1 = indices[labels == 1][np.argsort(dist_1)[SAMPLE_NUM//2:SAMPLE_NUM//2+5]]

# plot corresponding x with correpsonding log_likelihood
for idx in outliers_0:
    lbl = val_labels[idx]
    assert lbl == 0
    save_images(val_images[idx, :, :, :],
                save_path = save_path,
                filename = f"outlier_0_x_lbl{lbl}_log_ll{log_lls[indices == idx]}.png")

for idx in outliers_1:
    lbl = val_labels[idx]
    assert lbl == 1
    save_images(val_images[idx, :, :, :],
                save_path = save_path,
                filename = f"outlier_1_x_lbl{lbl}_log_ll{log_lls[indices == idx]}.png")

for idx in inliers_0:
    lbl = val_labels[idx]
    assert lbl == 0
    save_images(val_images[idx, :, :, :],
                save_path = save_path,
                filename = f"inlier_0_x_lbl{lbl}_log_ll{log_lls[indices == idx]}.png")

for idx in inliers_1:
    lbl = val_labels[idx]
    assert lbl == 1
    save_images(val_images[idx, :, :, :],
                save_path = save_path,
                filename = f"inlier_1_x_lbl{lbl}_log_ll{log_lls[indices == idx]}.png")