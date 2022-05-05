import os, sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

# Changable configs
MODEL_PATH = "NEED TO BE FILLED"
EPOCH = 200
SAMPLE_NUM = 500

# import internal libs
sys.path.insert(0, os.path.join(os.path.dirname(MODEL_PATH), "src/"))
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
flow.load_state_dict(torch.load(MODEL_PATH + f"/model_epoch{EPOCH}.pt"))
flow = flow.double()
flow.eval()

z_lst, label_lst = [], []
for batch_idx, data_ in enumerate(val_loader, 1):
    if len(z_lst) > SAMPLE_NUM:
        break
    x, lbls = data_
    # log-determinant of Jacobian from the logit transform
    x, _ = logit_transform(x)
    x = x.to(DEVICE).double()

    # get the z
    z, _ = flow.f(x)
    z = z.detach().cpu().numpy().reshape(len(z), -1)
    
    # update
    z_lst.extend(z)
    label_lst.extend(lbls.numpy())

Zs = np.array(z_lst)[:SAMPLE_NUM, :]
Zs_df = pd.DataFrame(Zs)
labels = np.array(label_lst)[:SAMPLE_NUM]

# now t-SNE
tsne = TSNE(n_components=2)
Zs_tsne = tsne.fit_transform(Zs_df)
outlier1 = np.where(np.abs(Zs_tsne[:, 0] - Zs_tsne[:, 0].mean()) > (3*Zs_tsne[:, 0].std()))[0]
outlier2 = np.where(np.abs(Zs_tsne[:, 1] - Zs_tsne[:, 1].mean()) > (3*Zs_tsne[:, 1].std()))[0]
outlier = set(np.concatenate([outlier1, outlier2]))
print(outlier1, outlier2)
print(outlier)

data_tsne = np.vstack((Zs_tsne.T, labels)).T
tsne_df = pd.DataFrame(data_tsne, columns=['Dim1', 'Dim2', 'class'])
tsne_df['class'] = tsne_df['class'].astype(str)

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

# plot tsne
plot_2D_scatter(save_path = os.path.dirname(MODEL_PATH),
                df = tsne_df,
                filename = "tsne_Zs_testset",
                title = "t-SNE on Zs produced from the testset")


