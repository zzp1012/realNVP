import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

def plot_multiple_curves(save_path: str, 
                         res_dict: dict,
                         name: str) -> None:
    """plot curves in one figure for each key in dictionary.
    Args:
        args (dict): set containing all program arguments
        res_dict (dict): dictionary containing pairs of key and val(list)
        name (str): the name of the plot.
    """
    # Initialise the figure and axes.
    fig, ax = plt.subplots(figsize=(7, 5))
    # Draw all the lines in the same plot, assigning a label for each one to be
    # shown in the legend.
    for key in res_dict.keys():
        ax.plot(range(100, epochs + 1, 100), np.asarray(res_dict[key]), label=key)
    ax.grid()
    ax.set(xlabel = 'epoch', title = name)
    # Add a legend, and position it on the lower right (with no box)
    plt.legend(frameon=True, prop={'size': 10})
    # save the fig
    path = os.path.join(save_path, "{}.png".format(name))
    fig.savefig(path)
    plt.close()

res_path = "/data/zzp1012/realNVP/outs/tmp/0505-220330-mnist-seed0-blocks8-bottle0-ln-label-lr0.001-bs64-wd0.0/train"
plot_dim = 15
epochs = 5000

para_names = [
    "s1_ckbd.0.coupling.out_bn.running_mean",
    "s1_ckbd.0.coupling.out_bn.running_var",
    "s1_chan.0.coupling.out_bn.running_mean",
    "s1_chan.0.coupling.out_bn.running_var",
    "s2_ckbd.0.coupling.out_bn.running_mean",
    "s2_ckbd.0.coupling.out_bn.running_var",
]

for pname in tqdm(para_names):
    val_lst = []
    for epoch in range(100, epochs + 1, 100):
        model_state = torch.load(os.path.join(res_path, "model_epoch{}.pt".format(epoch)))
        val_lst.append(model_state[pname].cpu().numpy())
    val_lst = np.asarray(val_lst)
    N, D = val_lst.shape
    val_dict = {f"dim{i}": val_lst[:, i] for i in range(min(D, plot_dim))}

    save_path = os.path.join(res_path, "bn_params")
    os.makedirs(save_path, exist_ok=True)
    plot_multiple_curves(save_path = save_path,
                         res_dict = val_dict,
                         name = f"{pname}_curve")