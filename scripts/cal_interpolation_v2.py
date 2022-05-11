import os, sys
from sklearn.metrics import mean_absolute_error
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# import internal libs
SRC_PATH = "/data2/zzp1012/realNVP/src"
sys.path.insert(0, SRC_PATH)
from data import load_data
from data.utils import logit_transform
from model import build_model
from utils import set_seed

# fixed params
DEVICE = torch.device("cuda:2")
SEED = 0
DATA_PATH = "/data2/zzp1012/realNVP/data"
DATASET = "mnist"
BOTTLENECK = 0
SKIP = 1
WEIGHT_NORM = 1
COUPLING_BN = 1
AFFINE = 1
BATCH_SIZE = 64

# Changable configs
RES_PATH = "/data2/zzp1012/realNVP/outs/0509"
MODEL = "fewerfeaturemap"
METHOD = "random"
EPOCH = 100

def main():
    # set the seed
    set_seed(SEED)

    # make a save dir
    save_root = os.path.join(RES_PATH, "interpolation", MODEL, METHOD, f"epoch{EPOCH}")
    os.makedirs(save_root, exist_ok=True)
    os.system(f"cp {os.path.abspath(__file__)} {save_root}")

    if MODEL == "origin":
        base_dim, res_blocks = 64, 8
    elif MODEL == "fewerblock":
        base_dim, res_blocks = 64, 4
    elif MODEL == "fewerfeaturemap":
        base_dim, res_blocks = 32, 8
    else:
        raise NotImplementedError

    # load data
    _, val_split, data_info = load_data(dataset = DATASET, root = DATA_PATH)
    val_loader = DataLoader(val_split,
        batch_size=len(val_split), shuffle=False, num_workers=2)
    val_images, val_labels = next(iter(val_loader))

    # get different alphas
    alphas = np.linspace(0.0, 1.0, num=10)
    np.save(os.path.join(save_root, "alphas.npy"), alphas)

    for bn_type in ["bn", "ln"]:
        print(f"calculate the curve for {bn_type}...")
        log_ll_dict = dict()
        for neg, pos in tqdm([(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]):
            # select the images by label
            indices_pos = np.where(val_labels == pos)[0]
            indices_neg = np.where(val_labels == neg)[0]
            sample_num = min(len(indices_pos), len(indices_neg))
            indices_pos = indices_pos[:sample_num]
            indices_neg = indices_neg[:sample_num]

            # clear the cache
            torch.cuda.empty_cache()

            # load the model
            flow = build_model(device = DEVICE,
                               data_info = data_info,
                               base_dim = base_dim,
                               res_blocks = res_blocks,
                               bottleneck = BOTTLENECK,
                               skip = SKIP,
                               weight_norm = WEIGHT_NORM,
                               coupling_bn = COUPLING_BN,
                               affine = AFFINE,
                               bn_type = bn_type)
            # load the pretained model
            model_path = os.path.join(RES_PATH, f"{neg}and{pos}", MODEL, METHOD, bn_type)
            assert len(os.listdir(model_path)) == 1, f"{model_path} should contain only one dir"
            model_path = os.path.join(model_path, os.listdir(model_path)[0], "train", f"model_epoch{EPOCH}.pt")
            flow.load_state_dict(torch.load(model_path))
            flow.eval()

            # calculate the mean_log_ll and std_log_ll by different alpha
            log_ll_alpha_lst = []
            for alpha in alphas:
                # create batches
                batch_size = 64
                np.random.seed(SEED) 
                indexes = np.arange(sample_num)
                np.random.shuffle(indexes)
                batches = np.array_split(indexes, sample_num // batch_size)

                log_ll_lst = []
                for batch in tqdm(batches):
                    # get the idxes
                    idxes_pos = indices_pos[batch]
                    idxes_neg = indices_neg[batch]
                    assert torch.unique(val_labels[idxes_pos]) == pos
                    assert torch.unique(val_labels[idxes_neg]) == neg

                    # get the data
                    x = alpha * val_images[idxes_pos] + (1 - alpha) * val_images[idxes_neg]

                    # log-determinant of Jacobian from the logit transform
                    x = x.to(DEVICE)
                    x, log_det = logit_transform(x, DEVICE)

                    # log-likelihood
                    log_ll, _ = flow(x)
                    log_ll = log_ll + log_det

                    # update the log_ll_lst
                    log_ll_lst.extend(log_ll.tolist())
                log_ll_lst = np.array(log_ll_lst) # (sample_num, ) for certain alpha, (neg, pos) and bn_type
                log_ll_alpha_lst.append(log_ll_lst)
            log_ll_alpha_lst = np.array(log_ll_alpha_lst)

            # save the log_ll_alpha_lst
            log_ll_dict[f"{neg}and{pos}"] = log_ll_alpha_lst

        # save the log_ll_dict
        np.save(os.path.join(save_root, f"log_ll_dict_{bn_type}.npy"), log_ll_dict)

        # cal mean and std
        mean_log_ll_lst, std_log_ll_lst = [], []
        for i in range(len(alphas)):
            log_ll_lst = []
            for key, val in log_ll_dict.items():
                log_ll_lst.extend(val[i])
            mean_log_ll_lst.append(np.mean(log_ll_lst))
            std_log_ll_lst.append(np.std(log_ll_lst))
        
        # save the mean_log_ll_lst and std_log_ll_lst
        np.save(os.path.join(save_root, f"mean_log_ll_lst_{bn_type}.npy"), mean_log_ll_lst)
        np.save(os.path.join(save_root, f"std_log_ll_lst_{bn_type}.npy"), std_log_ll_lst)


if __name__ == "__main__":
    main()