import os
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt

RES_PATH_LN = "/data2/zzp1012/batch-norm-CVPR2023-rebuttal/supplement2/outs/0128/6and7and8/origin/random/ln/0129-035631-mnist-lbls6+7+8-seed0-blocks8-bottle0-ln-random-lr0.001-bs64-wd0.0/train/interpolation/epoch100"
RES_PATH_BN = "/data2/zzp1012/batch-norm-CVPR2023-rebuttal/supplement2/outs/0128/6and7and8/origin/random/bn/0128-234245-mnist-lbls6+7+8-seed0-blocks8-bottle0-bn-random-lr0.001-bs64-wd0.0/train/interpolation/epoch100"

# load the two means
mean_ln = np.load(os.path.join(RES_PATH_LN, "mean_log_ll.npy"))
mean_bn = np.load(os.path.join(RES_PATH_BN, "mean_log_ll.npy"))

# load the alphas
alphas_ln = np.load(os.path.join(RES_PATH_LN, "alphas.npy"))
alphas_bn = np.load(os.path.join(RES_PATH_BN, "alphas.npy"))
assert np.allclose(alphas_ln, alphas_bn), "The two alphas are not the same!"
alphas = alphas_ln

# load the betas
betas_ln = np.load(os.path.join(RES_PATH_LN, "betas.npy"))
betas_bn = np.load(os.path.join(RES_PATH_BN, "betas.npy"))
assert np.allclose(betas_ln, betas_bn), "the two betas are not the same!"
betas = betas_ln

fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')
ax.plot_trisurf(alphas, betas, mean_ln, color="b", label='ln')
ax.plot_trisurf(alphas, betas, mean_bn, color="r", label='bn')
plt.xticks([0,1])
plt.yticks([0,1])
ax.set_xlabel(r'$\alpha$', fontsize=30, labelpad=0)
ax.set_ylabel(r'$\beta$', fontsize=30, labelpad=0)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('log-likelihood', rotation=90, fontsize=30, labelpad=30)
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)
ax.tick_params(axis='z', labelsize=24, pad=10)
# ax.legend(loc='upper left')
ax.view_init(10, 40)
# plt.tight_layout()

# save the fig
plt.savefig(os.path.join(".", "{}.pdf".format("surface")))
plt.savefig(os.path.join(".", "{}.png".format("surface")))
plt.close()