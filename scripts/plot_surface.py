import os
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

RES_PATH_LN = "/data2/zzp1012/batch-norm-CVPR2023-rebuttal/supplement2/outs/0128/0and1and2/origin/random/ln/0129-040116-mnist-lbls0+1+2-seed0-blocks8-bottle0-ln-random-lr0.001-bs64-wd0.0/train/interpolation/epoch100"
RES_PATH_BN = "/data2/zzp1012/batch-norm-CVPR2023-rebuttal/supplement2/outs/0128/0and1and2/origin/random/bn/0128-233959-mnist-lbls0+1+2-seed0-blocks8-bottle0-bn-random-lr0.001-bs64-wd0.0/train/interpolation/epoch100"

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

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(alphas, betas, mean_ln, color="b", label='ln')
ax.plot_trisurf(alphas, betas, mean_bn, color="r", label='bn')
ax.set_xlabel('alphas')
ax.set_ylabel('betas')
ax.set_zlabel('log_ll')
# ax.legend(loc='upper left')
ax.view_init(10, 40)

# save the fig
path = os.path.join(".", "{}.png".format("surface"))
plt.savefig(path)
plt.close()