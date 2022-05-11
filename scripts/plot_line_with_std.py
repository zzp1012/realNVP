import os
import numpy as np
import matplotlib.pyplot as plt

RES_PATH = "/data2/zzp1012/realNVP/outs/0509/interpolation/fewerblock/random/epoch100"

# load the two means
mean_ln = np.load(os.path.join(RES_PATH, "mean_log_ll_lst_ln.npy"))
mean_bn = np.load(os.path.join(RES_PATH, "mean_log_ll_lst_bn.npy"))

# load the two std
std_ln = np.load(os.path.join(RES_PATH, "std_log_ll_lst_ln.npy"))
std_bn = np.load(os.path.join(RES_PATH, "std_log_ll_lst_bn.npy"))

# load the alphas
alphas = np.load(os.path.join(RES_PATH, "alphas.npy"))

# the 1 sigma upper and lower analytic population bounds
lower_bound_ln = mean_ln - 1.0 * std_ln
upper_bound_ln = mean_ln + 1.0 * std_ln
lower_bound_bn = mean_bn - 1.0 * std_bn
upper_bound_bn = mean_bn + 1.0 * std_bn

fig, ax = plt.subplots(1)
ax.plot(alphas, mean_ln, lw=1, label='ln', color='blue')
ax.fill_between(alphas, lower_bound_ln, upper_bound_ln, alpha=0.2, color='blue')
ax.plot(alphas, mean_bn, lw=1, label='bn', color='red')
ax.fill_between(alphas, lower_bound_bn, upper_bound_bn, alpha=0.2, color='red')
ax.legend(loc='upper left')
ax.set_xlabel('alphas')
ax.set_ylabel('log_ll')
ax.grid()
# save the fig
path = os.path.join(RES_PATH, "{}.png".format("interpolation"))
fig.savefig(path)
plt.close()