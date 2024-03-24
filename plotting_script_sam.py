import numpy as np 
import matplotlib.pyplot as plt


######## LOAD ########
test_errors_mean = []
test_errors_std = []
test_errors_sam = []
test_errors_sam_std = []
Ms_mean = []
Ms_std = []
Ms_sam_mean = []
Ms_sam_std = []
ds = []
Ts = []
p = 10
choice_gamma = 'sgd_optimal'
# choice_gamma = 'sam'
data = np.load(f'computation-database/sam_vs_sgd_p={p}_gamma={choice_gamma}.npz', allow_pickle=True)
for key in data.keys():
    if key == 'ds':
        ds = data[key]
    elif key == 'Ts':
        Ts = data[key]
    elif key == 'test_errors_mean':
        test_errors_mean = data[key]
    elif key == 'test_errors_std':
        test_errors_std = data[key]
    elif key == 'test_errors_sam':
        test_errors_sam = data[key]
    elif key == 'test_errors_sam_std':
        test_errors_sam_std = data[key]
    elif key == 'Ms_mean':
        Ms_mean = data[key]
    elif key == 'Ms_std':
        Ms_std = data[key]
    elif key == 'Ms_sam_mean':
        Ms_sam_mean = data[key]
    elif key == 'Ms_sam_std':
        Ms_sam_std = data[key]

### PLOT ### 

fig, ax = plt.subplots(2,2, figsize=(20,5), sharey = 'row', sharex = 'col')
colors = ['b', 'r', 'g', 'm', 'c']
for i,d in enumerate(ds):
    xaxis = np.arange(Ts[i]+1)/d**2
    print(test_errors_mean[i])
    ax[0,0].errorbar(xaxis, test_errors_mean[i], yerr=test_errors_std[i], label=f'SGD - d = {d}', color=colors[i], marker='x')
    ax[0,1].errorbar(xaxis, test_errors_sam[i], yerr=test_errors_sam_std[i], label=f'SAM - d = {d}', color=colors[i], marker='o')
    for j in range(p):
        ax[1,0].errorbar(xaxis, Ms_mean[i][:,j,0], yerr=Ms_std[i][:,j,0], label=f'SGD - d = {d}', color=colors[i], marker='x')
        ax[1,1].errorbar(xaxis, Ms_sam_mean[i][:,j,0], yerr=Ms_sam_std[i][:,j,0], label=f'SAM - d = {d}', color=colors[i], marker='o')

    # ax[1,0].errorbar(xaxis, Ms_mean[i][:,0,0], yerr=Ms_std[i][:,0,0], label=f'SGD - d = {d}', color=colors[i], marker='x')
    # ax[1,1].errorbar(xaxis, Ms_sam_mean[i][:,0,0], yerr=Ms_sam_std[i][:,0,0], label=f'SAM - d = {d}', color=colors[i], marker='o')

ax[0,0].axvline(x=1, color='k', ls='--')
ax[0,0].set_xlabel('# Steps / d^2')
ax[0,0].set_ylabel('Test error')
ax[0,0].set_yscale('log')
ax[0,0].legend()

ax[0,1].axvline(x=1, color='k', ls='--')
ax[0,1].set_xlabel('# Steps / d^2')
ax[0,1].set_ylabel('Test error')
ax[0,1].legend()

ax[1,1].axvline(x=1, color='k', ls='--')
ax[1,1].set_xlabel('# Steps / d^2')
ax[1,1].set_ylabel('Overlap')
ax[1,1].legend()

ax[1,0].axvline(x=1, color='k', ls='--')
ax[1,0].set_xlabel('# Steps / d^2')
ax[1,0].set_ylabel('Overlap')
ax[1,0].legend()
plt.show()
fig.savefig(f'computation-database/sam_vs_sgd_p={p}_gamma={choice_gamma}.pdf')