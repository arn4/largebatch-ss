import numpy as np
import matplotlib.pyplot as plt
import os



def ensemble_plot(show = 1, save = 0):
    fig, ax = plt.subplots(1,3, figsize = (15,5), sharey = True)
    for i,d in enumerate(ds):
        ax[0].plot(xaxiss[i], error_simus[i], label=f'd={d}', marker='o')
        ax[1].plot(xaxiss[i], error_montecarlos[i], label=f'd={d} MC', marker='o')
        ax[2].plot(xaxiss[i], error_simus_noresample[i], label=f'd={d} no resample', marker='o')
    ax[0].set_xlabel('t / log(d)')
    ax[0].set_ylabel('Test Risk')
    ax[1].set_xlabel('t / log(d)')
    ax[2].set_xlabel('t / log(d)')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    if save:
        path =  f"./results/figures/info_exponent"
        os.makedirs(path, exist_ok=True) 
        fig.savefig(f'{path}/ensemble_plot',bbox_inches='tight')
    if show:
        plt.show()

# load data # 

data_path = './results/data/info_exponent'
xaxiss = np.load(f'{data_path}/xaxiss.npz', allow_pickle=True)
ds = np.load(f'{data_path}/ds.npz', allow_pickle=True)
error_simus = np.load(f'{data_path}/error_simus.npz', allow_pickle=True)
error_simus_noresample = np.load(f'{data_path}/error_simus_noresample.npz', allow_pickle=True)
error_montecarlos = np.load(f'{data_path}/error_montecarlos.npz', allow_pickle=True)

# now load xaxiss as a list of arrays
xaxiss = xaxiss['arr_0']
ds = ds['arr_0']
error_simus = error_simus['arr_0']
error_simus_noresample = error_simus_noresample['arr_0']
error_montecarlos = error_montecarlos['arr_0']

ensemble_plot(show = 1, save = 0)
