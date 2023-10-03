import numpy as np
import matplotlib.pyplot as plt
import os



def ensemble_plot(show = 1, save = 0):
    fig, ax = plt.subplots(1,3, figsize = (8,4), sharey = True)
    for i,d in enumerate(ds):
        ax[0].plot(xaxiss[i], error_simus[i], label=f'd={d}', marker='o')
        ax[1].plot(xaxiss[i], error_montecarlos[i], label=f'd={d} MC', marker='o')
        ax[2].plot(xaxiss[i], error_simus_noresample[i], label=f'd={d} no resample', marker='o')
        ax[0].set_xlabel('t / log(d)')
        ax[0].set_ylabel('Test Risk')
        ax[1].set_xlabel('t / log(d)')
        ax[1].set_ylabel('Test Risk')
        ax[2].set_xlabel('t / log(d)')
        ax[2].set_ylabel('Test Risk')

    if save:
        path =  f"./results/figures/info_exponent"
        os.makedirs(path, exist_ok=True) 
        fig.savefig(f'{path}/ensemble_plot',bbox_inches='tight')
    if show:
        plt.show()

# load data # 
data_path = './results/data/info_exponent'
xaxiss = np.load(f'{data_path}/xaxiss.npy', allow_pickle=True)
ds = np.load(f'{data_path}/ds.npy', allow_pickle=True)
error_simus = np.load(f'{data_path}/error_simus.npy', allow_pickle=True)
error_simus_noresample = np.load(f'{data_path}/error_simus_noresample.npy', allow_pickle=True)
error_montecarlos = np.load(f'{data_path}/error_montecarlos.npy', allow_pickle=True)



ensemble_plot(show = 1, save = 0)
