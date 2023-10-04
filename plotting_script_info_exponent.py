import numpy as np
import matplotlib.pyplot as plt
import os



def ensemble_plot(show = 1, save = 0):
    fig, ax = plt.subplots(1,3, figsize = (15,5), sharey = True)
    for i,d in enumerate(ds):
        ax[0].errorbar(xaxiss[i], Mupdates_simulation[i][:,0], label=f'GD simulation d={d}', marker='o')
        ax[1].errorbar(xaxiss[i], np.zeros(len(xaxiss[i])), label=f'MC approximation d={d}', marker='o')
        ax[2].errorbar(xaxiss[i], Mupdates_simulation_noresample[i][:,0], label=f'GD simulation no resampling d={d}', marker='o')
    ax[0].set_xlabel('t / log(d)')
    ax[0].set_ylabel('Test Risk')
    ax[1].set_xlabel('t / log(d)')
    ax[2].set_xlabel('t / log(d)')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    if save:
        path =  f"./results/figures/info_exp12"
        os.makedirs(path, exist_ok=True) 
        fig.savefig(f'{path}/ensemble_plot',bbox_inches='tight')
    if show:
        plt.show()

# load data # 

data_path = './results/data/info_exp_12'
xaxiss = np.load(f'{data_path}/xaxiss.npz', allow_pickle=True)
ds = np.load(f'{data_path}/ds.npz', allow_pickle=True)
error_simus = np.load(f'{data_path}/error_simus.npz', allow_pickle=True)
error_simus_noresample = np.load(f'{data_path}/error_simus_noresample.npz', allow_pickle=True)
error_montecarlos = np.load(f'{data_path}/error_montecarlos.npz', allow_pickle=True)
std_simus = np.load(f'{data_path}/std_simus.npz', allow_pickle=True)
std_simus_noresample = np.load(f'{data_path}/std_simus_noresample.npz', allow_pickle=True)
std_montecarlos = np.load(f'{data_path}/std_montecarlos.npz', allow_pickle=True)

# now load xaxiss as a list of arrays
xaxiss = xaxiss['arr_0']
ds = ds['arr_0']
Mupdates_simulation = error_simus['arr_0']
Mupdates_simulation_noresample = error_simus_noresample['arr_0']
Mupdates_montercalo = error_montecarlos['arr_0']
std_simus = std_simus['arr_0']  
std_simus_noresample = std_simus_noresample['arr_0']
std_montecarlos = std_montecarlos['arr_0']

ensemble_plot(show = 1, save = 0)
