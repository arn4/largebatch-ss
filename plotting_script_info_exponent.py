import numpy as np
import matplotlib.pyplot as plt
import os

def ensemble_plot(show = 1, save = 0):
    fig, ax = plt.subplots(len(ds),2, figsize = (15,5), sharey = True)
    for i,d in enumerate(ds):
        xaxis = xaxiss[i]
        for j,seed in enumerate(range(nseeds)):
            ax[i][0].plot(xaxis, Mupdates_simulation[i][j][:,0], label=f'GD seed = {seed}', marker='o')
            ax[i][1].plot(xaxis, Mupdates_simulation_noresample[i][j][:,0], label=f'GD no resampling seed = {seed}', marker='o')
        ax[i][0].set_xlabel('t / log(d)')
        ax[i][1].set_xlabel('t / log(d)')
        ax[i][0].set_ylabel('Magnetization / Overlap with target')
        ax[i][0].set_ylim(0,1)
        ax[i][1].set_ylim(0,1)
        ax[i][0].legend()
        ax[i][1].legend()
    if save:
        folder_path =  f"./results/figures/info_exp_12"
        hyper_path = f"/l={l}_noise={noise}_gamma0={gamma0}_activation={act_tkn}_p={p}_start={start}"
        fig_name = folder_path + hyper_path
        fig.savefig(f'{fig_name}_ensemble_plot.pdf',bbox_inches='tight')
    if show:
        plt.show()

# load data # 
l = 1.2 ; noise = 0. ; gamma0 = 0.01 ; p = 1
H2 = lambda z: z**2 - 1
activations = {"relu": lambda x: np.maximum(x,0), "h2": H2}
activation_derivatives = {"relu": lambda x: (x>0).astype(float), "h2": lambda x: 2*x}
act_tkn = "h2"
act_derivative_tkn = "h2"
activation = activations[act_tkn]
activation_derivative = activation_derivatives[act_derivative_tkn]
starts = ["random"]
for start in starts:
    folder_path =  f"./results_cluster/data/info_exp_12"
    hyper_path = f"/l={l}_noise={noise}_gamma0={gamma0}_activation={act_tkn}_p={p}_start={start}"
    data_path = folder_path + hyper_path
    xaxiss = np.load(f'{data_path}/xaxiss.npz', allow_pickle=True)
    ds = np.load(f'{data_path}/ds.npz', allow_pickle=True)
    error_simus = np.load(f'{data_path}/error_simus.npz', allow_pickle=True)
    error_simus_noresample = np.load(f'{data_path}/error_simus_noresample.npz', allow_pickle=True)
    error_montecarlos = np.load(f'{data_path}/error_montecarlos.npz', allow_pickle=True)
    
    # now load xaxiss as a list of arrays
    xaxiss = xaxiss['arr_0']
    ds = ds['arr_0']
    Mupdates_simulation = error_simus['arr_0']
    Mupdates_simulation_noresample = error_simus_noresample['arr_0']
    Mupdates_montercalo = error_montecarlos['arr_0']
    nseeds = len(Mupdates_simulation[0])

    ensemble_plot(show = 1, save = 1)
