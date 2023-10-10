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

# plot similarities
fig, ax = plt.subplots()
for i,d in enumerate(ds): 
    for j,seed in enumerate(range(nseeds)):
        ss = store_simulation[i][j]
        sa = store_analytical[i][j]
        idx = (i+j)*(i+j+1) // 2 + j
        ax.plot(np.arange(len(ss)) / np.log(d), abs( ss[:,0,0] ),  label = f'Simul d = {d} - seed = {seed}', color = colors[idx] )
        ax.plot(np.arange(len(ss)) / np.log(d), abs(sa[:,0,0]), label= f'Analytical d = {d}  - seed = {seed}', color = colors[idx], linestyle='dotted')
ax.set_xlabel('t / log(d)')
ax.set_ylabel('Overlap')
ax.legend()
plt.show()
