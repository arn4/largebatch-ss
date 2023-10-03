import numpy as np
import matplotlib.pyplot as plt
import os



def ensemble_plot(show = 1, save = 0):
    '' ''
    if save:
        path =  f""
        os.makedirs(path, exist_ok=True) 
        fig.savefig(f'{path}/',bbox_inches='tight')
    if show:
        plt.show()
    
colors = ['darkgreen', 'lime', 'olive']
tkns = ['120']
ds = [8000] 
choices = ['hypercube']
T = 2



### Plot all the results all toghether ###
fig, ax = plt.subplots(1,1)
np.load('a.npy', allow_pickle=True)
for i,d in enumerate(ds):
    ax.plot(xaxiss[i], error_simus[i], label=f'd={d}', marker='o')
    ax.plot(xaxiss[i], error_montecarlos[i], label=f'd={d} MC', marker='o')
    ax.plot(xaxiss[i], error_simus_noresample[i], label=f'd={d} no resample', marker='o')
ax.set_xlabel('t / log(d)')
ax.set_ylabel('Test Risk')
ax.legend()
plt.show()