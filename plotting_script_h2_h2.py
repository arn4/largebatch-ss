import numpy as np
import matplotlib.pyplot as plt
import os



# generate array of colors for plotting with 50 colors 
colors = []
for i in range(10):
    for j in range(i+1):
        colors.append(plt.cm.tab10(i))

# load data
start = "random"

path =  f"./results_cluster/h2_h2/start={start}"

store_simulation = np.load(f'{path}/store_simulation.npy', allow_pickle = True)
store_analytical = np.load(f'{path}/store_analytical.npy', allow_pickle = True)
ds = np.load(f'{path}/ds.npy')
nseeds = np.load(f'{path}/nseeds.npy')

# plot similarities
fig, ax = plt.subplots()
for i,d in enumerate(ds): 
    for j,seed in enumerate(range(nseeds)):
        ss = store_simulation[1][j]
        idx = (i+j)*(i+j+1) // 2 + j
        ax.plot(np.arange(len(ss)) , abs( ss[:,0,0] ),  label = f'Simul d = {d} - seed = {seed}', color = colors[idx], marker = 'o', linestyle = 'None')
sa = store_analytical[2][j]
ax.plot(np.arange(len(sa)) , abs(sa[:,0,0]), label= f'Analytical d = {d}  - seed = {seed}', color = colors[idx])
ax.set_xlabel('t')
ax.set_ylabel('Overlap')
ax.legend()
plt.show()
