import numpy as np
import matplotlib.pyplot as plt
import os

def twoDplot_at(t, show = 1, save = 0):
    # ion on the planes xy and xz
    fig, ax = plt.subplots(1,2, figsize = (8,4), sharey = True)
    ax[0].scatter(
        similarity_simulation[t,:,0],
        similarity_simulation[t,:,1],
        color = colors[t],
        label = f'{t+1} GD steps'
    )
    ax[1].scatter(
        similarity_simulation[t,:,2],
        similarity_simulation[t,:,1],
        color = colors[t],
        label = f'{t+1} GD steps'
    )
    ax[0].set_xlim(-1,1) 
    ax[0].set_ylim(-1,1)
    ax[1].set_xlim(-1,1) 
    ax[1].set_ylim(-1,1)

    circle00 = plt.Circle((0, 0), 1, color='black', fill=False)
    circle10 = plt.Circle((0, 0), 1, color='black', fill=False)
    
    ax[0].set_ylabel(r'$cos(G^{p}_i,w^*_2)$', fontsize = 10)
    ax[0].set_xlabel(r'$cos(G^{p}_i,w^*_1)$', fontsize = 10)
    ax[1].set_xlabel(r'$cos(G^{p}_i,w^*_3)$', fontsize = 10)
    # ax[1].set_ylabel(r'$cos(G^{\perp}_i,e_3)$', fontsize = 10)

    ax[0].add_artist(circle00)
    ax[1].add_artist(circle10)

    ax[0].legend(loc = 'center')
    # ax[1].legend()
    if save:
        path =  f"./results/figures/new_fig1_giant_step/tkn={target_tkn}_choice2={choice_2layer}"
        os.makedirs(path, exist_ok=True) 
        fig.savefig(f'{path}/2dim_plot_d={d}_T={t}',bbox_inches='tight')
    if show:
        plt.show()

def riskplot():
    # plot risk of GD and MC and GD no resample
    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(T+1), simulation.test_errors, label='GD', marker='o', color='orange')
    ax.plot(np.arange(T+1), simulation_noresample.test_errors, label='GD no resample', marker='o', color='green')
    ax.plot(np.arange(T+1), montecarlo.test_errors, label='MC', marker='o', color='blue')
    ax.set_xlabel('t') 
    ax.set_ylabel('Test Risk')
    ax.legend()
    plt.show()

def threeDplot_at(show = 1, save = 0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot the unit sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color='black',alpha=0.1)

    ax.set_xlim(-1,1) ; ax.set_ylim(-1,1) ; ax.set_zlim(-1,1)
    for t in range(T):
        ax.scatter(
            similarity_simulation[t,:,0],
            similarity_simulation[t,:,1],
            similarity_simulation[t,:,2],
            color = colors[t],
            label = f'{t+1} GD steps'
        )

    ax.legend()
    if save:
        path =  f"./results/figures/new_fig1_giant_step/tkn={target_tkn}_choice2={choice_2layer}"
        os.makedirs(path, exist_ok=True) 
        fig.savefig(f'{path}/3dim_plot_d={d}_T={t}',bbox_inches='tight')
    if show:
        plt.show()
    
colors = ['darkgreen', 'lime', 'olive']
tkns = ['111']
ds = [10] 
choices = ['hypercube']
T = 3
for target_tkn in tkns:
    for choice_2layer in choices:
        for d in ds:
            data = np.load(f'./results_cluster/data/new_fig1_giant_step/tkn={target_tkn}_choice2={choice_2layer}/d={d}.npz', allow_pickle=True)
            similarity_simulation = data['arr_0']
            for t in range(T):
                twoDplot_at(t, show = 1, save = 0)
            # threeDplot_at(show = 1, save = 1)


