from giant_learning.gradient_descent import GradientDescent
from giant_learning.montecarlo_overlaps import MonteCarloOverlaps
from giant_learning.erf_erf import ErfErfOverlaps

import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import matplotlib.pyplot as plt
from time import perf_counter_ns

alpha = 0. ; noise = 0.

k = 3 ; gamma0 = 4 ; T = 3 ; mc_samples = 100000 ; p = 100
H_2 = lambda z: z**2 - 1
def target(lft):
    if target_tkn == '111':
        return lft[...,0]/3 + 2*lft[...,0]*lft[...,1] + lft[...,1]*lft[...,2]
    if target_tkn == '120':
        return lft[...,0]/3 + 2*H_2(lft[...,0])*lft[...,1] + lft[...,0]*lft[...,2]
    if target_tkn == '100':
        return lft[...,0]/3 + 2*lft[...,0]*H_2(lft[...,1]) + lft[...,1]*lft[...,2]

# 3D plot T similarity matrices, last dimension are the 3 coordinates of the points
def twoDplot_at(t):
    # ion on the planes xy and xz
    fig, ax = plt.subplots(1,2)
    ax[0].scatter(
        similarity_simulation[t,:,0],
        similarity_simulation[t,:,1],
        color = 'orange',
        label = 'GD'
    )
    ax[1].scatter(
        similarity_simulation[t,:,2],
        similarity_simulation[t,:,1],
        color = 'orange',
        label = 'GD'
    )
    ax[0].set_xlim(-1,1) ; ax[0].set_ylim(-1,1)
    ax[1].set_xlim(-1,1) ; ax[1].set_ylim(-1,1)

    circle00 = plt.Circle((0, 0), 1, color='black', fill=False)
    circle10 = plt.Circle((0, 0), 1, color='black', fill=False)


    ax[0].add_artist(circle00)
    ax[1].add_artist(circle10)

    ax[0].legend()
    ax[1].legend()

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

def threeDplot_at(t):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot the unit sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color='black',alpha=0.1)

    ax.set_xlim(-1,1) ; ax.set_ylim(-1,1) ; ax.set_zlim(-1,1)

    ax.scatter(
        similarity_simulation[t,:,0],
        similarity_simulation[t,:,1],
        similarity_simulation[t,:,2],
        color = 'orange',
        label='GD'
    )

    ax.legend()

    plt.show()
second_layers = {'gaussian': 1/np.sqrt(p)*np.random.randn(p) , 'hypercube': np.sign(np.random.normal(size=(p,))) /np.sqrt(p) , 'ones': np.ones(p)/np.sqrt(p)}

tkns = ['111','120','100']
ds = [10,20,40] 
choices = ['gaussian', 'hypercube', 'ones']
for d in ds: 
    for target_tkn in tkns:
        for choice_2layer in choices:
            t1_start = perf_counter_ns()
            print(f'START d = {d}')
            l = np.log(4*d) / np.log(d)

            activation = lambda x: np.maximum(x,0)
            activation_derivative = lambda x: (x>0).astype(float)

            Wtarget = orth((normalize(np.random.normal(size=(k,d)), axis=1, norm='l2')).T).T
            W0 = 1/np.sqrt(d) * np.random.normal(size=(p,d))
            a0 = second_layers[choice_2layer]

            simulation = GradientDescent(
                target = target, W_target = Wtarget,
                activation = activation, W0 = W0, a0 = a0, activation_derivative = activation_derivative,
                gamma0 = gamma0, l = l, noise = noise,
                second_layer_update= False, alpha=alpha,
                resampling=True,
                test_size=mc_samples
            )
            
            simulation.train(T)

            Ws = np.array(simulation.W_s)
            Ms_simulation = np.einsum('tji,ri->tjr', Ws, Wtarget)
            Qs_simulation = np.einsum('tji,tlk->tjl', Ws, Ws)

            
            P = Wtarget @ Wtarget.T

            Wupdates = Ws[1:] - Ws[:-1]
            Mupdates_simulation = Wupdates @ Wtarget.T


            similarity_simulation = np.einsum(
                'tjr,tj,r->tjr',
                Mupdates_simulation,
                1/np.sqrt(np.einsum('tji,tji->tj', Mupdates_simulation, Mupdates_simulation)),
                1/np.sqrt(np.diag(P))
            )

            
            # for t in range(T):
            #     twoDplot_at(t)
            #     threeDplot_at(t)
            t1_stop = perf_counter_ns()
            print("Elapsed time:", (t1_stop - t1_start)*1e-9, 's')
            np.savez(f'./results_reproduce/data_d={d}_tkn={target_tkn}_choice2={choice_2layer}.npz',similarity_simulation)