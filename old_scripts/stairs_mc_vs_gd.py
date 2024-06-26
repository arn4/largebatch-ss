from giant_learning.gradient_descent import GradientDescent
from giant_learning.montecarlo_overlaps import MonteCarloOverlaps
from giant_learning.erf_erf import ErfErfOverlaps

import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import matplotlib.pyplot as plt

k = 3
p = 100
d = 4000
gamma0 = .5
l = 1.15
noise = 0.
T = 5
alpha=0.
mc_samples = 100000

def target(lft):
    return lft[...,0]/3 + 2*lft[...,0]*lft[...,1] + lft[...,1]*lft[...,2]

# def target(lft):
#     return lft[...,0]/6 + lft[...,0]*lft[...,1]  + 2*lft[...,1]*lft[...,2]*lft[...,0]

activation = lambda x: np.maximum(x,0)
activation_derivative = lambda x: (x>0).astype(float)

Wtarget = orth((normalize(np.random.normal(size=(k,d)), axis=1, norm='l2')).T).T
W0 = 1/np.sqrt(d) * np.random.normal(size=(p,d))
a0 = np.sign(np.random.normal(size=(p,))) /np.sqrt(p)

simulation = GradientDescent(
    target = target, W_target = Wtarget,
    activation = activation, W0 = W0, a0 = a0, activation_derivative = activation_derivative,
    gamma0 = gamma0, l = l, noise = noise,
    second_layer_update= False, alpha=alpha,
    resampling=True,
    test_size=mc_samples
)

simulation_noresample = GradientDescent(
    target = target, W_target = Wtarget,
    activation = activation, W0 = W0, a0 = a0, activation_derivative = activation_derivative,
    gamma0 = gamma0, l = l, noise = noise,
    second_layer_update= False, alpha=alpha,
    resampling=False,
    test_size=mc_samples
)

montecarlo = MonteCarloOverlaps(
    target, activation, activation_derivative,
    Wtarget @ Wtarget.T, W0 @ Wtarget.T, W0 @ W0.T, a0,
    gamma0, d, l, noise,
    False, alpha,
    mc_size = mc_samples
)

montecarlo.train(T)
simulation.train(T)
simulation_noresample.train(T)

Ws = np.array(simulation.W_s)
Ws_noresample = np.array(simulation_noresample.W_s)
Ms_simulation = np.einsum('tji,ri->tjr', Ws, Wtarget)
Qs_simulation = np.einsum('tji,tlk->tjl', Ws, Ws)
Ms_simulation_noresample = np.einsum('tji,ri->tjr', Ws_noresample, Wtarget)
Qs_simulation_noresample = np.einsum('tji,tlk->tjl', Ws_noresample, Ws_noresample)
Ms_montecarlo = np.array(montecarlo.Ms)
Qs_montecarlo = np.array(montecarlo.Qs)
P = Wtarget @ Wtarget.T



Wupdates = Ws[1:] - Ws[:-1]
Wupdates_noresample = Ws_noresample[1:] - Ws_noresample[:-1]
Mupdates_simulation = Wupdates @ Wtarget.T
Mupdates_simulation_noresample = Wupdates_noresample @ Wtarget.T

similarity_simulation = np.einsum(
    'tjr,tj,r->tjr',
    Mupdates_simulation,
    1/np.sqrt(np.einsum('tji,tji->tj', Mupdates_simulation, Mupdates_simulation)),
    1/np.sqrt(np.diag(P))
)

similarity_simulation_noresample = np.einsum(
    'tjr,tj,r->tjr',
    Mupdates_simulation_noresample,
    1/np.sqrt(np.einsum('tji,tji->tj', Mupdates_simulation_noresample, Mupdates_simulation_noresample)),
    1/np.sqrt(np.diag(P))
)

Mupdates_montercalo = Ms_montecarlo[1:] - Ms_montecarlo[:-1]
similarity_montecarlo = np.einsum(
    'tjr,tj,r->tjr',
    Mupdates_montercalo,
    1/np.sqrt(np.einsum('tji,tji->tj', Mupdates_montercalo, Mupdates_montercalo)),
    1/np.sqrt(np.diag(P))
)

# 3D plot T similarity matrices, last dimension are the 3 coordinates of the points
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
    ax.scatter(
        similarity_montecarlo[t,:,0],
        similarity_montecarlo[t,:,1],
        similarity_montecarlo[t,:,2],
        color = 'blue',
        label='MC'
    )
    ax.scatter(
        similarity_simulation_noresample[t,:,0],
        similarity_simulation_noresample[t,:,1],
        similarity_simulation_noresample[t,:,2],
        color = 'green',
        label='GD no resample'
    )
    ax.legend()

    plt.show()

    # 2d plot of the projection on the planes xy and xz
    fig, ax = plt.subplots(3,2)
    ax[0,0].scatter(
        similarity_simulation[t,:,0],
        similarity_simulation[t,:,1],
        color = 'orange'
    )
    ax[0,1].scatter(
        similarity_simulation[t,:,2],
        similarity_simulation[t,:,1],
        color = 'orange'
    )

    ax[1,0].scatter(
        similarity_montecarlo[t,:,0],
        similarity_montecarlo[t,:,1],
        color = 'blue'
    )
    ax[1,1].scatter(
        similarity_montecarlo[t,:,2],
        similarity_montecarlo[t,:,1],
        color = 'blue'
    )

    ax[2,0].scatter(
        similarity_simulation_noresample[t,:,0],
        similarity_simulation_noresample[t,:,1],
        color = 'green'
    )
    ax[2,1].scatter(
        similarity_simulation_noresample[t,:,2],
        similarity_simulation_noresample[t,:,1],
        color = 'green'
    )



    ax[0,0].set_xlim(-1,1) ; ax[0,0].set_ylim(-1,1)
    ax[0,1].set_xlim(-1,1) ; ax[0,1].set_ylim(-1,1)
    ax[1,0].set_xlim(-1,1) ; ax[1,0].set_ylim(-1,1)
    ax[1,1].set_xlim(-1,1) ; ax[1,1].set_ylim(-1,1)
    ax[2,0].set_xlim(-1,1) ; ax[2,0].set_ylim(-1,1)
    ax[2,1].set_xlim(-1,1) ; ax[2,1].set_ylim(-1,1)

    circle00 = plt.Circle((0, 0), 1, color='black', fill=False)
    circle10 = plt.Circle((0, 0), 1, color='black', fill=False)
    circle01 = plt.Circle((0, 0), 1, color='black', fill=False)
    circle11 = plt.Circle((0, 0), 1, color='black', fill=False)
    circle20 = plt.Circle((0, 0), 1, color='black', fill=False)
    circle21 = plt.Circle((0, 0), 1, color='black', fill=False)

    ax[0,0].add_artist(circle00)
    ax[1,0].add_artist(circle10)
    ax[0,1].add_artist(circle01)
    ax[1,1].add_artist(circle11)
    ax[2,0].add_artist(circle20)
    ax[2,1].add_artist(circle21)

    ax[0,0].legend()
    ax[1,0].legend()
    ax[0,1].legend()
    ax[1,1].legend()
    ax[2,0].legend()
    ax[2,1].legend()


    plt.show()

def riskplot():
    # plot risk of GD and MC and GD no resample
    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(T+1), simulation.test_errors, label='GD', marker='o', color='orange')
    ax.plot(np.arange(T+1), simulation_noresample.test_errors, label='GD no resample', marker='o', color='green')
    ax.plot(np.arange(T+1), montecarlo.test_errors, label='MC', marker='o', color='blue')
    ax.legend()
    plt.show()



for t in range(T):
    threeDplot_at(t)
riskplot()