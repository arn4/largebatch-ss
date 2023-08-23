from giant_learning.gradient_descent import GradientDescent
from giant_learning.montecarlo_overlaps import MonteCarloOverlaps
from giant_learning.erf_erf import ErfErfOverlaps

import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import matplotlib.pyplot as plt

k = 3
p = 80
d = 300
gamma0 = 4.
l = 1.15
noise = 0.
T = 3
alpha=0.
mc_samples = 1000000

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

Ws = np.array(simulation.W_s)
Ms = np.einsum('tji,ri->tjr', Ws, Wtarget)
Qs = np.einsum('tji,tlk->tjl', Ws, Ws)
P = Wtarget @ Wtarget.T
print('W',Ws.shape)
print('Q',Qs.shape)
print('P',P.shape)
print('Ms',Ms.shape)


Wupdates = Ws[1:] - Ws[:-1]
Wupdates_teachersubspace = Wupdates @ Wtarget.T

similarity_simulation = np.einsum(
    'tjr,tj,rr->tjr',
    Wupdates_teachersubspace,
    1/np.sqrt(np.einsum('tji,tji->tj', Wupdates_teachersubspace, Wupdates_teachersubspace)),
    1/np.sqrt(P)
)


Mupdates = Ms[1:] - Ms[:-1]
similarity_montecarlo = np.einsum(
    'tjr,tj,rr->tjr',
    Mupdates,
    1/np.sqrt(np.einsum('tji,tji->tj', Mupdates, Mupdates)),
    1/np.sqrt(P)
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
        similarity_simulation[t,:,2]
    )
    ax.scatter(
        similarity_montecarlo[t,:,0],
        similarity_montecarlo[t,:,1],
        similarity_montecarlo[t,:,2]
    )

    plt.show()

    # 2d plot of the projection on the planes xy and xz
    fig, ax = plt.subplots(2,2)
    ax[0,0].scatter(
        similarity_simulation[t,:,0],
        similarity_simulation[t,:,1],
    )
    ax[0,1].scatter(
        similarity_simulation[t,:,2],
        similarity_simulation[t,:,1],
    )

    ax[1,0].scatter(
        similarity_montecarlo[t,:,0],
        similarity_montecarlo[t,:,1],
    )
    ax[1,1].scatter(
        similarity_montecarlo[t,:,2],
        similarity_montecarlo[t,:,1],
    )


    ax[0].set_xlim(-1,1) ; ax[0].set_ylim(-1,1)
    ax[1].set_xlim(-1,1) ; ax[1].set_ylim(-1,1)

    circle0 = plt.Circle((0, 0), 1, color='black', fill=False)
    circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
    ax[0].add_artist(circle0)
    ax[1].add_artist(circle1)






    plt.show()



def twoDplot_at(t):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot the unit circle
    u = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(u), np.sin(u), color='black',alpha=0.1)

    ax.set_xlim(-1,1) ; ax.set_ylim(-1,1)

    ax.scatter(
        normalize_similarity_updates[t,:,0],
        normalize_similarity_updates[t,:,1],
    )
    plt.show()

threeDplot_at(0)
threeDplot_at(1)
threeDplot_at(2)