from giant_learning.gradient_descent import SphericalGradientDescent, GradientDescent, ProjectedGradientDescent
from giant_learning.montecarlo_overlaps import MonteCarloOverlaps
from giant_learning.poly_poly import H3H3Overlaps

import numpy as np
from scipy.special import erf
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import matplotlib.pyplot as plt

p = 1
k = 1
noise = 1e-6
predictor_interaction = True
target = H3H3Overlaps._target
activation = H3H3Overlaps._activation
activation_derivative = H3H3Overlaps._activation_derivative
nseeds = 1
ds = np.logspace(7,9,base=2,num=4,dtype=int)

### save test error as a function of time for each seed and each d ###
T = 2*max(ds)**2
simu_test_errors = np.zeros((nseeds, len(ds), T+1))
Ms_projected = np.zeros((nseeds, len(ds), T+1, p, k))
Ms_spherical = np.zeros((nseeds, len(ds), T+1, p, k))
for i,d in enumerate(ds):
    n = int(d)
    t = 1/np.sqrt(d)  ### fix initial overlap
    gamma = .01 * n * np.power(d,-3/2)
    print(f'NOW Running d = {d}')
    for seed in range(nseeds):
        print(f'Seed = {seed}')
        rng = np.random.default_rng(seed+1)
        Wtarget = orth((normalize(rng.normal(size=(k,d)), axis=1, norm='l2')).T).T
        Wtild = normalize(rng.normal(size=(p,d)), axis=1, norm='l2')
        Wtild_target = np.einsum('ji,ri,rh->jh', Wtild , Wtarget ,Wtarget)
        W0_orth =  normalize(Wtild - Wtild_target, axis=1, norm='l2')
        W0 = -(t*normalize(Wtild_target,norm='l2',axis=1) + np.sqrt(1-t**2)*W0_orth)
        a0 = np.ones(p) ### It is changed with the new version of the package. The 1/p is included in giant-learning ###

        P = Wtarget @ Wtarget.T
        M0 = W0 @ Wtarget.T
        Q0 = W0 @ W0.T

        print(f'P = {P}')
        print(f'M0 = {M0}')
        print(f'Q0 = {Q0}')

        # Create a gradient descent object
        gd_projected = ProjectedGradientDescent(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative,
            gamma, noise, predictor_interaction = predictor_interaction,
            test_size = None, analytical_error= 'H3H3'
        )

        gd_spherical = SphericalGradientDescent(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative,
            gamma, noise, predictor_interaction = predictor_interaction,
            test_size = None, analytical_error= 'H3H3'
        )

        gd_projected.train(T)
        gd_spherical.train(T)
        Ws_projected = np.array(gd_projected.W_s)
        Ws_spherical = np.array(gd_spherical.W_s)
        # Measure cosine similarity between Ws and Wtarget as a function of time
        Ms_projected[seed, i] = Ws_projected @ Wtarget.T
        Ms_spherical[seed, i] = Ws_spherical @ Wtarget.T

        print(f'Ms.shape = {Ms_projected.shape}')

        simu_test_errors[seed, i, :] = gd_projected.test_errors



######## PLOTS ########

### Plot the average test error with std error bars as a function of time for different d ###
for i,d in enumerate(ds):
    # pass
    # simu_plot = plt.errorbar(np.arange(T+1), np.mean(simu_test_errors[:,i,:], axis=0), yerr=np.std(simu_test_errors[:,i,:], axis=0), label=f'SGD Simulation', marker='x', ls='', color='red') 
    plt.axvline(x=d**2, color='black', linestyle='--', label=f'T=d^2 ({d})')
    plt.plot(np.arange(T+1), Ms_projected[0,i,:,0,0], label=f'Projected SGD d={d}')
    plt.plot(np.arange(T+1), Ms_spherical[0,i,:,0,0], label=f'Spherical SGD d={d}')
plt.xlabel('Steps')
plt.ylabel('Test error')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

