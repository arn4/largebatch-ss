from giant_learning.gradient_descent import SphericalGradientDescent, GradientDescent
from giant_learning.montecarlo_overlaps import MonteCarloOverlaps
from giant_learning.erf_erf import SphericalErfErfOverlaps, ErfErfOverlaps

import numpy as np
from scipy.special import erf
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import matplotlib.pyplot as plt

p = 1
# l = 1.
k = 1
T = 500
gamma = 1.
noise = 1e-6
alpha = 0.
target = lambda x: np.mean(erf(x/np.sqrt(2)), axis=-1)
activation = lambda x: erf(x/np.sqrt(2))
activation_derivative = lambda x: np.sqrt(2/np.pi) * np.exp(-x**2/2)
nseeds = 1
ds = np.logspace(8,9,base=2,num=3,dtype=int)

### save test error as a function of time for each seed and each d ###
simu_test_errors = np.zeros((nseeds, len(ds), T+1))
spherical_simu_test_errors = np.zeros((nseeds, len(ds), T+1))
theo_test_errors = np.zeros((nseeds, len(ds), T+1))
spherical_test_errors = np.zeros((nseeds, len(ds), T+1))
for i,d in enumerate(ds):
    n = d
    # t = 1/np.sqrt(d) ### fix initial overlap 
    t = .0
    print(f'NOW Running d = {d}')
    for seed in range(nseeds):
        print(f'Seed = {seed}')
        Wtarget = orth((normalize(np.random.normal(size=(k,d)), axis=1, norm='l2')).T).T
        Wtild = normalize(np.random.normal(size=(p,d)), axis=1, norm='l2')
        Wtild_target = np.einsum('ji,ri,rh->jh', Wtild , Wtarget ,Wtarget)
        W0_orth =  normalize(Wtild - Wtild_target, axis=1, norm='l2')
        W0 = (t*normalize(Wtild_target,norm='l2',axis=1) + np.sqrt(1-t**2)*W0_orth)
        a0 = np.ones(p) ### It is changed with the new version of the package. The 1/p is included in giant-learning ###

        P = Wtarget @ Wtarget.T
        M0 = W0 @ Wtarget.T
        Q0 = W0 @ W0.T

        print(P, M0, Q0)

        # Create a gradient descent object
        gd = GradientDescent(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative, 
            gamma, noise, second_layer_update = False, test_size = None, analytical_error= 'erferf'
        )
        spherical_gd = SphericalGradientDescent(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative, 
            gamma, noise, second_layer_update = False, test_size = None, analytical_error= 'erferf'
        )

        gd.train(T)
        spherical_gd.train(T)

        simu_test_errors[seed, i, :] = gd.test_errors
        spherical_simu_test_errors[seed, i, :] = spherical_gd.test_errors

        # Create a ErfErf object
        erferf = ErfErfOverlaps(
            P, M0, Q0, a0,
            gamma, noise,
            I4_diagonal=True, I4_offdiagonal=True,
            second_layer_update=False)
        
        erferf_spherical = SphericalErfErfOverlaps(
            P, M0, Q0, a0,
            gamma, noise,
            I4_diagonal=True, I4_offdiagonal=True,
            second_layer_update=False)

        erferf.train(T)
        erferf_spherical.train(T)

        print('m=',erferf_spherical.Ms[14])
        print('q=',erferf_spherical.Qs[14])

        theo_test_errors[seed, i, :] = erferf.test_errors
        spherical_test_errors[seed, i, :] = erferf_spherical.test_errors


######## PLOTS ########

### Plot the average test error with std error bars as a function of time for different d ###
for i,d in enumerate(ds):
    simu_plot = plt.errorbar(np.arange(T+1), np.mean(simu_test_errors[:,i,:], axis=0), yerr=np.std(simu_test_errors[:,i,:], axis=0), label=f'Simulation - d={d}', marker='o', ls='')
    spherical_simu_plot = plt.errorbar(np.arange(T+1), np.mean(spherical_simu_test_errors[:,i,:], axis=0), yerr=np.std(spherical_simu_test_errors[:,i,:], axis=0), label=f'Spherical Simulation - d={d}', marker='x', ls='')
theo_plot = plt.errorbar(np.arange(T+1), np.mean(theo_test_errors[:,i,:], axis=0), yerr=np.std(theo_test_errors[:,i,:], axis=0), label=f'Theory', marker='', color='black', linestyle='-')
spherical_plot = plt.errorbar(np.arange(T+1), abs(np.mean(spherical_test_errors[:,i,:], axis=0)), yerr=np.std(spherical_test_errors[:,i,:], axis=0), label=f'Spherical Theory', marker='', color='gray', linestyle='--')
plt.xlabel('Steps')
plt.ylabel('Test error')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

