from giant_learning.gradient_descent import GradientDescent
from giant_learning.staircase_overlaps import Hermite2Relu_Staricase2

import numpy as np
from scipy.special import erf
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import matplotlib.pyplot as plt

p = 2
k = 2
T = 100
gamma = .1
noise = 0.
alpha = 0.
target = Hermite2Relu_Staricase2._target
activation = Hermite2Relu_Staricase2._activation
activation_derivative = Hermite2Relu_Staricase2._activation_derivative
nseeds = 1
ds = np.logspace(4,10,base=2,num=4,dtype=int)

### save test error as a function of time for each seed and each d ###
simu_test_errors = np.zeros((nseeds, len(ds), T+1))
theo_test_errors = np.zeros((nseeds, len(ds), T+1))
theo_test_errors_alt = np.zeros((nseeds, len(ds), T+1))
theo_test_errors_alt_alt = np.zeros((nseeds, len(ds), T+1))
for i,d in enumerate(ds):
    n = d
    # t = 1/np.sqrt(d) ### fix initial overlap 
    t = .1
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

        # Create a gradient descent object
        gd = GradientDescent(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative, 
            gamma, noise, second_layer_update = False, test_size = None, analytical_error= 'hermite2ReLuStaircase2'
        )
        gd.train(T)
        simu_test_errors[seed, i, :] = gd.test_errors

        # Create a ErfErf object
        relustair = Hermite2Relu_Staricase2(
            P, M0, Q0, a0,
            gamma, noise,
            I4_diagonal=True, I4_offdiagonal=True,
            second_layer_update=False)
        resultstair_alt = Hermite2Relu_Staricase2(
            P, M0, Q0, a0,
            gamma, noise,
            I4_diagonal=False, I4_offdiagonal=True,
            second_layer_update=False)
        # resultstair_alt_alt = Hermite2Relu_Staricase2(
        #     P, M0, Q0, a0,
        #     gamma, noise,
        #     I4_diagonal=False, I4_offdiagonal=False,
        #     second_layer_update=False)
        relustair.train(T)
        resultstair_alt.train(T)
        # resultstair_alt_alt.train(T)

        print(gd.test_errors[0], relustair.test_errors[0])

        theo_test_errors[seed, i, :] = relustair.test_errors
        theo_test_errors_alt[seed, i, :] = resultstair_alt.test_errors
        # theo_test_errors_alt_alt[seed, i, :] = resultstair_alt_alt.test_errors


######## PLOTS ########

### Plot the average test error with std error bars as a function of time for different d ###
for i,d in enumerate(ds):
    simu_plot = plt.errorbar(np.arange(T+1), np.mean(simu_test_errors[:,i,:], axis=0), yerr=np.std(simu_test_errors[:,i,:], axis=0), label=f'Simulation - d={d}', marker='o', ls='')
theo_plot = plt.errorbar(np.arange(T+1), np.mean(theo_test_errors[:,i,:], axis=0), yerr=np.std(theo_test_errors[:,i,:], axis=0), label=f'Theory', marker='', color='black', linestyle='-')
theo_plot_alt = plt.errorbar(np.arange(T+1), np.mean(theo_test_errors_alt[:,i,:], axis=0), yerr=np.std(theo_test_errors_alt[:,i,:], axis=0), label=f'Theory alt', marker='', color='green', linestyle='--')
# theo_plot_alt_alt = plt.errorbar(np.arange(T+1), np.mean(theo_test_errors_alt_alt[:,i,:], axis=0), yerr=np.std(theo_test_errors_alt_alt[:,i,:], axis=0), label=f'Theory alt alt', marker='', color='red', linestyle=':')
plt.xlabel('Steps')
plt.ylabel('Test error')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

