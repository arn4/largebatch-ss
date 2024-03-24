from giant_learning.gradient_descent import SphericalGradientDescent, GradientDescent
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
predictor_interaction = False
target = H3H3Overlaps._target
activation = H3H3Overlaps._activation
activation_derivative = H3H3Overlaps._activation_derivative
nseeds = 1
ds = np.logspace(7,10,base=2,num=1,dtype=int)
T = 5*max(ds)**2
simu_test_errors = np.zeros((nseeds, len(ds), T+1))
simu_test_errors_sam = np.zeros((nseeds, len(ds), T+1))
for i,d in enumerate(ds):
    n = int(1)
    t = 4/np.sqrt(d)  ### fix initial overlap
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

        # Standard projected GD 
        gd = SphericalGradientDescent(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative,
            gamma, noise, predictor_interaction = not predictor_interaction,
            test_size = None, analytical_error= 'H3H3', resample_every = 1
        )
        gd.train(T) # Train for T steps
        Ws = np.array(gd.W_s) # Save Ws
        Ms = Ws @ Wtarget.T  # Overlap between Ws and Wtarget as a function of time
        simu_test_errors[seed, i, :] = gd.test_errors # Save test errors

        # Proxy for SAM -- resample every other step 
        sam = SphericalGradientDescent(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative,
            gamma, noise, predictor_interaction = not predictor_interaction,
            test_size = None, analytical_error= 'H3H3', resample_every = 2
        )
        sam.train(T) # Train for T steps
        Ws_sam = np.array(sam.W_s) # Save Ws
        Ms_sam = Ws_sam @ Wtarget.T # Overlap between Ws and Wtarget as a function of time
        simu_test_errors_sam[seed, i, :] = sam.test_errors # Save test errors



######## PLOTS ########

### Plot the average test error with std error bars as a function of time for different d ### 
### 2 subplots: first comparing the overlap os SAM vs SGD, second comparing test errors for different d ###
ax, fig = plt.subplots(1,2, figsize=(10,5))
for i,d in enumerate(ds):
    # Test errors
    ax[0].errorbar(np.arange(T+1), np.mean(simu_test_errors[:,i,:], axis=0), yerr=np.std(simu_test_errors[:,i,:], axis=0), label=f'SGD d={d}', marker='x', ls='')
    ax[0].errorbar(np.arange(T+1), np.mean(simu_test_errors_sam[:,i,:], axis=0), yerr=np.std(simu_test_errors_sam[:,i,:], axis=0), label=f'SAM d={d}', marker='x', ls='')
    # Overlaps
    ax[1].plot(np.arange(T+1), np.mean(Ms, axis=0), label=f'SGD d={d}')
    ax[1].plot(np.arange(T+1), np.mean(Ms_sam, axis=0), label=f'SAM d={d}')
ax[0].axvline(x=d**2, color='k', ls='--')
ax[0].set_xlabel('Steps')
ax[0].set_ylabel('Test error')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].legend()
ax[1].axvline(x=d**2, color='k', ls='--')
ax[1].set_xlabel('Steps')
ax[1].set_ylabel('Overlap')
ax[1].legend()
plt.show()

