from giant_learning.gradient_descent import SphericalGradientDescent, GradientDescent
from giant_learning.montecarlo_overlaps import MonteCarloOverlaps
from giant_learning.poly_poly import H3H3Overlaps

import numpy as np
from scipy.special import erf
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import matplotlib.pyplot as plt

p = 100
k = 1
noise = 1e-6
predictor_interaction = False
target = H3H3Overlaps._target
activation = H3H3Overlaps._activation
activation_derivative = H3H3Overlaps._activation_derivative
nseeds = 1
ds = np.logspace(5, 7, base = 2, num = 3, dtype=int)
# choice_gamma = 'sgd_optimal' 
choice_gamma = 'sam'
Ts = []
test_errors_mean = [] 
test_errors_std = []
test_errors_sam = []
test_errors_sam_std = []
Ms_mean = []
Ms_std = []
Ms_sam_mean = []
Ms_sam_std = []
for i,d in enumerate(ds):
    T = int(1.5*d**2)
    Ts.append(T)
    n = int(1)
    t = 1/np.sqrt(d)  ### fix initial overlap
    if choice_gamma == 'sgd_optimal':
        gamma = .01 * n * np.power(d,-3/2)
    else: 
        gamma = .01 * n * np.power(d,-1.)
    simu_test_errors = np.zeros((nseeds, T+1))
    simu_test_errors_sam = np.zeros((nseeds,  T+1))
    Ms = np.zeros((nseeds, T+1, p, k))
    Ms_sam = np.zeros((nseeds, T+1, p, k))
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
        if M0[0,0] < 0:
            W0 = -W0
            M0 = -M0
        # Standard spherical SGD 
        gd = SphericalGradientDescent(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative,
            gamma, noise, predictor_interaction = not predictor_interaction,
            test_size = None, analytical_error= 'H3H3', resample_every = 1
        )
        gd.train(T) # Train for T steps
        Ws = np.array(gd.W_s) # Save Ws
        simu_test_errors[seed, :] = gd.test_errors # Save test errors
        Ms[seed, :] = Ws @ Wtarget.T

        # Proxy for SAM -- resample every other step 
        sam = SphericalGradientDescent(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative,
            gamma, noise, predictor_interaction = not predictor_interaction,
            test_size = None, analytical_error= 'H3H3', resample_every = 2
        )
        sam.train(T) # Train for T steps
        Ws_sam = np.array(sam.W_s) # Save Ws
        simu_test_errors_sam[seed, :] = sam.test_errors # Save test errors
        Ms_sam[seed, :] = Ws_sam @ Wtarget.T
    test_errors_mean.append(np.mean(simu_test_errors, axis=0))
    test_errors_std.append(np.std(simu_test_errors, axis=0))
    test_errors_sam.append(np.mean(simu_test_errors_sam, axis=0))
    test_errors_sam_std.append(np.std(simu_test_errors_sam, axis=0))
    Ms_mean.append(np.mean(Ms, axis=0))
    Ms_std.append(np.std(Ms, axis=0))
    Ms_sam_mean.append(np.mean(Ms_sam, axis=0))
    Ms_sam_std.append(np.std(Ms_sam, axis=0))

### SAVE ###

np.savez(f'computation-database/sam_vs_sgd_p={p}_gamma={choice_gamma}.npz', ds=ds, Ts=Ts, test_errors_mean=test_errors_mean, test_errors_std=test_errors_std, test_errors_sam=test_errors_sam, test_errors_sam_std=test_errors_sam_std, Ms_mean=Ms_mean, Ms_std=Ms_std, Ms_sam_mean=Ms_sam_mean, Ms_sam_std=Ms_sam_std)
