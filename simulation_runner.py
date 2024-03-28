from giant_learning.gradient_descent import SphericalGradientDescent, ProjectedGradientDescent, GradientDescent
from giant_learning.montecarlo_overlaps import MonteCarloOverlaps
from giant_learning.poly_poly import ProjectedH3H3Overlaps, H3H3Overlaps

import numpy as np
from scipy.special import erf
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import matplotlib.pyplot as plt

import os

from simulation_conf import *

force_run = False

def get_params():
    params = []
    for mu in mus:
        for d in ds:
            delta = get_delta(mu, d)
            T = get_T(delta, mu, d)
            params.append((mu, delta, d, int(T), ic_seed))
    return params

#        (mu, delta, d, T, ic_seed)
params = get_params()
      
target = H3H3Overlaps._target
activation = H3H3Overlaps._activation
activation_derivative = H3H3Overlaps._activation_derivative

### RUN SIMULATIONS ###
for mu, delta, d, T, ic_seed in params:
    print(f'Now running: mu = {mu}, delta = {delta}, d = {d}, T = {T}, ic_seed = {ic_seed}')

    rng = np.random.default_rng(ic_seed)
    n = int(np.power(d, mu))
    gamma = np.power(d, -delta)
    
    ## Initial condition
    init_corr = 1/np.sqrt(d)
    Wtarget = orth((normalize(rng.normal(size=(k,d)), axis=1, norm='l2')).T).T
    Wtild = normalize(rng.normal(size=(p,d)), axis=1, norm='l2')
    Wtild_target = np.einsum('ji,ri,rh->jh', Wtild , Wtarget ,Wtarget)
    W0_orth =  normalize(Wtild - Wtild_target, axis=1, norm='l2')
    W0 = (init_corr*normalize(Wtild_target,norm='l2',axis=1) + np.sqrt(1-init_corr**2)*W0_orth)
    a0 = np.ones(p)

    P = Wtarget @ Wtarget.T
    M0 = W0 @ Wtarget.T
    Q0 = W0 @ W0.T

    # Assert same 
    if M0[0,0] < 0:
        W0 = -W0
        M0 = -M0
    
    for seed in range(nseeds):
        print(f'\tSeed = {seed}')
        gd_plain_errors = np.zeros((T+1))
        gd_correlation_errors = np.zeros((T+1))
        projected_gd_plain_errors = np.zeros((T+1))
        projected_gd_correlation_errors = np.zeros((T+1))
        spherical_gd_plain_errors = np.zeros((T+1))
        spherical_gd_correlation_errors = np.zeros((T+1))

        gd_plain_m = np.zeros((T+1, k))
        gd_correlation_m = np.zeros((T+1, k))
        projected_gd_plain_m = np.zeros((T+1, k))
        projected_gd_correlation_m = np.zeros((T+1, k))
        spherical_gd_plain_m = np.zeros((T+1, k))
        spherical_gd_correlation_m = np.zeros((T+1, k))

        # Create a gradient descent object
        gd_plain = GradientDescent(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative,
            gamma, noise, predictor_interaction=True,
            test_size = None, analytical_error= 'H3H3',
            seed = seed
        )
        gd_correlation = GradientDescent(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative,
            gamma, noise, predictor_interaction=False,
            test_size = None, analytical_error= 'H3H3',
            seed = seed
        )
        spherical_gd_plain = SphericalGradientDescent(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative,
            gamma, noise, predictor_interaction=True,
            test_size = None, analytical_error= 'H3H3',
            seed = seed
        )
        spherical_gd_correlation = SphericalGradientDescent(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative,
            gamma, noise, predictor_interaction=False,
            test_size = None, analytical_error= 'H3H3',
            seed = seed
        )
        projected_gd_plain = ProjectedGradientDescent(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative,
            gamma, noise, predictor_interaction=True,
            test_size = None, analytical_error= 'H3H3',
            seed = seed
        )
        projected_gd_correlation = ProjectedGradientDescent(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative,
            gamma, noise, predictor_interaction=False,
            test_size = None, analytical_error= 'H3H3',
            seed = seed
        )

        filename = f"computation-database/{'debug' if debug else ''}/H3H3_delta{delta}_mu{mu}_d{d}_T{T}_icseed{ic_seed}_seed{seed}.npz"

        if force_run or not os.path.exists(filename):
            spherical_gd_plain.train(T, verbose=True)
            print('Trained spherical_gd_plain')
            spherical_gd_correlation.train(T, verbose=True)
            print('Trained spherical_gd_correlation')
            projected_gd_plain.train(T, verbose=True)
            print('Trained projected_gd_plain')
            projected_gd_correlation.train(T, verbose=True)
            print('Trained projected_gd_correlation')
        
            # gd_plain_errors = gd_plain.test_errors
            # gd_correlation_errors = gd_correlation.test_errors
            projected_gd_plain_errors = projected_gd_plain.test_errors
            projected_gd_correlation_errors = projected_gd_correlation.test_errors
            spherical_gd_plain_errors = spherical_gd_plain.test_errors
            spherical_gd_correlation_errors = spherical_gd_correlation.test_errors

            # gd_plain_m = np.einsum('tja,ra->tjr', gd_plain.W_s, Wtarget).reshape(T+1,1)
            # gd_correlation_m = np.einsum('tja,ra->tjr', gd_correlation.W_s, Wtarget).reshape(T+1,1)
            projected_gd_plain_m = np.einsum('tja,ra->tjr', projected_gd_plain.W_s, Wtarget).reshape(T+1)
            projected_gd_correlation_m = np.einsum('tja,ra->tjr', projected_gd_correlation.W_s, Wtarget).reshape(T+1)
            spherical_gd_plain_m = np.einsum('tja,ra->tjr', spherical_gd_plain.W_s, Wtarget).reshape(T+1,1)
            spherical_gd_correlation_m = np.einsum('tja,ra->tjr', spherical_gd_correlation.W_s, Wtarget).reshape(T+1)

            ### Save data ###
            np.savez(filename,
                gd_plain_errors=gd_plain_errors,
                gd_correlation_errors=gd_correlation_errors,
                projected_gd_plain_errors=projected_gd_plain_errors,
                projected_gd_correlation_errors=projected_gd_correlation_errors,
                spherical_gd_plain_errors=spherical_gd_plain_errors,
                spherical_gd_correlation_errors=spherical_gd_correlation_errors,
                #
                gd_plain_m=gd_plain_m,
                gd_correlation_m=gd_correlation_m,
                projected_gd_plain_m=projected_gd_plain_m,
                projected_gd_correlation_m=projected_gd_correlation_m,
                spherical_gd_plain_m=spherical_gd_plain_m,
                spherical_gd_correlation_m=spherical_gd_correlation_m
            )
        else:
                print(f'File {filename} already exists. Skipping computation.')




