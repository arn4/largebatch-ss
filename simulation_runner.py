from giant_learning.gradient_descent import SphericalGradientDescent, ProjectedGradientDescent, GradientDescent
from giant_learning.montecarlo_overlaps import MonteCarloOverlaps
from giant_learning.poly_poly import ProjectedH2H2Overlaps, H2H2Overlaps

import numpy as np
from scipy.special import erf
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import matplotlib.pyplot as plt

run = True
show = True

mu = 0.
delta = 3/2.
T = 500
nseeds = 1
ds = np.logspace(8-1,10-1,base=2,num=3,dtype=int)

p = 1
k = 1
noise = 0.
predictor_interaction = False
target = H2H2Overlaps._target
activation = H2H2Overlaps._activation
activation_derivative = H2H2Overlaps._activation_derivative

filename = f'computation-database/H2H2_d{ds}_nseeds{nseeds}_T{T}_l{delta}_mu{mu}_k{k}.npz'.replace(' ',',')

### RUN SIMULATIONS ###
if run:
    gd_plain_errors = np.zeros((nseeds, len(ds), T+1))
    gd_correlation_errors = np.zeros((nseeds, len(ds), T+1))
    projected_gd_plain_errors = np.zeros((nseeds, len(ds), T+1))
    projected_gd_correlation_errors = np.zeros((nseeds, len(ds), T+1))
    spherical_gd_plain_errors = np.zeros((nseeds, len(ds), T+1))
    spherical_gd_correlation_errors = np.zeros((nseeds, len(ds), T+1))

    projected_gd_plain_m = np.zeros((nseeds, len(ds), T+1, k))
    projected_gd_correlation_m = np.zeros((nseeds, len(ds), T+1, k))
    spherical_gd_plain_m = np.zeros((nseeds, len(ds), T+1, k))
    spherical_gd_correlation_m = np.zeros((nseeds, len(ds), T+1, k))


    for i,d in enumerate(ds):
        n = int(np.power(d,mu))
        t = 1/np.sqrt(d)  ### fix initial overlap
        t = 1e-10
        gamma = np.power(d,-delta)
        print(f'NOW Running d = {d}')
        for seed in range(nseeds):
            print(f'Seed = {seed}')
            rng = np.random.default_rng(seed)
            Wtarget = orth((normalize(rng.normal(size=(k,d)), axis=1, norm='l2')).T).T
            Wtild = normalize(rng.normal(size=(p,d)), axis=1, norm='l2')
            Wtild_target = np.einsum('ji,ri,rh->jh', Wtild , Wtarget ,Wtarget)
            W0_orth =  normalize(Wtild - Wtild_target, axis=1, norm='l2')
            W0 = (t*normalize(Wtild_target,norm='l2',axis=1) + np.sqrt(1-t**2)*W0_orth)
            a0 = np.ones(p) ### It is changed with the new version of the package. The 1/p is included in giant-learning ###

            print(f'Wtarget = {np.linalg.norm(Wtarget, axis=1)}')
            print(f'W0 = {np.linalg.norm(W0, axis=1)}')
            P = Wtarget @ Wtarget.T
            M0 = W0 @ Wtarget.T
            Q0 = W0 @ W0.T

            if M0[0][0] < 0:
                W0 = -W0
                M0 = -M0

            print(f'P = {P}')
            print(f'M0 = {M0}')
            print(f'Q0 = {Q0}')

            # Create a gradient descent object
            gd_plain = GradientDescent(
                target, Wtarget, n,
                activation, W0, a0, activation_derivative,
                gamma, noise, predictor_interaction=True,
                test_size = None, analytical_error= 'H2H2'
            )
            gd_correlation = GradientDescent(
                target, Wtarget, n,
                activation, W0, a0, activation_derivative,
                gamma, noise, predictor_interaction=False,
                test_size = None, analytical_error= 'H2H2'
            )
            spherical_gd_plain = SphericalGradientDescent(
                target, Wtarget, n,
                activation, W0, a0, activation_derivative,
                gamma, noise, predictor_interaction=True,
                test_size = None, analytical_error= 'H2H2'
            )
            spherical_gd_correlation = SphericalGradientDescent(
                target, Wtarget, n,
                activation, W0, a0, activation_derivative,
                gamma, noise, predictor_interaction=False,
                test_size = None, analytical_error= 'H2H2'
            )
            projected_gd_plain = ProjectedGradientDescent(
                target, Wtarget, n,
                activation, W0, a0, activation_derivative,
                gamma, noise, predictor_interaction=True,
                test_size = None, analytical_error= 'H2H2'
            )
            projected_gd_correlation = ProjectedGradientDescent(
                target, Wtarget, n,
                activation, W0, a0, activation_derivative,
                gamma, noise, predictor_interaction=False,
                test_size = None, analytical_error= 'H2H2'
            )

            spherical_gd_plain.train(T)
            print('Trained spherical_gd_plain')
            spherical_gd_correlation.train(T)
            print('Trained spherical_gd_correlation')
            projected_gd_plain.train(T)
            print('Trained projected_gd_plain')
            projected_gd_correlation.train(T)
            print('Trained projected_gd_correlation')
            

            projected_gd_plain_errors[seed, i, :] = projected_gd_plain.test_errors
            projected_gd_correlation_errors[seed, i, :] = projected_gd_correlation.test_errors
            spherical_gd_plain_errors[seed, i, :] = spherical_gd_plain.test_errors
            spherical_gd_correlation_errors[seed, i, :] = spherical_gd_correlation.test_errors

            projected_gd_plain_m[seed, i, :] = np.einsum('tja,ra->tjr', projected_gd_plain.W_s, Wtarget).reshape(T+1,1)
            projected_gd_correlation_m[seed, i, :] = np.einsum('tja,ra->tjr', projected_gd_correlation.W_s, Wtarget).reshape(T+1,1)
            spherical_gd_plain_m[seed, i, :] = np.einsum('tja,ra->tjr', spherical_gd_plain.W_s, Wtarget).reshape(T+1,1)
            spherical_gd_correlation_m[seed, i, :] = np.einsum('tja,ra->tjr', spherical_gd_correlation.W_s, Wtarget).reshape(T+1,1)

    ### Save data ###
    np.savez(filename,
        ds=ds,
        gd_plain_errors=gd_plain_errors,
        gd_correlation_errors=gd_correlation_errors,
        projected_gd_plain_errors=projected_gd_plain_errors,
        projected_gd_correlation_errors=projected_gd_correlation_errors,
        spherical_gd_plain_errors=spherical_gd_plain_errors,
        spherical_gd_correlation_errors=spherical_gd_correlation_errors
    )


######## PLOTS ########
# Load data
data = np.load(filename)
ds = data['ds']
gd_plain_errors = data['gd_plain_errors']
gd_correlation_errors = data['gd_correlation_errors']
projected_gd_plain_errors = data['projected_gd_plain_errors']
projected_gd_correlation_errors = data['projected_gd_correlation_errors']
spherical_gd_plain_errors = data['spherical_gd_plain_errors']
spherical_gd_correlation_errors = data['spherical_gd_correlation_errors']


if show:
    seed_plot = 0
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['x', 'o', 's', 'v', '^', '<', '>', 'd', 'p', 'h']
    for i,d in enumerate(ds):
        plt.plot(np.arange(T+1), projected_gd_plain_errors[seed_plot,i,:], label=f'Projected GD Plain d={d}', marker='s', color=colors[i], ls='')
        plt.plot(np.arange(T+1), projected_gd_correlation_errors[seed_plot,i,:], label=f'Projected GD Correlation d={d}', marker='x', color=colors[i], ls='')
        plt.plot(np.arange(T+1), spherical_gd_plain_errors[seed_plot,i,:], label=f'Spherical GD Plain d={d}', marker='o', color=colors[i], ls='')
        plt.plot(np.arange(T+1), spherical_gd_correlation_errors[seed_plot,i,:], label=f'Spherical GD Correlation d={d}', marker='<', color=colors[i], ls='')

    plt.xlabel('Steps')
    plt.ylabel('Test error')
    # plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

    for i,d in enumerate(ds):
        plt.plot(np.arange(T+1), abs(projected_gd_plain_m[seed_plot,i,:,0]), label=f'Projected GD Plain d={d}', marker='s', color=colors[i], ls='')
        plt.plot(np.arange(T+1), abs(projected_gd_correlation_m[seed_plot,i,:,0]), label=f'Projected GD Correlation d={d}', marker='x', color=colors[i], ls='')
        plt.plot(np.arange(T+1), abs(spherical_gd_plain_m[seed_plot,i,:,0]), label=f'Spherical GD Plain d={d}', marker='o', color=colors[i], ls='')
        plt.plot(np.arange(T+1), abs(spherical_gd_correlation_m[seed_plot,i,:,0]), label=f'Spherical GD Correlation d={d}', marker='<', color=colors[i], ls='')


    plt.xlabel('Steps')
    plt.ylabel('Overlap')
    # plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()
