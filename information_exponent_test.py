from giant_learning.gradient_descent import GradientDescent
from giant_learning.montecarlo_overlaps import MonteCarloOverlaps
import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import matplotlib.pyplot as plt
from time import perf_counter_ns
import os 

experiment_tkns = ['12_log_conj', '12_linear_conj', '13','23']
alpha = 0. 
noise = 0.
k = 1 
gamma0 = 1 
H2 = lambda z: z**2 - 1
H3 = lambda z: z**3 - 3*z
mckey = False
mc_samples = 10000
activation = lambda x: np.maximum(x,0)
activation_derivative = lambda x: (x>0).astype(float)
ds = np.logspace(8,13,num = 5, base = 2, dtype = int) 
p = 1
folder_path =  f"./results_cluster/data/info_exp"
for d in ds:
    t1_start = perf_counter_ns()
    print(f'START d = {d}')
    if experiment_tkn == '12_log_conj':
        l = 1.15
        l_noresample = l
        target = lambda lft: np.mean(H2(lft))
        T = 20*np.log2(d).astype(int)
        xaxis = np.arange(T+1) / np.log2(d).astype(int)
    elif experiment_tkn == '12_linear_conj':
        l = 1.15
        l_noresample = l
        target = lambda lft: np.mean(H2(lft))
        T = 5*d.astype(int)
        xaxis = np.arange(T+1) / d
    elif experiment_tkn == '13':
        l = 1.15
        l_noresample = l
        target = lambda lft: np.mean(H3(lft))
        T = 5*d.astype(int)
        xaxis = np.arange(T+1) / d
    elif experiment_tkn == '23':
        l = 2
        l_noresample = l
        target = lambda lft: np.mean(H3(lft))
        T = 5*d.astype(int)
        xaxis = np.arange(T+1) / d
    # path to store the results
    hyper_path = f"/tkn={experiment_tkn}_d={d}_l={l}"
    path = folder_path + hyper_path
    # list to get the results
    store_error_simus = []
    store_error_simus_noresample = []
    store_error_montecarlos = []
    for seed in range(100):
        print(f'START seed = {seed}')
        Wtarget = orth((normalize(np.random.normal(size=(k,d)), axis=1, norm='l2')).T).T
        W0 = 1/np.sqrt(d) * np.random.normal(size=(p,d))
        a0 = np.sign(np.random.normal(size=(p,))) /np.sqrt(p)

        simulation = GradientDescent(
            target = target, W_target = Wtarget,
            activation = activation, W0 = W0, a0 = a0, activation_derivative = activation_derivative,
            gamma0 = gamma0, l = l, noise = noise,
            second_layer_update= False, alpha=alpha,
            resampling=True,
            test_size = mc_samples
        )
        
        simulation_noresample = GradientDescent(
            target = target, W_target = Wtarget,
            activation = activation, W0 = W0, a0 = a0, activation_derivative = activation_derivative,
            gamma0 = gamma0, l = l_noresample, noise = noise,
            second_layer_update= False, alpha=alpha,
            resampling=False,
            test_size=mc_samples
        )
        
        # train #
        simulation.train(T)
        simulation_noresample.train(T)

        # compute and store magnetizations # 

        Ws = np.array(simulation.W_s)
        Ws_noresample = np.array(simulation_noresample.W_s)
        Ms_simulation = np.einsum('tji,ri->tjr', Ws, Wtarget)
        Qs_simulation = np.einsum('tji,tlk->tjl', Ws, Ws)
        Ms_simulation_noresample = np.einsum('tji,ri->tjr', Ws_noresample, Wtarget)
        Qs_simulation_noresample = np.einsum('tji,tlk->tjl', Ws_noresample, Ws_noresample)

        # mcmc
        if mckey:
            montecarlo = MonteCarloOverlaps(
                target, activation, activation_derivative,
                Wtarget @ Wtarget.T, W0 @ Wtarget.T, W0 @ W0.T, a0,
                gamma0, d, l, noise,
                False, alpha,
                mc_size = mc_samples
            )
            montecarlo.train(T)
            Ms_montecarlo = np.array(montecarlo.Ms)
            Qs_montecarlo = np.array(montecarlo.Qs)
        else:
            Ms_montecarlo = np.zeros((T+1, k, p))
            Qs_montecarlo = np.zeros((T+1, k, k))

        # store the magnetizations 
        store_error_simus.append(np.abs(Ms_simulation))
        store_error_simus_noresample.append(np.abs(Ms_simulation_noresample))
        store_error_montecarlos.append(np.abs(Ms_montecarlo))

    # save the results with mean and std
    os.makedirs(path, exist_ok=True)
    np.savez(f'{path}/xaxis.npz', xaxis)
    np.savez(f'{path}/error_simus.npz',np.mean(store_error_simus, axis = 0))
    np.savez(f'{path}/error_simus_noresample.npz',np.mean(store_error_simus_noresample, axis = 0))
    np.savez(f'{path}/error_montecarlos.npz',np.mean(store_error_montecarlos, axis = 0))
    np.savez(f'{path}/std_simus.npz',np.std(store_error_simus, axis = 0))
    np.savez(f'{path}/std_simus_noresample.npz',np.std(store_error_simus_noresample, axis = 0))
    np.savez(f'{path}/std_montecarlos.npz',np.std(store_error_montecarlos, axis = 0))
    t1_stop = perf_counter_ns()
    print(f"Elapsed time for d={d}:", (t1_stop - t1_start)*1e-9, 's')


