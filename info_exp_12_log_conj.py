from giant_learning.gradient_descent import GradientDescent
from giant_learning.montecarlo_overlaps import MonteCarloOverlaps
import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import matplotlib.pyplot as plt
from time import perf_counter_ns
import os 

alpha = 0. 
noise = 0.
k = 1  
gamma0 = 0.01
H2 = lambda z: z**2 - 1
mckey = False
mc_samples = 10000
l = 1.2
l_noresample = l
def target(lft):
    return np.mean(H2(lft))
activations = {"relu": lambda x: np.maximum(x,0), "h2": H2}
activation_derivatives = {"relu": lambda x: (x>0).astype(float), "h2": lambda x: 2*x}
act_tkn = "h2"
activation = activations[act_tkn]
activation_derivative = activation_derivatives[act_tkn]
ds = np.logspace(10,12,num = 3, base = 2, dtype = int) 
p = 1
nseeds = 500
folder_path =  f"./results_cluster/data/info_exp_12"
starts = ["random", "warm", "tiepide", "cold"]
for start in starts:
    error_simus = [] 
    error_simus_noresample = []
    error_montecarlos = []
    std_simus = []
    std_simus_noresample = []
    std_montecarlos = []
    xaxiss = []
    hyper_path = f"/l={l}_noise={noise}_gamma0={gamma0}_activation={act_tkn}_p={p}_start={start}"
    path = folder_path + hyper_path
    if start == "warm":
        t = 0.99
    elif start == "tiepide":
        t = 0.7
    elif start == "cold":
        t = 0.3
    else:
        t = 0
    for d in ds:
        t1_start = perf_counter_ns()
        print(f'{start} START d = {d}')
        T = 10*np.log2(d).astype(int)
        xaxis = np.arange(T+1) / np.log2(d).astype(int)
        xaxiss.append(xaxis)
        store_error_simus = []
        store_error_simus_noresample = []
        store_error_montecarlos = []
        for seed in range(nseeds):
            Wtarget = orth((normalize(np.random.normal(size=(k,d)), axis=1, norm='l2')).T).T
            a0 = np.sign(np.random.normal(size=(p,))) /np.sqrt(p)
            W0 = t*Wtarget + np.sqrt(1-t**2)*1/np.sqrt(d) * np.random.normal(size=(p,d))
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

            # compute weights # 
            Ws = np.array(simulation.W_s)
            Ws_noresample = np.array(simulation_noresample.W_s)

            # compute magnetizations #
            Ms_simulation = np.einsum('tji,ri->tjr', Ws, Wtarget)
            similarity_simulation = np.einsum(
                'tjr,tj->tjr',
                Ms_simulation,
                1/np.sqrt(np.einsum('tji,tji->tj', Ws, Ws)),
            )

            Ms_simulation_noresample = np.einsum('tji,ri->tjr', Ws_noresample, Wtarget)
            similarity_simulation_noresample = np.einsum(
                'tjr,tj->tjr',
                Ms_simulation_noresample,
                1/np.sqrt(np.einsum('tji,tji->tj', Ws_noresample, Ws_noresample)),
            )
            ##### mcmc #####
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
            else:
                Ms_montecarlo = np.zeros((T+1, k, p))

            # store the magnetizations 
            store_error_simus.append(np.abs(similarity_simulation))
            store_error_simus_noresample.append(np.abs(similarity_simulation_noresample))
            ### temporary store whatever for mc ###
            store_error_montecarlos.append(np.abs(Ms_montecarlo))  
            print(f'At seed {seed} for d = {d} the error is {np.mean(np.abs(similarity_simulation[-1]))}')

        # save the different trajectories
        error_simus.append(store_error_simus)
        error_simus_noresample.append(store_error_simus_noresample)
        error_montecarlos.append(store_error_montecarlos)
        t1_stop = perf_counter_ns()
        print(f"elapsed time for d={d}:", (t1_stop - t1_start)*1e-9, 's')

    os.makedirs(path, exist_ok=True)
    np.savez(f'{path}/xaxiss.npz', np.array(xaxiss, dtype=object), allow_pickle = True)
    np.savez(f'{path}/ds.npz', np.array(ds, dtype=object), allow_pickle = True)
    np.savez(f'{path}/error_simus.npz', np.array(error_simus, dtype=object), allow_pickle = True)
    np.savez(f'{path}/error_simus_noresample.npz', np.array(error_simus_noresample, dtype=object), allow_pickle = True)
    np.savez(f'{path}/error_montecarlos.npz', np.array(error_montecarlos, dtype=object), allow_pickle = True)



