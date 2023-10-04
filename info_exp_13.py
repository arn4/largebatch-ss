from giant_learning.gradient_descent import GradientDescent
from giant_learning.montecarlo_overlaps import MonteCarloOverlaps
import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import matplotlib.pyplot as plt
from time import perf_counter_ns
import os 

alpha = 0. ; noise = 0.
k = 1 ; gamma0 = 1 ; mc_samples = 10000 ; p = 1
H2 = lambda z: z**2 - 1
H3 = lambda z: z**3 - 3*z
def target(lft):
    return np.mean(H3(lft))

ds = np.logspace(8,11,num = 4, base = 2, dtype = int) 
error_simus = [] 
error_simus_noresample = []
error_montecarlos = []
std_simus = []
std_simus_noresample = []
std_montecarlos = []
xaxiss = []
path =  f"./results_cluster/data/info_exp_13"
for d in ds:
    t1_start = perf_counter_ns()
    print(f'START d = {d}')
    T = 2*d.astype(int)
    xaxis = np.arange(T+1) / d
    xaxiss.append(xaxis)
    l = 1.15
    l_noresample = l
    activation = lambda x: np.maximum(x,0)
    activation_derivative = lambda x: (x>0).astype(float)
    store_error_simus = [] 
    store_error_simus_noresample = []
    store_error_montecarlos = []
    for seed in range(5):
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
        

        montecarlo = MonteCarloOverlaps(
            target, activation, activation_derivative,
            Wtarget @ Wtarget.T, W0 @ Wtarget.T, W0 @ W0.T, a0,
            gamma0, d, l, noise,
            False, alpha,
            mc_size = mc_samples
        )

        # train #
        montecarlo.train(T)
        simulation.train(T)
        simulation_noresample.train(T)

        # compute and store magnetizations # 

        Ws = np.array(simulation.W_s)
        Ws_noresample = np.array(simulation_noresample.W_s)
        Ms_simulation = np.einsum('tji,ri->tjr', Ws, Wtarget)
        Qs_simulation = np.einsum('tji,tlk->tjl', Ws, Ws)
        Ms_simulation_noresample = np.einsum('tji,ri->tjr', Ws_noresample, Wtarget)
        Qs_simulation_noresample = np.einsum('tji,tlk->tjl', Ws_noresample, Ws_noresample)
        Ms_montecarlo = np.array(montecarlo.Ms)
        Qs_montecarlo = np.array(montecarlo.Qs)

        Wupdates = Ws[1:] - Ws[:-1]
        Wupdates_noresample = Ws_noresample[1:] - Ws_noresample[:-1]

        Mupdates_simulation = Wupdates @ Wtarget.T
        Mupdates_simulation_noresample = Wupdates_noresample @ Wtarget.T        
        Mupdates_montercalo = Ms_montecarlo[1:] - Ms_montecarlo[:-1]
         
        store_error_simus.append(Mupdates_simulation)
        store_error_simus_noresample.append(Mupdates_simulation_noresample)
        store_error_montecarlos.append(Mupdates_montercalo)
        


    # get mean and std of the errors over the 10 seeds
    error_simus.append(np.mean(store_error_simus, axis = 0))
    error_simus_noresample.append(np.mean(store_error_simus_noresample, axis = 0))
    error_montecarlos.append(np.mean(store_error_montecarlos, axis = 0))
    std_simus.append(np.std(store_error_simus, axis = 0))
    std_simus_noresample.append(np.std(store_error_simus_noresample, axis = 0))
    std_montecarlos.append(np.std(store_error_montecarlos, axis = 0))


    t1_stop = perf_counter_ns()
    print(f"Elapsed time for d={d}:", (t1_stop - t1_start)*1e-9, 's')

os.makedirs(path, exist_ok=True)
np.savez(f'{path}/xaxiss.npz', np.array(xaxiss, dtype=object), allow_pickle = True)
np.savez(f'{path}/ds.npz', np.array(ds, dtype=object), allow_pickle = True)
np.savez(f'{path}/error_simus.npz', np.array(error_simus, dtype=object), allow_pickle = True)
np.savez(f'{path}/error_simus_noresample.npz', np.array(error_simus_noresample, dtype=object), allow_pickle = True)
np.savez(f'{path}/error_montecarlos.npz', np.array(error_montecarlos, dtype=object), allow_pickle = True)
np.savez(f'{path}/std_simus.npz', np.array(std_simus, dtype=object), allow_pickle = True)
np.savez(f'{path}/std_simus_noresample.npz', np.array(std_simus_noresample, dtype=object), allow_pickle = True)
np.savez(f'{path}/std_montecarlos.npz', np.array(std_montecarlos, dtype=object), allow_pickle = True)




