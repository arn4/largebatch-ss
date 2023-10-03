from giant_learning.gradient_descent import GradientDescent
from giant_learning.montecarlo_overlaps import MonteCarloOverlaps
import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import matplotlib.pyplot as plt
from time import perf_counter_ns
import os 

alpha = 0. ; noise = 0.
k = 1 ; gamma0 = 1 ; mc_samples = 10000 ; p = 100
H2 = lambda z: z**2 - 1

def target(lft):
    return np.mean(H2(lft))
ds = np.logspace(8,11,num = 4, base = 2, dtype = int) 
error_simus = [] 
error_simus_noresample = []
error_montecarlos = []
xaxiss = []
path =  f"./results_cluster/data/info_exponent"
for d in ds:
    t1_start = perf_counter_ns()
    print(f'START d = {d}')
    T = 2*np.log2(d).astype(int)
    xaxis = np.arange(T+1) / np.log2(d).astype(int)
    xaxiss.append(xaxis)
    l = 1.15
    l_noresample = l

    activation = lambda x: np.maximum(x,0)
    activation_derivative = lambda x: (x>0).astype(float)

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
    
    simulation.train(T)
    simulation_noresample.train(T)

    montecarlo = MonteCarloOverlaps(
        target, activation, activation_derivative,
        Wtarget @ Wtarget.T, W0 @ Wtarget.T, W0 @ W0.T, a0,
        gamma0, d, l, noise,
        False, alpha,
        mc_size = mc_samples
    )

    montecarlo.train(T)

    error_simus.append(simulation.test_errors)
    error_simus_noresample.append(simulation_noresample.test_errors)
    error_montecarlos.append(montecarlo.test_errors)
    
    t1_stop = perf_counter_ns()
    print(f"Elapsed time for d={d}:", (t1_stop - t1_start)*1e-9, 's')

os.makedirs(path, exist_ok=True)
np.savez(f'{path}/xaxiss.npz', np.array(xaxiss, dtype=object), allow_pickle = True)
np.savez(f'{path}/ds.npz', np.array(ds, dtype=object), allow_pickle = True)
np.savez(f'{path}/error_simus.npz', np.array(error_simus, dtype=object), allow_pickle = True)
np.savez(f'{path}/error_simus_noresample.npz', np.array(error_simus_noresample, dtype=object), allow_pickle = True)
np.savez(f'{path}/error_montecarlos.npz', np.array(error_montecarlos, dtype=object), allow_pickle = True)




