from giant_learning.gradient_descent import GradientDescent
from giant_learning.montecarlo_overlaps import MonteCarloOverlaps

import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import matplotlib.pyplot as plt
from time import perf_counter_ns
import os 
alpha = 0. ; noise = 0.

k = 3 ; gamma0 = 4 ; T = 3 ; mc_samples = 100000 ; p = 1000
H_2 = lambda z: z**2 - 1
H_3 = lambda z: z**3 - 3*z
def target(lft):
    if target_tkn == '111':
        return lft[...,0]/3 + 2*lft[...,0]*lft[...,1] + lft[...,1]*lft[...,2]
    elif target_tkn == '120':
        return lft[...,0]/3 + 2*H_2(lft[...,0])*lft[...,1] + lft[...,0]*lft[...,2]
    elif target_tkn == '100':
        return lft[...,0]/3 + 2*H_2(lft[...,0])*H_2(lft[...,1]) + lft[...,1]*lft[...,2]
    elif target_tkn == '000':
        return H_2(lft[...,2])*H_3(lft[...,0]/3) + 2*H_2(lft[...,0])*H_3(lft[...,1]) + lft[...,1]*lft[...,2]
    elif target_tkn == '100_stronger':
        return lft[...,0] + H_2(lft[...,2])*H_2(lft[...,1])/2 + lft[...,1]*lft[...,2]/2
    

second_layers = {'gaussian': 1/np.sqrt(p)*np.random.randn(p) , 'hypercube': np.sign(np.random.normal(size=(p,))) /np.sqrt(p) , 'ones': np.ones(p)/np.sqrt(p),
'2var': np.sqrt(2/p)*np.random.randn(p), '4var': np.sqrt(4/p)*np.random.randn(p), '8var': np.sqrt(8/p)*np.random.randn(p), 'uniform': np.random.uniform(-np.sqrt(12),np.sqrt(12),size=(p,)) }

tkns = ['100_stronger']
ds = [2000] 
choices = ['hypercube']
for d in ds: 
    for target_tkn in tkns:
        for choice_2layer in choices:
            t1_start = perf_counter_ns()
            print(f'START d = {d}')
            l = np.log(4*d) / np.log(d)

            activation = lambda x: np.maximum(x,0)
            activation_derivative = lambda x: (x>0).astype(float)

            Wtarget = orth((normalize(np.random.normal(size=(k,d)), axis=1, norm='l2')).T).T
            W0 = 1/np.sqrt(d) * np.random.normal(size=(p,d))
            # W0 = 1/1000 * np.random.normal(size=(p,d))
            a0 = second_layers[choice_2layer]

            simulation = GradientDescent(
                target = target, W_target = Wtarget,
                activation = activation, W0 = W0, a0 = a0, activation_derivative = activation_derivative,
                gamma0 = gamma0, l = l, noise = noise,
                second_layer_update= False, alpha=alpha,
                resampling=True,
                test_size=mc_samples
            )
            
            simulation.train(T)

            Ws = np.array(simulation.W_s)
            Ms_simulation = np.einsum('tji,ri->tjr', Ws, Wtarget)
            Qs_simulation = np.einsum('tji,tlk->tjl', Ws, Ws)

            
            P = Wtarget @ Wtarget.T

            Wupdates = Ws[1:] - Ws[:-1]
            Mupdates_simulation = Wupdates @ Wtarget.T


            similarity_simulation = np.einsum(
                'tjr,tj,r->tjr',
                Mupdates_simulation,
                1/np.sqrt(np.einsum('tji,tji->tj', Mupdates_simulation, Mupdates_simulation)),
                1/np.sqrt(np.diag(P))
            )

            t1_stop = perf_counter_ns()
            print("Elapsed time:", (t1_stop - t1_start)*1e-9, 's')
            path =  f"./results_cluster/data/new_fig1_giant_step/tkn={target_tkn}_choice2={choice_2layer}"
            os.makedirs(path, exist_ok=True) 
            np.savez(f'{path}/d={d}.npz',similarity_simulation)