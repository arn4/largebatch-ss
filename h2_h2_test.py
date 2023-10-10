import giant_learning
from giant_learning.h2_h2 import H2H2Overlaps
from giant_learning.gradient_descent import GradientDescent

import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import matplotlib.pyplot as plt
from tqdm import tqdm

p = 1
k = 1
gamma0 = .01
l = 1.15
noise = 0.
mc_samples = 100000
alpha = 0.
nseeds = 20
target = giant_learning.h2_h2._target
activation = giant_learning.h2_h2._activation
activation_derivative = giant_learning.h2_h2._activation_derivative
a0 = np.ones(shape=(p,)) /np.sqrt(p)
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
ds = np.logspace(10,12,num = 3, base = 2, dtype = int) 
store_simulation = []
store_analytical = []
starts = ["tiepide", "cold", "random"]
for start in starts:
    if start == "tiepide":
        t = 0.7
    elif start == "cold":
        t = 0.3
    else:
        t = 0
    print(f'{start} START')
    for d in ds:
        print(f'd = {d}')
        similarity_simulation = []
        similarity_analytical = []
        T = int(2*np.log(d))
        for _ in tqdm(range(nseeds)):
            print(f'seed = {_}')
            Wtarget = orth((normalize(np.random.normal(size=(k,d)), axis=1, norm='l2')).T).T
            Wtild = np.random.normal(size=(p,d)) / np.sqrt(d)
            A =  normalize(Wtild - np.einsum('ji,ri,rh->jh', Wtild , Wtarget ,Wtarget))
            W0 = t*Wtarget + np.sqrt(1-t**2)*A

            simulation = GradientDescent(
                target = target, W_target = Wtarget,
                activation = activation, W0 = W0, a0 = a0, activation_derivative = activation_derivative,
                gamma0 = gamma0, l = l, noise = noise,
                second_layer_update= False, alpha=alpha,
                resampling=True,
                test_size=mc_samples
            )
            simulation.train(T)

            analytical = H2H2Overlaps(
                Wtarget @ Wtarget.T, W0 @ Wtarget.T, W0 @ W0.T, a0,
                gamma0, d, l, noise,
                False, alpha,
            )
            analytical.train(T)

            # compute magnetizations 
            Ws = np.array(simulation.W_s)
            Ms_simulation = np.einsum('tji,ri->tjr', Ws, Wtarget)
            similarity_simulation.append(np.einsum(
                'tjr,tj->tjr',
                Ms_simulation,
                1/np.sqrt(np.einsum('tji,tji->tj', Ws, Ws)),
            ))
            similarity_analytical.append( analytical.Ms / np.sqrt(np.einsum('tjj,rr->tjr', analytical.Qs, analytical.P)) )
        store_simulation.append(similarity_simulation)
        store_analytical.append(similarity_analytical)

    ### save data in results cluster folder 
    path = f"./results_cluster/h2_h2/start={start}"
    os.makedirs(path, exist_ok=True)
    np.save(f'{path}/store_simulation.npy', store_simulation)
    np.save(f'{path}/store_analytical.npy', store_analytical)
    np.save(f'{path}/ds.npy', ds)
    np.save(f'{path}/nseeds.npy', nseeds)








