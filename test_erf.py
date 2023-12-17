from giant_learning.gradient_descent import GradientDescent
from giant_learning.montecarlo_overlaps import MonteCarloOverlaps
from giant_learning.erf_erf import ErfErfOverlaps

import numpy as np
from scipy.special import erf
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import matplotlib.pyplot as plt

p = 1
l = 1.
k = 1
T = 20
gamma = 0.1
noise = 0.
alpha = 0.
target = lambda x: np.mean(erf(x/np.sqrt(2)), axis=-1)
activation = lambda x: erf(x/np.sqrt(2))
activation_derivative = lambda x: np.sqrt(2/np.pi) * np.exp(-x**2/2)
nseeds = 1
ds = np.logspace(3,6,base=2,num=4,dtype=int)

### save test error as a function of time for each seed and each d ###
test_errors = np.zeros((nseeds, len(ds), T+1))
for i,d in enumerate(ds):
    n = d
    t = np.sqrt(d) ### fix initial overlap 
    print(f'NOW Running d = {d}')
    for seed in range(nseeds):
        print(f'Seed = {seed}')
        Wtarget = orth((normalize(np.random.normal(size=(k,d)), axis=1, norm='l2')).T).T
        Wt = orth((normalize(np.random.normal(size=(k,d)), axis=1, norm='l2')).T).T
        Wtild = np.random.normal(size=(p,d)) / np.sqrt(d)
        A =  normalize(Wtild - np.einsum('ji,ri,rh->jh', Wtild , Wt ,Wt))
        W0 = (t*Wt + np.sqrt(1-t**2)*A)
        a0 = np.ones(p)/p ### It is changed with the new version of the package ###

        # Create a gradient descent object
        gd = GradientDescent(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative, 
            gamma, noise, second_layer_update = False, test_size = None, analytical_error= 'erferf'
        )
        gd.train(T)
        test_errors[seed, i, :] = gd.test_errors

P = Wtarget @ Wtarget.T
M0 = W0 @ W0.T
Q0 = W0 @ Wt.T
# Create a ErfErf object
erferf = ErfErfOverlaps(
    P, M0, Q0, a0,
    gamma, noise,
    I4_diagonal=True, I4_offdiagonal=True,
    second_layer_update=False)

erferf.train(T)

######## PLOTS ########
plt.plot(erferf.test_errors, label='Theory', marker='o')

### Plot the average test error with std error bars as a function of time for different d ###
for i,d in enumerate(ds):
    plt.errorbar(np.arange(T+1), np.mean(test_errors[:,i,:], axis=0), yerr=np.std(test_errors[:,i,:], axis=0), label=f'Simulation - d={d}', marker='o')
plt.xlabel('Time')
plt.ylabel('Test error')
plt.legend()
plt.show()

