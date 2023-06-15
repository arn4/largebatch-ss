from giant_learning.gradient_descent import GradientDescent
from giant_learning.montecarlo_overlaps import MonteCarloOverlaps
from giant_learning.erf_erf import ErfErfOverlaps

import numpy as np
from scipy.special import erf
from sklearn.preprocessing import normalize
from scipy.linalg import orth

target = lambda x: np.mean(erf(x/np.sqrt(2)), axis=-1)
p = 10
ds = [100, 1000]
for d in ds:
    l = 1.
    # n = d**l
    k = 2
    Wtarget = orth((normalize(np.random.normal(size=(k,d)), axis=1, norm='l2')).T).T
    W0 = 1/np.sqrt(d) * np.random.normal(size=(p,d))
    # a0 = 1/np.sqrt(p) * np.random.normal(size=(p,))
    a0 = np.ones(p)/np.sqrt(p)
    Q0 = W0 @ W0.T
    M0 = W0 @ Wtarget.T
    P = Wtarget @ Wtarget.T
    T = 1000
    gamma0 = .5
    noise = 1e-3
    alpha = 0.
    mc_samples = 40000
    second_layer_update = False
    activation = lambda x: erf(x/np.sqrt(2))
    activation_derivative = lambda x: np.sqrt(2/np.pi) * np.exp(-x**2/2)

    # Create a gradient descent object
    gd = GradientDescent(
        target, Wtarget, 
        activation, W0, a0, activation_derivative, 
        gamma0, l, noise, second_layer_update, alpha,
        test_size = mc_samples, analytical_error= 'erferf'
    )
    gd.train(T)

    mc = MonteCarloOverlaps(
        target, activation, activation_derivative,
        P, M0, Q0, a0,
        gamma0, d, l, noise,
        second_layer_update, alpha,
        mc_size = mc_samples
    )
    # mc.train(T)

    erferf = ErfErfOverlaps(
        P, M0, Q0, a0,
        gamma0, d, l, noise,
        second_layer_update, alpha
    )

    erferf.train(T)


    import matplotlib.pyplot as plt
    # plt.ylim(0.,.12)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(gd.test_errors, label='GD', marker='o')
    plt.plot(mc.test_errors, label='MC', marker='o')
    plt.plot(erferf.test_errors, label='ErfErf', marker='o')
    plt.legend()
    plt.show()




