from giant_learning.gradient_descent import GradientDescent
from giant_learning.montecarlo_overlaps import MonteCarloOverlaps
from giant_learning.erf_erf import ErfErfOverlaps

import numpy as np
from scipy.special import erf
from sklearn.preprocessing import normalize
from scipy.linalg import orth

target = lambda x: np.mean(erf(x/np.sqrt(2)), axis=-1)
p = 2
ds = [1000]
for d in ds:
    l = 1.
    n = d**l
    k = 2
    # Wtarget = orth((normalize(np.random.normal(size=(k,d)), axis=1, norm='l2')).T).T
    Wtarget = 1/np.sqrt(d) * np.random.normal(size=(k,d))
    W0 = 1/np.sqrt(d) * np.random.normal(size=(p,d))
    a0 = 1/np.sqrt(p) * np.random.normal(size=(p,))
    # a0 = np.ones(p)/np.sqrt(p)
    Q0 = W0 @ W0.T
    M0 = W0 @ Wtarget.T
    P = Wtarget @ Wtarget.T
    T = 1
    gamma0 = 1.
    noise = 0.
    alpha = 0.
    mc_samples = 400000
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
    mc.train(T)

    # erferf = ErfErfOverlaps(
    #     P, M0, Q0, a0,
    #     gamma0, d, l, noise,
    #     second_layer_update, alpha
    # )

    # erferf.train(T)

    # eq6gd = np.array(gd.eq6)
    # eq7gd = np.array(gd.eq7)
    # eq6ee = np.array(erferf.eq6)
    # eq7ee = np.array(erferf.eq7)
    # print(eq7gd.shape)
    # print('eq6gd', np.mean(abs((eq6gd)), axis=(1,2)))
    # print('eq6ee', np.mean(abs((eq6ee)), axis=(1,2)))
    # print('eq6', np.mean(abs((eq6ee - eq6gd)/eq6gd), axis=(1,2)))
    
    # diag = np.einsum('tjluv,uv->tjl', eq7gd, np.eye(int(n)))
    # full = np.einsum('tjluv->tjl', eq7gd)

    print('MC M')
    print(mc.eq5[0])
    print('GD M ')
    print(gd.eq5[0])

    mc_eq5 = np.array(mc.eq5)
    gd_eq5 = np.array(gd.eq5)
    print('Relative error M')
    relative_M = (abs((mc_eq5 - gd_eq5)/abs(gd_eq5)))
    print(relative_M)
    print('Mean relative error M')
    print(np.mean(relative_M, axis=(1,2)))

    mc_eq6 = np.array(mc.eq6)
    gd_eq6 = np.array(gd.eq6)
    print('Relative error Q facile')
    relative_Q = (abs((mc_eq6 - gd_eq6)/abs(gd_eq6)))
    print(relative_Q)
    print('Mean relative error Q facile')
    print(np.mean(relative_Q, axis=(1,2)))

    mc_eq7 = np.array(mc.eq7)
    gd_eq7 = np.array(gd.eq7)
    print('Relative error Q difficile')
    relative_P = (abs((mc_eq7 - gd_eq7)/abs(gd_eq7)))
    print(relative_P)
    print('Mean relative error Q difficile')
    print(np.mean(relative_P, axis=(1,2)))





    import matplotlib.pyplot as plt
    # plt.ylim(0.,.12)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.plot(gd.test_errors, label='GD', marker='o')
    plt.plot(mc.test_errors, label='MC', marker='o')
    # plt.plot(erferf.test_errors, label='ErfErf', marker='o')
    plt.legend()
    # plt.show()




