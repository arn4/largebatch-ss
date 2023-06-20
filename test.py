from giant_learning.gradient_descent import GradientDescent
from giant_learning.montecarlo_overlaps import MonteCarloOverlaps
from giant_learning.erf_erf import ErfErfOverlaps

import numpy as np
from scipy.special import erf
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import matplotlib.pyplot as plt
target = lambda x: np.mean(erf(x/np.sqrt(2)), axis=-1)
p = 2
l = 1.
k = 2
T = 1
gamma0 = 1.
noise = 0.
alpha = 0.
mc_samples = 1000000
second_layer_update = False
activation = lambda x: erf(x/np.sqrt(2))
activation_derivative = lambda x: np.sqrt(2/np.pi) * np.exp(-x**2/2)
nseeds = 1
ds = np.logspace(8 ,10, base =2 , num = 3, dtype=int)
relative_Ms = [] ; relative_Qfaciles = [] ; relative_Qdifficiles = []
relativeMs_erf = [] ; relativeQs_erf = [] ; relativeQdiffs_erf = []
relativeMs_erf_mc = [] ; relativeQs_erf_mc = [] ; relativeQdiffs_erf_mc = []
for d in ds:
    n = d**l
    relative_M = np.zeros((T,p,k)) ; relative_Q = np.zeros((T,p,k)) ; relative_Qdiff = np.zeros((T,p,k))
    relative_M_erf = np.zeros((T,p,k)) ; relative_Q_erf = np.zeros((T,p,k)) ; relative_Qdiff_erf = np.zeros((T,p,k))        
    relative_M_erf_mc = np.zeros((T,p,k)) ; relative_Q_erf_mc = np.zeros((T,p,k)) ; relative_Qdiff_erf_mc = np.zeros((T,p,k))
    for _ in range(nseeds):
        print(f'NOW Running d = {d}')
        Wtarget = orth((normalize(np.random.normal(size=(k,d)), axis=1, norm='l2')).T).T
        # Wtarget = 1/np.sqrt(d) * np.random.normal(size=(k,d))
        W0 = 1/np.sqrt(d) * np.random.normal(size=(p,d))
        # a0 = 1/np.sqrt(p) * np.random.normal(size=(p,))
        a0 = np.ones(p)/np.sqrt(p)
        Q0 = W0 @ W0.T
        M0 = W0 @ Wtarget.T
        P = Wtarget @ Wtarget.T


        # Create a gradient descent object
        # gd = GradientDescent(
        #     target, Wtarget, 
        #     activation, W0, a0, activation_derivative, 
        #     gamma0, l, noise, second_layer_update, alpha,
        #     test_size = mc_samples, analytical_error= 'erferf'
        # )
        # gd.train(T)

        # Create a Monte Carlo object
        mc = MonteCarloOverlaps(
            target, activation, activation_derivative,
            P, M0, Q0, a0,
            gamma0, d, l, noise,
            second_layer_update, alpha,
            mc_size = mc_samples
        )
        mc.train(T)

        # Create a ErfErf object
        erferf = ErfErfOverlaps(
            P, M0, Q0, a0,
            gamma0, d, l, noise,
            second_layer_update, alpha
        )

        erferf.train(T)

        mc_eq5 = np.array(mc.eq5)
        # gd_eq5 = np.array(gd.eq5)
        erf_eq5 = np.array(erferf.eq5)

        print('Relative error M')
        # relative_M += (abs((mc_eq5 - gd_eq5)/abs(gd_eq5))) / nseeds
        # print(f'relative_M with mc = {relative_M}')  
        # relative_M_erf += (abs((erf_eq5 - gd_eq5)/abs(gd_eq5))) / nseeds 
        # print(f'relative_M with erf = {relative_M_erf}')    
        relative_M_erf_mc += (abs((erf_eq5 - mc_eq5)/abs(mc_eq5))) / nseeds
        print(f'relative_M with erf and mc = {relative_M_erf_mc}')

        mc_eq6 = np.array(mc.eq6)
        # gd_eq6 = np.array(gd.eq6)
        erf_eq6 = np.array(erferf.eq6)
        print('Relative error Q facile')
        # relative_Q += (abs((mc_eq6 - gd_eq6)/abs(gd_eq6))) / nseeds
        # print(f'relative_Q with mc = {relative_Q}')
        # relative_Q_erf += (abs((erf_eq6 - gd_eq6)/abs(gd_eq6))) / nseeds
        # print(f'relative_Q with erf = {relative_Q_erf}')
        relative_Q_erf_mc += (abs((erf_eq6 - mc_eq6)/abs(mc_eq6))) / nseeds
        print(f'relative_Q with erf and mc = {relative_Q_erf_mc}')
        

        mc_eq7 = np.array(mc.eq7)
        # gd_eq7 = np.array(gd.eq7)
        erf_eq7 = np.array(erferf.eq7)
        print('Relative error Q difficile')
        # relative_Qdiff += (abs((mc_eq7 - gd_eq7)/abs(gd_eq7))) / nseeds
        # print(f'relative_Qdiff with mc = {relative_Qdiff}')
        # relative_Qdiff_erf += (abs((erf_eq7 - gd_eq7)/abs(gd_eq7))) / nseeds
        # print(f'relative_Qdiff with  = {relative_Qdiff_erf}')
        relative_Qdiff_erf_mc += (abs((erf_eq7 - mc_eq7)/abs(mc_eq7))) / nseeds
        print(f'relative_Qdiff with erf and mc = {relative_Qdiff_erf_mc}')

        
    # relative_Ms.append(np.mean(relative_M, axis=(1,2)))
    # relative_Qfaciles.append(np.mean(relative_Q, axis=(1,2)))
    # relative_Qdifficiles.append(np.mean(relative_Qdiff, axis=(1,2)))
    # relativeMs_erf.append(np.mean(relative_M_erf, axis=(1,2)))
    # relativeQs_erf.append(np.mean(relative_Q_erf, axis=(1,2)))
    # relativeQdiffs_erf.append(np.mean(relative_Qdiff_erf, axis=(1,2)))
    relativeMs_erf_mc.append(np.mean(relative_M_erf_mc, axis=(1,2)))
    relativeQs_erf_mc.append(np.mean(relative_Q_erf_mc, axis=(1,2)))
    relativeQdiffs_erf_mc.append(np.mean(relative_Qdiff_erf_mc, axis=(1,2)))

######## PLOTS ########
# plt.ylim(0.,.12)
plt.xscale('log')
# plt.yscale('log')
# plt.plot(ds, relative_Ms, label='M', marker='o')
# plt.plot(ds, relative_Qfaciles, label='Q facile', marker='o')
# plt.plot(ds, relative_Qdifficiles, label='Q difficile', marker='o')
# plt.plot(ds, relativeMs_erf, label='M erf', marker='x')
# plt.plot(ds, relativeQs_erf, label='Q facile erf', marker='x')
# plt.plot(ds, relativeQdiffs_erf, label='Q difficile erf', marker='x')
plt.plot(ds,relativeMs_erf_mc, label='M erf mc', marker='x')
plt.plot(ds,relativeQs_erf_mc, label='Q facile erf mc', marker='x')
plt.plot(ds,relativeQdiffs_erf_mc, label='Q difficile erf mc', marker='x')
plt.legend()
plt.show()
### error plots 
# plt.plot(gd.test_errors, label='GD', marker='o')
# plt.plot(mc.test_errors, label='MC', marker='o')
# plt.plot(erferf.test_errors, label='ErfErf', marker='o')
