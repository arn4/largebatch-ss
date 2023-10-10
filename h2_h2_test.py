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
d = 500
gamma0 = .01
l = 1.15
noise = 0.
T = int(10*np.log(d))
alpha=0.
mc_samples = 100000

target = giant_learning.h2_h2._target
activation = giant_learning.h2_h2._activation
activation_derivative = giant_learning.h2_h2._activation_derivative

Wtarget = orth((normalize(np.random.normal(size=(k,d)), axis=1, norm='l2')).T).T
W0 = 1/np.sqrt(d) * np.random.normal(size=(p,d))
a0 = np.ones(shape=(p,)) /np.sqrt(p)

simulations = [GradientDescent(
    target = target, W_target = Wtarget,
    activation = activation, W0 = W0, a0 = a0, activation_derivative = activation_derivative,
    gamma0 = gamma0, l = l, noise = noise,
    second_layer_update= False, alpha=alpha,
    resampling=True,
    test_size=mc_samples
) for _ in range(10)]

simulation_noresample = GradientDescent(
    target = target, W_target = Wtarget,
    activation = activation, W0 = W0, a0 = a0, activation_derivative = activation_derivative,
    gamma0 = gamma0, l = l, noise = noise,
    second_layer_update= False, alpha=alpha,
    resampling=False,
    test_size=mc_samples
)

analytical = H2H2Overlaps(
    Wtarget @ Wtarget.T, W0 @ Wtarget.T, W0 @ W0.T, a0,
    gamma0, d, l, noise,
    False, alpha,
)

for simulation in tqdm(simulations):
    simulation.train(T)
simulation_noresample.train(T)
analytical.train(T)

similarity_simulation = []
for simulation in simulations:
    Ws = np.array(simulation.W_s)
    Ws_noresample = np.array(simulation_noresample.W_s)

    Ms_simulation = np.einsum('tji,ri->tjr', Ws, Wtarget)
    similarity_simulation.append(np.einsum(
        'tjr,tj->tjr',
        Ms_simulation,
        1/np.sqrt(np.einsum('tji,tji->tj', Ws, Ws)),
    ))

Ms_simulation_noresample = np.einsum('tji,ri->tjr', Ws_noresample, Wtarget)
similarity_simulation_noresample = np.einsum(
    'tjr,tj->tjr',
    Ms_simulation_noresample,
    1/np.sqrt(np.einsum('tji,tji->tj', Ws_noresample, Ws_noresample)),
)

similarity_analytical = analytical.Ms * np.sqrt(np.einsum('tjj,rr->tjr', analytical.Qs, analytical.P))

# plot similarities
fig, ax = plt.subplots()
for ss in similarity_simulation:
    ax.plot(np.arange(len(ss)), abs(ss[:,0,0]), color='blue', alpha=.2)
# ax.plot(np.arange(len(similarity_simulation)), abs(similarity_simulation[:,0,0]), label='simulation')
ax.plot(np.arange(len(similarity_simulation_noresample)), abs(similarity_simulation_noresample[:,0,0]), label='simulation no resample')
ax.plot(np.arange(len(similarity_analytical)), abs(similarity_analytical[:,0,0]), label='analytical')
ax.legend()
plt.show()




