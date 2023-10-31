import giant_learning
from giant_learning.poly_poly import H3H3Overlaps
from giant_learning.gradient_descent import GradientDescent

import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import matplotlib.pyplot as plt
from tqdm import tqdm
import os 
p = 1
k = 1
gamma0 = .005
l = 1
d = 2**8
noise = 0.
alpha = 0.
target = H3H3Overlaps._target
activation = H3H3Overlaps._activation
activation_derivative = H3H3Overlaps._activation_derivative
a0 = np.ones(shape=(p,)) /np.sqrt(p)
colors = []
# generate palette with 30 different colors 
# for i in range(10):
#     for j in range(i+1):
#         colors.append(plt.cm.tab10(i))

colors = ['blue', 'green', 'blue','brown', 'purple', 'orange', 'red', 'black', 'yellow', 'pink']
threshold_value = 0.15
# m0s = np.logspace(-100,-3,num = 2)
m0s = np.array([0,1e-10])
T = 3
similarity_analytical = []
crossing_times = []
norms = []
for m0 in m0s:
    M0 = np.array([[m0]])
    Q0 = np.array([[1/3]])
    P0 = np.array([[1]])
    analytical = H3H3Overlaps(
                P0, M0, Q0, a0,
                gamma0, d, l, noise,
                False, alpha,
            )
    analytical.train(T)
    similarity = analytical.Ms / np.sqrt(np.einsum('tjj,rr->tjr', analytical.Qs, analytical.P))
    similarity_analytical.append( similarity )
    crossing_times.append(np.argmax(similarity > threshold_value))
    norms.append(np.array(analytical.Qs))
    print(f'm0 = {m0} - crossing time = {crossing_times[-1]}')
fig, ax = plt.subplots(1,3, figsize = (15,5))
for i,m0 in enumerate(m0s):
    ax[0].plot(np.arange(T+1), abs(similarity_analytical[i][:,0,0]), label = f'm0 = {m0}', color = colors[i])
    ax[2].plot(np.arange(T+1), norms[i][:,0,0], color = 'black', marker = 'o', linestyle = 'None')
ax[1].plot(m0s, crossing_times, color = 'black', marker = 'o', linestyle = 'None')
from scipy.optimize import curve_fit
# def func(x, a, b, c):
#     return a + b*x**(c)
# popt, pcov = curve_fit(func, m0s, crossing_times 
# ,p0=[20, 35/94, -1], bounds = ([5,0,-2], [250,1e4,-0.03])
# )
def func(x, a, b):
    # return a + b*np.log(x)
    # return a + b/x**2
    return a + b*x
popt, pcov = curve_fit(func, m0s, crossing_times)
print(f'popt = {popt}')
fit = func(m0s, *popt)
ax[1].plot(m0s, fit, color = 'red', label = 'Best fit')


# print the parameter fit in the legend 
ax[1].plot([], [], ' ', label=f'fit = {popt}')
ax[0].set_xlabel('t')
ax[0].set_ylabel('Overlap')
# ax[0].legend()
ax[1].set_xlabel(' m0')
ax[1].set_ylabel('Crossing time')
# ax[1].set_xscale('log')
# ax[1].set_yscale('log')
ax[2].set_xlabel('t')
ax[2].set_ylabel('Q')
ax[1].legend()
plt.show()