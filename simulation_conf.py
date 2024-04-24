debug = False
l = 3.
mus = [
    0.,
    1.,
    l/2,
    (3*l-2)/4
]
ds = [64, 128, 256, 512,1024]
nseeds = 5
ic_seed = 0
ic_seed ^= 9031908 ## Do bit change the XOR seed!
gamma_prefactor = 0.01

noise = 0.
p = 1
k = 1

import numpy as np

def get_delta(mu, d):
    return l/2 - mu - np.log(gamma_prefactor)/np.log(d)

def get_T(delta, mu, d):
    T = int(np.power(d, delta + l/2 - 1))
    return T
