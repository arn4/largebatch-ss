import numpy as np
from scipy.special import erf

from .base import OverlapsBase
from .cython_erf_erf import erf_updates, erf_error

##################
from .cython_erf_erf import erf_I3, erf_I4
##################


def _target(local_fields):
    return np.mean(erf(local_fields/np.sqrt(2)))
def _activation(x):
    return erf(x/np.sqrt(2))
def _activation_derivative(x):
    return np.sqrt(2/np.pi) * np.exp(-x**2/2)

class ErfErfOverlaps(OverlapsBase):
    def __init__(self, P: np.array, M0: np.array, Q0: np.array, a0: np.array, gamma0: float, d: int, l: int, noise: float, second_layer_update: bool, alpha: float):
        super().__init__(_target, _activation, _activation_derivative, P, M0, Q0, a0, gamma0, d, l, noise, second_layer_update, alpha)
        self.measure()
        ##########
        self.eq6 = []
        self.eq7 = []
        ##########

    def update(self):
        dQ, dM = erf_updates(self.Q, self.M, self.P, self.a, self.gamma0, self.noise, pow(self.d, (self.l-1)/2))
        self.Qs.append(self.Q + dQ)
        self.Ms.append(self.M + dM)
        ##########
        self.eq6.append(erf_I3(self.Q, self.M, self.P, self.a, self.gamma0, self.noise, pow(self.d, (self.l-1)/2)))
        self.eq7.append(erf_I4(self.Q, self.M, self.P, self.a, self.gamma0, self.noise, pow(self.d, (self.l-1)/2)))
        ##########
    def measure(self):
        self.test_errors.append(erf_error(self.Q, self.M, self.P, self.a, self.noise))
        
