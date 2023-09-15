import numpy as np
from scipy.special import erf

from .base import OverlapsBase
# from .cython_erf_erf import erf_updates, erf_error




# def _target(local_fields):
#     return np.mean(erf(local_fields/np.sqrt(2)),axis=-1)
# def _activation(x):
#     return erf(x/np.sqrt(2))
# def _activation_derivative(x):
#     return np.sqrt(2/np.pi) * np.exp(-x**2/2)

# class ErfErfOverlaps(OverlapsBase):
#     def __init__(self, P: np.array, M0: np.array, Q0: np.array, a0: np.array, gamma0: float, d: int, l: int, noise: float, second_layer_update: bool, alpha: float):
#         super().__init__(_target, _activation, _activation_derivative, P, M0, Q0, a0, gamma0, d, l, noise, second_layer_update, alpha)
#         self.measure()
#     def compute_expected_values(self):
#         return erf_updates(self.Q, self.M, self.P, self.a, self.noise)
#     def measure(self):
#         self.test_errors.append(erf_error(self.Q, self.M, self.P, self.a, self.noise))
        
