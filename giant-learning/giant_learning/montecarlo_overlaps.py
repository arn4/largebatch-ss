import numpy as np
from numpy.linalg import inv as inverse_matrix
from tqdm import tqdm

from .base import GiantStepBase, OverlapsBase

class MonteCarloOverlaps(OverlapsBase):
    def __init__(self,
                 target: callable, activation: callable, activation_derivative: callable,
                 P: np.array, M0: np.array, Q0: np.array,  a0: np.array, 
                 gamma0: float, d: int, l: int, noise: float,
                 second_layer_update: bool, alpha: float,
                 seed: int = 0, mc_size = None):
        super().__init__(target, activation, activation_derivative, P, M0, Q0, a0, gamma0, d, l, noise, second_layer_update, alpha)

        self.rng = np.random.default_rng(seed)

        if mc_size is None:
            self.mc_size = self.n
        self.mc_size = mc_size

        self.measure()
    
    def local_fields_montecarlo(self, f, std = False):
        Omega = np.block([[self.Q, self.M], [self.M.T, self.P]])
        def compute_sample():
            local_fields = self.rng.multivariate_normal(np.zeros(Omega.shape[0]), Omega)
            network_local_field = local_fields[:self.p]
            target_local_field = local_fields[self.p:]
            noise_randomness =self.rng.normal()
            return f(network_local_field, target_local_field, noise_randomness)
        data = np.array([compute_sample() for _ in tqdm(range(self.mc_size))])
        if std:
            return np.mean(data, axis=0), np.std(data, axis=0)/np.sqrt(self.mc_size)
        else:
            return np.mean(data, axis=0)

    def update(self):
        def local_field_M_update(network_field, target_field, noise_randomness):
            return (
                (self.target(target_field)+np.sqrt(self.noise)*noise_randomness-self.network(network_field)) *
                np.einsum('j,j,r->jr', self.a, self.activation_derivative(network_field), target_field)
            )
        self.Ms.append(
            self.M + self.gamma0 * np.sqrt(self.p) * np.power(self.d, (self.l-1)/2) * self.local_fields_montecarlo(local_field_M_update)
        )

        def local_field_I3_Q_update(network_field, target_field, noise_randomness):
            return (
                (self.target(target_field)+np.sqrt(self.noise)*noise_randomness-self.network(network_field)) * (
                    np.einsum('j,j,l->jl', self.a, self.activation_derivative(network_field), network_field) +
                    np.einsum('l,l,j->jl', self.a, self.activation_derivative(network_field), network_field)
                )
            )
        def local_field_I4_Q_update(network_field, target_field, noise_randomness):
            return (
                (self.target(target_field)+np.sqrt(self.noise)*noise_randomness-self.network(network_field))**2 * 
                np.einsum('j,l,j,l->jl', self.a, self.a, self.activation_derivative(network_field), self.activation_derivative(network_field))
            )
        self.Qs.append(
            self.Q + self.gamma0 * np.sqrt(self.p) * np.power(self.d, (self.l-1)/2) * self.local_fields_montecarlo(local_field_I3_Q_update) 
            + self.gamma0**2 * self.p * self.local_fields_montecarlo(local_field_I4_Q_update)
        )
    
    def measure(self):
        def local_field_error(network_field, target_field, noise_randomness):
            return .5*(self.target(target_field)+np.sqrt(self.noise)*noise_randomness-self.network(network_field))**2
        
        self.test_errors.append(
            self.local_fields_montecarlo(local_field_error)
        )
