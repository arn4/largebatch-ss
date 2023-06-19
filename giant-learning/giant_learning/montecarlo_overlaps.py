import numpy as np
from tqdm import tqdm
from numpy.linalg import inv as inverse_matrix
from collections.abc import Iterable

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

        ###########
        self.eq5 = []
        self.eq6 = []
        self.eq7 = []
        ###########

        self.measure()
    
    def local_fields_montecarlo(self, fs, std = False):
        single_input = False
        if not isinstance(fs, Iterable):
            fs = [fs]
            single_input = True

        Omega = np.block([[self.Q, self.M], [self.M.T, self.P]])
        local_fields_samples = self.rng.multivariate_normal(np.zeros(Omega.shape[0]), Omega, size=(self.mc_size,))
        noise_randomness_samples = self.rng.normal(size=(self.mc_size,))
        def compute_samples(local_fields, noise_randomness):
            network_local_field = local_fields[:self.p]
            target_local_field = local_fields[self.p:]
            return [f(network_local_field, target_local_field, noise_randomness) for f in fs]
        datas = [compute_samples(local_fields, noise_randomness) for local_fields, noise_randomness in tqdm(zip(local_fields_samples, noise_randomness_samples), total=self.mc_size)]
        datas = list(zip(*datas))
        datas = [np.array(data) for data in datas]
        if std:
            raise NotImplementedError
            return np.mean(data, axis=0), np.std(data, axis=0)/np.sqrt(self.mc_size)
        else:
            if not single_input:
                return (np.mean(data, axis=0) for data in datas)
            else:
                return np.mean(datas[0], axis=0)


    def update(self):
        def local_field_term_target(network_field, target_field, noise_randomness):
            return (
                (self.target(target_field)+np.sqrt(self.noise)*noise_randomness-self.network(network_field)) *
                np.einsum('j,r->jr', self.activation_derivative(network_field), target_field)
            )
        
        def local_field_term_network(network_field, target_field, noise_randomness):
            return (
                (self.target(target_field)+np.sqrt(self.noise)*noise_randomness-self.network(network_field)) *
                np.einsum('j,l->jl', self.activation_derivative(network_field), network_field)
            )
        
        def local_field_term_orthogonal(network_field, target_field, noise_randomness):
            orthogonal_field = network_field - self.M @ self.inverse_P @ target_field
            return (
                (self.target(target_field)+np.sqrt(self.noise)*noise_randomness-self.network(network_field)) *
                np.einsum('j,l->jl', self.activation_derivative(network_field), orthogonal_field)
            )
        
        inverse_Qbot = inverse_matrix(self.Q - self.M @ self.inverse_P @ self.M.T)

        # expected_value_target = self.local_fields_montecarlo(local_field_term_target) # p x k
        # expected_value_network = self.local_fields_montecarlo(local_field_term_network) # p x p
        expected_value_target, expected_value_network = self.local_fields_montecarlo((local_field_term_target, local_field_term_network)) # p x k
        # expected_value_network = self.local_fields_montecarlo(local_field_term_network) # p x p
        expected_value_orthogonal = self.local_fields_montecarlo(local_field_term_orthogonal)
        # expected_value_orthogonal = expected_value_network - self.M @ self.inverse_P @ (expected_value_target.T)


        ##############
        self.eq5.append(
            self.gamma0 * np.sqrt(self.p) * np.power(self.d, (self.l-1)/2) * np.einsum('j,jr->jr', self.a, expected_value_target)
        )
        self.eq6.append(
            self.gamma0 * np.sqrt(self.p) * np.power(self.d, (self.l-1)/2) * (
                np.einsum('j,jl->jl', self.a, expected_value_network) +
                np.einsum('l,lj->jl', self.a, expected_value_network)
            )
        )
        self.eq7.append(
            self.gamma0**2 * self.p * np.power(self.d, (self.l-1)) * np.einsum('j,l->jl', self.a, self.a) * (
                np.einsum('jr,rt,lt->jl', expected_value_target, self.inverse_P, expected_value_target) +
                np.einsum('jm,mn,ln->jl', expected_value_orthogonal, inverse_Qbot, expected_value_orthogonal)
            )
        )
        ##############

        self.Ms.append(
            self.M + self.gamma0 * np.sqrt(self.p) * np.power(self.d, (self.l-1)/2) * np.einsum('j,jr->jr', self.a, expected_value_target)
        )
        self.Qs.append(
            self.Q + self.gamma0 * np.sqrt(self.p) * np.power(self.d, (self.l-1)/2) * (
                np.einsum('j,jl->jl', self.a, expected_value_network) +
                np.einsum('l,lj->jl', self.a, expected_value_network)
            ) + self.gamma0**2 * self.p * np.power(self.d, (self.l-1)) * np.einsum('j,l->jl', self.a, self.a) * (
                np.einsum('jr,rt,lt->jl', expected_value_target, self.inverse_P, expected_value_target) +
                np.einsum('jm,mn,ln->jl', expected_value_orthogonal, inverse_Qbot, expected_value_orthogonal)
            )
        )


        # def local_field_M_update(network_field, target_field, noise_randomness):
        #     return (
        #         (self.target(target_field)+np.sqrt(self.noise)*noise_randomness-self.network(network_field)) *
        #         np.einsum('j,j,r->jr', self.a, self.activation_derivative(network_field), target_field)
        #     )
        # self.Ms.append(
        #     self.M + self.gamma0 * np.sqrt(self.p) * np.power(self.d, (self.l-1)/2) * self.local_fields_montecarlo(local_field_M_update)
        # )

        # def local_field_I3_Q_update(network_field, target_field, noise_randomness):
        #     return (
        #         (self.target(target_field)+np.sqrt(self.noise)*noise_randomness-self.network(network_field)) * (
        #             np.einsum('j,j,l->jl', self.a, self.activation_derivative(network_field), network_field) +
        #             np.einsum('l,l,j->jl', self.a, self.activation_derivative(network_field), network_field)
        #         )
        #     )
        # def local_field_I4_Q_update(network_field, target_field, noise_randomness):
        #     bot_fields = network_field - self.M @ self.inverse_P @ target_field
        #     return (
        #         (self.target(target_field)+np.sqrt(self.noise)*noise_randomness-self.network(network_field))**2 * 
        #         np.einsum('j,l,j,l->jl', self.a, self.a, self.activation_derivative(network_field), self.activation_derivative(network_field))
        #     )
        # self.Qs.append(
        #     self.Q + self.gamma0 * np.sqrt(self.p) * np.power(self.d, (self.l-1)/2) * self.local_fields_montecarlo(local_field_I3_Q_update) 
        #     + self.gamma0**2 * self.p * self.local_fields_montecarlo(local_field_I4_Q_update)
        # )
    
    def measure(self):
        def local_field_error(network_field, target_field, noise_randomness):
            return .5*(self.target(target_field)+np.sqrt(self.noise)*noise_randomness-self.network(network_field))**2
        
        self.test_errors.append(
            self.local_fields_montecarlo(local_field_error)
        )
