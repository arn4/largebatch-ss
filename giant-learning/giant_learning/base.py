import numpy as np
from numpy.linalg import inv as inverse_matrix

class GiantStepBase():
    def __init__(self,
                 target: callable, p: int, k: int,
                 activation: callable, a0: np.array, activation_derivative: callable,
                 gamma: float, noise: float,
                 second_layer_update: bool
                ):
        self.target = target
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.gamma = gamma
        self.p = p
        self.k = k
        self.noise = noise
        self.second_layer_update = second_layer_update
        assert(a0.shape == (p,))

        self.a_s = [a0.copy()]
        self.test_errors = []

        self.network = lambda local_field: 1/np.sqrt(self.p) * np.dot(self.a, self.activation(local_field))

    @property
    def a(self):
        return self.a_s[-1]
    
class OverlapsBase(GiantStepBase):
    def __init__(self,
                 target: callable, activation: callable, activation_derivative: callable,
                 P: np.array, M0: np.array, Q0: np.array,  a0: np.array, 
                 gamma: float, noise: float,
                 I4_diagonal: bool, I4_offdiagonal: bool, second_layer_update: bool
                ):
        super().__init__(target,Q0.shape[0],P.shape[0], activation, a0, activation_derivative, gamma, noise, second_layer_update)

        self.I4_diagonal = I4_diagonal
        self.I4_offdiagonal = I4_offdiagonal

        self.Ms = [M0.copy()]
        self.Qs = [Q0.copy()]
        self.P = P.copy()
        self.inverse_P = inverse_matrix(P)

        self.measure()

    @property
    def M(self):
        return self.Ms[-1]
    
    @property
    def Q(self):
        return self.Qs[-1]
    
    def train(self, steps):
        for _ in range(steps):
            self.update()
            self.measure()

    def update(self):
        inverse_Qbot = inverse_matrix(self.Q - self.M @ self.inverse_P @ self.M.T)
        expected_value_target, expected_value_network, expected_I4 = self.compute_expected_values()
        expected_value_orthogonal = expected_value_network - self.M @ self.inverse_P @ (expected_value_target.T)

        newM = self.M + self.gamma/self.p * np.einsum('j,jr->jr', self.a, expected_value_target)
        self.Ms.append(newM)

        newQ = self.Q
        newQ += (self.gamma/self.p) * (
                np.einsum('j,jl->jl', self.a, expected_value_network) +
                np.einsum('l,lj->jl', self.a, expected_value_network)
            )
        if self.I4_diagonal:
            newQ += (self.gamma/self.p)**2 * np.einsum('j,l->jl', self.a, self.a ) * expected_I4

        if self.I4_offdiagonal:
            newQ += (self.gamma/self.p)**2 * np.einsum('j,l->jl', self.a, self.a) * (
                np.einsum('jr,rt,lt->jl', expected_value_target, self.inverse_P, expected_value_target) +
                np.einsum('jm,mn,ln->jl', expected_value_orthogonal, inverse_Qbot, expected_value_orthogonal)
            )
        self.Qs.append(newQ)

        if self.second_layer_update:
            raise NotImplementedError
        else:
            self.a_s.append(self.a)

    # abstact method
    def error(self):
        raise NotImplementedError('error() must be reimplemented!')
    
    def measure(self):
        self.test_errors.append(self.error())

    
class SpecializedOverlapsBase(OverlapsBase):
    def __init__(self, P: np.array, M0: np.array, Q0: np.array, a0: np.array, gamma: float, noise: float, I4_diagonal: bool=True, I4_offdiagonal: bool=True, second_layer_update: bool=False):
        super().__init__(self._target, self._activation, self._activation_derivative, P, M0, Q0, a0, gamma, noise, I4_diagonal, I4_offdiagonal, second_layer_update)


