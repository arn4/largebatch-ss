import numpy as np

class GiantStepBase():
    def __init__(self,
                 target: callable, p: int, k: int,
                 activation: callable, a0: np.array, activation_derivative: callable,
                 gamma0: float, d: int, l: float, noise: float,
                 second_layer_update: bool, alpha: float):
        self.target = target
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.gamma0 = gamma0
        self.p = p
        self.k = k
        self.d = d
        self.l = l
        self.n = int(d**l)
        self.noise = noise
        self.second_layer_update = second_layer_update
        self.alpha = alpha


        self.a_s = [a0]
        self.test_errors = []

        self.network = lambda local_field: 1/np.sqrt(self.p) * np.dot(self.a, self.activation(local_field))

    @property
    def a(self):
        return self.a_s[-1]
    
class OverlapsBase(GiantStepBase):
    def __init__(self,
                 target: callable, activation: callable, activation_derivative: callable,
                 P: np.array, M0: np.array, Q0: np.array,  a0: np.array, 
                 gamma0: float, d: int, l: int, noise: float,
                 second_layer_update: bool, alpha: float):
        super().__init__(target,Q0.shape[0],P.shape[0], activation, a0, activation_derivative, gamma0, d, l, noise, second_layer_update, alpha)

        self.Ms = [M0]
        self.Qs = [Q0]
        self.P = P

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

