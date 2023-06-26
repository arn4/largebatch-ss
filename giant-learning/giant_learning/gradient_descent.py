import numpy as np
from numpy.linalg import inv as inverse_matrix

from .base import GiantStepBase
from .cython_erf_erf import erf_error

class GradientDescent(GiantStepBase):
    def __init__(self,
                 target: callable, W_target: np.array,
                 activation: callable, W0: np.array, a0: np.array, activation_derivative: callable,
                 gamma0: float, l: int, noise: float,
                 second_layer_update: bool, alpha: float,
                 seed: int = 0, test_size = None,
                 analytical_error = None):
        super().__init__(target, W0.shape[0], W_target.shape[0], activation, a0, activation_derivative, gamma0, W0.shape[1], l, noise, second_layer_update, alpha)

        self.rng = np.random.default_rng(seed)

        self.W_target = W_target
        self.W_s = [W0]

        self.analytical_error = analytical_error
        if self.analytical_error is None:
            if test_size is None:
                self.test_size = self.n
            self.test_size = test_size
            self.zs_test, self.ys_test = self.samples(self.test_size)
        ##########
        self.eq5 = []
        self.eq6 = []
        self.eq7 = []
        self.zs_special, self.ys_special = self.samples(self.n)
        ##########
        self.measure()

    def samples(self, size):
        zs = self.rng.normal(size=(size,self.d))
        ys = np.apply_along_axis(self.target, -1, zs @ self.W_target.T) + np.sqrt(self.noise) * np.random.normal(size=(size,))
        return zs, ys

    def error(self, zs, ys):
        if self.analytical_error == 'erferf':
            return erf_error(self.W @ self.W.T, self.W @ self.W_target.T, self.W_target @ self.W_target.T, self.a, self.noise)
        else:
            if zs is None and ys is None:
                zs, ys = self.zs_test, self.ys_test
            return 1/2 * np.mean(
                (np.apply_along_axis(self.network, -1, zs @ self.W.T)- ys)**2
            )
    
    @property
    def W(self):
        return self.W_s[-1]

    def update(self, zs, ys):
        ##########
        self.eq5.append(self.equation5(self.zs_special, self.ys_special))
        self.eq6.append(self.equation6(self.zs_special, self.ys_special))
        self.eq7.append(self.equation7(self.zs_special, self.ys_special))
        ##########
        displacements = np.apply_along_axis(self.target, -1, zs @ self.W_target.T) + np.sqrt(self.noise)*np.random.normal(size=(self.n,)) - np.apply_along_axis(self.network, -1, zs @ self.W.T)
        
        self.W_s.append(
            self.W + self.gamma0 * np.sqrt(self.p) * np.power(self.d,((self.l-1)/2))* 1/self.n * np.einsum('j,uj,u,ui->ji',self.a,self.activation_derivative(zs @ self.W.T),displacements,zs)
        )

        if self.second_layer_update:
            X = self.activation(zs @ self.W.T)/np.sqrt(self.p)
            self.a_s.append(inverse_matrix(X.T @ X + self.alpha*np.eye(self.p)) @ X.T @ ys)
        else:
            self.a_s.append(self.a)

    def measure(self, zs = None, ys = None):
        self.test_errors.append(self.error(zs, ys))

    def train(self, steps):
        for step in range(steps):
            zs, ys = self.samples(self.n)
            self.update(zs, ys)
            self.measure()

#################################################

    def equation7(self, zs, ys):
        displacement = ys - np.apply_along_axis(self.network, -1, zs @ self.W.T)
        return self.gamma0**2 * self.p * self.d**(self.l-1) * (
            1 / self.n**2 * np.einsum('j,l,uj,vl,ui,vi,u,v->jl', self.a, self.a, self.activation_derivative(zs @ self.W.T), self.activation_derivative(zs @ self.W.T),zs,zs,displacement,displacement)
        )
    
    def equation6(self, zs, ys):
        displacement = ys - np.apply_along_axis(self.network, -1, zs @ self.W.T)
        return self.gamma0 * np.sqrt(self.p) * self.d**((self.l-1)/2) * (
            1/self.n * (
                np.einsum('j,uj,ul,u->jl', self.a, self.activation_derivative(zs @ self.W.T), zs @ self.W.T, displacement) +
                np.einsum('l,ul,uj,u->jl', self.a, self.activation_derivative(zs @ self.W.T), zs @ self.W.T, displacement)
            )
        )
    
    def equation5(self, zs, ys):
        displacement = ys - np.apply_along_axis(self.network, -1, zs @ self.W.T)
        return self.gamma0 * np.sqrt(self.p) * self.d**((self.l-1)/2) * (
            1/self.n * np.einsum('j,uj,ul,u->jl', self.a, self.activation_derivative(zs @ self.W.T), zs @ self.W_target.T, displacement)
        )

    