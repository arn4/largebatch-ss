import numpy as np
from scipy.special import erf

from .base import OverlapsBase
from .cython_erf_erf import erf_updates, erf_error

def _target(local_fields, a):
    k = len(local_fields) 
    return 1/np.sqrt(k) * np.sum(local_fields**2-1,axis=-1)
def _activation(x):
    return x**2 - 1
def _activation_derivative(x):
    return 2 * x

def _I2(Caa):
    return 2*Caa

def _I3(Caa, Cab, Cac, Cbb, Cbc, Ccc):
    return 2*Cab*Ccc + 4*Cac*Cbc - 2*Cab

def _I4(Caa, Cab, Cac, Cad, Cbb, Cbc, Cbd, Ccc, Ccd, Cdd):
    return 4*Cab*Ccc*Cdd + 8*Cab*Ccd**2 + 8*Cac*Cbc*Cdd + 16*Cac*Cbd*Ccd + 16*Cad*Cbc*Ccd + 8*Cad*Cbd*Ccc - 4*Cab*Cdd -8*Cac*Cbc - 4*Cab*Ccc - 8*Cad*Cbd + 4*Cab

class H2H2Overlaps(OverlapsBase):
    def __init__(self, P: np.array, M0: np.array, Q0: np.array, a0: np.array, gamma0: float, d: int, l: int, noise: float, second_layer_update: bool, alpha: float):
        super().__init__(_target, _activation, _activation_derivative, P, M0, Q0, a0, gamma0, d, l, noise, second_layer_update, alpha)
        self.measure()
    def compute_expected_values(self):
        ev_target = np.zeros(shape=(self.p, self.k))
        ev_network = np.zeros(shape=(self.p, self.p))
        ev_I4 = np.zeros(shape=(self.p, self.p))

        # EV target
        for j in range(self.p):
            for r in range(self.k):
                for t in range(self.k):
                    ev_target[j, r] += 1/np.sqrt(self.k) * _I3(
                        self.Q[j,j], self.M[j,r], self.M[j,t],
                        self.P[r,r], self.P[r,t],
                        self.P[t,t]
                    )
                for s in range(self.p):
                    ev_target[j, r] -= 1/np.sqrt(self.p) * self.a[s] * _I3(
                        self.Q[j,j], self.M[j,r], self.Q[j,s],
                        self.P[r,r], self.M[s,r],
                        self.Q[s,s]
                    )
        # EV network
        for j in range(self.p):
            for l in range(self.p):
                for t in range(self.k):
                    ev_network[j, l] += 1/np.sqrt(self.k) * _I3(
                        self.Q[j,j], self.Q[j,l], self.M[j,t],
                        self.Q[l,l], self.M[l,t],
                        self.P[t,t]
                    )
                for s in range(self.p):
                    ev_network[j, l] -= 1/np.sqrt(self.p) * self.a[s] * _I3(
                        self.Q[j,j], self.Q[j,l], self.Q[j,s],
                        self.Q[l,l], self.Q[l,s],
                        self.Q[s,s]
                    )
        # EV I4
        for j in range(self.p):
            for l in range(self.p):
                for r in range(self.k):
                    for t in range(self.k):
                        ev_I4[j, l] += 1/self.k * _I4(
                            self.Q[j,j], self.Q[j,l], self.M[j,r], self.M[j,t],
                            self.Q[l,l], self.M[l,r], self.M[l,t],
                            self.P[r,r], self.P[r,t],
                            self.P[t,t]
                        )
                for s in range(self.p):
                    for u in range(self.p):
                        ev_I4[j, l] += 1/self.p * self.a[s] * self.a[u] * _I4(
                            self.Q[j,j], self.Q[j,l], self.Q[j,s], self.Q[j,u],
                            self.Q[l,l], self.Q[l,s], self.Q[l,u],
                            self.Q[s,s], self.Q[s,u],
                            self.Q[u,u]
                        )

                for s in range(self.p):
                    for r in range(self.k):
                        ev_I4[j, l] -= 2/np.sqrt(self.p*self.k) * self.a[s] * _I4(
                            self.Q[j,j], self.Q[j,l], self.Q[j,s], self.M[j,r],
                            self.Q[l,l], self.Q[l,s], self.M[l,r],
                            self.Q[s,s], self.M[s,r],
                            self.P[r,r]
                        )
                
                ev_I4[j, l] += self.noise * _I2(self.Q[j,l])
        return ev_target, ev_network, ev_I4

    def measure(self):
        self.test_errors.append(
            0.
        )
        
