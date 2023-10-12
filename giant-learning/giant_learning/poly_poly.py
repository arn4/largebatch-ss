import numpy as np

from .base import OverlapsBase

class PolynomialPolynomialOverlapsBase(OverlapsBase):
    def __init__(self, P: np.array, M0: np.array, Q0: np.array, a0: np.array, gamma0: float, d: int, l: int, noise: float, second_layer_update: bool, alpha: float):
        super().__init__(self._target, self._activation, self._activation_derivative, P, M0, Q0, a0, gamma0, d, l, noise, second_layer_update, alpha)
        self.measure()
    def compute_expected_values(self):
        ev_target = np.zeros(shape=(self.p, self.k))
        ev_network = np.zeros(shape=(self.p, self.p))
        ev_I4 = np.zeros(shape=(self.p, self.p))

        # EV target
        for j in range(self.p):
            for r in range(self.k):
                for t in range(self.k):
                    ev_target[j, r] += 1/np.sqrt(self.k) * self._I3(
                        self.Q[j,j], self.M[j,r], self.M[j,t],
                        self.P[r,r], self.P[r,t],
                        self.P[t,t]
                    )
                for s in range(self.p):
                    ev_target[j, r] -= 1/np.sqrt(self.p) * self.a[s] * self._I3(
                        self.Q[j,j], self.M[j,r], self.Q[j,s],
                        self.P[r,r], self.M[s,r],
                        self.Q[s,s]
                    )
        # EV network
        for j in range(self.p):
            for l in range(self.p):
                for t in range(self.k):
                    ev_network[j, l] += 1/np.sqrt(self.k) * self._I3(
                        self.Q[j,j], self.Q[j,l], self.M[j,t],
                        self.Q[l,l], self.M[l,t],
                        self.P[t,t]
                    )
                for s in range(self.p):
                    ev_network[j, l] -= 1/np.sqrt(self.p) * self.a[s] * self._I3(
                        self.Q[j,j], self.Q[j,l], self.Q[j,s],
                        self.Q[l,l], self.Q[l,s],
                        self.Q[s,s]
                    )
        # EV I4
        for j in range(self.p):
            for l in range(self.p):
                for r in range(self.k):
                    for t in range(self.k):
                        ev_I4[j, l] += 1/self.k * self._I4(
                            self.Q[j,j], self.Q[j,l], self.M[j,r], self.M[j,t],
                            self.Q[l,l], self.M[l,r], self.M[l,t],
                            self.P[r,r], self.P[r,t],
                            self.P[t,t]
                        )
                for s in range(self.p):
                    for u in range(self.p):
                        ev_I4[j, l] += 1/self.p * self.a[s] * self.a[u] * self._I4(
                            self.Q[j,j], self.Q[j,l], self.Q[j,s], self.Q[j,u],
                            self.Q[l,l], self.Q[l,s], self.Q[l,u],
                            self.Q[s,s], self.Q[s,u],
                            self.Q[u,u]
                        )

                for s in range(self.p):
                    for r in range(self.k):
                        ev_I4[j, l] -= 2/np.sqrt(self.p*self.k) * self.a[s] * self._I4(
                            self.Q[j,j], self.Q[j,l], self.Q[j,s], self.M[j,r],
                            self.Q[l,l], self.Q[l,s], self.M[l,r],
                            self.Q[s,s], self.M[s,r],
                            self.P[r,r]
                        )
                
                ev_I4[j, l] += self.noise * self._I2noise(self.Q[j,j], self.Q[j,l], self.Q[l,l])
        return ev_target, ev_network, ev_I4

    def measure(self):
        population_error = 0.
        for j in range(self.p):
            for l in range(self.p):
                population_error += 1/self.p * self.a[j] * self.a[l] * self._I2(self.Q[j,j], self.Q[j,l], self.Q[l,l])
        for j in range(self.p):
            for r in range(self.k):
                population_error -= 2/np.sqrt(self.p*self.k) * self.a[j] * self._I2(self.Q[j,j], self.M[j,r], self.P[j,r])
        for r in range(self.k):
            for t in range(self.k):
                population_error += 1/self.k * self._I2(self.P[r,r], self.P[r,t], self.P[t,t])
        self.test_errors.append(population_error)
        

class H2H2Overlaps(PolynomialPolynomialOverlapsBase):
    def _target(local_fields):
        k = len(local_fields) 
        return 1/np.sqrt(k) * np.sum(local_fields**2-1,axis=-1)
    
    @staticmethod
    def _activation(x):
        return x**2 - 1
    
    @staticmethod
    def _activation_derivative(x):
        return 2 * x

    @staticmethod
    def _I2noise(Caa,Cab,Cbb):
        return 2*Cab

    @staticmethod
    def _I2(Caa, Cab, Cbb):
        return Caa*Cbb + 2*Cab**2 - Caa - Cbb + 1

    @staticmethod
    def _I3(Caa, Cab, Cac, Cbb, Cbc, Ccc):
        return 2*Cab*Ccc + 4*Cac*Cbc - 2*Cab

    @staticmethod
    def _I4(Caa, Cab, Cac, Cad, Cbb, Cbc, Cbd, Ccc, Ccd, Cdd):
        return 4*Cab*Ccc*Cdd + 8*Cab*Ccd**2 + 8*Cac*Cbc*Cdd + 16*Cac*Cbd*Ccd + 16*Cad*Cbc*Ccd + 8*Cad*Cbd*Ccc - 4*Cab*Cdd -8*Cac*Cbc - 4*Cab*Ccc - 8*Cad*Cbd + 4*Cab
    
class H3H3Overlaps(PolynomialPolynomialOverlapsBase):
    def _target(local_fields):
        k = len(local_fields) 
        return 1/np.sqrt(k) * np.sum(local_fields**3-local_fields,axis=-1)
    
    @staticmethod
    def _activation(x):
        return x**3 - x
    
    @staticmethod
    def _activation_derivative(x):
        return 3 * x**2 - 1

    @staticmethod
    def _I2noise(Caa,Cab,Cbb):
        return 1 - 3*Caa + 18*Cab**2 - 3*Cbb + 9*Caa*Cbb

    @staticmethod
    def _I2(Caa, Cab, Cbb):
        return Cab - 3*Caa*Cab + 6*Cab**3 - 3*Cab*Cbb + 9*Caa*Cab*Cbb

    @staticmethod
    def _I3(Caa, Cab, Cac, Cbb, Cbc, Ccc):
        return -6*Cab*Cac + Cbc - 3*Caa*Cbc + 18*Cac**2*Cbc + 18*Cab*Cac*Ccc - 3*Cbc*Ccc + 9*Caa*Cbc*Ccc

    @staticmethod
    def _I4(Caa, Cab, Cac, Cad, Cbb, Cbc, Cbd, Ccc, Ccd, Cdd):
        return (
            -6*Cac*Cad + 18*Cac*Cad*Cbb + 36*Cab*Cad*Cbc - 108*Cac*Cad*Cbc**2 + 36*Cab*Cac*Cbd - 6*Cbc*Cbd + 18*Caa*Cbc*Cbd - 108*Cac**2*Cbc*Cbd - 
            108*Cad**2*Cbc*Cbd - 108*Cac*Cad*Cbd**2 + 18*Cac*Cad*Ccc - 54*Cac*Cad*Cbb*Ccc - 108*Cab*Cad*Cbc*Ccc - 108*Cab*Cac*Cbd*Ccc + 18*Cbc*Cbd*Ccc - 
            54*Caa*Cbc*Cbd*Ccc + 324*Cad**2*Cbc*Cbd*Ccc + 324*Cac*Cad*Cbd**2*Ccc + Ccd - 3*Caa*Ccd + 18*Cab**2*Ccd + 18*Cac**2*Ccd + 18*Cad**2*Ccd - 3*Cbb*Ccd + 
            9*Caa*Cbb*Ccd - 54*Cac**2*Cbb*Ccd - 54*Cad**2*Cbb*Ccd - 216*Cab*Cac*Cbc*Ccd + 18*Cbc**2*Ccd - 54*Caa*Cbc**2*Ccd + 324*Cad**2*Cbc**2*Ccd - 
            216*Cab*Cad*Cbd*Ccd + 1296*Cac*Cad*Cbc*Cbd*Ccd + 18*Cbd**2*Ccd - 54*Caa*Cbd**2*Ccd + 324*Cac**2*Cbd**2*Ccd - 3*Ccc*Ccd + 9*Caa*Ccc*Ccd - 
            54*Cab**2*Ccc*Ccd - 54*Cad**2*Ccc*Ccd + 9*Cbb*Ccc*Ccd - 27*Caa*Cbb*Ccc*Ccd + 162*Cad**2*Cbb*Ccc*Ccd + 648*Cab*Cad*Cbd*Ccc*Ccd - 54*Cbd**2*Ccc*Ccd + 
            162*Caa*Cbd**2*Ccc*Ccd - 108*Cac*Cad*Ccd**2 + 324*Cac*Cad*Cbb*Ccd**2 + 648*Cab*Cad*Cbc*Ccd**2 + 648*Cab*Cac*Cbd*Ccd**2 - 108*Cbc*Cbd*Ccd**2 + 
            324*Caa*Cbc*Cbd*Ccd**2 + 6*Ccd**3 - 18*Caa*Ccd**3 + 108*Cab**2*Ccd**3 - 18*Cbb*Ccd**3 + 54*Caa*Cbb*Ccd**3 + 18*Cac*Cad*Cdd - 54*Cac*Cad*Cbb*Cdd - 
            108*Cab*Cad*Cbc*Cdd + 324*Cac*Cad*Cbc**2*Cdd - 108*Cab*Cac*Cbd*Cdd + 18*Cbc*Cbd*Cdd - 54*Caa*Cbc*Cbd*Cdd + 324*Cac**2*Cbc*Cbd*Cdd - 
            54*Cac*Cad*Ccc*Cdd + 162*Cac*Cad*Cbb*Ccc*Cdd + 324*Cab*Cad*Cbc*Ccc*Cdd + 324*Cab*Cac*Cbd*Ccc*Cdd - 54*Cbc*Cbd*Ccc*Cdd + 
            162*Caa*Cbc*Cbd*Ccc*Cdd - 3*Ccd*Cdd + 9*Caa*Ccd*Cdd - 54*Cab**2*Ccd*Cdd - 54*Cac**2*Ccd*Cdd + 9*Cbb*Ccd*Cdd - 27*Caa*Cbb*Ccd*Cdd + 
            162*Cac**2*Cbb*Ccd*Cdd + 648*Cab*Cac*Cbc*Ccd*Cdd - 54*Cbc**2*Ccd*Cdd + 162*Caa*Cbc**2*Ccd*Cdd + 9*Ccc*Ccd*Cdd - 27*Caa*Ccc*Ccd*Cdd + 
            162*Cab**2*Ccc*Ccd*Cdd - 27*Cbb*Ccc*Ccd*Cdd + 81*Caa*Cbb*Ccc*Ccd*Cdd
        )