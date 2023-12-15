from numpy import np 

from .base import OverlapsBase

# Constants
_pi = np.pi
_sqrtpi = np.sqrt(np.pi)
_pisqrtpi = np.pi * _sqrtpi
_pisquared = np._pisquared


class StudentBase_Staircase2(OverlapsBase):
    r"""
    Teacher is f^*(\vec{\lf}^*) = \lf^*_0 + \lf^*_0 * \lf^*_1

    Student is f(\vec{\lf}) = \frac1p \sum_{j=1}^p \sigma(\lf_j)

    Methods needed to work:
     - _activation
     - _activation_derivative
     - _actprime_lft0_target
     - _actprime_lft1_target
     - _actprime_lfs_target
     - _I3
     - _actprime_actprime_targetsquare
     - _actprime_actprime_target_act
     - _I4
     - _I2noise
    """
    @staticmethod
    def _target(local_fields):
        return local_fields[:,0] + local_fields[:,0] * local_fields[:,1]

    def __init__(self, P: np.array, M0: np.array, Q0: np.array, a0: np.array, gamma: float, noise: float, I4_diagonal:bool, I4_offdiagonal:bool, second_layer_update: bool):
        # Check that k=2
        assert(P.shape[0] == 2)
        assert(M0.shape[1] == 2)
        super().__init__(self._target, self._activation, self._activation_derivative, P, M0, Q0, a0, gamma, noise, I4_diagonal, I4_offdiagonal, second_layer_update)

    def compute_expected_values(self):
        ev_target = np.zeros(shape=(self.p, self.k))
        ev_network = np.zeros(shape=(self.p, self.p))
        ev_I4 = np.zeros(shape=(self.p, self.p))

        # EV target
        for j in range(self.p):
            # target part
            ev_target[j, 0] += self._actprime_lft0_target() 
            ev_target[j, 1] += self._actprime_lft1_target()
                
            # network part
            for s in range(self.p):
                ev_target[j, 0] -= 1./self.p * self.a[s] * (
                    self._I3()
                )
                ev_target[j, 1] -= 1./self.p * self.a[s] * (
                    self._I3()
                )
        
        # EV network
        for j in range(self.p):
            for l in range(self.p):
                # target part
                ev_network[j, l] += self._actprime_lfs_target()
                
                # network part
                for s in range(self.p):
                    ev_network[j, l] -= 1./self.p * self.a[s] * (
                        self._I3()
                    )
        
        # EV I4
        for j in range(self.p):
            for l in range(self.p):
                # target-target part
                ev_I4[j, l] += self._actprime_actprime_targetsquare()

                for s in range(self.p):
                    # target-network part
                    ev_I4[j, l] -= 1./self.p * self.a[s] * (
                        self._actprime_actprime_target_act()
                    )
                    # network-network part
                    for u in range(self.p):
                        ev_I4[j, l] += 1./self.p**2 * self.a[s] * self.a[u] * (
                            self._I4()
                        )
                
                # noise part
                ev_I4[j, l] += self.noise * (
                    self._I2noise()
                )


    @staticmethod
    def _actprime_lft0_target():
        raise NotImplementedError('StudentBase_Staircase2 is an abstract class, this methos is student specific')
    
    @staticmethod
    def _actprime_lft1_target():
        raise NotImplementedError('StudentBase_Staircase2 is an abstract class, this methos is student specific')
    
    @staticmethod
    def _I3():
        raise NotImplementedError('StudentBase_Staircase2 is an abstract class, this methos is student specific')
    
    @staticmethod
    def _actprime_lfs_target():
        raise NotImplementedError('StudentBase_Staircase2 is an abstract class, this methos is student specific')
    
    @staticmethod
    def _actprime_actprime_targetsquare():
        raise NotImplementedError('StudentBase_Staircase2 is an abstract class, this methos is student specific')
    
    @staticmethod
    def _actprime_actprime_target_act():
        raise NotImplementedError('StudentBase_Staircase2 is an abstract class, this methos is student specific')
    
    @staticmethod
    def _I4():
        raise NotImplementedError('StudentBase_Staircase2 is an abstract class, this methos is student specific')


class Hermite2Relu_Staricase2(StudentBase_Staircase2):
    r"""
    The studend activation \sigma is the truncated ReLU function,
    up to the second Hermite polynomial.
    """

    @staticmethod
    def _activation(x):
        return 1./(4*_sqrtpi) + x/2 + x**2/(2*_sqrtpi)
    
    @staticmethod
    def _activation_derivative(x):
        return 1./2 + x/_sqrtpi
    
    @staticmethod
    def _actprime_lft0_target(C11, C12, C13, C22, C23, C33):
        return C11/2. + (2*C12*C13)/_sqrtpi + (C11*C23)/_sqrtpi
                                                                              
    @staticmethod
    def _actprime_lft1_target(C11, C12, C13, C22, C23, C33):
        return C12/2. + (C13*C22)/_sqrtpi + (2*C12*C23)/_sqrtpi                                                               
                                                                                              
    @staticmethod
    def _actprime_lfs_target(C11, C12, C13, C14, C22, C23, C24, C33, C34, C44):
        C13/(4.*_pi) + C14/4. + (C14*C34)/_pi + (C13*C44)/(2.*_pi)

    @staticmethod
    def _I3(C33, C34, C35, C44, C45, C55): # Omming the arguments C1x, C2x since target is not involved
        return C34/(4.*_pi) + C45/4. + (C35*C45)/_pi + (C34*C55)/(2.*_pi)
                                                                                              
    @staticmethod
    def _actprime_actprime_targetsquare(C11, C12, C13, C14, C22, C23, C24, C33, C34, C44):
        return (
            C11/4. + C12**2/2. + (2*C12*C13)/_sqrtpi + (2*C12*C14)/_sqrtpi + (2*C13*C14)/_pi + (C11*C22)/4. + (2*C13*C14*C22)/_pi +
            (C11*C23)/_sqrtpi + (4*C12*C14*C23)/_pi + (C11*C24)/_sqrtpi + (4*C12*C13*C24)/_pi + (2*C11*C23*C24)/_pi + (C11*C34)/_pi +
            (2*C12**2*C34)/_pi + (C11*C22*C34)/_pi
        )
                                                                                              
    @staticmethod
    def _actprime_actprime_target_act(C11, C12, C13, C14, C15, C22, C23, C24, C25, C33, C34, C35, C44, C45, C55):
        return (
            C12/(16.*_sqrtpi) + C13/(8.*_pi) + C14/(8.*_pi) + C15/8. + (C14*C23)/(4.*_pisqrtpi) + (C15*C23)/(4.*_sqrtpi) +
            (C13*C24)/(4.*_pisqrtpi) + (C15*C24)/(4.*_sqrtpi) + (C13*C25)/(4.*_sqrtpi) + (C14*C25)/(4.*_sqrtpi) + (C15*C25)/(4.*_sqrtpi) + 
            (C12*C34)/(4.*_pisqrtpi) + (C15*C34)/(2.*_pi) + (C15*C25*C34)/_pisqrtpi + (C12*C35)/(4.*_sqrtpi) + (C14*C35)/(2.*_pi) + 
            (C15*C35)/(2.*_pi) + (C15*C24*C35)/_pisqrtpi + (C14*C25*C35)/_pisqrtpi + (C12*C45)/(4.*_sqrtpi) + (C13*C45)/(2.*_pi) + 
            (C15*C45)/(2.*_pi) + (C15*C23*C45)/_pisqrtpi + (C13*C25*C45)/_pisqrtpi + (C12*C35*C45)/_pisqrtpi + (C12*C55)/(8.*_sqrtpi) + 
            (C13*C55)/(4.*_pi) + (C14*C55)/(4.*_pi) + (C14*C23*C55)/(2.*_pisqrtpi) + (C13*C24*C55)/(2.*_pisqrtpi) + (C12*C34*C55)/(2.*_pisqrtpi)
        )
    
    @staticmethod
    def _I4(C33, C34, C35, C36, C44, C45, C46, C55, C56, C66):
        return (
            1/(64.*_pi) + C34/(16.*_pisquared) + C35/(16.*_pi) + C36/(16.*_pi) + C45/(16.*_pi) + (C35*C45)/(4.*_pisquared) + (C36*C45)/(4.*_pi) +
            C46/(16.*_pi) + (C35*C46)/(4.*_pi) + (C36*C46)/(4.*_pisquared) + C55/(32.*_pi) + (C34*C55)/(8.*_pisquared) + (C36*C55)/(8.*_pi) +
            (C46*C55)/(8.*_pi) + (C36*C46*C55)/(2.*_pisquared) + C55/16. + (C34*C55)/(4.*_pi) + (C35*C55)/(4.*_pi) + (C36*C55)/(4.*_pi) +
            (C45*C55)/(4.*_pi) + (C36*C45*C55)/_pisquared + (C46*C55)/(4.*_pi) + (C35*C46*C55)/_pisquared + C55**2/(8.*_pi) +
            (C34*C55**2)/(2.*_pisquared) + C66/(32.*_pi) + (C34*C66)/(8.*_pisquared) + (C35*C66)/(8.*_pi) + (C45*C66)/(8.*_pi) +
            (C35*C45*C66)/(2.*_pisquared) + (C55*C66)/(16.*_pi) + (C34*C55*C66)/(4.*_pisquared)
        )

    @staticmethod
    def _I2noise(C33, C34, C44):
        return 0.25 + C34/_pi
