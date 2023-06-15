import numpy as np
from scipy.special import erf
from sklearn.preprocessing import normalize
from scipy.linalg import orth

from numpy import sqrt as sqrt 
from numpy.linalg import inv as inverse_matrix

from committee_learning.ode.cython_erf import erf_updates


p = 10
d = 1000
l = 1
n = d**l
k = 2
T = 5
gamma0 = .6
noise = 0.
alpha = 0.
mc_samples = 40000
second_layer_update = True
activation = lambda x: erf(x/sqrt(2))
activation_derivative = lambda x: sqrt(2/np.pi) * np.exp(-x**2/2)
target_activation = activation

# activation = lambda x: np.maximum(x, 0)
# activation_derivative = lambda x: np.heaviside(x,1/2)
# target_activation = lambda x: np.maximum(x, 0)


def network(local_field, a, activation):
    p = len(a)
    return 1/sqrt(p) * np.dot(a, activation(local_field)) 

def target(local_fields):
    return np.mean(target_activation(local_fields))

def error(W, a, zs, ys):
    return 1/2 * np.mean(
        (np.apply_along_axis(network, -1, zs @ W.T, a, activation)- ys)**2
    )

def local_field_error(network_field, target_field, a, noise_randomness):
    return .5*(target(target_field)+sqrt(noise)*noise_randomness-network(network_field, a, activation))**2

def samples(n, W_star):
    zs = np.random.normal(size=(n,d))
    ys = np.apply_along_axis(target, -1, zs @ W_star.T) + sqrt(noise) * np.random.normal(size=(n,))
    return zs, ys

def local_fields_montecarlo(f, size, Q,M,P,a, std = False):
    p = Q.shape[0]
    Omega = np.block([[Q, M], [M.T, P]])
    def compute_sample():
        local_fields = np.random.multivariate_normal(np.zeros(Omega.shape[0]), Omega)
        network_local_field = local_fields[:p]
        target_local_field = local_fields[p:]
        noise_randomness = np.random.normal()
        return f(network_local_field, target_local_field, a, noise_randomness)
    data = np.array([compute_sample() for _ in range(size)])
    if std:
        return np.mean(data, axis=0), np.std(data, axis=0)/sqrt(size)
    else:
        return np.mean(data, axis=0)

def sufficent_statistics_error(Q,M,P,a, **kwargs):
    return local_fields_montecarlo(local_field_error, mc_samples, Q,M,P,a, **kwargs)

W0 = 1/sqrt(d) * np.random.normal(size=(p,d))
W_star = orth((normalize(np.random.normal(size=(k,d)), axis=1, norm='l2')).T).T
a0 = 1/sqrt(p) * np.random.normal(size=(p,))

# zs, ys = samples(n, W_star)
z_test, y_test = samples(mc_samples, W_star)

M0 = W0 @ W_star.T
Q0 = W0 @ W0.T
P = W_star @ W_star.T

Ws = [W0]
Ms = [M0]
Qs = [Q0]
as_GD = [a0]
as_SS = [a0]

zs, ys = samples(n, W_star)
train_errors = [error(W0, a0, zs, ys)]
test_errors = [error(W0, a0, z_test, y_test)]
pop_errors = [sufficent_statistics_error(Q0, M0, P, a0,std=True)]

print('Train error GD: ', train_errors[-1])
print('Test error GD: ', test_errors[-1])
print('Pop error SS: ', pop_errors[-1])
for tau in range(1,T):
    zs, ys = samples(n, W_star)
    print('tau: ', tau)

    ## Gradient descent on weights
    W = Ws[-1]
    a = as_GD[-1]
    displacements = np.apply_along_axis(target, -1, zs @ W_star.T) + sqrt(noise)*np.random.normal(size=(n,)) - np.apply_along_axis(network, -1, zs @ W.T, a, activation)
    updated_W = W + gamma0 * sqrt(p) * d**((l-1)/2) * np.einsum('j,uj,u,ui->ji',a,activation_derivative(zs @ W.T),displacements,zs)/n
    X = activation(zs @ updated_W.T) # updated_features
    if second_layer_update:
        updated_a = inverse_matrix(X.T @ X + alpha*np.eye(p)) @ X.T @ ys
    else:
        updated_a = a
    Ws.append(updated_W)
    as_GD.append(updated_a)
    train_errors.append(error(updated_W, updated_a, zs, ys))
    test_errors.append(error(updated_W, updated_a, z_test, y_test))

    ## Saad&Solla
    M = Ms[-1]
    Q = Qs[-1]
    a = as_SS[-1]

    upQ, upM = erf_updates(Q, M, P, noise_term = True, gamma_over_p = gamma0/p, noise=noise, quadratic_terms=True)

    # updated_M = M + gamma0 * sqrt(p) * d**((l-1)/2) *  
    def I3_Q_update(network_field, target_field, a, noise_randomness):
        return (
            (target(target_field)+sqrt(noise)*noise_randomness-network(network_field, a, activation)) * (
                np.einsum('j,j,l->jl', a, activation_derivative(network_field), network_field) +
                np.einsum('l,l,j->jl', a, activation_derivative(network_field), network_field)
            )
        )
    def I4_Q_update(network_field, target_field, a, noise_randomness):
        return (
            (target(target_field)+sqrt(noise)*noise_randomness-network(network_field, a, activation))**2 * 
            np.einsum('j,l,j,l->jl', a, a, activation_derivative(network_field), activation_derivative(network_field))
        )
    updated_Q = Q + gamma0 * sqrt(p) * d**((l-1)/2) * local_fields_montecarlo(I3_Q_update, mc_samples, Q, M, P, a) + gamma0 * p * local_fields_montecarlo(I4_Q_update, mc_samples, Q, M, P, a)
    
    def a_update_cross(network_field, target_field, a, noise_randomness):
        return np.einsum('j,l->jl',activation(network_field), activation(network_field))
    def a_update_target(network_field, target_field, a, noise_randomness):
        return activation(network_field) * (target(target_field)+sqrt(noise)*noise_randomness)

    if second_layer_update:
        # ## Separate expectation
        # updated_a = inverse_matrix(
        #     local_fields_montecarlo(a_update_cross, mc_samples, Q, M, P, a) + n*alpha*np.eye(p)
        # ) @ local_fields_montecarlo(a_update_target, mc_samples, Q, M, P, a)

        # ##Full expectation
        # updated_a = local_fields_montecarlo(
        #     lambda network_field, target_field, a, noise_randomness: (a_update_cross(network_field, target_field, a, noise_randomness) + n*alpha*np.eye(p)) @ a_update_target(network_field, target_field, a, noise_randomness), 
        #     mc_samples, Q, M, P, a
        # )

        ## Oracle
        def A_sample(network_field, target_field, a, noise_randomness):
            return 1/(p) * np.einsum('j,l->jl',activation(network_field), activation(network_field))
        A = local_fields_montecarlo(A_sample, mc_samples, Q, M, P, a)
        def b_sample(network_field, target_field, a, noise_randomness):
            return 1/sqrt(p) * activation(network_field) * (target(target_field)+sqrt(noise)*noise_randomness)
        b = 1/sqrt(p) * local_fields_montecarlo(b_sample, mc_samples, Q, M, P, a)
        updated_a = inverse_matrix(A) @ b
    else:
        updated_a = a
    as_SS.append(updated_a)
    Qs.append(updated_Q)
    Ms.append(updated_M)
    pop_errors.append(sufficent_statistics_error(updated_Q, updated_M, P, updated_a,std=True))
    
    print('Train error GD: ', train_errors[-1])
    print('Test error GD: ', test_errors[-1])
    print('Pop error SS: ', pop_errors[-1])

# make a plot of the errors
import matplotlib.pyplot as plt
plt.plot(train_errors, label='train')
plt.plot(test_errors, label='test')
# Plot the population error with error bars
pop_errors_mean, pop_errors_std = zip(*pop_errors)
plt.errorbar(range(T), pop_errors_mean, yerr=pop_errors_std, label='pop')
# plt.xscale('log')
# plt.yscale('log')
plt.legend()
plt.show()

from numpy.linalg import eigvalsh as spectrum

Ms = np.array(Ms)
Qs = np.array(Qs)

# print(Ms.shape, Qs.shape,np.diagonal(Qs, axis1=1, axis2=2).shape)

Gamma = np.einsum('tjr,tj->tjr', Ms, np.diagonal(Qs,axis1=1, axis2=2))
Correlation = np.einsum('tjr,tjq->trq', Gamma, Gamma)
# print(Correlation.shape)
# print(Correlation)
# eigenvalues = np.apply_along_axis(lambda x: x[0][1], arr=Correlation, axis=1)
# print(eigenvalues)
eigenvalues = []
for corr in Correlation:
    eigenvalues.append(
        spectrum(corr)
    )
eigenvalues = np.array(eigenvalues)
# print(eigenvalues)

metric = np.max(eigenvalues, axis = -1) / np.sum(eigenvalues,axis=-1)

plt.plot(metric[1:],label='(max eigenv.)/trace',marker='o')
plt.legend()
plt.xlabel('steps')
plt.ylim(0., 1.1)
plt.show()




