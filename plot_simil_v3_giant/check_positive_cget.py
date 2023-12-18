import numpy as np
from regression_functions import *
import pickle

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# Dictionary of possible student activation functions and its derivatives 
stud_acts = { 'relu': lambda x: np.maximum(x,0)}
stud_ders = {'relu': lambda x: (x>0).astype(int)}

to_int = lambda z: np.tanh(z) * z * gaussian(z,0,1)
fhct_tanh = quad(to_int, -np.inf, np.inf)[0] 
tanh_mod = lambda z: np.tanh(z) - fhct_tanh

def f_check_cget(Z):
    nsamples = Z.shape[0] 
    if ctrl == 'difficult':
        ret = Z[:,0] + Z[:,1]*Z[:,2] + np.sqrt(noise)*np.random.randn(nsamples)
    elif ctrl == '3dirs':
        ret = Z[:,0] + H2(Z[:,0])/2*Z[:,1] + Z[:,0]*Z[:,2] + np.sqrt(noise)*np.random.randn(nsamples)
    elif ctrl == 'tanh':
        ret = Z[:,0] + tanh_mod(Z[:,0])*Z[:,1] + Z[:,0]*Z[:,2] + np.sqrt(noise)*np.random.randn(nsamples)
    elif ctrl == '3dirs_difficult':
        ret =  ret = Z[:,0] + Z[:,1]*Z[:,2] + tanh_mod(Z[:,1])*Z[:,2] + np.sqrt(noise)*np.random.randn(nsamples)
    elif ctrl == '2_gd_then_fit':
        ret = Z[:,0] + Z[:,0]*Z[:,1] + tanh_mod(Z[:,1])*Z[:,2] + np.sqrt(noise)*np.random.randn(nsamples)
    elif ctrl == 'new':
        ret = Z[:,0] + Z[:,0]*Z[:,1] + tanh_mod(Z[:,2])*Z[:,3] + np.sqrt(noise)*np.random.randn(nsamples)
    else:
        ret = Z[:,0] + Z[:,0]*Z[:,1] + np.sqrt(noise)*np.random.randn(nsamples)
    return ret
def aux_fun(n,ntest,d):
    Z = np.random.randn(n,d) ; Ztest = np.random.randn(ntest,d)
    Y =  f_check_cget(Z) 
    Ytest = f_check_cget(Ztest)
    return Z,Ztest,Y,Ytest

H2 = lambda x: x**2 - 1

ctrl = 'tanh'
stud_act = 'relu' 
f = stud_acts[stud_act]
fprime = stud_ders[stud_act] 
fnn0 = lambda D , W , a : 1/np.sqrt(p)*f(D@W.T)@a
d = 2**8
ntest = int(5*1e4)
nseeds = 4
noise = 0

print(f'START CTRL = {ctrl}')

### VARY ALPHA ###

print('VARY ALPHA')
alphas = [4,8,16]
lamb = 1e-6
lr_strength = 5
list_eg_mean = {} ; list_eg_var = {} ; list_et_mean = {} ; list_et_var = {}
list_eg_mean_rf = {} ; list_eg_var_rf = {} ; list_et_mean_rf = {} ; list_et_var_rf = {}
list_eg_mean2 = {} ; list_eg_var2 = {} ; list_et_mean2 = {} ; list_et_var2 = {}
pss = []
for alpha in alphas:
    print(f'alpha = {alpha}')
    n = int(alpha*d)
    n2 = n
    list_name = f'alpha_{alpha}'
    list_eg_mean[list_name] = [] ; list_eg_var[list_name] = [] ; list_et_mean[list_name] = [] ; list_et_var[list_name] = []
    list_eg_mean_rf[list_name] = [] ; list_eg_var_rf[list_name] = [] ; list_et_mean_rf[list_name] = [] ; list_et_var_rf[list_name] = []
    # 2gd steps 
    list_eg_mean2[list_name] = [] ; list_eg_var2[list_name] = [] ; list_et_mean2[list_name] = [] ; list_et_var2[list_name] = []
    # change array of p to adapt to alpha
    ps = np.linspace(1,6*alpha*d,30).astype(int)
    pss.append(ps)
    for j,p in enumerate(ps):
        eta = lr_strength*np.sqrt(p)
        errgs, errts = [] , [] 
        errgs_rf, errts_rf = [] , []
        errgs2, errts2 = [] , []
        for trial in range(nseeds):
            a0 = 1/np.sqrt(p)*np.random.randn(p)
            W0 = 1/np.sqrt(d)*np.random.randn(p,d)
            print(f'START seed = {trial} p={p}')
            # GD on the RF weights 
            Z,Ztest,Y,Ytest = aux_fun(n,ntest,d) 
            G = 1/n * Z.T @ ( 1/np.sqrt(p) * np.outer( ( Y - fnn0(Z,W0,a0) ) , a0) * fprime(Z@W0.T))
            Wnew = W0 + eta*np.sqrt(p)*G.T
            # Train second layer with n2 samples
            ## 1GD step 
            print('1GD step model')
            Z,Ztest,Y,Ytest = aux_fun(n2,ntest,d) 
            X = f(Z@Wnew.T) ; Xtest = f(Ztest@Wnew.T) 
            e1,e2,s1,s2,w = get_errors_ridge(X,Xtest,Y,Ytest,lamb)
            errgs.append(e2) ; errts.append(e1)
            ## 2 GD step 
            print('2GD step model')
            Z,Ztest,Y,Ytest = aux_fun(n,ntest,d)
            G = 1/n * Z.T @ ( 1/np.sqrt(p) * np.outer( ( Y - fnn0(Z,Wnew,a0) ) , a0) * fprime(Z@Wnew.T))
            Wnew2 = Wnew + eta*np.sqrt(p)*G.T
            Z,Ztest,Y,Ytest = aux_fun(n2,ntest,d)
            X = f(Z@Wnew2.T) ; Xtest = f(Ztest@Wnew2.T)
            e1,e2,s1,s2,w = get_errors_ridge(X,Xtest,Y,Ytest,lamb)
            errgs2.append(e2) ; errts2.append(e1)
            ## RF model
            print('RF model')
            X = f(Z@W0.T) ; Xtest = f(Ztest@W0.T)
            e1,e2,s1,s2,w = get_errors_ridge(X,Xtest,Y,Ytest,lamb)
            errgs_rf.append(e2) ; errts_rf.append(e1)
        list_eg_mean[list_name].append(np.mean(errgs)) ; list_et_mean[list_name].append(np.mean(errts))
        list_eg_var[list_name].append(np.var(errgs)) ; list_et_var[list_name].append(np.var(errts))
        list_eg_mean_rf[list_name].append(np.mean(errgs_rf)) ; list_et_mean_rf[list_name].append(np.mean(errts_rf))
        list_eg_var_rf[list_name].append(np.var(errgs_rf)) ; list_et_var_rf[list_name].append(np.var(errts_rf))
        list_eg_mean2[list_name].append(np.mean(errgs2)) ; list_et_mean2[list_name].append(np.mean(errts2))
        list_eg_var2[list_name].append(np.var(errgs2)) ; list_et_var2[list_name].append(np.var(errts2))
        print(f'END p={p}')
# Save results
np.savez(f'check_positive_cget_ctrl={ctrl}_vary_alpha.npz',
list_eg_mean=list_eg_mean,list_eg_var=list_eg_var,list_et_mean=list_et_mean,list_et_var=list_et_var, 
list_eg_mean_rf=list_eg_mean_rf,list_eg_var_rf=list_eg_var_rf,list_et_mean_rf=list_et_mean_rf,list_et_var_rf=list_et_var_rf,
list_eg_mean2=list_eg_mean2,list_eg_var2=list_eg_var2,list_et_mean2=list_et_mean2,list_et_var2=list_et_var2,
pss=pss,lr_strengths=lr_strength,nseeds=nseeds,ntest=ntest,d=d,n=n,n2=n2,lambs = lamb,alphas=alphas
)
