"""
Created on Thu Sep  9 13:35:32 2021

@author: sbharath
"""
import os
import numpy as np
import rmm 
import cupy as cp
import torch
from torch.cuda import nvtx
    
nvtx.range_push("rmm_cupy_allocator")
cp.cuda.set_allocator(rmm.rmm_cupy_allocator)
nvtx.range_pop()
#rmm.reinitialize(pool_allocator=True)
mempool = rmm.mr.PoolMemoryResource(
  rmm.mr.CudaMemoryResource(),
  initial_pool_size = 2**33,
  maximum_pool_size = 2**35,
)
rmm.mr.set_current_device_resource(mempool)

import cudf 
import pandas as pd  
import statsmodels.api as sm
import time

#fh_timing = open('timing','w')

def minmax(x):
    return (x-x.min())/(x.max()-x.min())

data0 = cudf.read_csv('python_dta.csv').dropna()

cols=[
  'logasset',
  'logasset_l',
  'free_cash',
  'lag_free_cash',
  'omega',
  'lag_omega'
]
for col in cols:
    data0[col] = minmax(data0[col])

data1 = data0.values
h_data1 = cp.asnumpy(data1)
data1 = data0.values

M=900
np.random.seed(42)
cp.random.seed(42)
storage=cp.random.rand(M,4) 
q=0.03
r=0.03
beta=0.86
theta=0.916
share=0.01

delta=0.98
omega_seed=storage[:,1]
asset_seed=storage[:,2]
cash_seed =storage[:,3]

prior_rd=data1[:,13]
prior_cb=data1[:,5]
prior_prod=data1[:,10]
prior_asset=data1[:,6]
prior_cash=data1[:,11]
curr_asset=data1[:,3]
curr_cash=data1[:,8]
cb=data1[:,2]
rd=data1[:,12]
prod=data1[:,9]

h_prior_rd    = cp.asnumpy(prior_rd   )
h_prior_cb    = cp.asnumpy(prior_cb   )
h_prior_prod  = cp.asnumpy(prior_prod )
h_prior_asset = cp.asnumpy(prior_asset)
h_prior_cash  = cp.asnumpy(prior_cash )
h_curr_asset  = cp.asnumpy(curr_asset )
h_curr_cash   = cp.asnumpy(curr_cash  )
h_cb          = cp.asnumpy(cb         )
h_rd          = cp.asnumpy(rd         )
h_prod        = cp.asnumpy(prod       )

h_X = sm.add_constant(h_prior_asset)
h_model = sm.OLS(h_curr_asset,h_X)
h_results = h_model.fit()
print(h_results.summary())
h_const_a = h_results.params[0]
h_eta_a = h_results.params[1]
h_sig_a = np.std(h_curr_asset-h_results.predict(h_X))
# put on device
const_a = cp.array(h_const_a)
eta_a   = cp.array(h_eta_a) 
sig_a   = cp.array(h_sig_a) 

h_X = sm.add_constant(h_prior_cash)
h_model = sm.OLS(h_curr_cash,h_X)
h_results = h_model.fit()
print(h_results.summary())
h_const_c = h_results.params[0]
h_eta_c   = h_results.params[1]
h_sig_c   = np.std(h_curr_cash-h_results.predict(h_X))
# put on device
const_c = cp.array(h_const_c)
eta_c   = cp.array(h_eta_c  )
sig_c   = cp.array(h_sig_c  )

h_X=np.transpose([ 
  h_prior_prod,
  h_prior_rd,
  h_prior_cb, 
  h_prior_prod * h_prior_rd, 
  h_prior_prod * h_prior_cb,
  h_prior_cb   * h_prior_rd,
  h_prior_prod * h_prior_cb * h_prior_rd
])
h_X = sm.add_constant(h_X)
#h_prod=h_data1[:,9]
h_model = sm.OLS(h_prod,h_X)
h_results = h_model.fit()
print(h_results.summary())

alpha0=cp.array(h_results.params[0])
alpha1=cp.array(h_results.params[1])
alpha2=cp.array(h_results.params[2])
alpha3=cp.array(h_results.params[3])
alpha4=cp.array(h_results.params[4])
alpha5=cp.array(h_results.params[5])
alpha6=cp.array(h_results.params[6])
alpha7=cp.array(h_results.params[7])

sig_o = cp.array(np.std(h_prod-h_results.predict(h_X)))
fire_rate= 0.023/(1-cp.mean(cb))
lambda_D = cp.mean(rd)
lambda_S = cp.mean(cb)
params=cp.array([1, 1.16e+01 , 1.58e+00 , 2.58e+00, -1.26e+02, -3.46e+00])
beta_da, beta_dc, beta_sa,beta_sc, u_d, u_s=params

E00 = cp.array([ [1, 0], [0, 0] ])
E01 = cp.array([ [0, 1], [0, 0] ])
E10 = cp.array([ [0, 0], [1, 0] ])
E11 = cp.array([ [0, 0], [0, 1] ])

def profits(omega): 
    A=((1-beta)*q/(beta*r))**(1-beta)
    O=((beta*theta*(omega*A)**theta)/q)**(1/(1-theta))
    return cp.log((1/theta-1)*q*O/beta)

def normal_pdf(mu,sigx):
    # 1/sqrt(2*pi) = 0.39894228040143267794
    # WATCH FOR: cupy adoption of scipy.special.eval_hermite
    inv_sig = 1/sigx
    return cp.exp(-0.5*(mu*inv_sig)**2)*inv_sig*0.39894228040143267794

def expcdf(x, lambdax):
    e=cp.maximum(1-cp.exp(-(1/lambdax)*x),0)
    return e

def T_omega(omega,d,e): 
    Q     = cp.shape(omega)[0]
    omega = omega.reshape(Q,1)
    mu    = omega_seed-(alpha0+alpha1*omega+alpha2*d+alpha4*e+alpha5*d*omega+\
                    alpha6*e*omega+alpha7*d*e*omega)
    nvtx.range_push("normal_pdf")
    npdf  = normal_pdf(mu,sig_o)
    nvtx.range_pop()
    T     = npdf/npdf.sum()
    return T

def T_X(X_seed,x,eta,const,sigx):
    Q    = cp.shape(x)[0]
    mu   = X_seed-eta*x.reshape(Q,1)-const
    npdf = normal_pdf(mu,sigx)
    T    = npdf/npdf.sum()
    return T

def build_T_tensor(omega):
    nvtx.range_push("T_omega_E00")
    T  = cp.tensordot(E00,T_omega(omega,0,0),axes=0) 
    nvtx.range_pop()
    nvtx.range_push("T_omega_E01")
    T += cp.tensordot(E01,T_omega(omega,0,1),axes=0) 
    nvtx.range_pop()
    nvtx.range_push("T_omega_E10")
    T += cp.tensordot(E10,T_omega(omega,1,0),axes=0) 
    nvtx.range_pop()
    nvtx.range_push("T_omega_E11")
    T += cp.tensordot(E11,T_omega(omega,1,1),axes=0) 
    nvtx.range_pop()
    return T

## =====================================================================
## GLOBAL VARS ---------------------------------------------------------
# T tensors constants in likeli_fun
TLF = build_T_tensor(prod)
TLF_asset=T_X(asset_seed,prior_asset,eta_a,const_a,sig_a)
TLF_cash=T_X(cash_seed,prior_cash,eta_c,const_c, sig_c)
TLF_asset_cash=TLF_asset*TLF_cash
# ----------------------------------------------------------------------
 
def isnan(x):
    if int(x) == -9223372036854775808:
        return True
    else:
        return False

def value_function(omega,asset,cash,v, beta_da, beta_dc, beta_sa,
                   beta_sc, u_d, u_s): 
    ## function 10-13, page 7
    # T shape (2,2,M,M), v is shape (M,)
    nvtx.range_push("build_T_Tensor")
    T       = build_T_tensor(omega)
    nvtx.range_pop()
    nvtx.range_push("T_X")
    T_asset = T_X(asset_seed,asset,eta_a,const_a,sig_a)
    T_cash  = T_X(cash_seed,cash,eta_c,const_c, sig_c)
    nvtx.range_pop()
    V       = cp.einsum('ijkl,kl->ijk',T,T_asset*T_cash*v)
    V00, V01, V10, V11 = V[0,0], V[0,1], V[1,0], V[1,1]
    kappa_D = asset*beta_da+cash*beta_dc+u_d
    kappa_S = asset*beta_sa+cash*beta_sc+u_s
    share_kappa_D = share*kappa_D
#   pd=expcdf(delta*V11-share*kappa_D-delta*V01,lambda_D)
    pd=expcdf(delta*(V11-V01)-share_kappa_D,lambda_D)
#   Ve=pd*(delta*V11-share*kappa_D)+(1-pd)*delta*V01
    Ve=delta*(pd*V11+(1-pd)*V01)-pd*share_kappa_D
#   pd=expcdf((1-fire_rate)*delta*V10-share*kappa_D-(1-fire_rate)*delta*V00,lambda_D )
    pd=expcdf((1-fire_rate)*delta*(V10-V00)-share_kappa_D,lambda_D )
#   Vne=pd*((1-fire_rate)*delta*V10-share*kappa_D)+(1-pd)*(1-fire_rate)*delta*V00
    Vne=(1-fire_rate)*delta*(pd*V10+(1-pd)*V00)-pd*share_kappa_D
    ps=expcdf(Ve-share*kappa_S-Vne, lambda_S)
    V_final=share*profits(cp.exp(omega))+ps*(Ve-share*kappa_S)+(1-ps)*Vne
    return V_final

def v_fun(beta_da, beta_dc, beta_sa, beta_sc, u_d, u_s):     
    iter     = 0   
    tol      = 0.05
    diff     = 1
    maxiter  = 100
    v        = storage[:,0]
    while iter<maxiter and diff>tol:
        nvtx.range_push("value_function")
        v1=value_function(storage[:,1], storage[:,2], storage[:,3],
                          v,beta_da,beta_dc, beta_sa,beta_sc,u_d, u_s)
        nvtx.range_pop()
        mm=cp.absolute(v-v1)/(cp.absolute(v)+1e-10)
        diff=cp.max(mm)        
        iter=iter+1
        v=v1

    return v


def likeli_fun(params):     
    #start = time.time()
    beta_da,beta_dc, beta_sa, beta_sc, u_d, u_s= params  
    nvtx.range_push("v_fun")
    v=v_fun(beta_da,beta_dc, beta_sa, beta_sc, u_d, u_s)      
    nvtx.range_pop()
    V = cp.einsum('ijkl,kl->ijk',TLF,TLF_asset_cash*v)
    V00, V01, V10, V11 = V[0,0], V[0,1], V[1,0], V[1,1]

    kappa_D=prior_asset*beta_da+prior_cash*beta_dc+u_d
    kappa_S=prior_asset*beta_sa+prior_cash*beta_sc+u_s
    pd1=expcdf(delta*V11-share*kappa_D-delta*V01,lambda_D) # function 23
    pd2=expcdf((1-fire_rate)*delta*V10-share*kappa_D-(1-fire_rate)*delta*V00,lambda_D)  
    p_rd_cb=cb*pd1+(1-cb)*pd2
    likeli_rd=rd*p_rd_cb+(1-rd)*(1-p_rd_cb)       
    Ve=pd1*(delta*V11-share*kappa_D)+(1-pd1)*delta*V01
    Vne=pd2*((1-fire_rate)*delta*V10-share*kappa_D)+(1-pd2)*(1-fire_rate)*delta*V00   
    p_cb=expcdf(Ve-share*kappa_S-Vne,lambda_S)       # function 22
    likeli_cb=rd*p_cb+(1-rd)*(1-p_cb)               
    likeli=cp.mean(cp.log(cp.maximum(likeli_rd*likeli_cb, 1e-300)))
    likeli_agg=cp.mean(likeli)*100
    #print(likeli_agg)
    #print(params)
    #end = time.time()
    #print(end - start)
    return likeli_agg

def likeli_fun0(params):     
#   tic = time.perf_counter()
    beta_da, beta_dc, beta_sa, beta_sc, u_d, u_s= params
    
    def log_prior(theta):
        beta_da, beta_dc, beta_sa, beta_sc, u_d, u_s = theta
        value = -cp.inf
        if (
                42 < beta_da <   52 
          and   70 < beta_dc <   80 
          and   45 < beta_sa <   55 
          and   75 < beta_sc <   85 
          and -105 <    u_d  <  -95 
          and -112 <    u_s  < -102
        ):
            value = 0.0
        return value
    lp = log_prior(params)
    if not cp.isfinite(lp):
        return -cp.inf
    s=likeli_fun(params)+lp
#   toc = time.perf_counter()
#   print(toc-tic,file=fh_timing)
    return s

if __name__ == '__main__':
    torch.cuda.cudart().cudaProfilerStart()
    nvtx.range_push("likeli_fun")
    test=likeli_fun(params)
    nvtx.range_pop()
    params= [47.16020557,76.16743912,50.21942078,81.10831694,-100.85229483,-107.39444187]
    nvtx.range_push("likeli_fun0")
    H=likeli_fun0(params)
    nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print(H)
