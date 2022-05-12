# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 13:35:32 2021

@author: sbharath
"""
import os
import numpy as np
import pandas as pd 
import statsmodels.api as sm
#os.chdir(r'C:\Users\sbharath\Downloads\staggeredboardscode')
data0=pd.read_csv('python_dta.csv')
data0=data0.dropna()

def minmax(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

cols=[
  'logasset',
  'logasset_l',
  'free_cash',
  'lag_free_cash',
  'omega',
  'lag_omega'
]
for col in cols:
    data0[col]=minmax(data0[col])

data1=data0.values

M=900
np.random.seed(42)
storage=np.random.rand(M,4) 
q=0.03
r=0.03
beta=0.86
theta=0.916
share=0.01

delta=0.98
omega_seed=storage[:,1]
asset_seed=storage[:,2]
cash_seed=storage[:,3]

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

X = sm.add_constant(prior_asset)
model = sm.OLS(curr_asset,X)
results = model.fit()
print(results.summary())
const_a=results.params[0]
eta_a=results.params[1]
sig_a=np.std(curr_asset-results.predict(X))

X = sm.add_constant(prior_cash)
model = sm.OLS(curr_cash,X)
results = model.fit()
print(results.summary())
const_c=results.params[0]
eta_c=results.params[1]
sig_c=np.std(curr_cash-results.predict(X))

X=np.transpose([ prior_prod, prior_rd,prior_cb, 
                 prior_prod*prior_rd, prior_prod*prior_cb,prior_cb*prior_rd,
                 prior_prod*prior_cb*prior_rd])
X = sm.add_constant(X)
prod=data1[:,9]
model = sm.OLS(prod,X)
results = model.fit()
alpha0=results.params[0]
alpha1=results.params[1]
alpha2=results.params[2]
alpha3=results.params[3]
alpha4=results.params[4]
alpha5=results.params[5]
alpha6=results.params[6]
alpha7=results.params[7]
print(results.summary())
sig_o=np.std(prod-results.predict(X))
fire_rate=0.023/(1-np.mean(cb))
lambda_D=np.mean(rd)
lambda_S=np.mean(cb)
params=[1, 1.16e+01 , 1.58e+00 , 2.58e+00, -1.26e+02, -3.46e+00]
beta_da, beta_dc, beta_sa,beta_sc, u_d, u_s=params

E00 = np.array([ [1, 0], [0, 0] ])
E01 = np.array([ [0, 1], [0, 0] ])
E10 = np.array([ [0, 0], [1, 0] ])
E11 = np.array([ [0, 0], [0, 1] ])

def profits(omega): 
    A=((1-beta)*q/(beta*r))**(1-beta)
    O=((beta*theta*(omega*A)**theta)/q)**(1/(1-theta))
    return np.log((1/theta-1)*q*O/beta)

def normal_pdf(mu,sigx):
    return np.exp(-(mu)*(mu)/(2*sigx*sigx))/(sigx*np.sqrt(2*3.14))


def expcdf(x, lambdax):
    e=np.maximum((1-np.exp(-(1/lambdax)*x)),0)
    return e


def T_omega(omega,d,e): 
    Q=np.shape(omega)[0]
    omega=omega.reshape(Q,1)
    mu=omega_seed-(alpha0+alpha1*omega+alpha2*d+alpha4*e+alpha5*d*omega+\
                   alpha6*e*omega+alpha7*d*e*omega)
    npdf = normal_pdf(mu,sig_o)
    T=npdf/npdf.sum()
    return T

def T_X(X_seed,x,eta,const,sigx):
    Q=np.shape(x)[0]
    mu=X_seed-eta*x.reshape(Q,1)-const
    npdf = normal_pdf(mu,sigx)
    T=npdf/npdf.sum()
    return T

def build_T_tensor(omega):
    T  = np.tensordot(E00,T_omega(omega,0,0),axes=0) 
    T += np.tensordot(E01,T_omega(omega,0,1),axes=0) 
    T += np.tensordot(E10,T_omega(omega,1,0),axes=0) 
    T += np.tensordot(E11,T_omega(omega,1,1),axes=0) 
    return T

# T tensors constants in likeli_fun
TLF = build_T_tensor(prod)
TLF_asset=T_X(asset_seed,prior_asset,eta_a,const_a,sig_a)
TLF_cash=T_X(cash_seed,prior_cash,eta_c,const_c, sig_c)
TLF_asset_cash=TLF_asset*TLF_cash



def isnan(x):
    if int(x) == -9223372036854775808:
        return True
    else:
        return False

def value_function(omega,asset,cash,v, beta_da, beta_dc, beta_sa,
                   beta_sc, u_d, u_s): 
## function 10-13, page 7
    # T shape (2,2,M,M), v is shape (M,)
    T  = build_T_tensor(omega)
    T_asset=T_X(asset_seed,asset,eta_a,const_a,sig_a)
    T_cash=T_X(cash_seed,cash,eta_c,const_c, sig_c)
    V  = np.einsum('ijkl,kl->ijk',T,T_asset*T_cash*v)
    V00, V01, V10, V11 = V[0,0], V[0,1], V[1,0], V[1,1]
    kappa_D=asset*beta_da+cash*beta_dc+u_d
    kappa_S=asset*beta_sa+cash*beta_sc+u_s
    pd=expcdf(delta*V11-share*kappa_D-delta*V01,lambda_D)
    Ve=pd*(delta*V11-share*kappa_D)+(1-pd)*delta*V01
    pd=expcdf((1-fire_rate)*delta*V10-share*kappa_D-(1-fire_rate)*delta*V00,lambda_D )
    Vne=pd*((1-fire_rate)*delta*V10-share*kappa_D)+(1-pd)*(1-fire_rate)*delta*V00
    ps=expcdf(Ve-share*kappa_S-Vne, lambda_S)
    V_final=share*profits(np.exp(omega))+ps*(Ve-share*kappa_S)+(1-ps)*Vne
    return V_final


def v_fun(beta_da,beta_dc, beta_sa, beta_sc, u_d, u_s):     
    iter=0   
    tol=0.05
    diff=1
    maxiter=100
    v=storage[:,0]
    while iter<maxiter and diff>tol:
        v1=value_function(storage[:,1], storage[:,2],storage[:,3],
                          v,beta_da,beta_dc, beta_sa,beta_sc,u_d, u_s)
        mm=np.absolute(v-v1)/(np.absolute(v)+1e-10)
        diff=np.max(mm)        
        iter=iter+1
        v=v1

    return v


def likeli_fun(params):     
    #start = time.time()
    beta_da,beta_dc, beta_sa, beta_sc, u_d, u_s= params  
    v=v_fun(beta_da,beta_dc, beta_sa, beta_sc, u_d, u_s)      
    #start = time.time()
#   T11=T_omega(prod,1,1)
#   T10=T_omega(prod,1,0)
#   T01=T_omega(prod,0,1)
#   T00=T_omega(prod,0,0)
    # These only need to be computed once, as only v is param dependent
    # moved to preamble
#   TLF = build_T_tensor(prod)
#   TLF_asset=T_X(asset_seed,prior_asset,eta_a,const_a,sig_a)
#   TLF_cash=T_X(cash_seed,prior_cash,eta_c,const_c, sig_c)
#   TLF_asset_cash=TLF_asset*TLF_cash
    
    V = np.einsum('ijkl,kl->ijk',TLF,TLF_asset_cash*v)
    V00, V01, V10, V11 = V[0,0], V[0,1], V[1,0], V[1,1]
#   V11=np.sum(T11*T_asset*T_cash*v, axis=1)        
#   V01=np.sum(T01*T_asset*T_cash*v, axis=1)      
#   V10=np.sum(T10*T_asset*T_cash*v, axis=1)
#   V00=np.sum(T00*T_asset*T_cash*v, axis=1)
   
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
    likeli=np.mean(np.log(np.maximum(likeli_rd*likeli_cb, 1e-300)))
    likeli_agg=np.mean(likeli)*100
    #print(likeli_agg)
    #print(params)
    #end = time.time()
    #print(end - start)
    return likeli_agg

test=likeli_fun(params)


params= [47.16020557,76.16743912,50.21942078,81.10831694,-100.85229483,-107.39444187]

def likeli_fun0(params):     
    beta_da, beta_dc, beta_sa, beta_sc, u_d, u_s= params
    
    def log_prior(theta):
        beta_da, beta_dc, beta_sa, beta_sc, u_d, u_s = theta
        value = -np.inf
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
    #prior=np.zeros(6)
    #prior[0]=normal_pdf(beta_da,1)
    #prior[1]=normal_pdf(beta_dc,1)
    #prior[2]=normal_pdf(beta_sa,1)
    #prior[3]=normal_pdf(beta_sc,1)
    #prior[4]=normal_pdf(u_d,1)
    #prior[5]=normal_pdf(u_s,1)  
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    #s=likeli_fun(params)+np.log(np.prod(prior))
    s=likeli_fun(params)+lp
    return s

H=likeli_fun0(params)
print('Hello, please move forward')
