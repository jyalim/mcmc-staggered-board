# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 13:33:17 2021

@author: sbharath
"""
import matplotlib.pyplot as plt
import emcee
import likelifun1
import numpy as np
from multiprocessing import Pool
import pandas as pd

params= [47.16020557,76.16743912,50.21942078,81.10831694,-100.85229483,-107.39444187]
prior_para=params

p0 =  prior_para+1e-4*np.random.randn(24, 6)
nwalkers, ndim = p0.shape
print(p0.shape)

if __name__ == '__main__':
    import sys
    N = int(sys.argv[1])
    with Pool(processes=4) as pool:
        sampler = emcee.EnsembleSampler(
          nwalkers, 
          ndim, 
          likelifun1.likeli_fun0, 
          pool=pool,
          moves=[
            (emcee.moves.DEMove(), 0.8), 
            (emcee.moves.DESnookerMove(), 0.2),
          ]
        )
        sampler.run_mcmc(p0,N,progress=True); 


samples = sampler.get_chain()
print(samples.shape)
tau = sampler.get_autocorr_time()
print(tau)
flat_samples = sampler.get_chain(discard=0, flat=True)
print(flat_samples.shape)
pd.DataFrame(flat_samples).to_csv("results1.csv")
acceptance_fraction=sampler.acceptance_fraction
print("Acceptance Rate is: ", acceptance_fraction)

