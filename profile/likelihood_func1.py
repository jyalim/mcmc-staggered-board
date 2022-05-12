"""
Created on Thu Sep  9 13:33:17 2021

@author: sbharath
"""
import emcee
import numpy as np
import pandas as pd
import likelifun1
import torch
from torch.cuda import nvtx

params= [47.16020557,76.16743912,50.21942078,81.10831694,-100.85229483,-107.39444187]
prior_para=params

p0 =  prior_para+1e-4*np.random.randn(24, 6)
nwalkers, ndim = p0.shape
print(p0.shape)

if __name__ == '__main__':
    import sys
    N = 200 if len(sys.argv) < 2 else int(sys.argv[1])
    torch.cuda.cudart().cudaProfilerStart()
    nvtx.range_push("EMCEE ensemble sampler")
    sampler = emcee.EnsembleSampler(
      nwalkers, 
      ndim, 
      likelifun1.likeli_fun0, 
      moves=[
        (emcee.moves.DEMove(), 0.8), 
        (emcee.moves.DESnookerMove(), 0.2),
      ]
    )
    nvtx.range_pop();
    sampler.run_mcmc(p0,N,progress=True) 

    samples = sampler.get_chain()
    print(samples.shape)
    flat_samples = sampler.get_chain(discard=0, flat=True)
    print(flat_samples.shape)
    pd.DataFrame(flat_samples).to_csv("results1.csv")
    try:
        if N > 200:
            tau = sampler.get_autocorr_time()
            print(tau)
        acceptance_fraction=sampler.acceptance_fraction
        print("Acceptance Rate is: ", acceptance_fraction)
    except Exception as ex:
        print('Exception occured: ', ex) 
