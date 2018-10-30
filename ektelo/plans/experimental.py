from ektelo import workload, matrix, dataset_experimental
import pandas as pd
import numpy as np
from ektelo.private import measurement

def synthetic(cols, dom, N = 1000000):
    arr = np.zeros((N, len(cols)), dtype=int)
    for i, n in enumerate(dom):
        arr[:,i] = np.random.randint(0, n, N)
    return pd.DataFrame(arr, columns=cols)
    
def inference(measurements):
    # consume measurements over different projections
    # return an estimate of the data vector in factored form (as a ProductDist)
    pass 

def factored_example():
    prng = np.random.RandomState(seed=0)
    cols = ['a','b','c', 'd', 'e', 'f', 'g']
    # domain is far too large to fit data vector in memory (or on spark)
    dom = (128, 2, 10, 1024, 1024, 1024 1024)
    # create the data, in tabular form
    data = synthetic(cols, dom)

    # Workload needs to make projections explicit
    W = { 'a'       : workload.Prefix(128), 
          ('b','c') : workload.Identity(20) }

   
    # suppose we take the following measurements
    # Identity on x_a 
    # Idenity on x_bc
    
    measurement_cache = []
    
    a = data.project('a').toarray()
    # note we could run some complex sub-plan on a, but keep things simple here
    I = matrix.Identity(128)
    y = measurement.Laplace(I, 0.5).measure(a, prng)
   
    # for proper inference need to store projection, queries, answers, and epsilon 
    # maybe we can create a class to store measurements, where projection and epsilon
    # are optional
    measurement_cache.append( ('a', I, y, 0.5) )
    
    bc = data.project(('b', 'c')).toarray()
    I = matrix.Identity(20)
    y = measurement.Laplace(I, 0.5).measure(bc, prng)

    measurement_cache.append( (('b','c'), I, y, 0.5) ) 

    # estimate the data in factored form
    prod_dist = inference(measurement_cache)

    # answer the workload
    for key in W:
        x = prod_dist.project(key).toarray()
        ans = W[key].dot(x)

    # or answer some other new queries
    ans = prod_dist.project(('a','c')).toarray()

   
    # note that the tabular data has the same interface as the estimated data,
    # even though the underlying representation is totally different
    # both should be used in the same way: 
    # - project down to a small domain, 
    # - convert to a vector
    # - use that vector normally
