from ektelo import workload, matrix
import pandas as pd
import numpy as np
from ektelo.data import Domain
from ektelo.data import Relation
from ektelo.private import measurement
from ektelo.client.inference_projected import FactoredMultiplicativeWeights

def synthetic(cols, dom, N = 1000000):
    arr = np.zeros((N, len(cols)), dtype=int)
    for i, n in enumerate(dom):
        arr[:,i] = np.random.randint(0, n, N)
    df = pd.DataFrame(arr, columns=cols)
    config = {field: {'bins': bins, 'domain': (0, bins)} for field, bins in zip(cols, dom)}
    return Relation(Domain(config), df)
    
def factored_example():
    prng = np.random.RandomState(seed=0)
    cols = ['a','b','c', 'd', 'e', 'f', 'g']
    dom = (128, 2, 10, 1024, 1024, 1024, 1024)
    # domain is far too large to fit data vector in memory (or on spark)
    # create the data, in tabular form
    total = 1000000
    data = synthetic(cols, dom, N = total)

    # Workload needs to make projections explicit
    W = { 'a'       : workload.Prefix(128), 
          ('b','c') : workload.Identity(20) }

   
    # suppose we take the following measurements
    # Identity on x_a 
    # Idenity on x_bc
    
    measurement_cache = []
    
    a = data.project(['a']).datavector()
    # note we could run some complex sub-plan on a, but keep things simple here
    I = matrix.Identity(128)
    y = measurement.Laplace(I, 0.5).measure(a, prng)
   
    # for proper inference need to store projection, queries, answers, and epsilon 
    # maybe we can create a class to store measurements, where projection and epsilon
    # are optional
    measurement_cache.append( (I, y, 0.5, ('a',)) )
    
    bc = data.project(['b', 'c']).datavector()
    I = matrix.Identity(20)
    y = measurement.Laplace(I, 0.5).measure(bc, prng)

    measurement_cache.append( (I, y, 0.5, ('b','c')) ) 

    # estimate the data in factored form
    infer_engine = FactoredMultiplicativeWeights(data.domain)
    prod_dist = infer_engine.infer(measurement_cache, total)

    # answer the workload
    for key in W:
        x = prod_dist.project(key).datavector().flatten()
        ans = W[key].dot(x)

    # or answer some other new queries
    ans = prod_dist.project(['a','c']).datavector().flatten()

   
    # note that the tabular data has the same interface as the estimated data,
    # even though the underlying representation is totally different
    # both should be used in the same way: 
    # - project down to a small domain, 
    # - convert to a vector
    # - use that vector normally

if __name__ == '__main__':
    factored_example()
