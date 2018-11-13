from ektelo import workload, matrix
import pandas as pd
import numpy as np
from ektelo.data import Domain
from ektelo.data import Relation, RelationHelper
from ektelo.private import measurement
from ektelo.client import selection, inference_projected
from IPython import embed
import time
    
if __name__ == '__main__':
    prng = np.random.RandomState(seed=0)

    data = RelationHelper('ADULT').load()
    total = data.df.shape[0]
    
    measurement_cache = []
  
    # domain = 2 * 5 * 8 * 15 * 9 = 10800
    demographics = data.project(['sex', 'race', 'marital-status', 'occupation', 'workclass'])
    # measure all 3 way marginals of this contingency table
    marg = workload.DimKMarginals(demographics.domain.shape, 3)
    x = demographics.datavector()
    y = measurement.Laplace(marg, 0.25).measure(x, prng)
    measurement_cache.append( (marg, y, 1.0, demographics.domain) )
    
    
    age = data.project('age')
    x = age.datavector()
    #  take measurements on age for answering range queries
    hier = selection.H2(tuple(age.domain.shape)).select()
    y = measurement.Laplace(hier, 0.25).measure(x, prng)
    measurement_cache.append( (hier, y, 1.0, age.domain) )

    # domain = 100 * 100 = 10000    
    capital = data.project(['capital-gain', 'capital-loss'])
    x = capital.datavector()
    # take measurements on capital-gain and loss for answering 2D range queries
    quad = selection.QuadTree(tuple(capital.domain.shape)).select()
    y = measurement.Laplace(quad, 0.25).measure(x, prng)
    measurement_cache.append( (quad, y, 1.0, capital.domain) )
    
    # domain = 101 * 99 = 9999
    agehours = data.project(['age', 'hours-per-week'])
    x = agehours.datavector()
    # take measurements on age/hours-per-week
    ident = matrix.Identity(np.prod(agehours.domain.shape))
    y = measurement.Laplace(ident, 0.25).measure(x, prng)
    measurement_cache.append( (ident, y, 1.0, agehours.domain) )
    
    t0 = time.time()
    # total domain = 10000^3 = 10^12
    engine = inference_projected.FactoredMultiplicativeWeights(data.domain)
    est = engine.infer(measurement_cache, total)
    t1 = time.time()
    print('%.2f seconds' % (t1-t0))
    
    # inference is done, now we can answer new questions over the original data
    
    hours_occup = est.project(['hours-per-week', 'occupation'])
    x = hours_occup.datavector().flatten()
    I = workload.Identity(x.shape[0])
    y = I.dot(x)
    
    
    
    
    