import numpy as np
from ektelo import matrix
from scipy import sparse
from ektelo.client.inference import multWeightsFast

class ProductDist:
    """ factored representation of data from MWEM paper """
    def __init__(self, factors, domain, total):
        """
        :param factors: a list of contingency tables, 
                defined over disjoint subsets of attributes
        :param domain: the domain object
        :param total: known or estimated total
        """
        self.factors = factors
        self.domain = domain
        self.total = total

    def project(self, cols):
        domain = self.domain.project(cols)
        factors = []
        for factor in self.factors:
            pcol = [c for c in cols if c in factor.domain.attr]
            if pcol != []:
                factors.append(factor.project(pcol))
        return ProductDist(factors, domain, self.total)

    def datavector(self):
        domain = self.domain
        factors = []
        for factor in self.factors:
            shape = factor.domain.shape(domain.attr)
            factors.append(factor.counts.reshape(shape))
        # np.prod only works correctly if len(factors) >= 2
        return reduce(lambda x,y: x*y, factors, 1.0)

"""
This class is designed to do inference from measurements taken over different projections
of the data.  
"""

def projected_vstack(measurement_cache, domain):
    """ 
    Convert measurements over different projections of the data to measurements
    over the full domain (or some common domain)
    :param measurement_cache: a list of (M, y, noise_scale, projection) tuples
        M is an EkteloMatrix of queries
        y is a numpy array of noisy answers
        noise_scale is the sqrt of the variance of the noisy answers
        projection is the tuple of axes the measurements are taken over
    :param domain: the domain size tuple
    """
    Ms = []
    ys = []

    for M, y, noise_scale, projection in measurement_cache:
        c = 1.0 / noise_scale
        P = matrix.Project(domain, projection)
        Q = c * matrix._LazyProduct(M, P)
        Ms.append(Q)
        ys.append(c*y)
    return matrix.VStack(Ms), np.concatenate(ys)

def _cluster(measurement_cache):
    """
    Cluster the measurements into disjoint subsets by finding the connected components
    of the graph implied by the measurement projections
    """
    # create the adjacency matrix
    k = len(measurement_cache)
    G = sparse.dok_matrix((k,k))
    for i, (_, _, _, p) in enumerate(measurement_cache):
        for j, (_, _, _, q) in enumerate(measurement_cache):
            if len(set(p) & set(q)) >= 1:
                G[i,j] = 1
    # find the connected components and group measurements
    ncomps, labels = sparse.csgraph.connected_components(G)
    groups = [ [] for _ in range(ncomps) ]
    projections = [ set() for _ in range(ncomps) ]
    for i, group in enumerate(labels):
        groups[group].append(measurement_cache[i])
        projections[group] |= set(measurement_cache[i][3])
    projections = [tuple(p) for p in projections]
    return groups, projections


class FactoredMultiplicativeWeights:
    def __init__(self, domain):
        self.domain = domain

    def infer(self, measurement_cache, total):
        groups, proj = _cluster(measurement_cache) # cluster measurements

        for group, proj in zip(groups, proj):
            subdom = tuple(self.domain[i] for i in proj)
            n = np.prod(subdom)
            M, y = projected_vstack(group, subdom)
            hatx = np.ones(n) * total / n
            hatx = multWeightsFast(hatx, M, y, updateRounds=100) / total

if __name__ == '__main__':

    I = matrix.Identity(10)
    y = np.ones(10)*10

    measurement_cache = [(I, y, 1.0, (0,)), (I, y, 1.0, (1,))]
    
    mw = FactoredMultiplicativeWeights( (10, 10) )
    mw.infer(measurement_cache, 100) 
