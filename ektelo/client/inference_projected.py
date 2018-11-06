import numpy as np
from ektelo import matrix
from scipy import sparse
from ektelo.client.inference import multWeightsFast
from collections import OrderedDict
from functools import reduce

    
def projection_matrix(domain, proj):
    """ Construct the projection matrix P that projects a data vector from the
        full domain to a new datavector over a subset of attributes
        
    :param domain: A Domain object corresponding to the full domain
    :param proj: a tuple of attributes, corresponding to the projected domain
    :return: An ektelo matrix
    """ 
    new_domain = domain.project(proj)
    subs = []
    for a in domain.attrs:
        n = domain.size(a)
        if a in new_domain.attrs:
            subs.append(matrix.Identity(n))
        else:
            subs.append(matrix.Ones(1,n))
    
    return matrix.Kronecker(subs)
        

class ContingencyTable:
    def __init__(self, counts, domain):
        '''
        :param counts: ND histogram
        :parm domain: domain of the histogram
        '''
        self.counts = counts
        self.domain = domain
        
    def project(self, cols):

        project_attrs = set(self.domain.attrs) -  set(cols)
        ax = tuple([self.domain.attrs.index(att) for att in project_attrs])
        domain = self.domain.project(cols)
        counts = self.counts.sum(axis=ax)
        return ContingencyTable(counts, domain)
        
    def datavector(self):
        return self.counts.flatten()
    

class ProductDist:
    """ factored representation of data from MWEM paper """
    def __init__(self, factors, domain):
        """
        :param factors: a list of contingency tables, 
                defined over disjoint subsets of attributes
        :param domain: the domain object
        :param total: known or estimated total
        """
        self.factors = factors
        self.domain = domain

    def project(self, cols):
        domain = self.domain.project(cols)
        factors = []
        for factor in self.factors:
            pcol = [c for c in cols if c in factor.domain.attrs]
            if pcol != []:
                factors.append(factor.project(pcol))
        return ProductDist(factors, domain)

    def datavector(self):
        domain = self.domain
        factors = []
        for factor in self.factors:
            shape = []
            for att in domain.attrs:
                try:
                    shape.append(factor.domain.size(att))
                except:
                    shape.append(1)
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
    :param domain: the Domain object
    """
    Ms = []
    ys = []

    for M, y, noise_scale, proj in measurement_cache:
        c = 1.0 / noise_scale
        P = projection_matrix(domain, proj)
        Q = c * matrix._LazyProduct(M, P)
        Ms.append(Q)
        ys.append(c*y)
    return matrix.VStack(Ms), np.concatenate(ys)

def _cluster(measurement_cache):
    """
    Cluster the measurements into disjoint subsets by finding the connected 
    components of the graph implied by the measurement projections
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
        
        factors = []

        for group, proj in zip(groups, proj):
            subdom = self.domain.project(proj)
            n = np.prod(subdom.shape)
            M, y = projected_vstack(group, subdom)
            hatx = np.ones(n) * total / n
            hatx = multWeightsFast(hatx, M, y, updateRounds=100) / total
            factors.append(ContingencyTable(hatx.reshape(subdom.shape), subdom))
        return ProductDist(factors, self.domain)

if __name__ == '__main__':
    
    attrs = ('a', 'b')
    shape = (10, 10)
    domain = Domain(attrs, shape)

    I = matrix.Identity(10)
    y = np.ones(10)*10

    measurement_cache = [(I, y, 1.0, ('a',)), (I, y, 1.0, ('b',))]
    
    mw = FactoredMultiplicativeWeights( domain )
    mw.infer(measurement_cache, 100) 
