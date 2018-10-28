import numpy as np
from functools import reduce
import pandas as pd
from pandas.api.types import CategoricalDtype

class Domain:
    """
    an object that stores domain information
    Note(ryan): not the best internal representation, consider changing
    """
    def __init__(self, attr, univ):
        """
        :param attr: a list of attribute name strings
        :param univ: a list of possible values for each corresponding attribute
        """
        self.attr = attr
        if not univ==[] and univ[0] is int:
            univ = [list(range(n)) for n in univ]
        self.univ = univ
        
    def project(self, cols):
        univ = [self.univ[self.attr.index(c)] for c in cols]
        return Domain(cols, univ)
    
    def shape(self, cols=None):
        """
        if cols is None, return the domain shape
        
        if cols is a list, return the domain shape with 1s for the columns
        that are not in the domain.
        """
        if cols is None:
            return tuple([len(u) for u in self.univ])
        shape = []
        for c in cols:
            try:
                i = self.attr.index(c)
                shape.append(len(self.univ[i]))
            except:
                shape.append(1)
        return tuple(shape)
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return ','.join(self.attr)

class Tabular:
    """ A tabular representation of a dataset 
    
    data is stored internally by a pandas dataframe, df
    domain is a domain object
    """
    def __init__(self, df, domain):
        self.df = df
        self.domain = domain
    
    def project(self, cols):
        """ project dataset onto a subset of columns """
        df = self.df[:,cols]
        domain = self.domain.project(cols)
        return Tabular(df, domain)
    
    def contingency_table(self):
        attrs, univ = self.domain.attr, self.domain.univ
        types = { a : CategoricalDtype(u) for a, u in zip(attrs, univ) } 
        df = self.df.astype(types).apply(lambda x: x.cat.codes)
        vals = df.values
        
        bins = [range(n+1) for n in self.domain.shape()]
        counts = np.histogramdd(vals, bins)[0]
        return ContingencyTable(counts, self.domain)

    def datavector(self):
        """ return the database in vector-of-counts form """
        return self.contingency_table().datavector()
    
class ContingencyTable:
    def __init__(self, counts, domain):
        self.counts = counts
        self.domain = domain
        
    def project(self, cols):
        ax = [self.domain.attr.index(c) for c in cols]
        ax = tuple(set(range(self.counts.ndim)) - set(ax))
        domain = self.domain.project(cols)
        counts = self.counts.sum(axis=ax)
        return ContingencyTable(counts, domain)
        
    def datavector(self):
        return self.counts.flatten()
    
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

if __name__ == '__main__':
    attrs = ['a','b','c']
    univ = [list(range(n)) for n in [3,4,5]]
    domain = Domain(attrs, univ)
    
    counts = np.random.rand(3,4,5)
    counts /= counts.sum()
    
    ABC = ContingencyTable(counts, domain)
    
    A = ABC.project(['a'])
    B = ABC.project(['b'])
    C = ABC.project(['c'])
    
    prod = ProductDist([A,B,C], domain, 1.0)
    
    AA = prod.project(['a'])
    
    a = A.datavector()
    aa = AA.datavector()
    print(a)
    print(aa)
    
    df = pd.DataFrame()
    df['a'] = ['x','y','z']
    df['b'] = [1,2,1]
    
    domain = Domain(['a','b'], [['x','y','z'], [1,2]])
    
    data = Tabular(df, domain)
