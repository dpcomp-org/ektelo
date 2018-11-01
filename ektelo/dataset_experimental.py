import numpy as np
from functools import reduce
import pandas as pd
from pandas.api.types import CategoricalDtype
from collections import OrderedDict

class Domain:
    def __init__(self, attrs, shape):
        """ Construct a Domain object
        
        :param attrs: a list or tuple of attribute names
        :param shape: a list or tuple of domain sizes for each attribute
        """
        self.attrs = attrs
        self.shape = shape
        self.config = OrderedDict(zip(attrs, shape))
        
    def project(self, attrs):
        """ project the domain onto a subset of attributes
        
        :param attrs: the attributes to project onto
        :return: the projected Domain object
        """
        # return the projected domain
        shape = tuple(self.config[a] for a in attrs)
        return Domain(attrs, shape)
    
    def size(self, col):
        """ return the size of an individual attribute
        :param col: the attribute 
        """
        return self.config[col]

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
        df = self.df[list(cols)]
        domain = self.domain.project(cols)
        return Tabular(df, domain)

    def datavector(self):
        """ return the database in vector-of-counts form """
        vals = self.df.values
        bins = [range(n+1) for n in self.domain.shape]
        counts = np.histogramdd(vals, bins)[0]
        return counts.flatten()
    
if __name__ == '__main__':
    attrs = ['a','b','c']
    shape = (3,4,5)
    domain = Domain(attrs, shape)
    
    df = pd.DataFrame()
    df['a'] = [0,1,2]
    df['b'] = [0,1,0]
    
    domain = Domain(['a','b'], [3, 2])
    data = Tabular(df, domain)

    from IPython import embed
    a = data.project(['a'])
    b = data.project(['b'])
    
    print(a.datavector())
    print(b.datavector())
