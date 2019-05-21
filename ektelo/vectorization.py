from collections import OrderedDict
from ektelo.data import Relation
import numpy as np


class VectorizationDescription:

    def __init__(self, W_log, schema):
        self.W_log = W_log
        self.schema = schema

    def vectorize(self, R: Relation):
        attributes = self.schema.attributes

        for attr in attributes:
            assert self.schema.type(attr) == 'discrete'

        attr_dom_map = {attr: self.schema.domain(attr) for attr in attributes}
        attr_dsize_map = OrderedDict([(attr, attr_dom_map[attr][1]-attr_dom_map[attr][0]+1) for attr in attributes])
        vec = np.zeros((np.product(list(attr_dsize_map.values())),1)) 

        for index, row in R.df.iterrows():
            idx = 0
            mult = 1
            for i in reversed(range(len(attributes))):
                attr = attributes[i]
                #idx += int(np.product([dsize for dsize in list(attr_dsize_map.values())[i+1:]]) + row[attr])
                idx += mult*row[attr]
                if i > 0:
                    mult *= attr_dsize_map[attributes[i-1]]
            vec[idx] += 1

        return vec
