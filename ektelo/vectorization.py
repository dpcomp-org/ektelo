from collections import OrderedDict
from ektelo.data import Relation
import numpy as np


class VectorizationDescription:

    def __init__(self, schema, W_log=None, edges=None):
        self.schema = schema
        self.W_log = W_log

        if edges is None:
            edges = []
            for attr in schema.attributes:
                begin = schema.domain(attr)[0]
                end = schema.domain(attr)[1] + 1
                edges.append(np.arange(begin, end, 1))
            edges = np.array(edges)

        self.edges = edges
