from collections import OrderedDict
from ektelo.data import Relation
import numpy as np


class VectorizationDescription:

    def __init__(self, W_log, schema):
        self.W_log = W_log
        self.schema = schema
