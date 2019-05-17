from ektelo import support
from ektelo import util
from functools import reduce
import numpy as np


class VectorizationDescription:

    def __init__(self, W, schema):
        self.W = W
        self.schema = schema

    @property
    def shape(self):
        pass
