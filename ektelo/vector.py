from ektelo import util
from ektelo.matrix import EkteloMatrix
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator, lsqr
from functools import reduce

class EkteloVector:
    """
    An EkteloVector is a wrapper for various primitive vector types such as a numpy ndarray

    import numpy as np; from ektelo.matrix import Identity, Ones; from ektelo.vector import EkteloVector; v = EkteloVector(np.ones((3,1))); m = Identity(3); m2 = Ones(3, 1)

    works:
    m * v
    np.transpose(v) * m
    v - v
    v + v

    doesn't work:
    """
    def __init__(self, vector):
        """ Instantiate a EkteloVector from an explicitly represented backing vector
        
        :param vector: a 1d numpy ndarray
        """
        self.vector = vector
        self.dtype = vector.dtype
        self.shape = vector.shape
     
    def __abs__(self):
        return EkteloMatrix(self.vector.__abs__())

    def __add__(self, other):
        if np.isscalar(other):
            return self.vector + other

        assert self.shape == other.shape, 'incompatible dimensions for addition'

        if isinstance(other, EkteloVector):
            return EkteloVector(self.vector + other.vector)
        elif isinstance(other, EkteloMatrix):
            raise NotImplementedError('EkteloMatrix does not currently support addition')
        elif type(other) == np.ndarray:
            return EkteloVector(self.vector + other)
        else:
            raise TypeError('incompatible type %s for multiplication with EkteloVector')

    def _adjoint(self):
        return self._transpose()

    def __getitem__(self, key):
        return EkteloVector(self.vector.__getitem__(key))

    def __len__(self):
        return self.size()

    def __mul__(self, other):
        assert self.shape[-1] == other.shape[0], 'incompatible dimensions for multiplication'

        if np.isscalar(other):
            return self.vector * other
        elif isinstance(other, EkteloVector):
            return self.vector * other.vector
        elif isinstance(other, EkteloMatrix):
            return EkteloVector((other * self.transpose().vector).transpose())
        elif type(other) == np.ndarray:
            return EkteloMatrix(self.vector * other)
        else:
            raise TypeError('incompatible type %s for multiplication with EkteloVector')

    def __neg__(self):
        return EkteloVector(-self.vector)

    def __sub__(self, other):
        return -(-self + other)
    
    def asDict(self):
        d = util.class_to_dict(self, ignore_list=[])
        return d

    def flatten(self, order='C'):
        return EkteloVector(self.vector.flatten(order))

    def size(self):
        return int(np.prod(self.shape))

    def sum(self, axis=None):
        return EkteloVector(self.vector.sum())

    def to_array(self):
        return self.vector

    def transpose(self, *axes):
        vector = self.vector.T

        return EkteloVector(vector)
