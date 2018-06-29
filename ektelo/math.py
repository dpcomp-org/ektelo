import inspect
import numpy as np
import scipy.sparse
from scipy.sparse import spmatrix

class DelegateMatrix:
    # import numpy as np; from scipy import sparse; from ektelo.math import DelegateMatrix; s = sparse.csr_matrix((3, 4), dtype=np.int8); m = DelegateMatrix(s)
    #
    # This approach is very flexible. Not only do you automatically inherit many methods on the delegate object
    # but numpy is also capable of handling duck-typed objects, therefore many builtin methods also work.
    # The principal drawback is that the return type of every call is that of the delegate object. So we
    # cannot easily do things like m * m.T because the first object with be a DelegateMatrix and the second 
    # will be a scipy sparse matrix.
    #
    # Works:
    # m.shape
    # m + m
    # np.abs(m)
    # np.sum(m)
    # m.T
    # m * m.T
    # m * s.T
    #
    # Doesn't work:
    # s * m.T

    def __init__(self, mat):
        self._mat = mat

    def __abs__(self):
        print('absing', self)
        return self._mat.__abs__()

    def __add__(self, other):
        print('adding', self, other)
        if type(other) == DelegateMatrix:
            other = other._mat
        return DelegateMatrix(self._mat + other)

    def __mul__(self, other):
        print('multiplying', self, other)
        if type(other) == DelegateMatrix:
            other = other._mat
        return DelegateMatrix(self._mat * other)
 
    @property
    def dtype(self):
        return self._mat.dtype

    @property
    def ndim(self):
        return self._mat.ndim
    
    @property
    def T(self):
        return DelegateMatrix(self._mat.T)
 
    @property
    def shape(self):
        return self._mat.shape


import scipy.sparse.linalg
import scipy.sparse.linalg.interface
class LinopMatrix(scipy.sparse.linalg.LinearOperator):
    # import numpy as np; from scipy import sparse; from ektelo.math import LinopMatrix; s = sparse.csr_matrix((3, 4), dtype=np.int8); s[0,0] = 3; s[1,1] = 1; s[2,2] = 2; m = LinopMatrix(s)
    # Works:
    # m.shape
    # np.abs(m)
    # m.T
    # m * m.T
    # m * s.T
    # (m * m.T).todense()
    #
    # Doesn't work:
    # m + m
    # np.sum(m)
    # s * m.T

    def __init__(self, A):
        self.A = A

    def _transpose(self):
        return LinopMatrix(self.A.T)

    def __abs__(self):
        return LinopMatrix(self.A.__abs__())

    def matmat( self, x ):
        return LinopMatrix(self.A * x)

    def matvec( self, x ):
        return LinopMatrix(self.A * x)

    def _matvec(self, b):
        raise NotImplementedError( "_matvec" )

    def dot(self, x):
        if isinstance(x, scipy.sparse.linalg.LinearOperator):
            return scipy.sparse.linalg.interface._ProductLinearOperator(self, x)
        elif scipy.sparse.compressed._cs_matrix in type(x).mro():
            return self.matmat(x)
        elif np.isscalar(x):
            return scipy.sparse.linalg.interface._ScaledLinearOperator(self, x)
        else:
            x = np.asarray(x)

            if x.ndim == 1 or x.ndim == 2 and x.shape[1] == 1:
                return self.matvec(x)
            elif x.ndim == 2:
                return self.matmat(x)
            else:
                raise ValueError('expected 1-d or 2-d array or matrix, got %r'
                                 % x)

    @property
    def shape(self):
        return self.A.shape

    @property
    def dtype(self):
        return self.A.dtype

    @property
    def ndim(self):
        return self.A.ndim
