from __future__ import division
import math
import numpy as np
from ektelo import util
from ektelo.operators import MeasurementOperator
from ektelo.private_laplace import snap_laplace

snap_v = np.vectorize(snap_laplace)


class Laplace(MeasurementOperator):

    def __init__(self, A, eps, private_laplace=True):
        self.A = A
        self.eps = eps
        self.private_laplace = private_laplace

    def measure(self, X, prng):
        sensitivity = self.sensitivity_L1(self.A)
        laplace_scale = util.old_div(sensitivity, float(self.eps))

        if self.private_laplace:
            x_hat = snap_v(self.A.dot(X), laplace_scale)
        else:
            noise = prng.laplace(0.0, laplace_scale, self.A.shape[0])
            x_hat = self.A.dot(X) + noise

        return x_hat

    @staticmethod
    def sensitivity_L1(A):
        """Return the L1 sensitivity of input matrix A: maximum L1 norm of the columns."""
        return float(np.abs(A).sum(axis=0).max()) # works efficiently for both numpy arrays and scipy matrices
