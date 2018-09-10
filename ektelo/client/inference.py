from __future__ import division
import numpy as np
import math
from ektelo import matrix
from scipy import linalg, optimize
from scipy.sparse.linalg import lsmr, lsqr
from scipy import sparse
import ektelo
from ektelo import util
from ektelo.operators import InferenceOperator

def nls_lbfgs_b(A, y, l1_reg=0.0, l2_reg=0.0, maxiter = 15000):
    """
    Solves the NNLS problem min || Ax - y || s.t. x >= 0 using gradient-based optimization
    :param A: numpy matrix, scipy sparse matrix, or scipy Linear Operator
    :param y: numpy vector
    """
    M = sparse.linalg.aslinearoperator(A)

    def loss_and_grad(x):
        diff = M.matvec(x) - y
        res = 0.5 * np.sum(diff ** 2)
        f = res + l1_reg*np.sum(x) + l2_reg*np.sum(x**2)
        grad = M.rmatvec(diff) + l1_reg + l2_reg*x

        return f, grad

    xinit = np.zeros(A.shape[1])
    bnds = [(0,None)]*A.shape[1]
    xest,_,info = optimize.lbfgsb.fmin_l_bfgs_b(loss_and_grad,
                                                x0=xinit,
                                                pgtol=1e-4,
                                                bounds=bnds,
                                                maxiter=maxiter,
                                                m=1)
    xest[xest < 0] = 0.0
    return xest, info

def eval_x(hatx, q):
    """evaluation of a query in the form of t"""
    # note(ryan): the need for float(.) was due to pathologies in sparse matrix behavior
    # I think we can safely delete this with EkteloMatrix
    return float(q.dot(hatx)) 

def multWeightsUpdate(hatx, Q, Q_est, updateRounds = 1):
    """ Multiplicative weights update, supporting multiple measurements and repeated update rounds
    hatx: starting estimate of database
    Q: list of query arrays representing measurements
    Q_est: list of corresponding answers to query
    updateRounds: number of times to repeat the update of _all_ provided queries
    """
    assert Q.shape[0]==len(Q_est)

    total = sum(hatx)

    # TODO: this update rule shouldn't require materializing Q
    Q = sparse.csr_matrix(Q.sparse_matrix())

    for i in range(updateRounds):
        for q_est, q in zip(Q_est, Q):
            error = q_est - eval_x(hatx,q)        # difference between query ans on current estimated data and the observed answer
            update_vector = np.exp( q.toarray() * error / (2.0 * total) ).flatten()  # note that numpy broadcasting is happening here
            hatx = hatx * update_vector
            hatx = hatx * total / sum(hatx)     # normalize

    return hatx


class ScalableInferenceOperator(InferenceOperator):

    def _apply_scales(self, Ms, ys, scale_factors):
        if scale_factors is None:
            if type(Ms) is list:
                return matrix.VStack(Ms), np.concatenate(ys)
            return Ms, ys
        assert type(Ms) == list and type(ys) == list
        assert len(Ms) > 0 and len(Ms) == len(ys) and len(ys) == len(scale_factors)

        A = matrix.VStack([M*(1.0/w) for M, w in zip(Ms, scale_factors)])
        print(Ms, ys, scale_factors,[y/w for y, w in zip(ys, scale_factors)])
        y = np.concatenate([y/w for y, w in zip(ys, scale_factors)])
        
        return A, y

class LeastSquares(ScalableInferenceOperator):

    def __init__(self, method='lsmr', l2_reg=0.0):
        super(LeastSquares, self).__init__()

        self.method = method
        self.l2_reg = l2_reg

    def infer(self, Ms, ys, scale_factors=None):
        ''' Either:
            1) Ms is a single M and ys is a single y 
               (scale_factors ignored) or
            2) Ms and ys are lists of M matrices and y vectors
               and scale_factors is a list of the same length.
        '''
        A, y = self._apply_scales(Ms, ys, scale_factors)

        if self.method == 'standard':
            assert self.l2_reg == 0, 'l2 reg not supported with method=standard'
            (x_est, _, rank, _) = linalg.lstsq(A.dense_matrix(), y, lapack_driver='gelsy')
        elif self.method == 'lsmr':
            print(A.shape, y.shape)
            res = lsmr(A, y, atol=0, btol=0, damp=self.l2_reg)
            x_est = res[0]
        elif self.method == 'lsqr':
            res = lsqr(A, y, atol=0, btol=0, damp=self.l2_reg)
            x_est = res[0]

        x_est = x_est.reshape(A.shape[1])  # reshape to match shape of x

        return x_est

class NonNegativeLeastSquares(ScalableInferenceOperator):
    '''
    Non negative least squares (nnls)
    Note: undefined behavior when system is under-constrained
    '''

    def __init__(self, lasso=None):
        '''
        :param lasso:
            None for no regularization
            True for regularization as determined by total estimate give by least squares
            positive number for regularization strength (xest will have sum approximately equal to this number)
        '''
        super(NonNegativeLeastSquares, self).__init__()

        self.lasso = lasso

    def infer(self, Ms, ys, scale_factors=None):
        ''' Either:
            1) Ms is a single M and ys is a single y 
               (scale_factors ignored) or
            2) Ms and ys are lists of M matrices and y vectors
               and scale_factors is a list of the same length.
        '''
        A, y = self._apply_scales(Ms, ys, scale_factors)

        x_est, info = nls_lbfgs_b(A, y)

        x_est = x_est.reshape(A.shape[1])  # reshape to match shape of x
        return x_est


class MultiplicativeWeights(ScalableInferenceOperator):
    '''
    Multiplicative weights update with multiple update rounds and optional history
    useHistory is no longer available inside the operator. To use history measurements,
    use M and ans with full history.
    '''

    def __init__(self, updateRounds=50):
        super(MultiplicativeWeights, self).__init__()
        self.updateRounds = updateRounds

    def infer(self, Ms, ys, x_est, scale_factors=None):
        ''' Either:
            1) Ms is a single M and ys is a single y 
               (scale_factors ignored) or
            2) Ms and ys are lists of M matrices and y vectors
               and scale_factors is a list of the same length.
        '''
        M, y = self._apply_scales(Ms, ys, scale_factors)

        """ mult_weights is an update method which works on the original domain"""
        assert x_est is not None, 'Multiplicative Weights update needs a starting xest, but there is none.'

        x_est = multWeightsUpdate(x_est, M, y, self.updateRounds)
        return x_est


class AHPThresholding(ScalableInferenceOperator):
    '''
    Special update operator for AHP thresholding step.
    This operator assumes that the previous one is a Laplace measurement of the Identity workload.
    The xest is updated by answers from the Identity workload after thresholding. 
    To calculate the threshold, the eps used for the measurement is assumed to be ratio*_eps_total
    '''

    def __init__(self, eta, ratio):
        super(AHPThresholding, self).__init__()
        self.eta = eta
        self.ratio = ratio

    def infer(self, Ms, ys, eps_par, scale_factors=None):
        ''' Either:
            1) Ms is a single M and ys is a single y 
               (scale_factors ignored) or
            2) Ms and ys are lists of M matrices and y vectors
               and scale_factors is a list of the same length.
        '''
        A, y = self._apply_scales(Ms, ys, scale_factors)

        eps = eps_par * self.ratio
        x_est = lsmr(A, y.flatten())[0]
        x_est = x_est.reshape(A.shape[1])
        n = len(x_est)
        cutoff = self.eta * math.log(n) / eps
        x_est = np.where(x_est <= cutoff, 0, x_est)

        return x_est
