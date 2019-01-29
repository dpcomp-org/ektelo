from ektelo.client import selection
from ektelo.private import measurement
from ektelo.private_laplace import snap_laplace
from ektelo import util
from ektelo import workload
import numpy as np
from numpy import linalg as la

snap_v = np.vectorize(snap_laplace)
prng = np.random.RandomState(0)

# random data, workload, and selection matrix
eps = 0.1
X = np.random.uniform(0, 1, 100)
W = workload.RandomRange(None, X.shape[0], 25)
M = selection.Identity(X.shape).select()

# Laplace setup
sensitivity = measurement.Laplace.sensitivity_L1(M)
laplace_scale = util.old_div(sensitivity, float(eps))

# Error using regular Laplace
noise = prng.laplace(0.0, laplace_scale, M.shape[0])
x_hat_reg = M.dot(X) + noise
print('Regular Laplace error:', la.norm(X - x_hat_reg))

# Error using Laplace with snapping mechanism
x_hat_snap = snap_v(M.dot(X), laplace_scale)
print('Snapping Laplace error:', la.norm(X - x_hat_snap))
