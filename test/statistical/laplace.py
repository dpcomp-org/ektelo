from ektelo import private_laplace as pl
import math
import random
from scipy import stats
import unittest


class TestLaplace(unittest.TestCase):

    def setUp(self):
        self.threshold = 0.1

    def test_snapping(self):
        lm = 10

        xs = []
        ys = []
        for i in range(1000):
            r = 1e-52 + i
            xs.append(pl.snap_laplace(1, lm, r=r))
            ys.append(pl.snap_laplace(1.5, lm, r=r))

        D, pvalue = stats.ks_2samp(xs, ys)
        self.assertGreater(pvalue, self.threshold)
