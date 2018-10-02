import crlibm
import numpy as np
import random
from scipy.stats import geom


def clamp(x, B):
    if x > B:
        return B
    elif x < -B:
        return -B
    
    return x


def select_ulp_random():
    exp = geom.rvs(0.5) - 1
    sig = float('0.'+str(int(''.join(random.choices(('0','1'), k=52)), 2))) 

    return sig * 10**(-exp)


def Lambda(lm):
    return np.power(2, np.ceil(np.log2(lm)))
    

def Lambda_round(x, Lam):
    mult = x / Lam

    if mult - np.floor(mult) < 0.5:
        return Lam * np.floor(x / Lam)
    else:
        return Lam * np.ceil(x / Lam)


def snap_laplace(f, lm, sensitivity=1, B=None):
    if B is None:
        B = 2e20 * lm

    f = f / sensitivity
    Lam = Lambda(lm)
    s = 2*(random.random() > 0.5) - 1
    r = select_ulp_random()

    return clamp(Lambda_round(clamp(f, B) + s * lm * crlibm.log2_rd(r), Lam), B)
