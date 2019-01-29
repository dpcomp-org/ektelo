import crlibm
import decimal
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
    exp = geom.rvs(0.5)
    sig = float('0.'+str(int(''.join(random.choices(('0','1'), k=52)), 2))) 

    return sig * 10**(exp)


def Lambda(lm):
    return np.power(2, np.ceil(np.log2(lm)))
    

def Lambda_round(x, Lam):
    mult = x / Lam

    if mult - np.floor(mult) < 0.5:
        return Lam * np.floor(x / Lam)
    else:
        return Lam * np.ceil(x / Lam)


def snap_laplace(f, lm, sensitivity=1, B=None, log_type='decimal', r=None):
    """Implementation of snapping mechanism for private Laplace random variates
       See: Mironov 2012

    Args:
        f (float): Actual value of query f over database D, i.e. f(D)
        lm (float): parameter lambda intended for the Laplace distribution
        sensitivity (float, optional): the maximum possible value 
            |f(D) - f(D')| for neighboring D and D'
        B (float, optional): tunable security parameter for snapping mechanism
        log_type ({'decimal', 'crlibm'}, optional): module to use for 
            logarithm with exact rounding
        r (float, optional): value in (0, 1) to use for random instead of 
            actually generating it pseudo-randomly

    Returns: 
        float: A sample from the private Laplace distribution
    """
    if B is None:
        B = 2e20 * lm

    if r is None:
        r = select_ulp_random()

    if log_type == 'decimal':
        context = decimal.Context(rounding=decimal.ROUND_HALF_EVEN)
        log = float(context.create_decimal_from_float(r).ln())
    elif log_type == 'crlibm':
        log = crlibm.log_rn(r)
    else:
        raise ValueError('unknown log_type: "%s"' % log_type)

    f = f / sensitivity
    Lam = Lambda(lm)
    s = 2*(random.random() > 0.5) - 1

    return clamp(Lambda_round(clamp(f, B) + s * lm * log, Lam), B)
