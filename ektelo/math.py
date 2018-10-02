import crlibm
import decimal
import numpy as np
import random

MAX_PREC = 1e-54

def clamp(x, B):
    if x > B:
        return B
    elif x < -B:
        return -B
    
    return x

def ulp(x):
    x2 = np.nextafter(x, x+1)
    d = x2-x
    log = np.log10(d)

    if log < 0:
        return np.power(10, np.floor(log))
    else:
        return np.power(10, np.ceil(log))
    return 

def select_ulp_random():
    target = 2**-53
    cumm = 0.0

    while cumm < target:
        r = random.random()
        cumm += ulp(r)

    return r

def Lambda(lm):
    return np.power(2, np.ceil(np.log2(lm)))
    
def Lambda_round(x, Lam):
    mult = x / Lam

    if mult - np.floor(mult) < 0.5:
        return Lam * np.floor(x / Lam)
    else:
        return Lam * np.ceil(x / Lam)


def get_B(lm, eps):
    mult = 2**-49

    return 1/float(lm) * (1 - lm*eps)

def snap_laplace(f, lm, B, sensitivity=1):
    f = f / sensitivity
    #B = 2*lm
    print('B', B)
    Lam = Lambda(lm)
    print('Lam', Lam)
    s = 2*(random.random() > 0.5) - 1
    print('s', s)
    r = select_ulp_random()
    print('r', r)
    print(clamp(f, B), s * lm * crlibm.log2_rd(r))
    return clamp(Lambda_round(clamp(f, B) + s * lm * crlibm.log2_rd(r), Lam), B)
