"""
Authors: Amanda Landi (alandi@simons-rock.edu) and Joshua Landi (jlandi@ncsu.edu)
As part of a research project; results are forthcoming in a paper published in ___.

References:
(1) Segaran, Toby. 2007. Programming collective intelligence (First. ed.). O'Reilly. Chapter 11: Evolving Intelligence.
"""

import numpy as np

"""
typical functions that need to be evaluated with care during GP given the random nature:
exp function, log, division, sqrt
"""

def protected_exponent(x):
    with np.errstate(invalid='ignore'):
       return np.where(np.abs(x) < 100, np.exp(x), 0.0)

def protected_log(x):
    with np.errstate(invalid='ignore', divide='ignore'):
        y = np.nan_to_num(np.log(x))

    return y

def protected_div(a,b):
    with np.errstate(divide='ignore', invalid ='ignore'):
        y = np.nan_to_num(a/b)

    return y

def protected_sqrt(x):
    with np.errstate(invalid='ignore', over='ignore'):
        y = np.nan_to_num(np.sqrt(x))

    return y
