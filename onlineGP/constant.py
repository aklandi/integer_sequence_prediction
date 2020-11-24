"""
Authors: Amanda Landi (alandi@simons-rock.edu) and Joshua Landi (jlandi@ncsu.edu)
As part of a research project; results are forthcoming in a paper published in ___.

References:
(1) Segaran, Toby. 2007. Programming collective intelligence (First. ed.). O'Reilly. Chapter 11: Evolving Intelligence.
"""

import operator
import numpy as np
import typing

from classes.function_type import FunctionType
from functions import (protected_div, protected_exponent, protected_log,
                       protected_sqrt)

"""
A file that defines the global dictionary of functions to include in the search space for GP.
The functions include: addition, multiplication, division, negation, absolute value function,
sine, cosine, min, max, and protected division, natural log, natural exp, and square root.
"""

ADD = FunctionType('add', 'add', operator.add, 2)
MULTIPLY = FunctionType('multiply', 'mult', operator.mul, 2)
DIVISION = FunctionType('division', 'div', protected_div, 2)
NEGATE = FunctionType('negate', 'neg', operator.neg, 1)
ABSOLUTE_VALUE = FunctionType('absolute value', 'abs', operator.abs, 1)
LOG = FunctionType('log', 'log', protected_log, 1)
EXP = FunctionType('exp', 'exp', protected_exponent, 1)
SQRT = FunctionType('sqrt', 'sqrt', protected_sqrt, 1)
SIN = FunctionType('sine', 'sin', np.sin, 1)
COS = FunctionType('cosine', 'cos', np.cos, 1)
MIN = FunctionType('minimum', 'min', np.min, 1)
MAX = FunctionType('maximum', 'max', np.max, 1)


FUNCTION_SET: typing.Dict[str, FunctionType] = dict(
    [
        [ADD.long_name, ADD],
        [MULTIPLY.long_name, MULTIPLY],
        [DIVISION.long_name, DIVISION],
        [NEGATE.long_name, NEGATE],
        [ABSOLUTE_VALUE.long_name, ABSOLUTE_VALUE],
        [LOG.long_name, LOG],
        [EXP.long_name, EXP],
        [SQRT.long_name, SQRT],
        [SIN.long_name, SIN],
        [COS.long_name, COS],
        [MIN.long_name, MIN],
        [MAX.long_name, MAX]
    ]
)
