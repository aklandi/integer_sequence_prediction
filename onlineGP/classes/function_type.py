"""
Authors: Amanda Landi (alandi@simons-rock.edu) and Joshua Landi (jlandi@ncsu.edu)
As part of a research project; results are forthcoming in a paper published in ___.

References:
(1) Segaran, Toby. 2007. Programming collective intelligence (First. ed.). O'Reilly. Chapter 11: Evolving Intelligence.
"""

class FunctionType:
    """
    A class that defines a function object with features long_name: str, short_name: str, function, and arity: int
    """
    def __init__(self, long_name: str, short_name: str, function, arity: int):
        self.long_name = long_name
        self.short_name = short_name
        self.function = function
        self.arity = arity