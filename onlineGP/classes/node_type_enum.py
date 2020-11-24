"""
Authors: Amanda Landi (alandi@simons-rock.edu) and Joshua Landi (jlandi@ncsu.edu)
As part of a research project; results are forthcoming in a paper published in ___.

References:
(1) Segaran, Toby. 2007. Programming collective intelligence (First. ed.). O'Reilly. Chapter 11: Evolving Intelligence.
"""
from enum import Enum

class NodeType(Enum):
    """
    a class that determines the node type

    1: Function node
    2: Variable node
    3: Constant node
    4: Any of the node types

    """
    Function = 1,
    Variable = 2,
    Constant = 3,
    Any = 4