"""
Authors: Amanda Landi (alandi@simons-rock.edu) and Joshua Landi (jlandi@ncsu.edu)
As part of a research project; results are forthcoming in a paper published in ___.

References:
(1) Segaran, Toby. 2007. Programming collective intelligence (First. ed.). O'Reilly. Chapter 11: Evolving Intelligence.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from classes.random import Random
from nptyping import NDArray
from classes.node import Node

class VariableNode(Node):
    """
    a class that defines the variable node object
    """

    def __init__(self, index):
        self.index = index

    def __str__(self):
        return f"X{self.index}"

    # inputs should be numpy array
    def evaluate(self, inputs: NDArray) -> int:

        # either inputs is n rows by m columns
        if len(inputs.shape) > 1:
            return inputs[:,self.index]
        # or it is a single row with those m columns
        else:
            return inputs[self.index]

    def spawn_children(self, max_depth: int, random_instance: Random = None) -> 'VariableNode':
        return self