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

class ConstantNode(Node):
    """
    A class for nodes that represent constant values.  These nodes are child nodes, but do not have
    their own children.

    Constant nodes print their own values, and will evaluate to their own values given any input.
    They should not spawn any children, so they return themselves.

    """

    def __init__(self, function):
        self.function = function

    def __str__(self):
        return f"{self.function}"

    def evaluate(self, inputs: NDArray) -> float:
        return self.function

    def spawn_children(self, max_depth: int, random_instance: Random = None) -> 'ConstantNode':
        return self