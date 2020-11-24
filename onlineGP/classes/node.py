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
import abc
from abc import ABCMeta
from typing import NDArray

class Node(metaclass=ABCMeta):
    """
    a parent class for node object type;
    each node type has a __str__ function that prints its expression,
    an evaluate function that produces a value at that node given some input,
    and a spawn_children function that produces children nodes
    """

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    def evaluate(self, inputs: NDArray[int]) -> int:
        pass

    @abc.abstractmethod
    def spawn_children(self, max_depth: int, random_instance: Random = None) -> 'Node':
        pass
