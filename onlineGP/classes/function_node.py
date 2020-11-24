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
from random import choice
from classes.node import Node
from classes.function_type import FunctionType
from typing import List
from typing import NDArray
from classes.node_type_enum import NodeType
from classes.node import Node

class FunctionNode(FunctionType, Node):
    """
    A class for nodes that represent functions.  
    This is the only node type that can produce children.
    """
    children: List[Node] = []

    def __init__(self, function, name: str, arity: int):
        super().__init__(None, name, function, arity)

    def __str__(self):
        return f"{self.short_name}{() if self.arity == 1 else ({},{})}"

    def evaluate(self, inputs: NDArray[float]) -> float:
        
        if self.arity == 2:
            result = self.function(inputs[0], inputs[1])
        else:
            result = self.function(inputs)

        return result

    def spawn_children(self, max_depth: int, random_instance: Random) -> 'FunctionNode':
        
        while max_depth > 1:
            max_depth -= 1
            self.children = [random_instance.node(NodeType.Any) for _ in range(self.arity)]
            temp = [self.spawn_children(max_depth, random_instance) for _ in self.children]
            return self
        else:
            self.children = [random_instance.node(choice([NodeType.Constant, NodeType.Variable])) for _ in range(self.arity)]
            return self