"""
Authors: Amanda Landi (alandi@simons-rock.edu) and Joshua Landi (jlandi@ncsu.edu)
As part of a research project; results are forthcoming in a paper published in ___.

References:
(1) Segaran, Toby. 2007. Programming collective intelligence (First. ed.). O'Reilly. Chapter 11: Evolving Intelligence.
"""

from __future__ import annotations
from typing import List, Tuple
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from classes.node import Node
    from classes.function_node import FunctionNode
from classes.node_builder import NodeBuilder
from classes.node_type_enum import NodeType
from classes.population_builder import PopulationBuilder

class Random:
    """
    A class for random production of nodes, programs, and populations for GP.
    """

    def __init__(self, probability_of_function: float = 0.5, probability_of_variable: float = 0.78, max_variable_count: int = 2):
        self.probability_of_function = probability_of_function
        self.probability_of_variable = probability_of_variable
        self.max_variable_count = max_variable_count

    def node(self, node: NodeType) -> Node:
        return NodeBuilder().build(node, self.probability_of_function, self.probability_of_variable, self.max_variable_count)

    def program(self, max_depth: int) -> FunctionNode:
        node = self.node(NodeType.Function)
        children = node.spawn_children(max_depth, self)
        return children

    def population(self, population_size = 100, selected_programs: Tuple[List[Node], List[int]] = None) -> Tuple[List[Node], List[int]]:
        return PopulationBuilder().set_population_size(population_size).build(self, selected_programs)
