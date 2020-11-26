"""
Authors: Amanda Landi (alandi@simons-rock.edu) and Joshua Landi (jlandi@ncsu.edu)
As part of a research project; results are forthcoming in a paper published in ___.

References:
(1) Segaran, Toby. 2007. Programming collective intelligence (First. ed.). O'Reilly. Chapter 11: Evolving Intelligence.
"""

from random import randint, random, sample, uniform
from constant import FUNCTION_SET
from classes.node_type_enum import NodeType
from classes.function_node import FunctionNode
from classes.constant_node import ConstantNode
from classes.variable_node import VariableNode
from classes.node import Node

class NodeBuilder:

    """
    a class that builds a node, either with type specified or randomized
    """

    # the largest constant a constant node can be (this is arbitrarily chosen)
    _max_constant = 10
    _sample_size = 1

    def __init__(self):
        pass

    def set_constant_size(self, size: int) -> 'NodeBuilder':
        self._max_constant = size
        return self

    def set_sample_size(self, size: int) -> 'NodeBuilder':
        self._sample_size = size
        return self

    def build(self, node: NodeType, probability_of_function: float, probability_of_variable: float, max_variable_count: int) -> Node:
        # if specified type is function, return random function node
        if node == NodeType.Function:
            f = FUNCTION_SET[ sample( list(FUNCTION_SET), self._sample_size)[0]]
            return FunctionNode(f.function, f.short_name, f.arity)
        # if specified type is constant, return random constant node
        elif node == NodeType.Constant:
            return ConstantNode( uniform(0, self._max_constant) )
        # if specified type is variable, return random variable node
        elif node == NodeType.Variable:
            return VariableNode(index = randint(0, max_variable_count - 1))
        # if type is not specified
        else:
            # check to make sure the node shouldn't be a function node
            if random() < probability_of_function:
                f = FUNCTION_SET[ sample( list(FUNCTION_SET), self._sample_size)[0]]
                return FunctionNode(f.function, f.short_name, f.arity)
            else:
                # if not a function node, check probability it is variable
                if random() < probability_of_variable:
                    return VariableNode(index = randint(0, max_variable_count- 1))
                # when those checks fail, it's a constant node
                else:
                    return ConstantNode( uniform(0, self._max_constant) )