"""
Authors: Amanda Landi (alandi@simons-rock.edu) and Joshua Landi (jlandi@ncsu.edu)
As part of a research project; results are forthcoming in a paper published in ___.

References:
(1) Segaran, Toby. 2007. Programming collective intelligence (First. ed.). O'Reilly. Chapter 11: Evolving Intelligence.
"""
from copy import deepcopy
from random import choice, random, sample
from typing import List, Tuple
from nptyping import NDArray
import numpy as np
from classes.constant_node import ConstantNode
from classes.function_node import FunctionNode
from classes.node import Node
from classes.node_type_enum import NodeType
from classes.random import Random
from classes.tournament_configuration import TournamentConfiguration
from sklearn.metrics import mean_absolute_error

from classes.variable_node import VariableNode

"""
a class to manage the genetic part of GP
"""

class GeneticManagement:

    _max_depths: List[int]
    _independent_variables: NDArray
    _random: Random

    def __init__(self, random: Random) -> None:
        self._random = random
        pass

    def set_max_depths(self, value: List[int]) -> 'GeneticManagement':
        self._max_depths = value
        return self

    def set_independent_variables(self, value: NDArray) -> 'GeneticManagement':
        self._independent_variables = value
        return self

    def evaluate_tree(self, center_node: FunctionNode, inputs: NDArray) -> Tuple[float, str]:

        values: List[float] = []
        function_string = [center_node.short_name , "(", "\t"]
        
        for index, child in enumerate(center_node.children):
            if index == 1:
                function_string.append(", ")
            
            if isinstance(child, FunctionNode):
                value, _function_string = self.evaluate_tree(child, inputs)
                values.append(np.array(value))
                function_string.extend([_function_string , "\t"])
            else:
                values.append(child.evaluate(inputs))
                function_string.extend([child.__str__() , "\t"])

        if center_node.arity == 1:
            value = center_node.evaluate([values[0]])
        else:
            value = center_node.evaluate([values[0],values[1]])

        function_string.append(")")

        return (np.nan_to_num(value), ''.join(function_string))

    def fitness(self, random_program: FunctionNode, y: NDArray) -> float:

        value, _ = self.evaluate_tree(random_program, self._independent_variables)
        
        fit: float or NDArray = 0.0
        if np.isscalar(value):
            temp = value*np.ones(y.shape)
            fit = mean_absolute_error(temp, y)
        else:
            fit = mean_absolute_error(np.reshape(value, y.shape), y)

        return fit

    def tournament(
        self,
        population: List[FunctionNode],
        y: NDArray,
        config: TournamentConfiguration
    ) -> Tuple[List, List]:
        
        winners = []; fit = []; maxdep = []
        random_index = sample([i for i in range(len(population))], len(population))

        for i in range(len(population)):
            f = self.fitness(population[i], y)
            fit.append(f)

        for i in range(0, len(population), config.tournament_size):

            idx = random_index[i:(i+config.tournament_size)]
            f = np.argsort([fit[k] for k in idx])
            if f[0] < config.fit_threshold:
                winners.append(population[idx[f[0]]])
                maxdep.append(self._max_depths[idx[f[0]]])
            else:
                win1 = population[idx[f[0]]]
                win2 = population[idx[f[1]]]
                w1_depth = self._max_depths[idx[f[0]]]
                w2_depth = self._max_depths[idx[f[1]]]
                winners.extend([win1, win2])
                maxdep.extend([w1_depth, w2_depth])

                if config.train is False:
                    child, c_max_d = self.evolve(
                        [win1, win2],
                        [w1_depth, w2_depth],
                        config.train
                    )
                    winners.append(child[0])
                    maxdep.append(c_max_d[0])

        return (winners, maxdep)

    def mutate(self, program: Node, max_depth: int) -> Node:

        isFunctionNode = isinstance(program, FunctionNode)
        if random() < self._random.probability_of_mutation:
            if isFunctionNode:
                new_center_node: FunctionNode = self._random.node(NodeType.Function)
                if new_center_node.arity == program.arity:
                    new_center_node.children = deepcopy(program.children)
                elif new_center_node.arity < program.arity:
                    new_center_node.children = deepcopy([choice(program.children)])
                else:
                    max_depth -= 1
                    new_child = self._random.node(NodeType.Any).spawn_children(max_depth, self._random)
                    old_child = deepcopy(program.children[0])
                    new_center_node.children = [old_child, new_child]
                return new_center_node
            else:
                new_center_node = self._random.node(choice([NodeType.Variable, NodeType.Constant]))
                return new_center_node
        else:
            if isFunctionNode:
                new_children_node = [self.mutate(k, max_depth) for k in program.children]
                program.children = new_children_node
            return program

    def crossover(self, program1: FunctionNode, program2: FunctionNode) -> FunctionNode:

        if random() < self._random.probability_of_crossover:

            program1 = deepcopy(program2)
            return program1

        else:

            if isinstance(program1, FunctionNode) and isinstance(program2, FunctionNode):
                c1 = program1.children
                c2 = program2.children
                change_in_children = [self.crossover(k, choice(c2)) for k in c1]
                program1.children = change_in_children

            return program1

    def evolve(self, winners, max_depths = None, train: bool = False) -> Tuple[List[Node], List[int]]:

        new_gen = []
        new_gen_depths = []

        if train is False:

            for i in range(len(winners)-1):
                max_d = min(max_depths[i], max_depths[i+1])
                crossover = self.crossover(winners[i], winners[i+1])
                mutation = self.mutate(crossover, max_d)
                new_gen.append(mutation)
                new_gen_depths.append(max(max_depths[i], max_depths[i+1]))
        else:

            for i in range(len(winners)-1):
                new_gen.append(self.mutate(self.crossover(winners[i], winners[i+1]), max_depth = min(self._max_depths[i], self._max_depths[i+1])))
                new_gen_depths.append(max(self._max_depths[i], self._max_depths[i+1]))

        return (new_gen, new_gen_depths)
