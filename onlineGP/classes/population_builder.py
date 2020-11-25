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
from classes.node import Node

from random import randint
from typing import List, Tuple

class PopulationBuilder:
    """
    A class for an object that creates a population of random programs
    """

    _min_depth = 4
    _max_depth = 10
    _population_size = 100

    def __init__(self):
        pass

    def set_min_depth(self, size: int) -> 'PopulationBuilder':
        self._min_depth = size
        return self

    def set_max_depth(self, size: int) -> 'PopulationBuilder':
        self._max_depth = size
        return self

    def set_population_size(self, size: int) -> 'PopulationBuilder':
        self._population_size = size
        return self

    def build(self, random_instance: Random, selected_programs: Tuple[List[Node], List[int]] = None) -> Tuple[List[Node], List[int]]:
        if selected_programs is None:
            random_max_depths = [randint(self._min_depth, self._max_depth) for _ in range(self._population_size)]
            POP =  [random_instance.program(random_max_depths[k])for k in range(self._population_size)]
        else:
            programs, program_depth = selected_programs
            selected_programs_length = len(programs)
            random_max_depths: List[int] = [randint(self._min_depth, self._max_depth) for _ in range(self._population_size-selected_programs_length)]
            random_max_depths.extend(program_depth)
            POP: List[Node] = [random_instance.program(random_max_depths[k]) for k in range(self._population_size-selected_programs_length)]
            POP.extend(programs)

        return (POP, random_max_depths)