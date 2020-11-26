"""
Authors: Amanda Landi (alandi@simons-rock.edu) and Joshua Landi (??@blackbaud.com)
As part of a research project; results are forthcoming in a paper published in ___.

References:
(1) Segaran, Toby. 2007. Programming collective intelligence (First. ed.). O'Reilly. Chapter 11: Evolving Intelligence.
"""

from typing import Dict, List, Tuple
from nptyping import NDArray
from classes.genetic_management import GeneticManagement
from classes.node import Node
from classes.random import Random
import numpy as np

from classes.tournament_configuration import TournamentConfiguration

class SymbolicRegression:

    genetic_management: GeneticManagement = None

    def __init__(self, pop_size: int = 100, tourn_size: int = 5, n_gen: int = 10, p_mutate: float = 0.2, p_crossover: float = 0.8, fit_threshold: float = 0.01):

        self.pop_size = pop_size
        self.tourn_size = tourn_size
        self.n_gen = n_gen
        self.p_mutate = p_mutate
        self.p_crossover = p_crossover
        self.fit_threshold = fit_threshold
        self.best_program = []
        self.best_program_string = " "


    @staticmethod
    def parse_from_file(file: Dict) -> 'SymbolicRegression':
        return SymbolicRegression(
            file.get('population_size'),
            file.get('tournament_size'),
            file.get('generation_count'),
            file.get('mutation_probability'),
            file.get('crossover_probability'),
            file.get('fit_threshold')
        )

    #
    # a function modeled after keras ML to fit the data to a particular program
    #
    # X: NDArray = an array containing the data such that each column represents a different variable and each
    #               row represents a different observation (size n_samples x n_variables)
    # y: NDArray = an array containing the true outcome for each observation (size n_samples x n_outcomes = 1)
    # batch_size: int = an integer determining how many observations at a time we use during training
    #
    # returns the best program from the population in model.best_program in the form of the center function node
    #           the string of the best program, print(model.best_program_string) will display the mathematical expression
    #
    def fit(self, X: NDArray, y: NDArray, batch_size: int):

        n_samples, n_variables = X.shape
        random_instance = Random(0.5, 0.6, n_variables, self.p_mutate, self.p_crossover)
        selected_programs: Tuple[List[Node], List[int]] = ([], [])
        self.genetic_management = GeneticManagement(random_instance)
        tournament_configuration = TournamentConfiguration(self.fit_threshold, self.tourn_size)

        # pre-trains the population incrementally on the data
        for i in range(0, n_samples, batch_size):
            population, max_depths = random_instance.population(self.pop_size, selected_programs)
            self.genetic_management.set_max_depths(max_depths)
            self.genetic_management.set_independent_variables(X[i:(i+batch_size),:])
            winners: List[Node] = []
            winner_max_depth: List[int] = []
            winners, winner_max_depth = self.genetic_management.tournament(population, y[i:(i+batch_size)], tournament_configuration)
            selected_programs = (winners, winner_max_depth)

        # with the set of programs that are pre-trained, and with "all" data seen,
        # find the best one through multiple generations
        next_generation: List[Node] = None
        max_depths: List[int] = None
        next_generation, max_depths = random_instance.population(self.pop_size, selected_programs)
        self.genetic_management.set_max_depths(max_depths)
        self.genetic_management.set_independent_variables(X)
        for _ in range(self.n_gen):
            winners: List[Node] = []
            winner_max_depth: List[int] = []
            tournament_configuration.pretrain = False
            winners, winner_max_depth = self.genetic_management.tournament(next_generation, y, tournament_configuration)
            next_generation, _ = self.genetic_management.evolve(winners, winner_max_depth, tournament_configuration.pretrain)

            if len(next_generation) < self.tourn_size:
                break

        # if evolution left more than one program, then we determine the best among them
        if len(next_generation) > 0:
            best_fit = 2**10; best_fit_indx = 0
            for k in range(len(next_generation)):
                tree = next_generation[k]
                t = self.genetic_management.fitness(tree, y)
                if t < best_fit:
                    best_fit = t
                    best_fit_indx = k
            self.best_program = next_generation[best_fit_indx]
            _,s = self.genetic_management.evaluate_tree(self.best_program, X)
            self.best_program_string = s
        else:
            self.best_program = next_generation
            _,s = self.genetic_management.evaluate_tree(self.best_program, X)
            self.best_program_string = s

    #
    # a function modeled after keras ML to predict output based on input using the model found in model.fit
    #
    # X: NDArray = an array containing the data such that each column represents a different variable and each
    #               row represents a different observation (size n_samples x n_variables)
    # y: NDArray = an array containing the true outcome for each observation (size n_samples x n_outcomes = 1)
    #
    # returns a dictionary containing the prediction yhat = f(X) and the fitness of this prediction to the truth, y
    #
    def predict(self, X: NDArray, y: NDArray):

        n_samples, n_variables = X.shape
        value = []; f = []

        for k in range(n_samples):
            v,s = self.genetic_management.evaluate_tree(self.best_program, X[k,:])
            value.append(v)
            self.genetic_management.set_independent_variables(X[k,:])
            f.append(self.genetic_management.fitness(self.best_program, y[k,:]))

        return {"prediction(s)": np.reshape(np.array(value), y.shape), "fitness": np.array(f)}
