"""
Authors: Amanda Landi (alandi@simons-rock.edu) and Joshua Landi (??@blackbaud.com)
As part of a research project; results are forthcoming in a paper published in ___.

References:
(1) Segaran, Toby. 2007. Programming collective intelligence (First. ed.). O'Reilly. Chapter 11: Evolving Intelligence.
"""

from typing import Dict
from classes.genetic_management import GeneticManagement
from classes.random import Random
import numpy as np

class SymbolicRegression:

    def __init__(self, pop_size = 100, tourn_size = 5, n_gen = 10, p_mutate = 0.2, p_crossover = 0.8, fit_threshold = 0.01):
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

    def fit(self, X, y, batch_size):

        n_samples, n_variables = X.shape
        random_instance = Random(0.5, 0.6, n_variables)
        selected_programs = None
        genetic_management = GeneticManagement(random_instance)
        
        # pre-trains the population incrementally on the data
        for i in range(0, n_samples, batch_size):
            population, max_depths = random_instance.population(self.pop_size, selected_programs)
            genetic_management.set_max_depths(max_depths)
            genetic_management.set_independent_variables(X[i:(i+batch_size),:])
            win, w_max_depths = genetic_management.tournament(population, y[i:(i+batch_size)], self.fit_threshold, self.tourn_size, self.p_mutate, self.p_crossover)
            selected_programs = [win, w_max_depths]

        # with the set of programs that are pre-trained, and with "all" data seen,
        # find the best one through multiple generations
        next_gen, max_depths = random_instance.population(self.pop_size, selected_programs)
        genetic_management.set_max_depths(max_depths)
        genetic_management.set_independent_variables(X)
        for g in range(self.n_gen):
            win, w_max_depths = genetic_management.tournament(next_gen, y, self.fit_threshold, self.tourn_size, self.p_mutate, self.p_crossover, train = 1)
            next_gen, next_gen_depths = genetic_management.evolve(winners = win, p_mutate = self.p_mutate, p_crossover = self.p_crossover, train = 1)
            if len(next_gen) < self.tourn_size:
                break

        if len(next_gen) > 0:
            best_fit = 2**10; best_fit_indx = 0
            for k in range(len(next_gen)):
                tree = next_gen[k]
                t = genetic_management.fitness(tree, y)
                if t < best_fit:
                    best_fit = t
                    best_fit_indx = k
            self.best_program = next_gen[best_fit_indx]
            v,s = genetic_management.evaluate_tree(self.best_program, X)
            self.best_program_string = s
        else:
            self.best_program = next_gen
            v,s = genetic_management.evaluate_tree(self.best_program, X)
            self.best_program_string = s

    def predict(self, X, y):
        n_samples, n_variables = X.shape
        random_instance = Random(0.5, 0.6, n_variables)
        genetic_management = GeneticManagement(random_instance)
        value = []; f = []
        
        for k in range(n_samples):
            v,s = genetic_management.evaluate_tree(self.best_program, X[k,:])
            value.append(v)
            f.append(genetic_management.fitness(self.best_program, y[k,:]))

        return {"prediction(s)": np.reshape(np.array(value), y.shape), "fitness": np.array(f)}
