from classes.random import Random
from classes.genetic_management import GeneticManagement
from classes.symbolic_regression import SymbolicRegression

import numpy as np


X= np.array([[1,2,3],[2,3,-1]]).T
y = np.array([[1,0,-1]]).T
pop_size = 2000
fit_threshold = 0.1
tourn_size = 5
p_mutate = 0.08
p_crossover = 0.9
n_gen = 10

# est_gp = SymbolicRegression(pop_size, tourn_size, n_gen, p_mutate, p_crossover, fit_threshold)
# est_gp.fit(indep_variable = X, truth = y, batch_size = 1)
# validate = est_gp.predict(X, y)
# print("Fidelity on training data: %f" % np.linalg.norm(validate['prediction(s)'] - y, ord = 1)/ np.linalg.norm(y, ord = 1))


