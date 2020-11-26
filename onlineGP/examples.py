"""
Authors: Amanda Landi (alandi@simons-rock.edu)
As part of a research project; results are forthcoming in a paper published in ___.

References:
(1) Segaran, Toby. 2007. Programming collective intelligence (First. ed.). O'Reilly. Chapter 11: Evolving Intelligence.
"""

import numpy as np

from classes.random import Random
from classes.genetic_management import GeneticManagement
from classes.symbolic_regression import SymbolicRegression

# a simple example script
X = np.array([[1,2,3],[2,3,-1]]).T
y = np.array([[1,0,-1]]).T
pop_size = 2000
fit_threshold = 0.1
tourn_size = 5
p_mutate = 0.08
p_crossover = 0.9
n_gen = 10

est_gp = SymbolicRegression(pop_size, tourn_size, n_gen, p_mutate, p_crossover, fit_threshold)
est_gp.fit(X, y, batch_size = 1)
validate = est_gp.predict(X, y)
print("Predictions: " + str(validate['prediction(s)']))
print("Fidelity on training data: ", np.linalg.norm(validate['prediction(s)'] - y, ord = 1)/ np.linalg.norm(y, ord = 1))

# using the integer sequence data

# pre-precess the data:
DF = pd.read_csv("./train.csv", delimiter = ",")
data = []
for rows in DF['Sequence']:
    fix = rows.split(",")
    fix = [int(elem) for elem in fix]
    data.append(fix)
# remove lists with length less than 20
for rows in data:
    if len(rows) < 20:
        data.remove(rows)

j = 2
seqn = data[j]
train_size = int(np.ceil(len(seqn)*0.70)); test_size = len(seqn) - train_size
x = [i for i in range(len(seqn))]

# can change the time window size for each observation
time_step = 1
trainX, trainY = [], []
for i in range(0, train_size - time_step):
    trainX.append(x[i:(i+time_step)])
    trainY.append(seqn[i+time_step])

trainX = np.array(trainX)
trainY = np.reshape(np.array(trainY), (len(trainY),1))

testX, testY = [], []
for i in range( (train_size - time_step), (train_size - time_step)+(test_size - time_step)):
    testX.append(x[i:(i+time_step)])
    testY.append(seqn[i+time_step])
testX = np.array(testX)
testY = np.reshape(np.array(testY), (len(testY),1))

pop_size = 2000
fit_threshold = 0.1
tourn_size = 5
p_mutate = 0.08
p_crossover = 0.9
n_gen = 10

est_gp = SymbolicRegression(pop_size, tourn_size, n_gen, p_mutate, p_crossover, fit_threshold)
est_gp.fit(X = trainX, y = trainY, batch_size = 1)
validate = est_gp.predict(trainX, trainY)
accuracy = est_gp.predict(testX, testY)

print("Predictions for test set are: \n" + str(accuracy['prediction(s)']))
