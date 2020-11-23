import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, random, pickle
import keras as krs
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, LSTM, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from htm.bindings.encoders import RDSE, RDSE_Parameters
from htm.bindings.sdr import SDR
from htm.algorithms import SpatialPooler as SP
from htm.algorithms import TemporalMemory as TM
from htm.algorithms import Predictor
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

import genetic as gp
from genetic import function_node, constant_node, variable_node, SymbolicRegression

----------------------------------------
# change to directory with the data set
os.chdir('.\\Python Scripts')
# load the csv file as a pandas dataframe
# info: maximum length of sequence is 348, and minimim length is 1
#       on average, the length of a sequence is approximately 41,
#       total number of sequences is 113,845
DF = pd.read_csv("./integer_seqn/train.csv", delimiter = ",")

# since the sequences are represented as a string in column 2
# of the pandas dataframe, we convert that column into a list
# of lists
data = []
for rows in DF['Sequence']:
    fix = rows.split(",")
    fix = [int(elem) for elem in fix]
    data.append(fix)

# remove lists with length less than 20 so each sequence has
# enough terms for "good" training
# need to run this more than once for some reason
for rows in data:
    if len(rows) < 20:
        data.remove(rows)

# can create a randomized list of sequences
# and save for all methods
indx = [i for i in range(len(data))]
rand_indx = random.sample(indx, 100)
with open('rand_indx.txt', 'wb') as f:
    pickle.dump(rand_indx, f)

# to load again:
# with open('rand_indx.txt', 'rb') as f:
#    rand_indx = pickle.load(f)
----------------------------------------
#
# Dynamic ANN
#
# variables that can change: train_size, time_step, number of hidden nodes, number of hidden layers, activation function, learning rate, optimizing algorithm, and number of epochs.

# create a list to collect the deviation between true final element and predicted final element per sequence in D
compare_ragniklein = []
for j in rand_indx:

    # working with one sequence at a time
    # though this process might be paralyzed for faster
    # run time
    D = data[j]
    D = np.reshape(D, (len(D),1))
    D = D.astype('float64')

    # Ragni and Klein project data between -1 and 1
    scaler = MinMaxScaler((-1,1))
    X_train_scaled = scaler.fit_transform(D)

    # split each sequence into a training and a test set
    # the test set is used to validate the model
    # use 70% pretty consistently because it yields better
    # results on average
    train_size = int(len(X_train_scaled)*(0.70))
    test_size = len(X_train_scaled) - train_size
    train, test = X_train_scaled[0:train_size], X_train_scaled[train_size:len(X_train_scaled)]

    # create a data set so that the first column is
    # the first i:(i+time_step) elements in a sequence,
    # and the second column the next (i+time_step)
    # element in the sequence
    time_step = 3

    trainX, trainY = [], []
    for i in range(0,(len(train)-time_step)):

        trainX.append(train[i:(i+time_step),0])
        trainY.append(train[i+time_step, 0])
    # cast as array for later functions
    trainX = np.array(trainX)
    trainY = np.array(trainY)

    testX, testY = [], []
    for i in range(0,(len(test)-time_step)):

        testX.append(test[i:(i+time_step),0])
        testY.append(test[i+time_step,0])

    testX = np.array(testX)
    testY = np.array(testY)

    # authors have better results with one hidden layer with 4 nodes
    # activation is hyperbolic tangent
    model = Sequential()
    model.add(Dense(4, input_dim = time_step, activation = 'tanh'))
    model.add(Dense(1, activation = 'tanh'))
    # authors used lr = 0.375, but it was too large for this particular
    # architecture (not clear what function/algorithm authors use)
    optim = krs.optimizers.Adam(lr = 0.1)
    # loss is not specified in paper, MSE is common
    model.compile(loss='mse', optimizer = optim)
    # 10 epochs seems sufficient in literature
    # batch_size = 1 reads one sample at a time, so this mimics on-line learning
    model.fit(trainX, trainY, epochs = 10, batch_size = 1, verbose = 0)

    # validation = model.evaluate(trainX,trainY)
    # accuracy = model.evaluate(testX,testY)
    # print("Accuracy for %d: %f"% (j, accuracy))

    # make predictions
    trainpredictions = model.predict(trainX)
    testpredictions = model.predict(testX)
    # unscale the data
    projected_trainpredictions = scaler.inverse_transform(trainpredictions)
    projected_testpredictions = scaler.inverse_transform(testpredictions)
    original_trainY = scaler.inverse_transform(np.reshape(trainY,(trainY.shape[0],1)))
    original_testY = scaler.inverse_transform(np.reshape(testY,(testY.shape[0],1)))

    # compare prediction deviation
    fidelity_train = np.linalg.norm(original_trainY - projected_trainpredictions, ord = 1)/np.linalg.norm(original_trainY, ord = 1)
    fidelity_test = np.linalg.norm(original_testY - projected_testpredictions, ord = 1)/np.linalg.norm(original_testY, ord = 1)
    compare_ragniklein.append([fidelity_train, fidelity_test])

----------------------------------------
#
# LSTM
#
# variables that change: train_size, time_step, number of LSTM nodes, number of hidden layers, dropout rate, activation function, learning rate, optimizing algorithm, and number of epochs.

# create a list to collect the deviation between true final element and predicted final element per sequence in D
compare_LSTM = []
# using the same rand_indx as before, except in specific cases!
for j in rand_indx:

    # same set-up as before
    D = data[j]
    D = np.reshape(D, (len(D),1))
    D = D.astype('float64')
    scaler = MinMaxScaler((-1,1))
    X_train_scaled = scaler.fit_transform(D)
    train_size = int(len(X_train_scaled)*(0.70))
    test_size = len(X_train_scaled) - train_size
    train, test = X_train_scaled[0:train_size], X_train_scaled[train_size:len(X_train_scaled)]

    time_step = 3
    trainX, trainY = [], []
    for i in range(0,(len(train)-time_step)):

        trainX.append(train[i:(i+time_step),0])
        trainY.append(train[i+time_step, 0])

    trainX = np.array(trainX)
    trainY = np.array(trainY)

    testX, testY = [], []
    for i in range(0,(len(test)-time_step)):

        testX.append(test[i:(i+time_step),0])
        testY.append(test[i+time_step,0])

    testX = np.array(testX)
    testY = np.array(testY)

    # need to reshape the data so that model.fit will accept the
    # LSTM input data has dimensions [num_samples,time_step, num_features]
    trainX = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))
    testX = np.reshape(testX, (testX.shape[0],testX.shape[1],1))

    # build RNN-LSTM for a sequence
    # we stick to one hidden layer with 4 nodes and tanh activation, Adam with lr = 0.1, and MSE
    # add a dropout layer, common for LSTM to reduce extraneous weights
    model = Sequential()
    model.add(LSTM(units = 4, input_shape = (time_step,1), activation = 'tanh'))
    model.add(Dropout(0.10))
    model.add(Dense(1, activation = 'tanh'))
    optim = krs.optimizers.Adam(lr = 0.1)
    model.compile(loss="mean_squared_error", optimizer = optim)

    # train the model, same parameters as before
    model.fit(trainX, trainY, epochs = 10, batch_size = 1, verbose = 0)
    # validation = model.evaluate(trainX,trainY)
    # accuracy = model.evaluate(testX,testY)

    # predict using the model (on test and train sets)
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # "unscale" the data
    trainPredict = scaler.inverse_transform(trainPredict)
    D_trainY = scaler.inverse_transform(np.reshape(trainY,(len(trainY),1)))
    testPredict = scaler.inverse_transform(testPredict)
    D_testY = scaler.inverse_transform(np.reshape(testY, (len(testY),1)))

    # compare prediction deviation
    fidelity_train = np.linalg.norm(D_trainY - trainPredict, ord = 1)/np.linalg.norm(D_trainY, ord = 1)
    fidelity_test = np.linalg.norm(D_testY - testPredict, ord = 1)/np.linalg.norm(D_testY, ord = 1)
    compare_LSTM.append([fidelity_train, fidelity_test])

----------------------------------------
#
# CONV+LSTM
#
# There are many more parameters to be changed for conv+lstm, so they're set  ahead of time.
time_step = 3
train_percent = 0.70
n_features = 1
n_outputs = 1
filter_size = 1
batch = 1
# since we have an extra layer for the convolutions, we set it to be 8, and then
# have a second hidden layer with 4 nodes
num_CONV_nodes = 8
num_LSTM_nodes = 4
learning_rate = 0.1
inputshape = (time_step, n_features)

compare_CONV_LSTM = []
for j in rand_indx:

    D = data[j]
    D = np.reshape(D, (len(D),1))
    D = D.astype('float64')
    scaler = MinMaxScaler((-1,1))
    X_train_scaled = scaler.fit_transform(D)
    n = len(X_train_scaled)
    train_size = int(np.ceil(n*train_percent))
    train = np.reshape(X_train_scaled[0:(train_size)], (train_size, 1))
    trainX, trainY = [], []
    for i in range(0,(len(train)-time_step)):

        trainX.append(train[i:(i+time_step),0])
        trainY.append(train[i+time_step, 0])

    n_samples = len(trainX)
    # reshape so that conv layer will accept data; conv1d
    # expects data size (n_samples, time_steps, num_features)
    trainX = np.reshape(np.array(trainX), (n_samples,time_step, n_features))
    trainY = np.reshape(np.array(trainY), (n_samples,1,n_features))

    test = np.reshape(X_train_scaled[(train_size):n], (n - train_size, 1))
    testX, testY = [], []
    for i in range(0,(len(test)-time_step)):

        testX.append(test[i:(i+time_step),0])
        testY.append(test[i+time_step,0])
    # again, we need to reshape so when testing the trained model, all data
    # is of the expected size
    testX = np.reshape(np.array(testX), (len(testX),time_step, n_features))
    testY = np.reshape(np.array(testY), (len(testY),1,n_features))

    # build the model: first a convolution layer to extract more complex features
    # then input to a LSTM layer to learn sequence features
    model = Sequential()
    # the convolution layer
    model.add(Conv1D(num_CONV_nodes, filter_size, activation = 'tanh', padding = 'valid', input_shape = inputshape))
    model.add(MaxPooling1D(pool_size = 1))
    model.add(Dropout(0.1))
    # the LSTM layer
    model.add(LSTM(num_LSTM_nodes, activation = 'tanh'))
    model.add(Dropout(0.1))
    # followed by a common dense layer
    model.add(Dense(1, activation = 'tanh'))
    # same minimization method and learning rate
    optim = krs.optimizers.Adam(lr = learning_rate)
    # compile the model using MSE as the cost function
    model.compile(loss = 'mse', optimizer = optim)
    # train the model with 10 epochs, batch size is 1, using the training data
    model.fit(trainX, trainY, epochs = 10, batch_size = batch, verbose = 0)
    # validation = model.evaluate(trainX, trainY)
    # accuracy = model.evaluate(testX, testY)
    # print("Accuracy for %d: %f"% (j, accuracy))

    # get predictions
    trainPredict = model.predict(trainX).reshape(trainY.shape[0], trainY.shape[1])
    testPredict = model.predict(testX).reshape(testY.shape[0],testY.shape[1])
    # reshape original data for comparison
    trainY_reshaped = trainY.reshape(trainY.shape[0], trainY.shape[1])
    testY_reshaped = testY.reshape(testY.shape[0], testY.shape[1])
    #un-scale the data
    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)
    trainY_reshaped = scaler.inverse_transform(trainY_reshaped)
    testY_reshaped = scaler.inverse_transform(testY_reshaped)

    # compare prediction deviation
    fidelity_train = np.linalg.norm(trainY_reshaped - trainPredict, ord = 1)/np.linalg.norm(trainY_reshaped, ord = 1)
    fidelity_test = np.linalg.norm(testY_reshaped - testPredict, ord = 1)/np.linalg.norm(testY_reshaped, ord = 1)

    compare_CONV_LSTM.append([fidelity_train, fidelity_test])

----------------------------------------
#
# HTM
#
# there are so many parameters to modify below
compare_htm = []
for next_seqn in rand_indx:

    seqn = data[next_seqn]
    seqn = np.reshape(seqn, (len(seqn),1))
    seqn = seqn.astype('float64')

    # scaling must be between 0 and 1 because the line
    # "predict.learn(count, tm_actCells, label)" throws an error
    # when the label is a negative number
    scaler = MinMaxScaler((0,1))
    X_train_scaled = scaler.fit_transform(seqn)
    train_size = int(np.ceil(len(seqn)*0.70))
    train_set = X_train_scaled[0:train_size]

    # encode the integer sequence using the pre-built RDSE in htm.core,
    # it apparently uses a hash function
    params = RDSE_Parameters()
    # sparsity is actually recommended to be around 2%
    params.sparsity = 0.10
    # this radius will change depending on the values of the sequence
    # since the integers are scaled to be between 0 and 1, 0.1 seemed like
    # a good radius, but could be too large
    params.radius = 0.1
    # professionals from HTMForum recommended a larger encoder
    params.size = 1000
    rdseEncoder = RDSE(params)

    # set up the spatial pooler
    # if your encoded numbers are already sparse,
    # the SP isn't really necessary; it is used when the encoder
    # does not produce a sparse representation (which happens according
    # to resources)
    sp = SP(inputDimensions  = (rdseEncoder.size,),
        columnDimensions = (1000,),
        localAreaDensity = 0.1,
        globalInhibition = True,
        synPermActiveInc   = 0.15,
        synPermInactiveDec = 0.01,
        stimulusThreshold = 1,
    )

    # setting up the temporal memory architecture
    tm = TM(columnDimensions = (sp.getColumnDimensions()[0],),
        # number of cells in a column
        cellsPerColumn=10,
        # the initial level of permanence for a connection
        initialPermanence=0.5,
        # the level of permanence needed for a connection
        # to actually be considered connected
        # this must be permanenceIncrement away from initialPermanence
        connectedPermanence=0.6,
        # the number of potential active connections needed
        # for a segment to be eligible for learning
        minThreshold=1,
        # maximum number of synapses allowed per segment per
        # learning cycle
        maxNewSynapseCount=40,
        # how much a permanence per segment is increased
        # during each learning cycle
        permanenceIncrement=0.15,
        # how much a permanence per segment is decreased
        # during each learning cycle
        permanenceDecrement=0.01,
        # number of active connection needed
        # for a segment to be considered active
        # this may need to be modified if tm.getPredictiveCells() is
        # producing 0's
        activationThreshold=1,
        predictedSegmentDecrement=0.001,
        # maximum connections allowed per cell
        maxSegmentsPerCell = 1,
        # maximum active connections allowed per cell
        maxSynapsesPerSegment = 1
    )

    predict = Predictor([1])
    # training happens here
    for epoch in range(10):

        # if run without the reset, eventually get the following error
        # RuntimeError: Exception: SDRClassifier.cpp(228)
        #   message: CHECK FAILED: "recordNum >= lastRecordNum"
        #   The record number must increase monotonically.
        predict.reset()
        # need this for predictions later
        pred_actCells = []
        # anamoly_forall = []
        for count, i in enumerate(range(len(train_set))):

            # encode the current integer
            rdseSDR = rdseEncoder.encode(train_set[i])
            # create an SDR for SP output
            activeColumns = SDR( dimensions = tm.getColumnDimensions()[0] )

            # convert the SDR to SP
            # this is optional if the output from the encoder is
            # already a sparse binary representation
            # otherwise this step may be skipped as seen in
            # tutorials online
            sp.compute(rdseSDR, True, activeColumns)
            tm.compute(activeColumns, learn=True)
            tm.activateDendrites(True)
            tm_actCells = tm.getActiveCells()
            pred_actCells.append(tm_actCells)
            # anamoly_forall.append(tm.anomaly)

            label = int(train_set[i])
            # this is a neural network being trained to
            # know which SDR corresponds to which integer
            predict.learn(count, tm_actCells, label)

    predict.reset()
    next_elem = []
    # need to interpret the predictions made by TM (these predictions
    # are in sparse representations and the brain does not need this added
    # step because it automatically recognizes what the SDRs represent; however,
    # the SDRs from TM are not in our brain, so we needed an added step to inter-
    # pret)
    for elem in pred_actCells:

        pdf = predict.infer(elem)
        next_elem.append( np.argmax(pdf[1]) )

    s = np.array(seqn[:train_size])
    n_e = scaler.inverse_transform(np.reshape(np.array(next_elem),(len(next_elem),1)))
    # calculate validation
    fidelity_train = np.linalg.norm(s - n_e, ord = 1)/ np.linalg.norm(s, ord = 1)

    # now we take the model from above and make predictions for the test set
    for items in seqn[train_size:]:
            rdseSDR = rdseEncoder.encode(items)
            activeColumns = SDR( dimensions = tm.getColumnDimensions()[0] )
            sp.compute(rdseSDR, True, activeColumns)
            # when learn = False, the TM is not learning the pattern
            # but making predictions based on already-learned patterns
            tm.compute(activeColumns, learn=False)
            tm.activateDendrites(True)
            tm_actCells = tm.getActiveCells()
            pred_actCells.append(tm_actCells)

    predict.reset()
    remaining_next_elem = []
    for elem in pred_actCells[train_size:]:

        pdf = predict.infer(elem)
        remaining_next_elem.append( np.argmax(pdf[1]) )

    s = np.array(seqn[train_size:])
    n_e = scaler.inverse_transform(np.reshape(np.array(remaining_next_elem),(len(remaining_next_elem),1)))
    # calculate accuracy
    fidelity_test = np.linalg.norm(s - n_e, ord = 1)/ np.linalg.norm(s, ord = 1)

    compare_htm.append([fidelity_train, fidelity_test])

----------------------------------------
#
# Genetic Programming with gplearn
#

# to add an exponential function to the function set,
# I needed to create one from "scratch", and to prevent overflow
# errors from the gplearn method a "protected" exponential function
# is needed
def _protected_exponent(x):
    with np.errstate(over='ignore'):
       return np.where(np.abs(x) < 100, np.exp(x), 0.)

e_function = make_function(function = _protected_exponent, name = 'e_func', arity = 1)
# the function set must be specified, otherwise the gplearn method
# will only look at ['add', 'sub', 'mul', 'div']
f_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min', 'sin', 'cos', e_function]

time_step = 3
fidelity = []
for j in rand_indx:

    seqn = data[j]
    train_size = int(np.ceil(len(seqn)*0.70))
    x = np.reshape(np.array([i for i in range(len(seqn))]), (len(seqn), 1))

    # population_size = initial number of random programs to examine
    # generations = max number of times programs will continue on to earn a chance at becoming fit
    # tournament_size = number of programs that will compete
    # stopping_criteria = if a program's fitness is less than or equal to this value,
    #                       then it wins and will not continue on in generation
    # max_samples = percentage of samples to be used in learning
    # parsimony_coefficient = punishment for length
    est_gp = SymbolicRegressor(population_size = 5000, generations = 20, tournament_size = 10, stopping_criteria = 0.05, function_set = f_set, p_crossover = 0.7, p_subtree_mutation = 0.1, p_hoist_mutation = 0.05, p_point_mutation = 0.1, max_samples = 0.9, verbose = 1, parsimony_coefficient = 0.01, random_state = 0)

    # do sliding window, width time_step, to mimic on-line learning after the 3rd
    # element is received
    for k in range(0, (train_size-time_step-1)):
        est_gp.fit(x[k:(k+time_step)], seqn[k:(k+time_step)])

    #validate on the training set
    fidelity_train = np.linalg.norm(est_gp.predict(x[:train_size]) - seqn[:train_size], ord = 1)/np.linalg.norm(seqn[:train_size], ord = 1)
    # determine accuracy with the test set
    fidelity_test = np.linalg.norm(est_gp.predict(x[train_size:]) - seqn[train_size:], ord = 1)/np.linalg.norm(seqn[train_size:], ord = 1)

    fidelity.append([fidelity_train, fidelity_test])

---------------------
#
# on-line Genetic Programming with my own algorithm in oGP.zip
#
#
fidelity = []
for j in rand_indx:

    seqn = data[j]
    train_size = int(np.ceil(len(seqn)*0.70))
    test_size = len(seqn) - train_size
    x = [i for i in range(len(seqn))]
    #d = seqn

    time_step = 1
    trainX, trainY = [], []
    for i in range(0, train_size - time_step):
        #trainX.append(d[i:(i+time_step)])
        trainX.append(x[i:(i+time_step)])
        trainY.append(seqn[i+time_step])

    trainX = np.array(trainX)
    trainY = np.reshape(np.array(trainY), (len(trainY),1))

    testX, testY = [], []
    for i in range( (train_size - time_step), (train_size - time_step)+(test_size - time_step)):
        #testX.append(d[i:(i+time_step)])
        testX.append(x[i:(i+time_step)])
        testY.append(seqn[i+time_step])
    testX = np.array(testX)
    testY = np.reshape(np.array(testY), (len(testY),1))

    est_gp = SymbolicRegression(pop_size = 2000, tourn_size = 5, n_gen = 10, p_mutate = 0.08, p_crossover = 0.9, fit_threshold = 0.1)
    est_gp.fit(X = trainX, y = trainY, batch_size = 1)
    validate = est_gp.predict(trainX, trainY)
    fidelity_train = np.linalg.norm(validate['prediction(s)'] - trainY, ord = 1)/ np.linalg.norm(trainY, ord = 1)
    accuracy = est_gp.predict(testX, testY)
    fidelity_test = np.linalg.norm(accuracy['prediction(s)'] - testY, ord = 1)/np.linalg.norm(testY, ord = 1)

    fidelity.append([fidelity_train, fidelity_test])