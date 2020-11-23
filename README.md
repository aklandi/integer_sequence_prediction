# A Project in Integer Sequence Prediction

The data sets *train.csv* and *test.csv* come from Kaggle's Knowledge contest **Integer Sequence Learning** (https://www.kaggle.com/c/integer-sequence-learning), but they are based on the **On-line Encyclopedia of Integer Sequences** (https://oeis.org/).  

A findings report based on an exploration of the learning capabilities of dynamic artificial neural network, LSTM neural network, CONV+LSTM neural network, hierarchical temporal memory (HTM) method, and genetic programming (GP) is forthcoming.  It is submitted to *a journal* for publication.  It is well known that neural networks attempt to model learning based on a simplified version of neural activity in a human brain, but HTM and GP are other algorithms supported by biological and learning theories that have yet to be as fully explored as neural networks.  I was very curious what HTM and GP can do in comparison to NNs!  Additionally, the integer sequence data is used to validate these methods, but it does not appear regularly in sequence prediction literature.  So I expect findings will produce new understanding with respect to the learning algorithms' abilities.

In this repository, there is 

* *train.csv* = the training data of integer sequences available from Kaggle
* *test.csv* = the testing data of integer sequences available from Kaggle
* *integer_seqn_methods.py* = the code (commented) used in the report on the findings; architecture from keras, htm.core, and gplearn is used  
* *25_seqns.png* = a graph of 25 sequences from train.csv to give an idea of the functional forms that appear in the data set, though this is not an exhaustive list 
* *last_iteration_validation019100_conv8lstm4.png* = the predictions made by CONV+LSTM, 1 CONV layer with 8 nodes and 1 LSTM layer with 4 nodes and time_step 4
* *last_iteration_validation035751_ragniklein.png* = the predictions made by dynamic ANN, 1 hidden layer with 4 nodes and time_step 4
* *last_iteration_validation234593_LSTM1H4N.png* = the predictions made by LSTM, 1 hidden layer with 4 nodes and time_step 4
* *deviation_compare_LSTM_CONVLSTM_ragniklein.png* = comparison of the deviation between predicted next term and true next term by dynamic ANN, LSTM, and CONV+LSTM
* *ragniklein4_prediction_oscillating.png* = dynamic ANN predictions on oscillating sequence, with 1 hidden layer with 4 nodes and time_step 4
* *LSTM4_prediction_oscillating.png* = LSTM predictions on oscillating sequence, with 1 LSTM layer with 4 nodes and time_step 1
* *CONV8_LSTM4_prediction_oscillating.png* = CONV+LSTM predictions on oscillating sequence, with 1 CONV layer with 8 nodes and 1 LSTM layer with 4 nodes and time_step 1
* *htm_prediction_oscillating.png* = HTM predictions on oscillating sequence, with encoder size 1000 and encoder sparsity 10%, TM 1000 columns and 10 cells per column.
* *ragniklein4_prediction_monotoneincreasing.png* = dunamic ANN predictions on monotone increasing sequence, with one hidden layer and 4 nodes and time_step 4
* *htm_prediction_2.png* = HTM predictions on a monotone increasing sequence, with encoder size 1000 and encoder sparsity 10%, TM 1000 columns and 10 cells per column.
* *gp_prediction_monotoneincreasing.png* = GP predictions on a monotone increasing sequence, with population = 5000, tournament_size = 10, and generations = 20
* *gp_prediction_online_monotone_increasing.png* = on-line GP predictions on a monotone increasing sequence, with population = 2000, tournament_size = 5, and generations = 10
* *gp_prediction_oscillate_without_outlier.png* = GP predictions on oscillating sequence with outlier removed, with population = 5000, tournament_size = 10, and generations = 20
* *gp_prediction_oscillate_withoutlier.png* = GP predictions on oscillating sequence, with population = 5000, tournament_size = 10, and generations = 20
* *gp_prediction_online_oscillate.png* = on-line GP predictions on a oscillating sequence, with population = 2000, tournament_size = 5, and generations = 10
* *gp_program_output.md* = an example of what a program looks like that is created by gplearn
* *fidelity_all_24sequences.png* = the fidelity of all five methods' predictions on the same 25 sequences
* *fidelity_all_24sequences_without_outliers.png* = the fidelity of all five methods' predictions on the same 25 sequences, minus outliers
* *fidelity_all_methods.png* = fidelity of all methods; same as the images above, but includes the on-line GP, rather than vanilla GP with sliding window.
* *oeis_sequence.py* = a function that computes elements of a sequence from the OEIS
    
