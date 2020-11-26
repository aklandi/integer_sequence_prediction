"""
Author: Joshua Landi (jlandi@ncsu.edu)
As part of a research project; results are forthcoming in a paper published in ___.

References:
(1) Segaran, Toby. 2007. Programming collective intelligence (First. ed.). O'Reilly. Chapter 11: Evolving Intelligence.
"""
import sys, getopt, json
from typing import Dict
from nptyping import NDArray
import numpy as np
from classes.symbolic_regression import SymbolicRegression

def main(argv):
    skip_tofile_output = False
    input_path = ''
    output_path = ''
    input: SymbolicRegression = None
    independent_variables: NDArray = None
    try:
        opts, args = getopt.getopt(argv, "c", ["input=", "output="])
    except getopt.GetoptError:
        print('genetic.py -c input=<filepath> output=<filepath>')
        return
    for opt, arg in opts:
        if opt == '-c':
            skip_tofile_output = True
        elif opt in ("--input"):
            input_path = arg
        elif opt in ("--output"):
            output_path = arg
    
    if (input_path == ''):
        print("Must specify a settings input path.")
        return
    if (output_path == '' and skip_tofile_output == False):
        print("Must specify an output_path if -c is not provided as an argument.")
        return
    with open(input_path, "r") as read_file:
        print('Opening input file')
        file: Dict = json.load(read_file)
        input: SymbolicRegression = SymbolicRegression.parse_from_file(file)
        print('Reading independent variables')
        independent_variables = np.array(file["independent_variables"]).T
        print('Reading truth/dependent variables')
        truth: NDArray = np.array(file["truth"]).T
        print('Reading batch size')
        batch_size: int = file["batch_size"]
        print('Running program...')
        input.fit(independent_variables, truth, batch_size)
        read_file.close()
    if skip_tofile_output:
        print("Making online GP predictions...")
        validate = input.predict(independent_variables, truth)
        deviation = np.linalg.norm(validate['prediction(s)'] - truth, ord = 1)
        length = np.linalg.norm(truth, ord = 1)
        print("Fidelity on training data: " + str(deviation / length))
    else:
        print("Opening output file")
        with open(output_path, "w") as write_file:
            print("Making online GP predictions...")
            validate = input.predict(independent_variables, truth)
            deviation = np.linalg.norm(validate['prediction(s)'] - truth, ord = 1)
            length = np.linalg.norm(truth, ord = 1)
            output = "Fidelity on training data: " + str(deviation / length)
            print("Writing result to output file")
            write_file.write(output)
            print("Closing output file")
            write_file.close()
    
if __name__ == "__main__":
   main(sys.argv[1:])