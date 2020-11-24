"""
Author: Joshua Landi (jlandi@ncsu.edu)
As part of a research project; results are forthcoming in a paper published in ___.

References:
(1) Segaran, Toby. 2007. Programming collective intelligence (First. ed.). O'Reilly. Chapter 11: Evolving Intelligence.
"""

from typing import Dict
from typing import NDArray
import numpy
from classes.symbolic_regression import SymbolicRegression
import sys, getopt, json

def main(argv):
    skip_tofile_output = False
    input_path = ''
    output_path = ''
    try:
        opts, args = getopt.getopt(argv, "c", ["input=", "output="])
    except getopt.GetoptError:
        print('genetic.py input=<filepath> output=<filepath>')
    for opt, arg in opts:
        if opt == '-i':
            skip_tofile_output = True
        elif opt in ("--input"):
            input_path = arg
        elif opt in ("--output"):
            output_path = arg
    with open(input_path, "r") as read_file:
        print('Opening input file')
        file: Dict = json.load(read_file)
        input: SymbolicRegression = SymbolicRegression.parse_from_file(file)
        print('Reading independent variables')
        independent_variables = numpy.array(file["independent_variables"]).T
        print('Reading truth/dependent variables')
        truth: NDArray[int] = numpy.array(file["truth"]).T
        print('Reading batch size')
        batch_size: int = file["batch_size"]
        print('Running program...')
        input.fit(independent_variables, truth, batch_size)
        read_file.close()
    if skip_tofile_output:
        print('done')
    else:
        with open(output_path, "w") as write_file:
            write_file.write('done')
            write_file.close()
    
if __name__ == "__main__":
   main(sys.argv[1:])