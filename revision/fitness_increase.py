import itertools
import numpy as np
import os
from datalogger import DataLogger
from fitnessFunction import fitnessFunction
from pathlib import Path
from learningFunction import learn


#if true will save npz file in data folder with the end fitness as filename
log_data = True

#if verbose in [0.0, 1.0], also prints out the % of trial completed
#also prints out end fitness after each trial
#if verbose>=1, only prints out end fitness
verbose = -1
#if true, prints end fitness after every trial

starting_genome =np.array([0.99388489,  -0.19977217,   0.80557307,  0.66176187, -0.41946752,  0.00756486, -0.72451768, -0.50670193])
# times to try each element in the permutation of parameters
trials = 10
param_list = {
    "window_size": [4000],
    "point": [0.1],  # the starting fitnesses: "starting point"
    "learn_rate": [0.008],
    "conv_rate": [0.004],
    "min_period": [300],
    "max_period": [400],
    "init_flux": [2.75],  # ], 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "max_flux": [10],
    "duration": [2000],
    "size": [2]
}

# parameters to track and their order to save into npz file
tracking_parameters = ["name", "init_flux", "starting_fitness", "end_fitness"]

# itertools creates a list of all permutations for each list in the dictionary
# https://stackoverflow.com/questions/24594313/permutations-of-list-of-lists

for x in itertools.product(*param_list.values()):
    # x is an element from the full permutation of all lists
    # creates dictionary for specific instance of values
    params = {}
    for i, key in enumerate(param_list.keys()):
        params[key] = x[i]

    # list of filenames for perturbed genomes
    #iterate through each starting genome
    for key in params.keys():
        if key in tracking_parameters:
            print(f"{key}:{params[key]}", end=" ", flush=False)

    # load in perturbed genome
    starting_fitness = fitnessFunction(starting_genome)
    print(f"\nTrial:", end= " ")
    for i in range(trials):
        print(f" {i}", end=" ", flush=False)
        #weights and biases are initialized the same way
        learn(params["duration"],
              params["size"],
              params["window_size"],
              params["init_flux"],
              params["max_flux"],
              params["min_period"],
              params["max_period"],
              params["conv_rate"],
              params["learn_rate"],
              params["init_flux"],
              params["max_flux"],
              params["min_period"],
              params["max_period"],
              params["conv_rate"],
              starting_genome,
              log_data,
              verbose
              )
