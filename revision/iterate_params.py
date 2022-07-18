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
verbose =  1.0
#if true, prints end fitness after every trial

starting_genome =np.array([0.99388489,  -0.19977217,   0.80557307,  0.66176187, -0.41946752,  0.00756486, -0.72451768, -0.50670193])
starting_genome += 0.3
tracking_parameters = []
with open("tracking_parameters.txt", "r") as f:
    for line in f:
        if "#" in line or line=="\n":
            continue
        tracking_parameters.append(line.replace("\n", ""))
print(tracking_parameters)


#starting_genome+=0.5
# times to try each element in the permutation of parameters
trials = 1
param_list = {
    "window_size": [4000],
    "point": [0.1],  # the starting fitnesses: "starting point"
    "learn_rate": [0.008],
    "conv_rate": [0.004],
    "min_period": [300],
    "max_period": [400],
    "init_flux": [2.75],  # ], 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "max_flux": [20],
    "duration": [2000],
    "size": [2],
    "generator_type":["RPG"],
    "prob": [0.5]
}

# parameters to track and their order to save into npz file
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
        learn(starting_genome,
              duration = params["duration"],
              size = params["size"],
              windowsize=params["window_size"],
              init_flux=params["init_flux"],
              max_flux=params["max_flux"],
              min_period=params["min_period"],
              max_period=params["max_period"],
              conv_rate=params["conv_rate"],
              learn_rate=params["learn_rate"],
              bias_init_flux=params["init_flux"],
              bias_max_flux=params["max_flux"],
              bias_min_period=params["min_period"],
              bias_max_period=params["max_period"],
              bias_conv_rate=params["conv_rate"],
              log_data=log_data,
              verbose=verbose,
              generator_type=params['generator_type'],
              prob=params['prob'],
              tracking_parameters=tracking_parameters
              )

