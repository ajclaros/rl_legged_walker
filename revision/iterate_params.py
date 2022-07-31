import itertools
import numpy as np
import os
from datalogger import DataLogger
from fitnessFunction import fitnessFunction
from pathlib import Path
from learningFunction import learn
from walking_task import WalkingTask


folderName = "duration1"
#if true will save npz file in data folder with the end fitness as filename
log_data = True
#if verbose in [0.0, 1.0], also prints out the % of trial completed
#also prints out end fitness after each trial
#if verbose>=1, only prints out end fitness
verbose = -1
track_fitness = False
#if true, prints end fitness after every trial

tracking_parameters = []
with open("tracking_parameters.txt", "r") as f:
    for line in f:
        if "#" in line or line=="\n":
            continue
        tracking_parameters.append(line.replace("\n", ""))
print("Tracking:")
print(tracking_parameters)


# times to try each element in the permutation of parameters
trials = 1
param_list = {
    "window_size": [400],
    "learn_rate": [0.008],
    "conv_rate": [0.004],
    "min_period": [300],
    "max_period": [400],
    "init_flux": [0],  # ], 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "max_flux": [0],
    "duration": [2000],
    "size": [4],
    "generator_type":["RPG"],
    "tolerance": [0.00],
    "neuron_configuration": [[0]]
}

if not os.path.exists(Path(f"./data/{folderName}")):
    os.mkdir(f"./data/{folderName}")
with open(f"./data/{folderName}/params.txt", 'w') as f:
    for key in param_list.keys():
        f.writelines(f"{key}:{param_list[key]}\n")

size = param_list['size'][0]

#starting_Genome: fitness:.434 size 4, RPG, configuration=[0]
starting_genome = np.array([0.23346257, -0.30279292, 0.34302416, -0.03512043, 0.80039391, -0.36072524,
-0.49741529, 0.33465454, 0.40609191, -0.2660889, 0.41499235, -0.26798221,
-0.57463584, 0.53038157, 0.22581106, -0.82549032, 0.33720579, -0.26231516,
-0.30053218, 0.66658017, 0.21483684, -0.65461579, 0.89240772, -0.71878452])


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
        print(f"{key}:{params[key]} ", end=" ", flush=False)
    print("")

    # load in perturbed genome
    starting_fitness = fitnessFunction(starting_genome, N=params["size"],
                                       generator_type=params['generator_type'],
                                       configuration=params['neuron_configuration'], verbose=verbose)
    print(starting_fitness)
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
              tolerance=params['tolerance'],
              tracking_parameters=tracking_parameters,
              track_fitness=track_fitness,
              folderName=folderName
              )
