import numpy as np
import os
from datalogger import DataLogger
from learningFunction import learn
from visdata import *


# runn a single configuration "num_trials" times
# verbose=-1: do not print
# verbose>=0: print out starting and ending fitness
# verbose in (0,1), print out progress of trial every % time passes for example
verbose = 0
log_data = True
num_trials = 10
randomize_genomes = True
num_random_genomes = 1

#if visualize is true, print the parameters to visualize
visualize = True
vis_params = ['averaged running_average']


params = {
    "window_size": 4000,
    "learn_rate": 0.008,
    "conv_rate": 0.004,
    "min_period": 300,
    "max_period": 400,
    "init_flux": 6,
    "max_flux": 10,
    "duration": 2000,
    "size": 2,
    "generator_type": "RPG",
    "tolerance": 0.00,
    "neuron_configuration": [0]
}


#hard coded genomes
if not randomize_genomes:
    starting_genome =np.array([0.99388489,  -0.19977217,   0.80557307,  0.66176187, -0.41946752,  0.00756486, -0.72451768, -0.50670193])
#starting_genome = np.array([0.23346257, -0.30279292, 0.34302416, -0.03512043, 0.80039391, -0.36072524,
#-0.49741529, 0.33465454, 0.40609191, -0.2660889, 0.41499235, -0.26798221,
#-0.57463584, 0.53038157, 0.22581106, -0.82549032, 0.33720579, -0.26231516,
#-0.30053218, 0.66658017, 0.21483684, -0.65461579, 0.89240772, -0.71878452])

tracking_parameters = []
with open("tracking_parameters.txt", "r") as f:
    for line in f:
        if "#" in line or line=="\n":
            continue
        tracking_parameters.append(line.replace("\n", ""))
if verbose>=0:
    print("Tracking:")
    print(tracking_parameters)

if randomize_genomes:
    for i in range(num_random_genomes):
        print(f"Random genome:{i}")
        N = params['size']
        starting_genome = np.random.uniform(-1, 1, size=N*N+2*N)
        print("Trial:")
        for trial in range(num_trials):
            print(trial)
            filename = learn(starting_genome,
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
                             tracking_parameters=tracking_parameters)

files = os.listdir('./data')
files = [name for name in files if '.npz' in name]
data =np.load(f"./data/{files[0]}")
if visualize:
    for tracked in vis_params:
        if "averaged" in tracked:
            tracked = tracked.split(' ')[-1]
            plotAverageParam(tracked, show=True)
        else:
            plotChosenParam(filename, tracked, show=True)

