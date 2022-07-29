import numpy as np
import os
from datalogger import DataLogger
from learningFunction import learn
from pathlib import Path
from visdata import *


# runn a single configuration "num_trials" times
# verbose=-1: do not print
# verbose>=0: print out starting and ending fitness
# verbose in (0,1), print out progress of trial every % time passes for example
verbose = 0.1
log_data = True
track_fitness = False
num_trials = 1
randomize_genomes = False
num_random_genomes = 1

#if visualize is true, print the parameters to visualize
# "averaged [param_name]" will print the average of the parameter across all trials
visualize = True
vis_everything = False
vis_params = ["averaged running_average"]

#w
folderName = "experiment0"

params = {
    "window_size": 400,             #unit seconds
    "learn_rate": 0.008,
    "conv_rate": 0.004,
    "min_period": 300,              #unit seconds
    "max_period": 400,              #unit seconds
    "init_flux":4,
    "max_flux": 8,
    "duration": 8000,               #unit seconds
    "size": 2,
    "generator_type": "RPG",
    "tolerance": 0.00,
    "neuron_configuration": [0],
}
if not os.path.exists(Path(f"./data/{folderName}")):
    os.mkdir(f"./data/{folderName}")

with open(f"./data/{folderName}/params.txt", 'w') as f:
    for key in params.keys():
        f.writelines(f"{key}:{params[key]}\n")

# hard coded genomes
if not randomize_genomes:
    # size = 2
    starting_genome =np.array([0.99388489,  -0.19977217,   0.80557307,  0.66176187, -0.41946752,  0.00756486, -0.72451768, -0.50670193])

    #size = 4
#    starting_genome = np.array([0.23346257, -0.30279292, 0.34302416, -0.03512043, 0.80039391, -0.36072524,
#    -0.49741529, 0.33465454, 0.40609191, -0.2660889, 0.41499235, -0.26798221,
#    -0.57463584, 0.53038157, 0.22581106, -0.82549032, 0.33720579, -0.26231516,
#    -0.30053218, 0.66658017, 0.21483684, -0.65461579, 0.89240772, -0.71878452])

    for i, val in enumerate(starting_genome):

        #add noise to genome, keep within bounds [-1,1]
        perturb = 0.2
        #perturb  = np.random.binomial(1, p=0.5)*0.2
        if val+perturb>1 or val+perturb<-1:
            starting_genome[i]+=-perturb
        else:
            starting_genome[i]+=perturb
tracking_parameters = []
with open("tracking_parameters.txt", "r") as f:
    for line in f:
        if "#" in line or line=="\n":
            continue
        tracking_parameters.append(line.replace("\n", ""))
if verbose>=0:
    print("Tracking:")
    print(tracking_parameters)

if not randomize_genomes:
    for trial in range(num_trials):
        print("Trial:")
        print(trial)
        filename = learn(starting_genome,
                         duration=params["duration"],
                         size=params["size"],
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
                         track_fitness=track_fitness, folderName=folderName)

else:
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
                             tracking_parameters=tracking_parameters,
                             track_fitness=track_fitness,
                             max_perf = params['max_perf'],
                             folderName=folderName)

pathname = f"./data/{folderName}"
files = os.listdir(pathname)
files = [name for name in files if '.npz' in name]
data = np.load(f"{pathname}/{filename}.npz")
time = np.arange(0, data['duration'], data['stepsize'])
plt.plot(time[:-1], np.diff(data['distance']))
plt.title("diff distance")
if visualize:
    for tracked in vis_params:
        if "averaged" in tracked:
            tracked = tracked.split(' ')[-1]
            plotAverageParam(tracked, show=False, b=-1, pathname=pathname)
    if vis_everything:
        plotBehavior(data, show=False, save=True)
        plotWeightsBiases(data, show=False, extended=True, save=True)
        plotChosenParam(pathname+"/"+filename, params=['reward','flux_amp', 'distance', ('running_average', 'track_fitness')], save=True)
#plotAverageParam('running_average', show=True, b=1000, pathname=pathname)
