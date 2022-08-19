import numpy as np
import os
from datalogger import DataLogger
from learningFunction import learn
from pathlib import Path
from visdata import *
import concurrent.futures

# run a single configuration "num_trials" times
# verbose=-1: do not print
# verbose>=0: print out starting and ending fitness
# verbose in (0,1), print out progress of trial every % time passes for example

verbose = 1
log_data = True
track_fitness = False
num_trials = 1
num_processes = 8
num_sets = int(np.floor(num_trials/num_processes))
randomize_genomes = True
num_random_genomes = 1

#if visualize is true, print the parameters to visualize
# "averaged [param_name]" will print the average of the parameter across all trials
visualize = True
vis_everything = True
vis_params = ["averaged running_average"]


params = {
    "window_size": 400,             #unit seconds
    "learn_rate": 0.008,
    "conv_rate": 0.004,
    "min_period": 300,              #unit seconds
    "max_period": 800,              #unit seconds
    "init_flux": 6,
    "max_flux": 8,
    "duration": 1000,               #unit seconds
    "size": 2,
    "generator_type": "RPG",
    "tolerance": 0.00,
    "neuron_configuration": [0]
}
folderName = f"{params['generator_type']}_d{params['duration']}_initfx{params['init_flux']}_00_window{params['window_size']}_max_p{params['max_period']}"
if not os.path.exists(Path(f"./data/{folderName}")):
    print(f"creating folder:{folderName}")
    os.mkdir(f"./data/{folderName}")

with open(f"./data/{folderName}/params.txt", 'w') as f:
    for key in params.keys():
        f.writelines(f"{key}:{params[key]}\n")

N = params['size']
genome_list = [ ]
# hard coded genomes
if not randomize_genomes:
    # size = 2
    genome_list.append(np.array([0.99388489,  -0.19977217,   0.80557307,  0.66176187, -0.41946752,  0.00756486, -0.72451768, -0.50670193]))
    starting_genome = genome_list[0]
    for i, val in enumerate(starting_genome):

        #add noise to genome, keep within bounds [-1,1]
        perturb = 0.2
        #perturb  = np.random.binomial(1, p=0.5)*0.2
        if val+perturb>1 or val+perturb<-1:
            starting_genome[i]+=-perturb
        else:
            starting_genome[i]+=perturb

else:
    genome_list = np.random.uniform(-1, 1, size=(num_random_genomes, N*N+2*N))

tracking_parameters = []
with open("tracking_parameters.txt", "r") as f:
    for line in f:
        if "#" in line or line=="\n":
            continue
        tracking_parameters.append(line.replace("\n", ""))
if verbose>=0:
    print("Tracking:")
    print(tracking_parameters)

params['bias_init_flux'] = params['init_flux']
params['init_flux'] = params['init_flux']
params['max_flux'] = params['max_flux']
params['bias_init_flux'] = params['init_flux']
params['bias_init_flux'] = params['init_flux']
params['bias_max_flux'] = params['max_flux']
params['bias_max_flux'] = params['max_flux']
params['bias_min_period'] = params['min_period']
params['bias_max_period'] = params['max_period']
params['bias_conv_rate'] = params['conv_rate']
with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    for i, starting_genome in enumerate(genome_list):
        print(f"Genome:{i}")
        N = params['size']
        results = []
        print("trial:")
        for m in range(num_sets):
            for n in range(num_processes):
                print(int(m*num_processes+n), end=' ')
                results.append(executor.submit(learn, starting_genome, log_data=log_data,
                                               verbose=verbose,
                                               tracking_parameters=tracking_parameters,
                                               track_fitness=track_fitness,
                                               folderName=folderName,
                                               trial=int(m*num_processes+n),
                                               print_done=False,
                                               **params))
        for future in concurrent.futures.as_completed(results):
            print(future.result())
        results = []
        for m in range(int(num_trials-num_sets*num_processes)):
            print(int(m+num_sets*num_processes), end=" ")
            #print(int(m+num_sets*num_processes), end=' ')
            results.append(executor.submit(learn, starting_genome, log_data=log_data,
                                           verbose=verbose,
                                           tracking_parameters=tracking_parameters,
                                           track_fitness=track_fitness,
                                           folderName=folderName,
                                           trial=int(m+num_sets*num_processes),
                                           print_done=False,
                                           **params))

        for future in concurrent.futures.as_completed(results):
            filename = future.result()
            print(filename)
pathname = f"./data/{folderName}"
files = os.listdir(pathname)
files = [name for name in files if '.npz' in name]
data = np.load(f"{pathname}/{files[0]}")
time = np.arange(0, data['duration'], data['stepsize'])
if visualize:
    for tracked in vis_params:
        if "averaged" in tracked:
            tracked = tracked.split(' ')[-1]
            plotAverageParam(tracked, show=False, b=-1, pathname=pathname)

    if vis_everything:
        plotBehavior(data, show=False, save=True)
        plotWeightsBiases(data, show=False, extended=True, save=True)
        plotChosenParam(pathname+"/"+filename, params=['reward','flux_amp', 'distance', ('running_average', 'track_fitness')], save=True)
    #else:
    #    plot_params = [p for p in vis_params if "averaged" not in p]
    #    plotChosenParam(pathname+"/"+filename(), params=plot_params)

#plotAverageParam('running_average', show=True, b=1000, pathname=pathname)
plt.show()
