import itertools
import numpy as np
import pandas as pd
import os
from datalogger import DataLogger
from fitnessFunction import fitnessFunction
from pathlib import Path
from learningFunction import learn
from walking_task import WalkingTask
import concurrent.futures
import time


verbose = -1
log_data = False
track_fitness = False
num_trials = 4
num_processes = 7#how many parallel processes to run, set 1 for no parallization
randomize_genomes = True
num_random_genomes = 3
folderName = "iter_"
record_csv = True
csv_name = "data.csv"


# times to try each element in the permutation of parameters

param_list = {
    "window_size": [400, 600, 800, 1000],#, 600, 800, 1000],
    "learn_rate": [0.008],
    "conv_rate": [0.004],
    "min_period": [300],
    "max_period": [400,  600, 1000, 1200, 1400, 1600],
    "init_flux": [5+0.5*i for i in range(5)],
    "max_flux": [8],
    "duration": [2000],
    "size": [2],
    "generator_type":["RPG"],
    "tolerance": [0.00],
    "neuron_configuration": [[0]]
}
N= param_list['size'][0]
start_time = time.time()
csv_elements= ["start_fit", "end_fit", "generator", "configuration",
            "init_flux", "min_period", "max_period", "window_size",
            "genome_num"]

genome_list = []
if not randomize_genomes:
    genome_list.append(np.array([0.99388489,  -0.19977217,   0.80557307,  0.66176187, -0.41946752,  0.00756486, -0.72451768, -0.50670193]))
    #starting_genome = np.array([0.23346257, -0.30279292, 0.34302416, -0.03512043, 0.80039391, -0.36072524,
    #                            -0.49741529, 0.33465454, 0.40609191, -0.2660889, 0.41499235, -0.26798221,
    #                            -0.57463584, 0.53038157, 0.22581106, -0.82549032, 0.33720579, -0.26231516,
    #                            -0.30053218, 0.66658017, 0.21483684, -0.65461579, 0.89240772, -0.71878452])
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
    genome_list = np.random.uniform(-1, 1, size= (num_random_genomes, N*N+2*N))

if record_csv:
    with open(f"./data/{csv_name}",'w') as f:
        line = ", ".join(csv_elements)
        line+="\n"
        f.writelines(line)
    with open(f"./data/{csv_name}_params.txt", 'w') as f:
        for key in param_list.keys():
            f.writelines(f"{key}:{param_list[key]}\n")
        if randomize_genomes:
            f.writelines("Genomes:\n")
            for i in range(genome_list.shape[0]):
                genome = ", ".join(genome_list[i].astype(str).tolist())
                genome+="\n\n"
                f.writelines(genome)

if log_data:
    for key in param_list.keys():
        if "_" in key:
            words = key.split("_")

            if "min" in key or "max" in key or "init" in key:
                k = words[0][0]+words[0][-1]+words[1][0].capitalize()
            else:
                k = words[0][0]+words[1][0]
        else:
            k = key[0].capitalize()
        if len(param_list[key]) == 1:
            if "neuron_configuration" in key:
                conf = [str(val) for val in param_list[key][0]]
                conf = "".join(conf)
                folderName+=f"{k}.{conf}"
            else:
                folderName+=f"{k}.{param_list[key][0]}"
        else:
            if "neuron_configuration" in key:
                conf = [str(val) for val in param_list[key][0]]
                conf = "".join(conf)
                conf2 = [str(val) for val in param_list[key][-1]]
                conf2 = "".join(conf)
                folderName+=f"{k}.{conf}-{conf2}"
            else:
                folderName+=f"{k}.{param_list[key][0]}-{param_list[key][-1]}"

        folderName+="_"
    folderName=folderName[:-1]
    if not os.path.exists(Path(f"./data/{folderName}")):
        os.mkdir(f"./data/{folderName}")
    with open(f"./data/{folderName}/params.txt", 'w') as f:
        for key in param_list.keys():
            f.writelines(f"{key}:{param_list[key]}\n")
        if randomize_genomes:
            f.writelines("Genomes:\n")
            for i in range(genome_list.shape[0]):
                genome = ", ".join(genome_list[i].astype(str).tolist())
                genome+="\n\n"
                f.writelines(genome)

tracking_parameters = []
with open("tracking_parameters.txt", "r") as f:
    for line in f:
        if "#" in line or line=="\n":
            continue
        tracking_parameters.append(line.replace("\n", ""))
if verbose>=0:
    print("Tracking:")
    print(tracking_parameters)

start_time = time.time()
prev = list()
results = []
result_genome = []
result_fit= []
total_configurations = len(list(itertools.product(*param_list.values())))
with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
# parameters to track and their order to save into npz file
# itertools creates a list of all permutations for each list in the dictionary
# https://stackoverflow.com/questions/24594313/permutations-of-list-of-lists
    for completed, x in enumerate(itertools.product(*param_list.values())):
        # x is an element from the full permutation of all lists
        # creates dictionary for specific instance of values
        params = {}
        for i, key in enumerate(param_list.keys()):
            params[key] = x[i]
        # list of filenames for perturbed genomes
        #iterate through each starting genome
        print(f"{np.round(completed/total_configurations,2)} completed:")
        for key in params.keys():
            if params[key] not in prev:
                print(f"{key}:{params[key]}", end=" ")
        prev = params.values()
        for i, genome in enumerate(genome_list):
            starting_genome = genome
            if log_data:
                subFolderName = ""
                print(f"Genome: {i}")
                for key in params.keys():
                    if "_" in key:
                        words = key.split("_")
                        if "min" in key or "max" in key or "init" in key:
                            k = words[0][0]+words[0][-1]+words[1][0].capitalize()
                        else:
                            k = words[0][0]+words[1][0]
                        if 'neuron' in key:

                            conf = [str(val) for val in param_list[key][0]]
                            conf = "".join(conf)
                            subFolderName+=f"{k}.{conf}"
                    else:
                        k = key[0].capitalize()
                    subFolderName+=f"{k}.{params[key]}"
                    subFolderName+="_"

                subFolderName= subFolderName[:-1]
                if not os.path.exists(Path(f"./data/{folderName}/{subFolderName}")):
                    os.mkdir(f"./data/{folderName}/{subFolderName}")
                with open(f"./data/{folderName}/{subFolderName}/params.txt", 'w') as f:
                    for key in params.keys():
                        f.writelines(f"{key}:{params[key]}\n")

            starting_fitness = fitnessFunction(starting_genome, N=params["size"],
                                               generator_type=params['generator_type'],
                                               configuration=params['neuron_configuration'], verbose=verbose)

            print(f"genome start fit:{starting_fitness}")
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
            for trial in range(num_trials):
                np.random.seed(np.random.randint(10000))
                results.append(executor.submit(learn , starting_genome, **params,
                                               tracking_parameters=tracking_parameters,
                                               track_fitness=track_fitness,
                                               #folderName=f"{folderName}/{subFolderName}",
                                               print_done=False,
                                               trial=trial,
                                               log_data=log_data,
                                               verbose=verbose, genome_num=i,
                                               csv_name=csv_name))

                if len(results) == num_processes:
                    for future in concurrent.futures.as_completed(results):
                        num, fitness = future.result()
                        result_genome.append(num)
                        result_fit.append(fitness)
                    print(pd.Series(data=result_fit, index=result_genome))
                    result_genome = []
                    result_fit= []

                    results = []


print(f"Finished in:{np.round(time.time()-start_time, 2)} seconds")
