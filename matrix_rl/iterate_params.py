import itertools
import numpy as np

import pandas as pd
import os
from fitnessFunction import fitnessFunction
from pathlib import Path
from learningFunction import learn
from walking_task import WalkingTask
import concurrent.futures
import time

verbose = -1
log_data = False
num_trials = 10
num_processes = 128  # how many parallel processes to run, set 1 for no parallization
randomize_genomes = False
USE_GENOME_LIST = True
LOAD_GENOMES = True
num_random_genomes = 0
folderName = "parameter_explore"
record_csv = True
csv_name = "data"
if USE_GENOME_LIST:
    fit_low, fit_high = (0.16, 0.6)
    num_genomes_per_configuration = 20

# times to try each element in the permutation of parameters

# add = 2
# start = 0 + add
# end = start + 1
r_start = 0
init_f_start = 0
max_f_start = 0

r_end = 3
init_f_end = 1
max_f_end = 1
divide = 3
# rates: np.arange(0, .15, 0.01)
# init_flux: np.arange(0, 1.0)
# max_flux: np.arange(0, 0.6)
rates = np.arange(0.01, 0.05, 0.01)
init_flux = np.arange(0.1, 0.5, 0.02)
max_flux = np.arange(0.05, 0.5, 0.02)
param_list = {
    "window_size": [440],  # , 600, 800, 1000],
    "rates": rates[rates.size // divide * r_start : rates.size // divide * r_end],
    "min_period": [440],
    "max_period": [4400],
    "init_flux": init_flux[
        init_flux.size // divide * init_f_start : init_flux.size // divide * init_f_end
    ],
    "max_flux": max_flux[
        max_flux.size // divide * max_f_start : max_flux.size // divide * max_f_end
    ],
    "duration": [1500],
    "size": [3],
    "learning_start": [800],
    "record_every": [10],
    "generator_type": ["RPG"],
    "neuron_configuration": [[0, 1]],
}
N = param_list["size"][0]
start_time = time.time()
csv_elements = [
    "start_fit",
    "end_fit",
    "generator",
    "configuration",
    "init_flux",
    "min_period",
    "max_period",
    "window_size",
    "genome_num",
    "end_perf",
    "max_flux",
    "rates",
]

genome_list = []
if not randomize_genomes and USE_GENOME_LIST and not LOAD_GENOMES:
    print("USING GENOME LIST")
    genome_list = []
    conf_str = list(map(str, param_list["neuron_configuration"][0]))
    conf_str = "".join(conf_str)
    fitnesses = []
    genome_fitnesses = [
        float(name.split("-")[1].split(".")[0]) / 100000
        for name in os.listdir(
            f"./evolved/{param_list['generator_type'][0]}/{param_list['size'][0]}/{conf_str}"
        )
    ]
    hard_coded_genomes = [
        (name, fitnesses.append(name))
        for i, name in enumerate(
            os.listdir(
                f"./evolved/{param_list['generator_type'][0]}/{param_list['size'][0]}/{conf_str}"
            )
        )
        if genome_fitnesses[i] > fit_low and genome_fitnesses[i] < fit_high
    ]
    samples = np.random.randint(
        0, len(hard_coded_genomes), size=num_genomes_per_configuration
    )
    for samp in samples:
        genome_list.append(
            np.load(
                f"./evolved/{param_list['generator_type'][0]}/{param_list['size'][0]}/{conf_str}/{hard_coded_genomes[samp][0]}"
            )
        )
    genome_list = np.array(genome_list)
    print(fitnesses)
    np.save("genome_list.npy", genome_list)
    quit()

elif LOAD_GENOMES:
    genome_list = np.load("genome_list.npy")

else:
    genome_list = np.random.uniform(-1, 1, size=(num_random_genomes, N * N + 2 * N))
# if record_csv:
#     for i in range(num_trials):
#         for j in range(num_genomes_per_configuration):
#             with open(f"./data/csv_folder/{csv_name}{j}_{i}.csv", "w") as f:
#                 line = ", ".join(csv_elements)
#                 line += "\n"
#                 f.writelines(line)
#             with open(f"./data/csv_folder/{csv_name}_params.txt", "w") as f:
#                 for key in param_list.keys():
#                     f.writelines(f"{key}:{param_list[key]}\n")
#                 if randomize_genomes:
#                     f.writelines("Genomes:\n")
#                     for i in range(genome_list.shape[0]):
#                         genome = ", ".join(genome_list[i].astype(str).tolist())
#                         genome += "\n\n"
#                         f.writelines(genome)

if log_data:
    for key in param_list.keys():
        if "_" in key:
            words = key.split("_")

            if "min" in key or "max" in key or "init" in key:
                k = words[0][0] + words[0][-1] + words[1][0].capitalize()
            else:
                k = words[0][0] + words[1][0]
        else:
            k = key[0].capitalize()
        if len(param_list[key]) == 1:
            if "neuron_configuration" in key:
                conf = [str(val) for val in param_list[key][0]]
                conf = "".join(conf)
                folderName += f"{k}.{conf}"
            else:
                folderName += f"{k}.{param_list[key][0]}"
        else:
            if "neuron_configuration" in key:
                conf = [str(val) for val in param_list[key][0]]
                conf = "".join(conf)
                conf2 = [str(val) for val in param_list[key][-1]]
                conf2 = "".join(conf)
                folderName += f"{k}.{conf}-{conf2}"
            else:
                folderName += f"{k}.{param_list[key][0]}-{param_list[key][-1]}"

        folderName += "_"
    folderName = folderName[:-1]
    if not os.path.exists(Path(f"./data/{folderName}")):
        os.mkdir(f"./data/{folderName}")
    with open(f"./data/{folderName}/params.txt", "w") as f:
        for key in param_list.keys():
            f.writelines(f"{key}:{param_list[key]}\n")
        if randomize_genomes:
            f.writelines("Genomes:\n")
            for i in range(genome_list.shape[0]):
                genome = ", ".join(genome_list[i].astype(str).tolist())
                genome += "\n\n"
                f.writelines(genome)

tracking_parameters = []
with open("tracking_parameters.txt", "r") as f:
    for line in f:
        if "#" in line or line == "\n":
            continue
        tracking_parameters.append(line.replace("\n", ""))
if verbose >= 0:
    print("Tracking:")
    print(tracking_parameters)

start_time = time.time()
prev = list()
results = []
result_genome = []
result_fit = []
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
            if key == "rates":
                params["learn_rate"] = x[i]
                params["conv_rate"] = x[i]
            else:
                params[key] = x[i]
        # list of filenames for perturbed genomes
        # iterate through each starting genome
        print(f"{np.round(completed/total_configurations,2)} completed:")
        print(f"{completed}/{total_configurations}")
        print(params)

        for i, genome in enumerate(genome_list):
            starting_genome = genome
            if log_data:
                subFolderName = ""
                print(f"Genome: {i}")
                for key in params.keys():
                    if "_" in key:
                        words = key.split("_")
                        if "min" in key or "max" in key or "init" in key:
                            k = words[0][0] + words[0][-1] + words[1][0].capitalize()
                        else:
                            k = words[0][0] + words[1][0]
                        if "neuron" in key:

                            conf = [str(val) for val in param_list[key][0]]
                            conf = "".join(conf)
                            subFolderName += f"{k}.{conf}"
                    else:
                        k = key[0].capitalize()
                    subFolderName += f"{k}.{params[key]}"
                    subFolderName += "_"

                subFolderName = subFolderName[:-1]
                if not os.path.exists(Path(f"./data/{folderName}/{subFolderName}")):
                    os.mkdir(f"./data/{folderName}/{subFolderName}")
                with open(f"./data/{folderName}/{subFolderName}/params.txt", "w") as f:
                    for key in params.keys():
                        f.writelines(f"{key}:{params[key]}\n")

            starting_fitness = fitnessFunction(
                starting_genome,
                N=params["size"],
                generator_type=params["generator_type"],
                configuration=params["neuron_configuration"],
                verbose=verbose,
            )

            # print(f"genome start fit:{starting_fitness}")
            params["bias_init_flux"] = params["init_flux"]
            params["init_flux"] = params["init_flux"]
            params["max_flux"] = params["max_flux"]
            params["bias_init_flux"] = params["init_flux"]
            params["bias_init_flux"] = params["init_flux"]
            params["bias_max_flux"] = params["max_flux"]
            params["bias_max_flux"] = params["max_flux"]
            params["bias_min_period"] = params["min_period"]
            params["bias_max_period"] = params["max_period"]
            params["bias_conv_rate"] = params["conv_rate"]
            csv_name = int(starting_fitness * 100000)
            for trial in range(num_trials):
                np.random.seed(np.random.randint(10000))
                results.append(
                    executor.submit(
                        learn,
                        starting_genome,
                        **params,
                        tracking_parameters=tracking_parameters,
                        # folderName=f"{folderName}/{subFolderName}",
                        print_done=False,
                        trial=trial,
                        log_data=log_data,
                        verbose=-1,
                        genome_num=i,
                        csv_name=f"{csv_name}_{trial}_{r_start}{init_f_start}{max_f_start}_{r_end}{init_f_end}{max_f_end}.csv",
                        starting_fitness=starting_fitness,
                    )
                )

                if len(results) == num_processes:
                    for future in concurrent.futures.as_completed(results):
                        num, fitness = future.result()
                        result_genome.append(num)
                        result_fit.append(fitness)
                    # print(pd.Series(data=result_fit, index=result_genome))
                    result_genome = []
                    result_fit = []
                    results = []

for future in concurrent.futures.as_completed(results):
    num, fitness = future.result()
    result_genome.append(num)
    result_fit.append(fitness)
# print(pd.Series(data=result_fit, index=result_genome))
result_genome = []
result_fit = []
results = []

print(f"Finished in:{np.round(time.time()-start_time, 2)} seconds")
