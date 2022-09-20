import numpy as np
import os
from datalogger import DataLogger
from learningFunction import learn
from pathlib import Path
from visdata import *
import concurrent.futures
import matplotlib.pyplot as plt
from fitnessFunction import fitnessFunction
import time

# run a single configuration "num_trials" times
# verbose=-1: do not print
# verbose>=0: print out starting and ending fitness
# verbose in (0,1), print out progress of trial every % time passes for example

verbose = 0.1
log_data = True
record_csv = True
track_fitness = False
num_trials = 8
num_processes = 16
num_sets = int(np.floor(num_trials / num_processes))
randomize_genomes = True
num_random_genomes = 2
# if visualize is true, print the parameters to visualize
# "averaged [param_name]" will print the average of the parameter across all trials
visualize = True
vis_behavior = True
vis_weights = True
vis_agent = True
vis_params = [
    "averaged performance_average_hist",
    #    "distribution flux_amp",
    #    "distribution performance_hist",
    "averaged flux_amp",
]
csv_name = "single_genome.csv"

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
]

params = {
    "window_size": 440,  # unit seconds
    "learn_rate": 0.9,
    "conv_rate": 0.9,
    "min_period": 440,  # unit seconds
    "max_period": 4400,  # unit seconds
    "init_flux": 2,
    "max_flux": 1,
    "duration": 30000,  # unit seconds
    "size": 3,
    "generator_type": "CPG",
    "tolerance": 0.00000,
    "neuron_configuration": [0, 1, 2],
    "record_every": 3,
}

# params = {
#     "reward_func": "secondary",
#     "performance_func": "secondary",
#     "window_size": 400,  # unit seconds
#     "learn_rate": 0.008,
#     "conv_rate": 0.004,
#     "min_period": 300,  # unit seconds
#     "max_period": 400,  # unit seconds
#     "init_flux": 6.0,
#     "max_flux": 8,
#     "duration": 60000,  # unit seconds
#     "size": 2,
#     "generator_type": "RPG",
#     "tolerance": 0.00000,
#     "neuron_configuration": [0],
# }
folderName = f"{params['generator_type']}_d{params['duration']}_initfx{params['init_flux']}_00_window{params['window_size']}_max_p{params['max_period']}"
folderName += "recording"
if not os.path.exists(Path(f"./data/{folderName}")):
    print(f"creating folder:{folderName}")
    os.mkdir(f"./data/{folderName}")


def getAmp(end_fitness, start_fitness, max_flux):
    start_flux = 0
    if start_fitness == 0.0:
        return max_flux
    else:
        return max_flux * start_fitness / end_fitness


N = params["size"]
genome_list = []
# hard coded genomes
if not randomize_genomes:
    # size = 2
    genome_list.append(
        np.array(
            [
                0.99388489,
                -0.19977217,
                0.80557307,
                0.66176187,
                -0.41946752,
                0.00756486,
                -0.72451768,
                -0.50670193,
            ]
        )
    )
    genome_list = np.array(genome_list)
    starting_genome = genome_list[0]

    for i, val in enumerate(starting_genome):

        # add noise to genome, keep within bounds [-1,1]
        perturb = 0.2
        # perturb  = np.random.binomial(1, p=0.5)*0.2
        if val + perturb > 1 or val + perturb < -1:
            starting_genome[i] += -perturb
        else:
            starting_genome[i] += perturb

else:
    genome_list = np.random.uniform(-1, 1, size=(num_random_genomes, N * N + 2 * N))

tracking_parameters = []
with open("tracking_parameters.txt", "r") as f:
    for line in f:
        if "#" in line or line == "\n":
            continue
        tracking_parameters.append(line.replace("\n", ""))
if verbose >= 0:
    print("Tracking:")
    print(tracking_parameters)

if log_data:
    with open(f"./data/{folderName}/params.txt", "w") as f:
        for key in params.keys():
            f.writelines(f"{key}:{params[key]}\n")

start_time = time.time()
if record_csv:
    with open(f"./data{csv_name}", "w") as f:
        line = ",".join(csv_elements)
        line += "\n"
        f.writelines(line)
    with open(f"./data{csv_name}_params.txt", "w") as f:
        for key in params.keys():
            f.writelines(f"{key}:{params[key]}\n")
        for i in range(genome_list.shape[0]):
            genome = ", ".join(genome_list[i].astype(str).tolist())
            genome = "[" + genome + "]"
            genome += "\n\n"
            f.writelines(genome)

filename = ""
params["bias_init_flux"] = params["init_flux"]
params["bias_init_flux"] = params["init_flux"]
params["max_flux"] = params["max_flux"]
params["bias_init_flux"] = params["init_flux"]
params["bias_max_flux"] = params["max_flux"]
params["bias_min_period"] = params["min_period"]
params["bias_max_period"] = params["max_period"]
params["bias_conv_rate"] = params["conv_rate"]
results = []
with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    for i, starting_genome in enumerate(genome_list):
        print(f"Genome:{i}")
        # start_fitness = fitnessFunction(
        #     starting_genome,
        #     N=params["size"],
        #     generator_type=params["generator_type"],
        #     configuration=params["neuron_configuration"],
        # )
        # params["init_flux"] = getAmp(0.625, start_fitness, params["max_flux"])
        N = params["size"]
        for trial in range(num_trials):
            np.random.seed(np.random.randint(10000))
            if len(results) == 0:
                print_verbose = verbose
            else:
                print_verbose = -1
            results.append(
                executor.submit(
                    learn,
                    starting_genome,
                    **params,
                    tracking_parameters=tracking_parameters,
                    folderName=folderName,
                    track_fitness=track_fitness,
                    print_done=False,
                    trial=trial,
                    log_data=log_data,
                    verbose=print_verbose,
                    genome_num=i,
                    csv_name=csv_name,
                    stepsize=0.10,
                )
            )
            if len(results) == num_processes:
                for future in concurrent.futures.as_completed(results):
                    (filename := future.result())

                results = []
for future in concurrent.futures.as_completed(results):
    (filename := future.result())

print(folderName)
if visualize:
    pathname = f"./data/{folderName}"
    files = os.listdir(pathname)
    files = [name for name in files if ".npz" in name]
    filename = files[0].split(".")[0]
    print(filename)
    data = np.load(f"{pathname}/{files[0]}")
    Time = np.arange(0, data["duration"], data["stepsize"] / data["sample_rate"])
    if visualize:
        for tracked in vis_params:
            if "averaged" in tracked:
                tracked = tracked.split(" ")[-1]
                plotAverageParam(
                    tracked, show=False, b=-1, pathname=pathname, save=True
                )
            if "distribution" in tracked:
                tracked = tracked.split(" ")[-1]
                plotDistributionParam(
                    tracked, show=False, pathname=pathname, bins=10, save=True
                )

        if vis_behavior:
            plotBehavior(data, show=False, save=True)
        if vis_weights:
            plotWeightsBiases(data, show=False, extended=True, save=True)
        if vis_agent:
            plotChosenParam(
                pathname + "/" + filename + ".npz",
                params=[
                    "reward",
                    "flux_amp",
                    "distance",
                    "performance_average_hist",
                ],
                save=True,
            )
    plt.show()
    # else:
    #    plot_params = [p for p in vis_params if "averaged" not in p]
    #    plotChosenParam(pathname+"/"+filename(), params=plot_params)

# plotAverageParam('running_average', show=True, b=1000, pathname=pathname)
print(f"Finished in:{np.round(time.time()-start_time, 2)} seconds")
