import itertools
import leggedwalker
import numpy as np
from jason.rl_ctrnn import RL_CTRNN
from jason.ctrnn import CTRNN
from walking_task2 import WalkingTask
import warnings
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import datetime
from datalogger import DataLogger
from fitnessFunction import fitnessFunction
import pandas as pd
import seaborn as sns
from dataloggervis import visualize
import os



save_path = "./data/data5.csv"
save_csv = True
# size of network
N = 2
# save data in ./data/starting_fitness/{point}/end-fitness{end_fitness}.npy
log_data = True
# parameters to track and their order to save into csv
tracking_parameters = ["name", "init_flux", "starting_fitness", "end_fitness"]

# to track all parameters uncomment next 3 lines
#for key in param_list.keys():
#    if key not in tracking_parameters:
#        tracking_parameters.append(key)


#if logfitness==True: runs fitnessfunction every track_percent of the run
#if track_percent>0, also prints out the % of trial completed
log_fitness = False
track_percent = -1 #0.1

#if true, prints end fitness after every trial
print_end_fitness = False

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
    "duration": [2000],
}


# row to be appended to the csv at end of each iteration
row = dict()
# if log data is true: saves to "./data/durations/point/{end_fitness}.npy"
def get_data(path):
    """
    1. Reads in csv if exists, else data=None
    2. adds parameters not already tracked to tracking_parameters
    3. adds new tracked parameters not already listed in DF
    4. returns data as array
    """
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        [
            tracking_parameters.append(col)
            for col in df.columns
            if col not in tracking_parameters
        ]
        for col in tracking_parameters:
            if col not in df.columns:
                df[col] = np.NAN

        df = df[[col for col in tracking_parameters]]
    else:
        df = pd.DataFrame(columns=tracking_parameters)
    return df


if save_csv:
    df = get_data(save_path)


# itertools creates a list of all permutations for each list in the dictionary
# https://stackoverflow.com/questions/24594313/permutations-of-list-of-lists
for x in itertools.product(*param_list.values()):
    # x is an element from the full permutation of all lists
    # creates dictionary for specific instance of values
    params = {}
    for i, key in enumerate(param_list.keys()):
        params[key] = x[i]


    # list of filenames for perturbed genomes
    starting_genomes = os.listdir(f"./perturbations/{params['point']}")
    print(f"\n\nStartingPoint: {params['point']}")

    #iterate through each starting genome
    for i, filename in enumerate(starting_genomes):
        if i > 0:  # increase number if you want to increase the amount of genomes to
            break  # run the algorithm in. Comment out both lines to run on all genomes

        #print the parameters that are being tracked
        print(f"name:{filename}:")
        for key in params.keys():
            if key in tracking_parameters:
                print(f"{key}:{params[key]}", end=" ", flush=False)

        # load in perturbed genome
        start = np.load(f"./perturbations/{params['point']}/{filename}")
        starting_fitness = fitnessFunction(start)
        row["starting_fitness"] = starting_fitness
        print(f"\nTrial:")
        for i in range(trials):
            print(f" {i}", end=" ", flush=False)
            learner = WalkingTask(
                size=N,
                duration=params["duration"],
                stepsize=0.1,
                reward_func=None,
                performance_func=None,
                running_window_mode=True,
                running_window_size=params["window_size"],
                performance_update_rate=0.05,
                init_flux_amp=params["init_flux"],
                max_flux_amp=40,
                flux_period_min=params["min_period"],
                flux_period_max=params["max_period"],
                flux_conv_rate=params["conv_rate"],
                learn_rate=params["learn_rate"],
                bias_init_flux_amp=params["init_flux"],
                bias_max_flux_amp=40,
                bias_flux_period_min=params["min_period"],
                bias_flux_period_max=params["max_period"],
                bias_flux_conv_rate=params["conv_rate"],
            )
            weights = start[0 : N * N]
            learner.setWeights(weights.reshape((N, N)))
            learner.setBiases(start[N * N : N * N + N])
            learner.setTimeConstants(start[N * N + N :])
            learner.initializeState(np.zeros(N))
            body = leggedwalker.LeggedAgent()
            if log_data:
                datalogger = DataLogger()
                datalogger.data.update(
                    {key: params[key] for key in tracking_parameters if key in params.keys()}
                )
                learner.simulate(
                    body,
                    learning_start=4000,
                    datalogger=datalogger,
                    trackpercent=track_percent,
                    logfitness=log_fitness,
                )
            else:
                learner.simulate(body, learning_start=4000, trackpercent=-1)
            end_fitness = fitnessFunction(learner.recoverParameters())
            if print_end_fitness:
                print(f"endFitness: {end_fitness}")
            if log_data:
                # if data is being saves, save the end fiteness
                if log_fitness:
                    datalogger.data["trackPercent"] = track_percent
                datalogger.data["end_fitness"] = end_fitness
                # datalogger.save(f"./data/startingfitness/{params['point']}/endfit-{int(np.round(end_fitness, 5)*100000)}")
                datalogger.save(
                    f"./data/startingfitness/endfit-{int(np.round(end_fitness, 5)*100000)}"
                )

            row["name"] = filename.split(".")[0]
            row["end_fitness"] = end_fitness
            row.update(
                {
                    key: params[key]
                    for key in tracking_parameters
                    if key in params.keys()
                }
            )
            if df is None:
                df = pd.DataFrame(pd.Series(row)).T
            else:
                df = df.append(pd.Series(row), ignore_index=True)

        if save_csv:
            csv = df.fillna(value=np.nan)
            csv.to_csv(save_path)

if save_csv:
    csv = df.fillna(value=np.nan)
    csv.to_csv(save_path)

