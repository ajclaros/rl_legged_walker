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


param_list = {
    "window_size": [4000],
    "point": [0.4], #the starting fitnesses: "starting point"
    "learn_rate": [0.008],
    "conv_rate": [0.004],
    "min_period": [300],
    "max_period": [400],
    "init_flux": [0.35],
    "duration":[8000]

}
#times to try each element in the permutation of parameters
trials = 5
# size of network
N = 2

#parameters to track and their order
tracking_parameters = ["name", "init_flux", "starting_fitness", "end_fitness"]

# track everything
for key in param_list.keys():
    if key not in tracking_parameters:
        tracking_parameters.append(key)
# row to be appended at end of each iteration
row = dict()

# if log data is true: saves to "./data/durations/point/{end_fitness}.npy"
log_data = False

def get_data(path):
    """
    1. Reads in csv if exists, else data=None
    2. adds parameters not already tracked to tracking_parameters
    3. adds new tracked parameters not already listed in DF
    4. returns data as array
    """
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        [tracking_parameters.append(col) for col in df.columns if col not in tracking_parameters]
        for col in tracking_parameters:
            if col not in df.columns:
                df[col] = np.NAN
        df = df[[col for col in tracking_parameters]]
    else:
        df = None
    return df
df = get_data("./data/data.csv")

# itertools creates a list of all permutations for each list in the dictionary
# https://stackoverflow.com/questions/24594313/permutations-of-list-of-lists
for x in itertools.product(*param_list.values()):
    # x is an element of the full permutation of all lists
    # creates new dictionary for specific instance of values
    params = {}
    for i, key in enumerate(param_list.keys()):
        params[key] = x[i]

    # list of filenames for perturbed genomes
    starting_genomes = os.listdir(f"./durations/8000/{params['point']}")
    print(f"\n\nlocation: {params['point']}")
    for filename in starting_genomes:
        print(f"name:{filename}:")
        for key in params.keys():
            if key in tracking_parameters:
                print(f"{key}:{params[key]}", end=" ", flush=False)
        print(" ")
#        print(f"init_flux:{params['init_flux']}\ntrial:")

        # load in perturbed genome
        start = np.load(f"./durations/8000/{params['point']}/{filename}")
        starting_fitness = fitnessFunction(start)
        row["starting_fitness"] = starting_fitness
        for i in range(trials):
            print(f"{i}", end=" ", flush=False)
            learner = WalkingTask(size=N, duration=params["duration"], stepsize=0.1,
                                  reward_func=None,
                                  performance_func=None,
                                  running_window_mode=True,
                                  running_window_size= params["window_size"],
                                  performance_update_rate=0.05,
                                  init_flux_amp=params["init_flux"], max_flux_amp=40,
                                  flux_period_min=params["min_period"],
                                  flux_period_max=params["max_period"],
                                  flux_conv_rate=params["conv_rate"],
                                  learn_rate=params["learn_rate"],
                                  bias_init_flux_amp=params["init_flux"], bias_max_flux_amp=40,
                                  bias_flux_period_min=params["min_period"],
                                  bias_flux_period_max=params["max_period"],
                                  bias_flux_conv_rate=params["conv_rate"])
            weights = start[0:N*N]
            learner.setWeights(weights.reshape((N, N)))
            learner.setBiases(start[N*N:N*N+N])
            learner.setTimeConstants(start[N*N+N:])
            learner.initializeState(np.zeros(N))
            body = leggedwalker.LeggedAgent()
            learner.simulate(body, learning_start = 4000, trackpercent=1.00)
            if log_data:
                datalogger = DataLogger()
                datalogger.data.update({key:params[key] for key in tracking_parameters if key in params.keys()})
                learner.simulate(body, learning_start = 4000, datalogger=datalogger, trackpercent=1.00)
            else:
                learner.simulate(body, learning_start = 4000, trackpercent=1.00)
            end_fitness = fitnessFunction(learner.recoverParameters())
            if log_data:
                datalogger.data["end_fitness"] = end_fitness
                datalogger.save("./data/starting_fitness/{params['point']}/endfit-{int(np.round(end_fitness, 5)*100000)}")

            row["name"] = int(filename.split(".")[0])
            row["end_fitness"] = end_fitness
            row.update({key: params[key] for key in tracking_parameters if key in params.keys()})
            if df is None:
                df = pd.DataFrame(pd.Series(row)).T
            else:
                df = df.append(pd.Series(row), ignore_index=True)

csv = df.fillna(value=np.nan)
csv.to_csv("./data/data.csv")
