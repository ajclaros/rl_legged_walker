import numpy as np
import os
from datalogger import DataLogger
from learningFunction import learn
from pathlib import Path
from visdata import *
import concurrent.futures
import matplotlib.pyplot as plt
from fitnessFunction import fitnessFunction
import leggedwalker
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

# run a single configuration "num_trials" times
# verbose=-1: do not print
# verbose>=0: print out starting and ending fitness
# verbose in (0,1), print out progress of trial every % time passes for example

duration = 220
verbose = 0.1
log_data = True
record_csv = False
num_trials = 8
num_processes = 8
randomize_genomes = False
num_random_genomes = 1
# if visualize is true, print the parameters to visualize
# "averaged [param_name]" will print the average of the parameter across all trials
visualize = True
vis_behavior = False
vis_weights = True
vis_agent = False
vis_params = [
    "averaged performance_hist",
    # "distribution flux_amp",
    # "distribution performance_hist",
    # "averaged flux_amp",
]
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
    "window_size": 800,  # unit seconds
    "learn_rate": 0.800,
    "conv_rate": 0.800,
    "min_period": 440,  # unit seconds
    "max_period": 4400,  # unit seconds
    "init_flux": 0.0010000,
    "max_flux": 0.0050000,
    "duration": 4000,  # unit seconds
    "size": 3,
    "generator_type": "RPG",
    "tolerance": 0.00000000,
    "neuron_configuration": [0],
    "learning_start": 200,
    "record_every": 1,
    "stepsize": 0.1,
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
# folderName = f"{params['generator_type']}_d{params['duration']}_initfx{params['init_flux']}_00_window{params['window_size']}_max_p{params['max_period']}"
# folderName += "recording"
folderName = "test"
if not os.path.exists(Path(f"./data/{folderName}")):
    print(f"creating folder:{folderName}")
    os.mkdir(f"./data/{folderName}")


N = params["size"]
genome_list = []
# hard coded genomes
load_genome = np.load("./evolved/fit-3426.5.npy")
if not randomize_genomes:
    # size = 3
    genome_list.append(load_genome)
    genome_list = np.array(genome_list)
    starting_genome = genome_list[0]
else:
    genome_list = np.random.uniform(-1, 1, size=(num_random_genomes, N * N + 2 * N))

import leggedwalker
from walking_task import WalkingTask

learner = WalkingTask(
    duration=params["duration"],
    size=params["size"],
    stepsize=params["stepsize"],
    running_window_mode=True,
    running_window_size=params["window_size"],
    init_flux_amp=params["init_flux"],
    max_flux_amp=params["max_flux"],
    flux_period_min=params["min_period"],
    flux_period_max=params["max_period"],
    flux_conv_rate=params["conv_rate"],
    learn_rate=params["learn_rate"],
    bias_init_flux_amp=params["init_flux"],
    bias_max_flux_amp=params["max_flux"],
    bias_flux_period_min=params["min_period"],
    bias_flux_period_max=params["max_period"],
    bias_flux_conv_rate=params["conv_rate"],
    record_every=params["record_every"],
)
datalogger = DataLogger()
size = params["size"]
weights = starting_genome[0 : size * size]
learner.setWeights(weights.reshape((size, size)))
learner.setBiases(starting_genome[size * size : size * size + size])
learner.setTimeConstants(starting_genome[size * size + size :])
learner.initializeState(np.zeros(size))
body = leggedwalker.LeggedAgent()

time = np.arange(0, duration, 0.1)
x = np.arange(-5, 0, 0.1)
y_vec = np.zeros(x.size)

line1 = []


def live_plotter(x_vec, y1_data, line1, identfier="", pause_time=0.1):
    if line1 == []:
        plt.ion()
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        (line1,) = ax.plot(x_vec, y1_data, "-o", alpha=0.8)
        plt.ylabel = "time"
        plt.title("Distance traveled v time")
        plt.ylim((0, 1))
        plt.show()

    line1.set_ydata(y1_data)
    line1.set_xdata(x_vec)
    plt.xlim((x_vec[0], x_vec[-1]))
    plt.pause(pause_time)
    return line1


past_dist = 0
for i in time:
    # learner.step(0.1)
    learner.setInputs(np.array([body.anglefeedback()] * 3))
    learner.step(0.1)
    body.stepN(0.1, learner.outputs, params["neuron_configuration"])
    y_vec[-1] = body.cx - past_dist
    past_dist = body.cx
    if i - int(i) == 0 and i % 5 == 0:
        line1 = live_plotter(x, y_vec, line1)
    x = np.roll(x, -1)
    x[-1] = x[-2] + 0.1
    y_vec = np.roll(y_vec, -1)
print(body.cx / duration)
