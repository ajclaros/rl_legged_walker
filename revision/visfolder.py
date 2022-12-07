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
import time

visualize = True
vis_behavior = False
vis_weights = False
vis_agent = False
vis_behavior = False
vis_weights = False
vis_agent = False

vis_params = [
    "averaged performance_hist",
    # "averaged performance_average_hist",
    # "distribution flux_amp",
    "distribution performance_hist",
    "distribution end_fitness",
    # "averaged flux_amp",
]
folderName = "test"

pathname = f"./data/{folderName}"
files = os.listdir(pathname)
files = [name for name in files if ".npz" in name]
filename = files[0].split(".")[0]
pathname = f"./data/{folderName}"
print(filename)
data = np.load(f"{pathname}/{files[0]}")
Time = np.arange(0, data["duration"], data["stepsize"] / data["sample_rate"])
if visualize:
    for tracked in vis_params:
        if "averaged" in tracked:
            tracked = tracked.split(" ")[-1]
            plotAverageParam(
                tracked,
                show=False,
                b=-1,
                pathname=pathname,
                save=True,
                baseline=True,
            )
        if "distribution" in tracked:
            print("distribution")
            tracked = tracked.split(" ")[-1]
            plotDistributionParam(
                tracked,
                show=False,
                pathname=pathname,
                b=-1,
                bins=10,
                save=True,
                baseline=True,
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
                "performance_hist",
            ],
            save=True,
        )
