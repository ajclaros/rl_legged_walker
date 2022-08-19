import numpy as np
import matplotlib.pyplot as plt
import os
import dataloggervis as dv
from matplotlib import cm, colors
from collections import deque
from fitnessFunction import fitnessFunction
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
MAP= 'rainbow'
colormap = plt.get_cmap(MAP)

cmap= cm.viridis
files = os.listdir('./data/startingfitness/')
for i, name in enumerate(files):
    if 'git' in name:
        continue
    data = np.load(f"./data/startingfitness/{name}")
    if i==0:
        break
#data =np.load(f"./data/startingfitness/0.2/{files[0]}")
weight_hist =data['weightHist']
extended_weight_hist = data['extendedWeightHist']
c = np.linspace(0,400,len(weight_hist[: data['learningStart']]))

def plotWeightsBiases(data, show=False):
    cmap = plt.get_cmap('tab20').colors
    fig, ax = plt.subplots(figsize=(8,4))
    time = np.arange(0, data['duration'], 0.1)
    ax.plot(time, data['extendedWeightHist'].T[0,0],         color=cmap[0], label='w_00')
    ax.plot(time, data['weightHist'].T[0,0],                 color=cmap[1 ])
    ax.plot(time, data['extendedWeightHist'].T[0,1],         color=cmap[2], label='w_01')
    ax.plot(time, data['weightHist'].T[0,1],                 color=cmap[3 ])
    ax.plot(time, data['extendedWeightHist'].T[1,0],         color=cmap[4], label='w_10')
    ax.plot(time, data['weightHist'].T[1,0],                 color=cmap[5 ])
    ax.plot(time, data['extendedWeightHist'].T[1,1],         color=cmap[6], label='w_11')
    ax.plot(time, data['weightHist'].T[1,1],                 color=cmap[7 ])
    ax.plot(time, data['biasHist'].T[0], color=cmap[8], label='bias_0')
    ax.plot(time, data['extendedBiasHist'].T[0], color=cmap[9])
    ax.plot(time, data['biasHist'].T[1], color=cmap[10], label="bias_1")
    ax.plot(time, data['extendedBiasHist'].T[1], color=cmap[11])
    ax.title.set_text("Weight and Bias change during Trial")
    if show:
        plt.legend()
        plt.savefig("./images/weight-bias.png")
        plt.show()

def vis_frozen_fitness(data, show=False): 
    fig, ax = plt.subplots(figsize= (3,3))
    #messy, but works and generalizes to any tracked percent
    track_percent = data['trackPercent']
    duration = data['duration']
    scale = track_percent *duration*10# scale window search size
    spotty_fitness = data['trackFitness']
    filled_fitness = np.lib.stride_tricks.sliding_window_view(spotty_fitness, int(scale)).sum(axis=1)
    smaller_time = np.arange(0, filled_fitness.size/10,0.1)
    time = np.arange(0, data['duration'], 0.1)
    ax.plot(time, data['runningAverage'])
    ax.plot(smaller_time, filled_fitness)
    ax.title.set_text("Runnning Average Performance\nvs.\nFrozen Fitness")
    ax.set_xlabel("Time")
    ax.set_ylabel("Fitness/Running Average Performance")
    if show:
        plt.show()
#        plt.savefig("./images/frozenfitness.png", bbox_inches='tight', dpi=600)

def plot_NeuralOutputs(data, show=False):
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6,3))
    points = np.array(data['neuralOutputs'][:data['learningStart']].T).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    time = np.arange(0, data['learningStart']/10, 0.1)
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(time.min(), time.max())
    lc = LineCollection(segments, cmap='rainbow', norm=norm, linewidths=20)
    # Set the values used for colormapping
    lc.set_array(time)
    lc.set_linewidth(4)
    line = ax[0].add_collection(lc)
    fig.colorbar(line, ax=ax[0])
    points = np.array(data['neuralOutputs'][data['learningStart']:].T).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    time = np.arange(data['learningStart']/10, data['duration'], 0.1)
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(time.min(), time.max())
    lc = LineCollection(segments, cmap='rainbow', norm=norm, linewidths=20)
    # Set the values used for colormapping
    lc.set_array(time)
    lc.set_linewidth(4)
    line = ax[1].add_collection(lc)
    fig.colorbar(line, ax=ax[1])
    ax[0].set_xlabel('Neuron 0')
    ax[0].set_ylabel('Neuron 1')
    ax[1].set_xlabel('Neuron 0')
    ax[1].set_ylabel('Neuron 1')
    plt.suptitle("Neural Outputs")
    if show==True:
        plt.savefig("./images/neuraloutputs.png", dpi=600, bbox_inches='tight')
        plt.show()
def frozen_fitness(data, show=True):
    time = np.arange(0, data['duration'], 0.1)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time, data['runningAverage'], label='Average Performance')
    ax.plot(time, data['trackFitness'], label='Frozen Fitness')
    if show:
        ax.title.set_text("Performance vs. Fitness")
        plt.legend()
        plt.savefig("./images/fitness-perf.png")
        plt.show()
#frozen_fitness(data, show=True)
#vis_frozen_fitness(data, show=True)
time = np.arange(0, 4000, 0.1)
frozen_fitness(data, show=True)
plotWeightsBiases(data, show=True)
