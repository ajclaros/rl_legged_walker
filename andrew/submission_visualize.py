import numpy as np
import matplotlib.pyplot as plt
import os
import dataloggervis as dv
from matplotlib import cm, colors
from collections import deque
from fitnessFunction import fitnessFunction

MAP= 'rainbow'
colormap = plt.get_cmap(MAP)

cmap= cm.viridis
files = os.listdir('./data/startingfitness/0.2/')
for name in files:
    data = np.load(f"./data/startingfitness/0.2/{name}")
    if data['duration']==4000:
        break
#data =np.load(f"./data/startingfitness/0.2/{files[0]}")
weight_hist =data['weightHist']
extended_weight_hist = data['extendedWeightHist']
c = np.linspace(0,400,len(weight_hist[: data['learningStart']]))

def plotWeightsBiases(data, show=False):
    cmap = plt.get_cmap('tab20').colors

    fig, ax = plt.subplots()
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
    if show:
        plt.show()
def plotNeuralOutputs(data, show=False):
    cmap= cm.jet
    learning_start = int(data['learningStart'])
    duration = data['duration']
    norm= colors.Normalize(vmin=0,  vmax=learning_start/10)
    sm = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
    fig, ax = plt.subplots(ncols=2)
    time = np.arange(0,learning_start/10, 0.1)
    c = np.linspace(0,learning_start/10, len(data['neuralOutputs'][:int(learning_start)]))
    fig.colorbar(sm, ticks=np.linspace(0, learning_start/10, 10), ax=ax[0])
    ax[0].scatter(data['neuralOutputs'].T[0][:learning_start], data['neuralOutputs'].T[1][:learning_start], c=c, cmap='jet')
    ax[0].set_xlabel("Neuron 0")
    ax[0].set_ylabel("Neuron 1")
    ax[0].set_title("Before Learning")
    norm= colors.Normalize(vmin=learning_start/10,  vmax=duration)
    sm = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
    c = np.linspace(learning_start/10, duration,len(data['neuralOutputs'][data['learningStart']:]))
    ax[1].scatter(data['neuralOutputs'].T[0][learning_start:], data['neuralOutputs'].T[1][learning_start:], c=c, cmap='jet')
    ax[1].set_title("After Learning")
    ax[1].set_xlabel("Neuron 0")
    ax[1].set_ylabel("Neuron 1")
    fig.colorbar(sm, ticks=np.linspace(learning_start/10, duration, 10), ax=ax[1])
    plt.suptitle("Neural Outputs")
    if show:
        plt.show()

def vis_frozen_fitness(data , show= False):
    #messy, but works and generalizes to any tracked percent
    track_percent = data['trackPercent']
    duration = data['duration']
    scale = track_percent *duration*10# scale window search size
    spotty_fitness = data['trackFitness']
    filled_fitness = np.lib.stride_tricks.sliding_window_view(spotty_fitness, int(scale)).sum(axis=1)
    smaller_time = np.arange(0, filled_fitness.size/10,0.1)
    time = np.arange(0, data['duration'], 0.1)
    plt.plot(time, data['runningAverage'])
    plt.plot(smaller_time, filled_fitness)
    if show:
        plt.show()

vis_frozen_fitness(data, show=True)
