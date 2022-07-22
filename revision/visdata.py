import numpy as np
import os
import matplotlib.pyplot as plt
files = os.listdir("data")
generator_type = "RPG"
files = [name for name in files if ".npz" in name  and generator_type in name]
data= ""
for i, name in enumerate(files):
    data= np.load(f"./data/{name}")
data = np.load("./data/noreward.npz")



def plotWeightsBiases(data, show=False, legend=True):
    cmap = plt.get_cmap('tab20').colors
    fig, ax = plt.subplots(figsize=(8,4))
    time = np.arange(0, data['duration'], 0.1)
    for i in range(data["size"]):
        for j in range(data['size']):
            ax.plot(time, data['extended_weights'].T[i,j],         color=cmap[i], label=f'w_{i}{j}')
            ax.plot(time, data['inner_weights'].T[i,j],                 color=cmap[i+1])

        ax.plot(time, data['biases'].T[i], color=cmap[i], ls='dotted', label='bias_0')
        ax.plot(time, data['extended_biases'].T[i], ls='dotted', color=cmap[i+1])
    ax.axvline(data['learning_start']*data['stepsize'], color='k', lw='1')
    ax.title.set_text(f"Weight and Bias change during Trial:{generator_type}")
    if show:
        if legend:
            plt.legend()
        plt.savefig("./data/images/weight-bias.png")
        plt.show()
def plotPerformance(data, show=False):
    fig, ax = plt.subplots(nrows=2, ncols=2)
    time = np.arange(0, data['duration'], 0.1)
    ax[0].plot(time, data['running_average_performances'])
    ax.axvline(data['learning_start']*data['stepsize'], color='k')
    ax.title.set_text(f"Average performance over time:{generator_type}\ntol:{data['tolerance']}")
    if show:
        plt.legend()
        plt.savefig("./data/images/weight-bias.png")
        plt.show()
def plotBehavior(data, show=False):
    fig, ax = plt.subplots(nrows=2, ncols=2)
    time = np.arange(0, data['duration'], 0.1)
    ax[0][0].plot(time, data['outputs'])
    ax[0][0].set_title("Neural outputs")
    ax[0][1].plot(time, data['distance'])
    ax[0][1].set_title("Distance")
    ax[1][0].plot(time, data['omega'])
    ax[1][0].set_title("Omega")
    ax[1][1].plot(time, data['angle'])
    ax[1][1].set_title("Angle")
    fig.suptitle(f"Duration:{data['duration']},\nStartFit:{data['start_fitness']}\nEndFit:{data['end_fitness']}\nSize:{data['size']}")
    if show:
        plt.show()


plotWeightsBiases(data, show=True)
plotBehavior(data, show=True)
