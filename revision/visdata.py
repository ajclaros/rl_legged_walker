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



def plotWeightsBiases(data, show=False):
    cmap = plt.get_cmap('tab20').colors
    fig, ax = plt.subplots(figsize=(8,4))
    time = np.arange(0, data['duration'], 0.1)
    ax.plot(time, data['extended_weights'].T[0,0],         color=cmap[0], label='w_00')
    ax.plot(time, data['inner_weights'].T[0,0],                 color=cmap[1 ])
    ax.plot(time, data['extended_weights'].T[0,1],         color=cmap[2], label='w_01')
    ax.plot(time, data['inner_weights'].T[0,1],                 color=cmap[3 ])
    ax.plot(time, data['extended_weights'].T[1,0],         color=cmap[4], label='w_10')
    ax.plot(time, data['inner_weights'].T[1,0],                 color=cmap[5 ])
    ax.plot(time, data['extended_weights'].T[1,1],         color=cmap[6], label='w_11')
    ax.plot(time, data['inner_weights'].T[1,1],                 color=cmap[7 ])
    ax.plot(time, data['biases'].T[0], color=cmap[8], label='bias_0')
    ax.plot(time, data['extended_biases'].T[0], color=cmap[9])
    ax.plot(time, data['biases'].T[1], color=cmap[10], label="bias_1")
    ax.plot(time, data['extended_biases'].T[1], color=cmap[11])
    ax.axvline(data['learning_start']*data['stepsize'], color='k', lw='1')
    ax.title.set_text(f"Weight and Bias change during Trial:{generator_type}")
    if show:
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
