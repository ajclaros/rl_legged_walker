import numpy as np
import os
import matplotlib.pyplot as plt
files = os.listdir()
data= ""
generator_type = "RPG"
data = ""
for name in files:
    if generator_type and "npz" in name:
        print(name)
        data= np.load(f"{name}")


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
    ax.plot(time, data['inner_biases'].T[0], color=cmap[8], label='bias_0')
    ax.plot(time, data['extended_biases'].T[0], color=cmap[9])
    ax.plot(time, data['inner_biases'].T[1], color=cmap[10], label="bias_1")
    ax.plot(time, data['extended_biases'].T[1], color=cmap[11])
    ax.title.set_text("Weight and Bias change during Trial")
    if show:
        plt.legend()
        plt.savefig("./images/weight-bias.png")
        plt.show()

plotWeightsBiases(data, show=True)
