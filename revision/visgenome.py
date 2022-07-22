from fitnessFunction import fitnessFunction
import matplotlib.pyplot as plt
import numpy as np
from datalogger import DataLogger
import os

# enter CPG or RPG to visualize the generator with cofigurations "neuron_configuration", "size"
generator_type = "RPG"
neuron_configuration = [0, 1, 2]
size = 4

RPG = np.load(f"./data/microbial/genomes/RPG-s{size}-c{'_'.join(str(n) for n in neuron_configuration)}.npz")
CPG = np.load(f"./data/microbial/genomes/CPG-s{size}-c{'_'.join(str(n) for n in neuron_configuration)}.npz")
files = os.listdir("./data/microbial")

RPG_fit = fitnessFunction(RPG['bestind'], N=size, generator_type='RPG', configuration = neuron_configuration, stepsize = 0.01, record=True)
CPG_fit = fitnessFunction(CPG['bestind'], N=size, generator_type='CPG', configuration = neuron_configuration, stepsize = 0.01, record=True)

CPG_filename = [name for name in files if "CPG" in name and ".npz" in name][0]
RPG_filename = [name for name in files if "RPG" in name and ".npz" in name][1]
behavior = {}
behavior['CPG']= np.load(f"./data/microbial/{CPG_filename}")
behavior["RPG"]= np.load(f"./data/microbial/{RPG_filename}")
time = np.arange(0,220, 0.01)
fig, ax = plt.subplots(nrows = 2, ncols=2)

ax[0][0].plot(time, behavior[generator_type]['outputs'])
ax[0][0].set_title("Neural outputs")
ax[0][1].plot(time, behavior[generator_type]['distance'])
ax[0][1].set_title("Distance")
ax[1][0].plot(time, behavior[generator_type]['omega'])
ax[1][0].set_title("Omega")
ax[1][1].plot(time, behavior[generator_type]['angle'])
ax[1][1].set_title("Angle")
fig.suptitle(f"{generator_type}, fitness:{CPG_fit}")
plt.savefig(f"./data/microbial/behavior-{generator_type}-s{size}-c{'_'.join(str(num) for num in neuron_configuration)}")
plt.show()
