from fitnessFunction import fitnessFunction
import matplotlib.pyplot as plt
import numpy as np
from datalogger import DataLogger
size = 4
generator_type = "CPG"#, "RPG",
neuron_configuration = [0, 1, 2]

RPG = np.load('./data/microbial/genomes/RPG-s4-c0_1_2.npz')
CPG = np.load('./data/microbial/genomes/CPG-s4-c0_1_2.npz')

RPG_fit = fitnessFunction(RPG['bestind'], N=size, generator_type='RPG', configuration = neuron_configuration, stepsize = 0.01, record=True)
CPG_fit = fitnessFunction(CPG['bestind'], N=size, generator_type='CPG', configuration = neuron_configuration, stepsize = 0.01, record=True)
CPG= np.load("./data/microbial/behavior-CPG-17438-s4-c0_1_2.npz")
RPG= np.load("./data/microbial/behavior-RPG-46928-s4-c0_1_2.npz")
time = np.arange(0,220, 0.01)

fig, ax = plt.subplots(nrows = 2, ncols=2)
ax[0][0].plot(time, CPG['outputs'])
ax[0][0].set_title("Neural outputs")
ax[0][1].plot(time, CPG['distance'])
ax[0][1].set_title("Distance")
ax[1][0].plot(time, CPG['omega'])
ax[1][0].set_title("Omega")
ax[1][1].plot(time, CPG['angle'])
ax[1][1].set_title("Angle")
fig.suptitle(f"{generator_type}, fitness:{CPG_fit}")
plt.savefig(f"./data/microbial/behavior-{generator_type}-s{size}-c{'_'.join(str(num) for num in neuron_configuration)}")
plt.show()
