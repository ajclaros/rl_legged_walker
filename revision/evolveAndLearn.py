import ea
from fitnessFunction import fitnessFunction
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(formatter={"float": "{:.4f}".format})
cmap = plt.get_cmap("tab10").colors
# Nervous System Parameters
# Task Parameters
popsize = 32  # Population size
recombProb = 1.0  # Recombination probability
mutatProb = 1.5  # Mutation probability
demesize = 2  # Neighborhood size
generations = 100  # Number of generations

num_processes = 16
params = {
    "window_size": 440,  # unit seconds
    "learn_rate": 0.001,
    "conv_rate": 0.001,
    "min_period": 440,  # unit seconds
    "max_period": 4400,  # unit seconds
    "init_flux": 0.1,
    "max_flux": 0.1,
    "duration": 2000,  # unit seconds
    "size": 3,
    "generator_type": "CPG",
    "neuron_configuration": [0, 1, 2],
    "learning_start": 600,
    "record_every": 1,
    "stepsize": 0.1,
}

# Evolve and visualize fitness over generations
# x is an element from the full permutation of all lists
# creates dictionary for specific instance of values
size = params["size"]
genesize = size * size + 2 * size
ga = ea.GaEliteLearn(
    fitnessFunction,
    popsize,
    genesize,
    recombProb,
    mutatProb,
    demesize,
    generations,
    generator_type=params["generator_type"],
    neuron_configuration=params["neuron_configuration"],
    size=size,
    num_processes=num_processes,
    params=params,
    num_trials=5,
)
ga.run()
ga.showFitness(c=cmap[0], label="evol and learning")

ga2 = ea.GaElite(
    fitnessFunction,
    popsize,
    genesize,
    recombProb,
    mutatProb,
    demesize,
    generations,
    generator_type=params["generator_type"],
    neuron_configuration=params["neuron_configuration"],
    size=size,
)
ga2.run(savenp=True)
ga2.showFitness(c=cmap[1], label="evolution")

plt.title("Evolution vs Evolution and Learning")
plt.show()
