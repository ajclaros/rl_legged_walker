import ea
import ctrnn
import leggedwalker
import numpy as np
import itertools
from fitnessFunction import fitnessFunction

import matplotlib.pyplot as plt
from matplotlib import cm, colors
# Nervous System Parameters
N = 2      # Number of neurons in the nervous system
WR = 16    # Weight range - maps from [-1, 1] to: [-16,16]
BR = 16    # Bias range - maps from [-1, 1] to: [-16,16]
TR = 5.0   # Time range - maps from [-1, 1] to: [-5, 5]
TA = 6.0   # Time add - maps from [-5, 5] to: [1,11]

# Task Parameters

popsize = 50# Population size
recombProb = 0.5        # Recombination probability
mutatProb = 0.5         # Mutation probability
demesize = 2            # Neighborhood size
generations = 200# Number of generations

cmap = plt.get_cmap('tab10').colors
param_list = {
    "size": [4],
    "generator_type":["CPG", "RPG"],
    "neuron_configuration":[[0, 1,2]]
}

# Evolve and visualize fitness over generations

for i, x in enumerate(itertools.product(*param_list.values())):
    # x is an element from the full permutation of all lists
    # creates dictionary for specific instance of values
    params = {}
    for j, key in enumerate(param_list.keys()):
        params[key] = x[j]
    size = params['size']
    if len(params['neuron_configuration'])>size:
        continue
    genesize = size*size + 2*size
    ga = ea.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demesize, generations, generator_type = params['generator_type'], neuron_configuration=params['neuron_configuration'], size = size)
    ga.run()
    ga.showFitness(c = cmap[i])
    ga.save()
plt.show()
