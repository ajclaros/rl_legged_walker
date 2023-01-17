import ea
import ctrnn
import leggedwalker
import numpy as np
import itertools
from fitnessFunction import fitnessFunction
import time

import matplotlib.pyplot as plt
from matplotlib import cm, colors

# Nervous System Parameters
# Task Parameters

popsize = 100  # Population size
recombProb = 0.5  # Recombination probability
mutatProb = 0.05  # Mutation probability
demesize = 2  # Neighborhood size
generations = 100  # Number of generations

cmap = plt.get_cmap("tab10").colors
param_list = {
    "size": [3],
    "generator_type": ["CPG"],
    "neuron_configuration": [[0, 1]],
}

# Evolve and visualize fitness over generations

for i, x in enumerate(itertools.product(*param_list.values())):
    # x is an element from the full permutation of all lists
    # creates dictionary for specific instance of values
    params = {}
    for j, key in enumerate(param_list.keys()):
        params[key] = x[j]
    size = params["size"]
    if len(params["neuron_configuration"]) > size:
        continue
    genesize = size * size + 2 * size
    ga = ea.Microbial(
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
        verbose=True,
    )

    print("Starting evolution")
    t1 = time.time()
    ga.run()
    ga.showFitness(c=cmap[i])
    print("Time taken: ", time.time() - t1)

plt.title(
    f"Microbial: Best and average fitness\nN:{size},neuron_config:{param_list['neuron_configuration'][0]}"
)
ga.showFitness(c=cmap[1], save=True)
plt.show()
