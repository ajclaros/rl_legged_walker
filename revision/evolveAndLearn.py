import ea
from fitnessFunction import fitnessFunction
import matplotlib.pyplot as plt
cmap = plt.get_cmap("tab10").colors
# Nervous System Parameters
# Task Parameters
popsize = 16# Population size
recombProb = 0.5  # Recombination probability
mutatProb = 0.01  # Mutation probability
demesize = 2  # Neighborhood size
generations = 100  # Number of generations

num_processes = 16
params = {
    "window_size": 440,  # unit seconds
    "learn_rate": 0.8,
    "conv_rate": 0.8,
    "min_period": 440,  # unit seconds
    "max_period": 4400,  # unit seconds
    "init_flux": 0.5,
    "max_flux": 2.5,
    "duration": 4000,  # unit seconds
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
    num_processes=num_processes,
    params=params,

)
ga.run()
ga.showFitness(c=cmap[0], label = "evol and learning")

ga2 = ea.Microbial2(
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
ga2.run()
ga2.showFitness(c=cmap[1], label='evolution')

plt.title(
    "Evolution vs Evolution and Learning"
)
plt.show()
