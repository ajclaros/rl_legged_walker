import ea
from fitnessFunction import fitnessFunction
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures

np.set_printoptions(formatter={"float": "{:.4f}".format})
cmap = plt.get_cmap("tab10").colors
# Nervous System Parameters
# Task Parameters
popsize = 50  # Population size
recombProb = 0.5  # Recombination probability
mutatProb = 0.05  # Mutation probability
demesize = 2  # Neighborhood size
generations = 300  # Number of generations
runs = 16
COMPARE = False

num_processes = 16
params = {
    "window_size": 440,  # unit seconds
    "learn_rate": 0.0005,
    "conv_rate": 0.0005,
    "min_period": 440,  # unit seconds
    "max_period": 4400,  # unit seconds
    "init_flux": 0.1,
    "max_flux": 0.125,
    "duration": 1,  # unit seconds
    "size": 3,
    "generator_type": "CPG",
    "neuron_configuration": [0, 1],
    "learning_start": 800,
    "record_every": 1,
    "stepsize": 0.1,
}

# Evolve and visualize fitness over generations
# x is an element from the full permutation of all lists
# creates dictionary for specific instance of values
# size = params["size"]
paramlist = [
    (2, [0, 1]),
]


def evolve(verbose=False, idx=0, p=0):
    print(idx)
    print("Params:")
    print(paramlist[p])
    (size, nc) = paramlist[p]
    genesize = size * size + 2 * size
    ga = ea.GaElite(
        fitnessFunction,
        popsize,
        genesize,
        recombProb,
        mutatProb,
        demesize,
        generations,
        generator_type=params["generator_type"],
        neuron_configuration=nc,
        size=size,
        verbose=verbose,
    )
    ga.run(savenp=True)
    ga.showFitness(c=cmap[1])


if COMPARE:
    ga = ea.GaElite(
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

else:
    for j, elt in enumerate(paramlist):
        results = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_processes
        ) as executor:
            for i in range(runs):
                np.random.seed(np.random.randint(10000))
                if i == 0:
                    verbose = True
                else:
                    verbose = False
                results.append(executor.submit(evolve, verbose, idx=i, p=j))

            for i, future in enumerate(concurrent.futures.as_completed(results)):
                future.result()

plt.title("Evolution vs Evolution and Learning")
plt.show()
