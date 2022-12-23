import os

import ea
from fitnessFunction import fitnessFunction
import numpy as np
import concurrent.futures

# TODO finish automatic saving and add into single_trial
np.set_printoptions(formatter={"float": "{:.4f}".format})
popsize = 50  # Population size
recombProb = 0.5  # Recombination probability
mutatProb = 0.05  # Mutation probability
demesize = 2  # Neighborhood size
generations = 300  # Number of generations
runs = 16

num_processes = 1
min_files_in_range = 100
paramlist = [
    ("RPG", 3, [0, 1], (0.16, 0.3))  # Generator, size, neuron configuration, fit range
]


def evolve(
    verbose=False,
    idx=0,
    p=0,
    generator="RPG",
    size=2,
    conf=[0],
    fit_range=(0, 1),
    pathname=None,
):
    genesize = size * size + 2 * size
    np.random.seed()
    ga = ea.Microbial(
        fitnessFunction,
        popsize,
        genesize,
        recombProb,
        mutatProb,
        demesize,
        generations,
        generator_type=generator,
        neuron_configuration=conf,
        pathname=pathname,
        size=size,
        verbose=verbose,
        fit_range=fit_range,
    )
    ga.run(savenp=True)


def checkdir(pathname, fit_range, min_files=100):
    (generator, size, conf_str) = pathname.split("/")[2:]
    size = int(size)
    conf = list(map(int, conf_str))
    run = False
    if not os.path.exists(pathname):
        print(f"Creating:{pathname}")
        os.makedirs(pathname)
        run = True
    else:
        files = os.listdir(pathname)
        fitnesses = [float(name.split("-")[1].split(".")[0]) / 100000 for name in files]
        files_in_range = [
            fit for fit in fitnesses if fit > fit_range[0] and fit < fit_range[1]
        ]
        if len(files_in_range) < min_files:
            run = True

        if run:
            print("Running initial genome generation.")
            print("--------------------------")
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
                    results.append(
                        executor.submit(
                            evolve,
                            verbose,
                            idx=i,
                            generator=generator,
                            size=size,
                            conf=conf,
                            fit_range=fit_range,
                            pathname=pathname,
                        )
                    )

                for i, future in enumerate(concurrent.futures.as_completed(results)):
                    future.result()
        else:
            print("Skipping genome generation")
            print("--------------------------")
