import numpy as np
import os
from fitnessFunction import fitnessFunction
import pandas as pd

filename = "two_neuron-step1.npy"

genome_arr = np.load(filename)
best = genome_arr[3]
index = 0
lower_fitness = np.load("./scalinggenome.npy")

# uncomment to generate new scaling genome
#lower_fitness = np.zeros((best.shape))
#for i in range(lower_fitness[:-2].size):
#    lower_fitness[i] = - np.random.uniform(0, 0.2)
#np.save("scalinggenome.npy",  lower_fitness)

points = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
for i in range(10000):
    i+=529
    save_fitness = best+lower_fitness*2*i*(10**-3)
    fit = fitnessFunction(save_fitness)
    #print(i, end=" ", flush=False)
    print(np.round(fit, 3), end=" ", flush=False)

    for point in points:
        if fit<point+0.04 and fit> point-0.04:
            print(f"index{i}")
            print("saving fit of {} to {}".format(fit, 0.3))
            np.save("./durations/8000/{}/{}.npy".format(0.3, i), save_fitness)
