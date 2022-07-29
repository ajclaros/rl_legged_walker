import numpy as np
import os
from fitnessFunction import fitnessFunction
import pandas as pd

#filename = "two_neuron-step1.npy"

genome_arr = np.load(filename)
best = genome_arr[3]
#index = 0
select_genome = 7
#lower_fitness = np.load("./genomes/scalinggenome-{select_genome}.npy")
# uncomment to generate new scaling genome

for j in range(1):
    #lower_fitness = np.zeros((best.shape))
    #for i in range(lower_fitness.size):
    #    lower_fitness[i] = - np.random.uniform(0, 0.4)
    #np.save(f"./genomes/scalinggenome-{select_genome}.npy",lower_fitness)
    #genome9 start 6130
    #delete 7
    print(j)
    points = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for i in range(10000):
        save_fitness = best+lower_fitness*(i*10**-4)
        fit = fitnessFunction(save_fitness)
        print(fit)
        #print(fit, end=" ", flush=False)
        point_delete = []
        if not (i % 100):
            print(i, end=' ', flush=False)
        for point in points:
            if fit<point+0.02 and fit> point-0.02:
                print(f"index{i}")
                print("saving fit of {} to {}".format(fit, point))
    #            np.save(f"./perturbations/{point}/p-{select_genome}-{i}.npy", save_fitness)
                point_delete.append(points.index(point))
                break
        for ind in point_delete:
            points.pop(ind)
    print(f"points left:{points}")

#find genome with fitness in range 0.1 genome 5,
#lowest fitness before before given range
