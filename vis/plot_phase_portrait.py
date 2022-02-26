
from jason.ctrnn import CTRNN
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
import sys
import json
import os
from util.fitness_functions import fitness_maximize_output_change, fitness_frequency_match




def plot_grid_freq_fit(target_period=2.5,min=-16,max=16,inc=0.1, size=2, seed=1,stepsize=0.01):
    eval_duration=50
    filename = f"data/evolved_solutions/mga_pop-20_gen-1000/ALL/discovery_mga_best_nn{size}_seed-{seed}.json"  
    frozen_ctrnn = CTRNN( size)
    frozen_ctrnn.load_json( filename )
    weights00 = np.arange(min,max,inc)
    weights11 = np.arange(min,max,inc)
    for w00 in weights00:
        print(w00)
        for w11 in weights11:
            frozen_ctrnn.inner_weights[0][0] = w00
            frozen_ctrnn.inner_weights[1][1] = w11
            fit, freq_perf, change_perf = fitness_frequency_match(frozen_ctrnn,target_period=target_period, stepsize=stepsize, test_duration=eval_duration)
            if fit > 1.0:
                fit = 1.0
            plt.scatter(w00,w11,color=(fit, fit, fit))
    
    #TODO show bias0 vs. bias1 
    #TODO show w01 vs. w10
    

def plot_grid_max_change_no_transients_fit(min=16,max=16,inc=0.1, size=2, seed=1,stepsize=0.01):
    filename = f"data/evolved_solutions/mga_pop-20_gen-1000/ALL/discovery_mga_best_nn{size}_seed-{seed}.json"  
    frozen_ctrnn = CTRNN( size)
    frozen_ctrnn.load_json( filename )
    weights00 = np.arange(min,max,inc)
    weights11 = np.arange(min,max,inc)
    for w00 in weights00:
        print(w00)
        for w11 in weights11:
            frozen_ctrnn.inner_weights[0][0] = w00
            frozen_ctrnn.inner_weights[1][1] = w11
            fit = fitness_maximize_output_change(frozen_ctrnn, init_duration=250, test_duration=50)
            plt.scatter(w00,w11,color=(fit, fit, fit))

def plot_grid_max_change_fit(min=-16,max=16,inc=0.1, duration=1, size=2, seed=1,stepsize=0.01):
    filename = f"data/evolved_solutions/mga_pop-20_gen-1000/ALL/discovery_mga_best_nn{size}_seed-{seed}.json"  
    frozen_ctrnn = CTRNN( size)
    frozen_ctrnn.load_json( filename )
    weights00 = np.arange(min,max,inc)
    weights11 = np.arange(min,max,inc)
    for w00 in weights00:
        print(w00)
        for w11 in weights11:
            frozen_ctrnn.inner_weights[0][0] = w00
            frozen_ctrnn.inner_weights[1][1] = w11
            fit = fitness_maximize_output_change(frozen_ctrnn,test_duration=duration,stepsize=stepsize)
            plt.scatter(w00,w11,color=(fit, fit, fit))

def plot_freq_fitness(inc=0.25, stepsize=0.01, size=2, seeds=[1], target_periods=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10], show_plots=True):
    
    for seed in seeds:
        for target_period in target_periods:
                plt.figure()
                plot_grid_freq_fit(target_period=target_period, min=-16,max=16,inc=inc,stepsize=stepsize, size=size, seed=seed)
                plt.legend()
                plt.xlabel("w00")
                plt.ylabel("w11")
                plt.title(f"Fitness by weight values for target period={target_period}")
                plt.savefig(f"vis/plots/phase_portraits/freq_match/target_freq_period-{target_period}_seed-{seed}_inc-{inc}.png")
                if show_plots:
                    plt.show()

def plot_max_change_fitness(inc=0.25, stepsize=0.01, size=2, seeds=[1], durations=[1,2,3,4,5,6,7,8,9,10], show_plots=True):
    
    for seed in seeds:
        for duration in durations:
            plt.figure()
            plot_grid_max_change_fit(min=-16,max=16,inc=inc,stepsize=stepsize, size=size, seed=seed, duration=duration)
            plt.legend()
            plt.xlabel("w00")
            plt.ylabel("w11")
            plt.title(f"Fitness by weight values")
            plt.savefig(f"vis/plots/phase_portraits/max_change/seed-{seed}_size-{size}_phase_portrait_eval-{duration}_inc-{inc}.png")
            if show_plots:
                plt.show()

def plot_max_change_no_transients_fitness(inc=0.25, stepsize=0.01, size=2, seeds=[1], show_plots=True):
    
    for seed in seeds:
        plt.figure()
        plot_grid_max_change_no_transients_fit(min=-16,max=16,inc=inc,stepsize=stepsize, size=size, seed=seed)
        plt.legend()
        plt.xlabel("w00")
        plt.ylabel("w11")
        plt.title(f"Fitness by weight values")
        plt.savefig(f"vis/plots/phase_portraits/max_change_no_transients/seed-{seed}_size-{size}_phase-portrait-no_transients_inc-{inc}.png")
        if show_plots:
            plt.show()

seeds=[6]
#incs=[ 0.2, 0.1]
incs=[1,0.5, 0.2]
#eval_durations=[1,2,3,4,5,6,7,8,9,10]
eval_durations=[10]
#target_periods=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10]
#target_periods=[1, 2, 4, 8 ]
#target_periods=[ 1,2,3,4 ]

for inc in incs:
    show_plots=False
    plot_max_change_fitness(inc=inc,seeds=seeds, durations=eval_durations, show_plots=show_plots)
    #plot_freq_fitness(inc=inc,seeds=seeds, target_periods=target_periods, show_plots=show_plots)
    

for inc in incs:
    show_plots=False
    plot_max_change_no_transients_fitness(inc=inc,seeds=seeds, show_plots=show_plots)

