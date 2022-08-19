import ea
import leggedwalker
import numpy as np
import math
from jason.rl_ctrnn import RL_CTRNN
from jason.ctrnn import CTRNN
from walking_task2 import WalkingTask
import warnings
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import time
from matplotlib.colors import ListedColormap
import concurrent.futures #multiprocessing


np.seterr(all='warn')
warnings.simplefilter("always")
# Nervous System Parameters
N = 2 # Number of neurons in the nervous system
WR = 16    # Weight range - maps from [-1, 1] to: [-16,16]
BR = 16    # Bias range - maps from [-1, 1] to: [-16,16]
TR = 5.0   # Time range - maps from [-1, 1] to: [-5, 5]
TA = 6.0   # Time add - maps from [-5, 5] to: [1,11]

# Task Parameters
stepsize = 0.1
#time = np.arange(0.0, duration, stepsize)

x = cm.get_cmap('tab10')
colors = x.colors
duration = 2000
def fitnessFunction(genotype):
    # Create the agent's body
    legged = leggedwalker.LeggedAgent()
    # Create the nervous system
    ns = CTRNN(N)
    # Set the parameters of the nervous system according to the genotype-phenotype map
    weights = genotype[0:N*N]
    ns.setWeights(weights.reshape((N, N)))
    ns.setBiases(genotype[N*N:N*N+N])
    ns.setTimeConstants(genotype[N*N+N:])
    # Initialize the state of the nervous system to some value
    ns.initializeState(np.zeros(N))
    #learner = RL_CTRNN(ns)
    # Loop through simulated time, use Euler Method to step the nervous system and body
    time = np.arange(0.0, duration, stepsize)
    for i, t in enumerate(time):
        ns.setInputs(np.array([legged.anglefeedback()]*N))  # Set neuron input to angle feedback based on current body state
        ns.step(stepsize)                               # Update the nervous system based on inputs
        legged.step1(stepsize, ns.outputs)                  # Update the body based on nervous system activity
#        fitness_arr[i] = body.cx                        # track position of body
                                                        #update neurons based on speed of movement (cx(t)-cx(t-1))/dt
    # Calculate the fitness based on distance covered over the duration of time
#    fit = legged.cx/duration
    return legged.cx/duration


popsize = 25
genesize = N*N + 2*N
recombProb = 0.5
mutatProb = 0.01
demesize = 2
generations = 30




init_flux = 0.1
def learningFunction(genotype):
    weights = genotype[0:N*N]
    learner = WalkingTask(size=2,
                          duration=duration,
                          stepsize=0.1,
                          reward_func=None,
                          performance_func=None,
                          running_window_mode=True,
                          running_window_size=4000,
                          performance_update_rate=0.05,
                          init_flux_amp= init_flux,
                          max_flux_amp=40,
                          flux_period_min=300,
                          flux_period_max=400,
                          flux_conv_rate=0.004, learn_rate=0.008,
                          bias_init_flux_amp=init_flux,
                          bias_max_flux_amp=40,
                          bias_flux_period_min=300,
                          bias_flux_period_max=400,
                          bias_flux_conv_rate=0.004,
                          )
    learner.setWeights(weights.reshape((N, N)))
    learner.setBiases(genotype[N*N:N*N+N])
    learner.setTimeConstants(genotype[N*N+N:])
    learner.initializeState(np.zeros(N))
    body = leggedwalker.LeggedAgent()
    learner.simulate(body, learning_start=4000, trackpercent=1.00)
    return body.cx/duration


#create dictionary of 10 parallel processes
#each process is alternating between evo and evo+learn
#num_process = 4
#function = {'evo': fitnessFunction, 'learn':learningFunction}
#function_keys = list(function.keys())
#genetic = {function_keys[i%2]+f'{i//2}':ea.Microbial(function[function_keys[i%2]], popsize, genesize, recombProb, mutatProb, demesize, generations) for i in range(num_process)}
#with concurrent.futures.ProcessPoolExecutor() as executor:
#    s = [executor.submit(genetic[function_keys[i%2]+f"{i//2}"].run) for i in range(num_process)]
#    for p in s:
#        print('done')
#style = ['-', ':']
#
#results = dict()
#for i in range(2):
#    results[function_keys[i%2]+'best'] = []
#    results[function_keys[i%2]+'avg'] = []
#
#
#for i in range(num_process):
#    plt.plot(genetic[function_keys[i%2]+f"{i//2}"].bestHistory, label=function_keys[i%2], color='r', ls = style[i%2])
#    results[function_keys[i%2]+'best'].append(genetic[function_keys[i%2]+f"{i//2}"].bestHistory)
#    plt.plot(genetic[function_keys[i%2]+f"{i//2}"].avgHistory, label=function_keys[i%2], color='k', ls = style[i%2])
#    results[function_keys[i%2]+'avg'].append(genetic[function_keys[i%2]+f"{i//2}"].avgHistory)
#plt.plot(np.mean(results['evobest']), label='avgEvoBest', color='c', ls=':')
#plt.plot(np.mean(results['evoavg']), label='avgEvoAvg', color='y', ls=':')
#plt.plot(np.mean(results['learnbest']), label='avgLearnBest', color='c', ls='-')
#plt.plot(np.mean(results['learnavg']), label='avgLearnAvg', color='y', ls='-')
#plt.xlabel("Generations")
#plt.ylabel("Fitness")
#plt.title(f"Microbial: Best and average fitness\nBest evo+learn \ninit flux:{init_flux}\nT:{duration}s")
#plt.legend()
#plt.show()
#
