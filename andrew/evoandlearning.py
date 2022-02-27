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
    return legged.cx


popsize = 25
genesize = N*N + 2*N
recombProb = 0.5
mutatProb = 0.01
demesize = 2
generations = 100




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
                          init_flux_amp= 0.4,
                          max_flux_amp=40,
                          flux_period_min=300,
                          flux_period_max=400,
                          flux_conv_rate=0.004, learn_rate=0.008,
                          bias_init_flux_amp=0.4,
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
    return body.cx

ga = ea.Microbial(learningFunction, popsize, genesize, recombProb, mutatProb, demesize, generations)
ga.run()
#ga.showFitness(label = 'learn+evo')
ga2 = ea.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demesize, generations)
ga2.run()
#ga2.showFitness(label = 'evo')

plt.plot(ga.bestHistory, label = 'evo+learn', color='r', ls='-')
plt.plot(ga2.bestHistory, label = 'evo', color='r', ls=':')
plt.plot(ga.avgHistory, label = 'evo+learn', color='k', ls='-')
plt.plot(ga2.avgHistory, label = 'evo', color='k', ls=':')
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("Microbial: Best and average fitness")
plt.legend()
plt.show()

#    af, bf, bi = ga.fitStats()
#    ga.save("./data/learn-pop25/{}".format(i))
#    results['learnavg'].append(ga.avgHistory)
#    results['learnbest'].append(ga.bestHistory)
#    if i==0:
#        ax.plot(ga.bestHistory, label="best learning", color='cadetblue', linestyle = 'solid')
#        ax.plot(ga.avgHistory, color='steelblue', linestyle = 'solid', label='avg learning')
#    else:
#        ax.plot(ga.bestHistory, color='cadetblue', linestyle='solid')
#ax.plot(np.mean(results['evobest'], axis=0), color='red', linestyle='dashed', linewidth=4, label='avg best evo')
#ax.plot(np.mean(results['evoavg'], axis=0), color='orangered', linestyle='dashed', linewidth=4, label='avg-avg evo')
#ax.plot(np.mean(results['learnbest'], axis=0), color='blue', linestyle='solid', linewidth=4, label ='avg best learn')
#ax.plot(np.mean(results['learnavg'], axis=0), color='deepskyblue', linestyle='solid', linewidth=4, label='avg-avg learn')
#plt.legend()
#plt.savefig("./images/learning-250-evo.png".format(i))
#plt.show()
