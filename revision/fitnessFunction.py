
import leggedwalker
import numpy as np
from ctrnn import CTRNN
#N = 2 # Number of neurons in the nervous system
WR = 16    # Weight range - maps from [-1, 1] to: [-16,16]
BR = 16    # Bias range - maps from [-1, 1] to: [-16,16]
TR = 5.0   # Time range - maps from [-1, 1] to: [-5, 5]
TA = 6.0   # Time add - maps from [-5, 5] to: [1,11]

# Task Parameters
duration = 220.0
stepsize = 0.1
#time = np.arange(0.0, duration, stepsize)

def fitnessFunction(genotype, duration=220.0, N=2, generator_type='RPG', configuration = [0], verbose=0):
    # Create the agent's body
    legged = leggedwalker.LeggedAgent()
    if verbose>0:
        print(f"Running fitnessFunction:{generator_type}")
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

        if generator_type=='RPG':
            ns.setInputs(np.array([legged.anglefeedback()]*N))  # Set neuron input to angle feedback based on current body state
        else:
            ns.setInputs(np.array([0]* ns.size))
#        ns.setInputs(np.array([0.0]*N))  # Set neuron input to angle feedback based on current body state
        ns.step(stepsize)                               # Update the nervous system based on inputs
        legged.stepN(stepsize, ns.outputs, configuration)                  # Update the body based on nervous system activity
#        fitness_arr[i] = body.cx                        # track position of body
                                                        #update neurons based on speed of movement (cx(t)-cx(t-1))/dt
    # Calculate the fitness based on distance covered over the duration of time
    fit = legged.cx/duration
    if verbose==1:
        print(ns.recoverParameters())
    return fit
