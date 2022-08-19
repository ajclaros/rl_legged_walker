import leggedwalker
import numpy as np
from jason.rl_ctrnn import RL_CTRNN
from jason.ctrnn import CTRNN
from walking_task2 import WalkingTask
import warnings
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import datetime
from datalogger import DataLogger
from dataloggervis import visualize, vis2



np.seterr(all='warn')
warnings.simplefilter("always")
# Nervous System Parameters
N = 2 # Number of neurons in the nervous system
WR = 16    # Weight range - maps from [-1, 1] to: [-16,16]
BR = 16    # Bias range - maps from [-1, 1] to: [-16,16]
TR = 5.0   # Time range - maps from [-1, 1] to: [-5, 5]
TA = 6.0   # Time add - maps from [-5, 5] to: [1,11]

# Task Parameters
duration = 220.0
stepsize = 0.1
time = np.arange(0.0, duration, stepsize)
fitness_arr = np.zeros(len(time))

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
    for i, t in enumerate(time):

        ns.setInputs(np.array([legged.anglefeedback()]*N))  # Set neuron input to angle feedback based on current body state
        ns.step(stepsize)                               # Update the nervous system based on inputs
        legged.step1(stepsize, ns.outputs)                  # Update the body based on nervous system activity
#        fitness_arr[i] = body.cx                        # track position of body
                                                        #update neurons based on speed of movement (cx(t)-cx(t-1))/dt
    # Calculate the fitness based on distance covered over the duration of time
    fit = legged.cx/duration
    return fit


timelengths = [8000]
print("Trying:{}".format(timelengths))
fitnesses = []
fig, ax = plt.subplots(ncols=2)
cmap = cm.get_cmap("Paired").colors
conv_rate = 0.01
conv_mult = 10
init_flux = 0.5
flux_mult= 0.5
learn_rate = 0.8
learn_mult = 10
window_size = 1000
window_mult= 1000

for t in timelengths:
    if not os.path.exists("./durations/{}".format(t)):
        os.mkdir("./durations/{}".format(t))
    file_path = "./durations/{}".format(t)
    print("T:{}".format(t))
    for a in range(1):
        j = 41
        #if j//divide_by!= location:
        #    continue
        print("\ntrial:{}".format(j))
        lower_fitness = np.load('./durations/{}/{}/genome.npy'.format(8000, j))
        if not os.path.exists("./durations/{}/{}".format(t, j)):
            os.mkdir("./durations/{}/{}/".format(t,j))
            np.save("./durations/{}/{}/genome".format(t,j), lower_fitness)
        for b in range(5):
            i = b
            #if i>4:
            #    break

            #if os.path.exists("./durations/{}/{}/run-{}.npz".format(t, j, i)):
            #    continue
            print("{}".format(i), end=" ", flush=True)
            np.savez("./durations/{}/{}/run-{}".format(t, j,i))
            weights = lower_fitness[0:N*N]
            learner = WalkingTask(size=2, duration=t, stepsize=0.1,
                                  reward_func=None,
                                  performance_func=None,
                                  running_window_mode=True,
                                  running_window_size= 5000
                                  performance_update_rate=0.05,
                                  performance_bias=0.007,
                                  init_flux_amp=init_flux, max_flux_amp=40,
                                  flux_period_min= 300,
                                  flux_period_max=400, flux_conv_rate= 0.004 , learn_rate=0.008,
                                  bias_init_flux_amp=init_flux, bias_max_flux_amp=40,
                                  bias_flux_period_min=300,
                                  bias_flux_period_max=400,
                                  bias_flux_conv_rate= 0.004
                                  )
            learner.setWeights(weights.reshape((N, N)))
            learner.setBiases(lower_fitness[N*N:N*N+N])
            learner.setTimeConstants(lower_fitness[N*N+N:])
            learner.initializeState(np.zeros(N))
            body = leggedwalker.LeggedAgent()
            datalogger = DataLogger()
            learner.simulate(body, learning_start = 4000, datalogger=datalogger, trackpercent=1.00)
            fitnesses.append(fitnessFunction(learner.recoverParameters()))

            text = f"{np.round(fitnessFunction(datalogger.data['startgenome']), 3)}, change window size"
#            vis2(datalogger.data, ax, color=cmap[i], label=f"{window_size+(i*window_mult)}", alpha=0.9,
#                 text=text)

            print(np.round(fitnesses, 3))
            print(np.round(np.mean(fitnesses), 3))

            #save_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            #datalogger.data['savename'] = save_name
            #datalogger.data['geneID'] = j
            #datalogger.save("./data/durations/{}/{}/{}.npz".format(t, j, save_name))
            #learner.save2("./durations/{}/{}/run-{}".format(t, j,i))
plt.show()
