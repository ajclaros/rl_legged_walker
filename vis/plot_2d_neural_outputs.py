
from jason.ctrnn import CTRNN
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
import sys
import json
import os
import math
from util.fitness_functions import fitness_maximize_output_change, fitness_frequency_match


def main():
    trial_seed=1
    sol_seed=6
    size=2
    directory=f"data/perturbed_networks/nnsize-{size}_sol-seed-{sol_seed}/seed{trial_seed}/"
    directory=f"jason/data/ctrnn_snapshots_recovery/"
    

    #plot_2d_neural_outputs

    
    #filename = f"data/evolved_solutions/mga_pop-20_gen-1000/ALL/discovery_mga_best_nn{size}_seed-{seed}.json"  
    plot_2d_neural_outputs( directory, size=2)



def plot_2d_neural_outputs(directory, size=2, stepsize=0.01):
    
    filenames = os.listdir(directory)

    rows=int(  math.ceil( math.sqrt(len(filenames)) ))
    print(rows)
    fig, axs = plt.subplots(rows, rows)

    count=0
    for filename in filenames:
        count+=1
        #r=count/2
        #c=count%2+1

        filepath=f"{directory}{filename}"

        ctrnn = CTRNN( size)
        ctrnn.load_json( filepath )
        mid_point=50

        fitness, output_history = simulate_ctrnn(ctrnn, stepsize=0.01, init_duration=0, test_duration=100)
        output_history = output_history.transpose(1,0)
        ax1 = plt.subplot(rows,rows,count)
        
        start_of_test=int(mid_point/stepsize)

        ax1.plot(output_history[0][0:start_of_test],output_history[1][0:start_of_test], color='r')
        ax1.plot(output_history[0][start_of_test:],output_history[1][start_of_test:], color='b')
        ax1.set_xlim(0,1)
        ax1.set_ylim(0,1)

        ax1.set_title(f"{filename}\n{fitness:0.2f}")
    plt.show()


def simulate_ctrnn(ctrnn, stepsize=0.01, init_duration=0, test_duration=10):
    """This function simply provides an average change in output per neuron per time. Optionally can include initial duration to prevent transient changes at start of simulation."""


    init_time = np.arange(0.0, init_duration, stepsize)
    test_time = np.arange(0.0, test_duration, stepsize)

    output_history=np.zeros((len(test_time),ctrnn.size))

    #allow transients to clear
    ctrnn.initializeState( np.zeros( ctrnn.size ))

    #ctrnn.initializeState( np.asarray( [-5.0, 10.0] ))

    for i in range(len(init_time)):
        ctrnn.step(stepsize)
    
    #evaluate after transient period
    change_in_output=0
    for i in range(len(test_time)):
        output_history[i] = ctrnn.outputs
        pastOutputs = ctrnn.outputs
        ctrnn.step(stepsize)
        currentOutputs = ctrnn.outputs
        change_in_output += np.sum(abs(currentOutputs - pastOutputs) ) 
        
    #average over time and per neuron
    return change_in_output / ctrnn.size / test_duration, output_history



main()
