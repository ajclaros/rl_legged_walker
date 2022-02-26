import matplotlib.pyplot as plt
import numpy as np
import json
import math
from jason.rl_ctrnn import RL_CTRNN

"""
Generating similar plots from the paper: 
A Bio-inspired Reinforcement Learning Rule to Optimise Dynamical Neural Networks for Robot Control
Tianqi Wei and Barbara Webb (2018)
2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)

# Testing that works as desired and can reproduce results from paper
# Generates similar plots to Fig. 1, 2, 3 from the paper
# The period for Fig. 1 changes during the sim., so there is a slight difference here
"""

def main():

    #create network
    size=1
    nn1 = RL_CTRNN( size, weight_range=1, bias_range=1, tc_min=1, tc_max=2, \
        max_flux_amp=1, init_flux_amp=1, flux_period_min=20, flux_period_max=20, flux_conv_rate=0.00008, learn_rate=0.0005)
    nn1.randomize_parameters_with_seed( 4 )   #tested a bit to find this seed that gives a close approximation to fig.
    #set self-loop weight to be zero
    nn1.inner_weights[0] = 0
    #simulate
    time, weights, fluxweights, amps, rewards = sim(nn1)
    # Show Fig. 1 from paper
    plot( size, time, weights, fluxweights, amps, rewards )

    size=2
    nn2 = RL_CTRNN( size, weight_range=1, bias_range=1, tc_min=1, tc_max=2, \
        max_flux_amp=2, init_flux_amp=1, flux_period_min=10, flux_period_max=20, flux_conv_rate=0.0008, learn_rate=0.0005)
    nn2.randomize_parameters_with_seed( 4 )
    
    #set self-loop weight to be zero
    time, weights, fluxweights, amps, rewards = sim(nn2, duration=200, reward_func=fig2_reward)

    #plot( size, time, weights, fluxweights, amps, rewards )
    plot_exploration2d( size, time, fluxweights )
    plot_exploration3d( size, time, fluxweights )

    
    # This shows how fluctuations change with reward with more extreme values

    size=1
    nn1 = RL_CTRNN( size, weight_range=1, bias_range=1, tc_min=1, tc_max=2, \
        max_flux_amp=1, init_flux_amp=0.5, flux_period_min=2, flux_period_max=10, flux_conv_rate=0.001, learn_rate=0.001)
    nn1.randomize_parameters_with_seed( 4 )
    #set self-loop weight to be zero
    nn1.inner_weights[0] = 0
    #simulate
    time, weights, fluxweights, amps, rewards = sim(nn1, duration=125, reward_func=osc_converge_reward)
    # Show Fig. 1 from paper
    plot( size, time, weights, fluxweights, amps, rewards )

    #TODO could add in an actual task showing convergence, maybe even reward as a color to visualize the trajectory



def fig1_reward( timestep ):
    # There is no task here, just demonstration of effects of positive/negative reward on network weights
    # These values are meant to match fig. 1 from the paper's values
    if timestep < 10:
        reward = 0
    elif timestep < 20:
        reward = 1
    elif timestep < 40:
        reward = 0
    elif timestep < 50:
        reward = 1
    elif timestep < 60:
        reward = 0
    else:
        reward = -1
    return reward

#maximize exploration
def fig2_reward( timestep ):
    return -1


def osc_converge_reward( timestep ):
    # There is no task here, just demonstration of effects of positive/negative reward on network weights
    # Should show that the synaptic weights deviate increasingly further in the first half of sim
    # # but then should converge in the second half
    if timestep < 50:
        reward = 0
    elif timestep < 100:
        reward = -1
    else:
        reward = 1
    return reward

def sim(nn, duration=60, reward_func = fig1_reward):
    size = nn.size
    stepsize = 0.01
    time = np.arange(0.0, duration, stepsize)
    # inst. reward/performance
    weights = np.zeros((len(time), size, size))
    fluxweights = np.zeros((len(time), size, size))
    amps = np.zeros(len(time) )
    rewards = np.zeros(len(time) )

    for i in range(len(time)):
        nn.step(stepsize)
        
        reward=reward_func( time[i] )

        nn.update_weights_and_flux_amp_with_reward( reward )

        #record useful information
        weights[i] = nn.inner_weights
        fluxweights[i] = nn.calc_inner_weights_with_flux()
        amps[i] = nn.flux_amp
        rewards[i] = reward
    return time, weights, fluxweights, amps, rewards

def plot(size, time, weights, fluxweights, amps, rewards):
    #flip the weights and time for easy plotting
    weights = weights.transpose(1,2,0)
    fluxweights = fluxweights.transpose(1,2,0)

    plt.plot(time,rewards,label="instantaenous reward" )
    # Plot Synaptic Weights over time
    for i in range(size):
        for j in range(size):
            plt.plot(time,weights[i][j],label="weight center {}->{}".format(i,j) )
            plt.plot(time,fluxweights[i][j],label="flux weight {}->{}".format(i,j) )
    
    plt.plot(time,amps,label="flux amplitude")

    plt.xlabel("Time")
    plt.ylabel("Reward, Flux, Weights")
    plt.ylim(-2,2)
    plt.legend()
    plt.title("Reward, Flux, Synaptic Weights over time")
    plt.show()

def plot_exploration2d(size, time, fluxweights):
    #flip the weights and time
    fluxweights = fluxweights.transpose(1,2,0)

    fig, axs = plt.subplots(size, size*(size-1), sharex=True, sharey=True)
    
    # Plot Synaptic Weights over time
    for i in range(size):  #  self-connections
        j_x=0
        j_y=1
        for j in range( size*(size-1) ):    # other connections
            axs[i,j].plot(fluxweights[i][i], fluxweights[j_x][j_y]  )
            
            if i == size - 1:
                axs[i,j].set_xlabel("{}->{}".format(j_x,j_y) )

            if j == 0:
                axs[i,j].set_ylabel("{}->{}".format(i,i) )

            #plt.plot(time,amps,label="flux amplitude")
            axs[i,j].set_xlim(-2,2)
            axs[i,j].set_ylim(-2,2)

            j_y += 1
            if j_x == j_y:
                j_y += 1

            if j_y == size:
                j_y = 0
                j_x += 1
    plt.suptitle("Exploration Trajectory of 2 synapses")
    plt.show()

def plot_exploration3d(size, time, fluxweights):

    if size < 2:
        print("Cannot plot with less than size 2 network")
        return
    #flip the weights and time
    fluxweights = fluxweights.transpose(1,2,0)

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    #This is just one single slice of the exploration, there are many weights not show here (3x3 = 9 weights)
    x1, x2, x3, y1, y2, y3 = 0, 0, 1, 0, 1, 0

    ax.plot( fluxweights[x1][y1], fluxweights[x2][y2], fluxweights[x3][y3] )
    plt.title("Exploration Trajectory of 3 synapses")
    plt.show()


if __name__ == "__main__":
     main()