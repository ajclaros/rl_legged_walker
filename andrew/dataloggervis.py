import numpy as np
import matplotlib.pyplot as plt
from fitnessFunction import fitnessFunction
import os

#files = os.listdir("./data/startingfitness/0.1/")
#learner= np.load(f"./data/startingfitness/0.1/{files[-1]}")

def visualize(learner):
    fig, ax = plt.subplots(nrows = 4, ncols=2, figsize=(20,10))
    time = np.arange(0,learner['duration'], learner['stepsize'])
    for i in range(2):
        for j in range(2):
            ax[i, j].plot(time, learner['extendedWeightHist'][:, i, j], color='r', label="{},{} fluctuation".format(i,j))
            ax[i,j].axvline(time[learner['learningStart']], color='k')
            ax[i, j].plot(time, learner['weightHist'][:, i,j], color='k', label = "{},{} weight center".format(i,j))
            ax[i,j].legend()
            ax[i,j].title.set_text("Neuron:{},{} weights".format(i,j))
        ax[2, i].plot(time, learner['extendedBiasHist'][:, i], color='r', label= "{} bias fluctuation".format(i))
        ax[2,i].axvline(time[learner['learningStart']], color='k')
        ax[2, i].plot(time, learner['biasHist'][:, i], color='k', label= "{} bias center".format(i))
        ax[2, i].legend()
        ax[2,i].title.set_text("bias {}".format(i))
    ax[3, 0].plot(time, learner['trackFitness'], label="trackedFitness")
    ax[3, 0].axvline(time[learner['learningStart']], ymax= max(learner['rewardHist']), color='k')
    ax[3, 1].plot(time, learner['fluxAmpArr'], label="FluxSize")
    ax[3,1].axvline(time[learner['learningStart']], color='k')
    ax[3,1].set_ylim((0, 5))
    ax[3,0].title.set_text("Running AvgPerf")
    ax[3,1].title.set_text("Flux Size")
    #plt.suptitle("\nfilename:{}\nperturbed:{}, learned:{}\ngeneID:{}, T:{},".format(
        #learner['savename'],
    #    fitnessFunction(learner['startgenome']),
    #    fitnessFunction(learner['endGenome']),
        #learner['geneID'],
    #    learner['duration']
    #))
    plt.tight_layout()
    #plt.savefig("outputs.png", dpi=300)
    plt.show()


def vis2(learner, ax, color, label, alpha, text):
    time = np.arange(0,learner['duration'], learner['stepsize'])
    ax[0].plot(time, learner['fluxAmpArr'], color=color, label=label, alpha=alpha)
    ax[0].set_ylabel("Fluctuation Amplitude")
    ax[1].plot(time, learner['runningAverage'], color=color, label=label, alpha=alpha)
    ax[1].set_ylabel("Running Average")
    plt.suptitle(text)
    ax[1].legend()
