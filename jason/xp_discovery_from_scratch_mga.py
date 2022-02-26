import numpy as np
import numpy as np
import random
from jason.ctrnn import CTRNN
from jason.mga import MicrobialGA
import matplotlib.pyplot as plt
import sys
import json
import os
from jason.simple_oscillator_task import SimpleOscillatorTask
from util.fitness_functions import fitness_maximize_output_change

def main():

    main_demo()
    main_sweep()


def main_demo():
    show_progress=True
    #  Using these makes it more viable to evolve from scratch
    nnsize=2
    weight_range=16
    bias_range=16
    popsize=20
    seed=0
    total_iterations=200
    avgfit, bestfit, beer_best_fit = run_experiment(nnsize=nnsize, show_progress=show_progress, popsize=popsize, seed=seed, weight_range=weight_range, bias_range=bias_range,total_iterations=total_iterations )
    print(  f"avgfit:{avgfit} bestfit:{bestfit} beer_best_fit:{beer_best_fit}" ) 

def main_sweep():
    show_progress=True
    #  Using these makes it more viable to evolve from scratch
    nnsizes=[2]
    weight_ranges=[16]
    bias_ranges=[16]  
    popsizes=[20]
    seeds=[1]
    total_iterations=200
    save_filename=f"jason/data/discovery_mga_sizes-{nnsizes}_popsize-{popsizes}_iters-{total_iterations}.csv"
    line="seed,nnsize,popsize,weight_range,bias_range,best_fit,avg_fit,beer_best_fitness,"
    if not os.path.exists(save_filename):
        print("File does not exist, writing to new file")
        write_to_file( save_filename,line,'w' )
        print(line)
    
    for popsize in popsizes:
        for nnsize in nnsizes:
            for weight_range in weight_ranges:
                for bias_range in bias_ranges:
                    for seed in seeds:
                        avgfit, bestfit, beer_best_fit = run_experiment(nnsize=nnsize, show_progress=show_progress, popsize=popsize, seed=seed, weight_range=weight_range, bias_range=bias_range,total_iterations=total_iterations )
                        line=f"{seed},{nnsize},{popsize},{weight_range},{bias_range},{bestfit},{avgfit},{beer_best_fit},"
                        print( line )
                        write_to_file( save_filename, line,'a' )


def run_experiment(nnsize=3, seed=0, weight_range=16, bias_range=16,popsize=20, total_iterations=400,show_progress=False  ):
    
    np.random.seed(seed)
    random.seed(seed)

    # parameters for all CTRNNs
    tc_min=1
    tc_max=1
    # All Tasks
    generations=int(total_iterations/popsize)

    recombProb = 0.5
    mutatProb = 0.05

    #this is true of both the rl_ctrnn and the ctrnn
    genesize = nnsize * nnsize + 2 * nnsize


    def ctrnn_genotype_ff(genotype):
        ctrnn = CTRNN(nnsize, weight_range=weight_range, bias_range=bias_range, tc_min=tc_min, tc_max=tc_max )
        ctrnn.set_normalized_parameters( genotype )
        return fitness_maximize_output_change(ctrnn)

   

    ##############################################################################
    # Evolve a regular CTRNN
    ga = MicrobialGA(ctrnn_genotype_ff, popsize, genesize, recombProb, mutatProb)
    tournaments = generations * popsize
    ga.run_tournaments( tournaments, show_progress=show_progress )
    avgfit, bestfit, best_ind_genotype = ga.fitStats(show_progress=show_progress)


    best_nn = CTRNN(nnsize, weight_range=weight_range, bias_range=bias_range, tc_min=tc_min, tc_max=tc_max )
    best_nn.set_normalized_parameters( best_ind_genotype )
    beer_best_fit = fitness_maximize_output_change(best_nn,init_duration=250, test_duration=50)

    best_nn.save_json(f"data/jason/solutions/discovery_mga_best_nn{nnsize}_seed-{seed}.json")

    return avgfit, bestfit, beer_best_fit

def write_to_file(save_filename, line, flag='a'):
    with open( save_filename, flag) as filehandle:
        filehandle.write( line+"\n" )
    filehandle.close()

main()
