import numpy as np
import random
from jason.ctrnn import CTRNN
from jason.rl_ctrnn import RL_CTRNN
from jason.xp_perturb_solutions import run_perturb_experiment
from jason.mga import MicrobialGA
import matplotlib.pyplot as plt
import sys
import json
import os
from jason.simple_oscillator_task import SimpleOscillatorTask
from util.fitness_functions import fitness_maximize_output_change
from tqdm.contrib.concurrent import process_map

def main():
    THREADS=8
    sweep = get_sweep()

    #rl_discover_new_solutions_using_flux_biases(size=size, seed=seed)
    r = process_map(rl_discover_new_solutions_using_flux_biases, sweep, max_workers=THREADS, chunksize=1)

   

def get_sweep() :
    params=  []

    for size in [2,3,4,5,6,7,8,9,10]:
        for seed in range(100):
            params.append((size, seed))
    
    # for size in [2,3,4,5,6,7,8,9,10]:
    #     for seed in range(10,100):
    #         params.append((size, seed))
        
    return params


def rl_discover_new_solutions_using_flux_biases( params ):

    size=params[0]
    seed=params[1]
    performance_bias=0.1
    init_flux=4
    flux_convergence= 1.5           #1.5

    #data to record to files
    record_array=["performances", "running_average_performances"]
    #record_array=["performances", "running_average_performances", "amps", "weights", "flux_weights", "biases", "flux_biases"]

    #how frequently to record data
    record_every_n_steps=100

    #how long to ignore the intial transients
    ignore_transients=100  #20
    
    #############
    ignore_transients=100  #20
    show_plots=False
    show_subplots=False
    nnsize=size
    
               #0.03
    performance_update_rate=0.001   #0.05   0.03
    
    learning_duration=10000  #in seconds

    running_window_mode=True
    running_window_size=2000   # 2000 = 20 seconds ~= 0.001

    random_filename=f"cehong/paper_figure/data/conf_size{size}/random_size-{size}_seed-{seed}.json"

    #check if directory exists, if not create it
    #create folders for each size
    if not os.path.exists(f"cehong/paper_figure/data/duration-{learning_duration}"):
        os.makedirs(f"cehong/paper_figure/data/duration-{learning_duration}")
    for i in range(2,11):
        if not os.path.exists(f"cehong/paper_figure/data/duration-{learning_duration}/size{i}"):
            os.makedirs(f"cehong/paper_figure/data/duration-{learning_duration}/size{i}")

    save_recover_data_filename=f"cehong/paper_figure/data/duration-{learning_duration}/size{size}/learn_data_random_size-{size}_seed-{seed}.csv"
    

    if os.path.exists( save_recover_data_filename ):
        print(f"{save_recover_data_filename} exists not going to re-run it")
        return

    #1. generate random ctrnn
    np.random.seed(seed)
    ctrnn = CTRNN(size)
    ctrnn.randomize_parameters_with_seed(seed)

    #2. save random ctrnn to file
    ctrnn.save_json(random_filename)

    #. Call alternative recovery mode on it
    final_fitness, plot_info = run_recovery( ctrnn.get_normalized_parameters(), \
            init_flux=init_flux,running_window_mode=running_window_mode, running_window_size=running_window_size, \
            performance_bias=performance_bias, performance_update_rate=performance_update_rate, \
            nnsize=nnsize,learning_duration=learning_duration, flux_convergence=flux_convergence, \
            show_plots=show_plots, show_subplots=show_subplots, save_recover_data_filename=save_recover_data_filename, \
            ignore_transients=ignore_transients, 
            flux_bias_mode=True,\
            record_array=record_array,\
            record_every_n_steps=record_every_n_steps )
    print( final_fitness )


def run_recovery( norm_params, init_flux=1, nnsize=2, weight_range=16, bias_range=16,learning_duration=2000, performance_bias=0.005, \
    performance_update_rate=0.002, flux_convergence=1.0, show_plots=False, show_subplots=False, save_recover_data_filename=False,\
        ignore_transients=0, running_window_mode=False, running_window_size=2000,\
        learning_rate=1,\
        flux_period_min=4, \
        flux_bias_mode=False,\
        record_array=None,\
        record_every_n_steps=None ):
    # Parameters RL-CTRNN specific
    init_flux_amp=init_flux

    
    
    #init_flux_amp=0.1
    max_flux_amp=8
    flux_period_min=4
    flux_period_max=  flux_period_min*2

    flux_conv_rate=flux_convergence
    learn_rate=learning_rate
    # could be tuned
    bias_init_flux_amp=0
    bias_max_flux_amp=0
    bias_flux_period_min=0
    bias_flux_period_max=0
    bias_flux_conv_rate=0
    if flux_bias_mode:
        bias_init_flux_amp=init_flux_amp
        bias_max_flux_amp=max_flux_amp
        bias_flux_period_min=flux_period_min
        bias_flux_period_max=flux_period_max
        bias_flux_conv_rate=flux_conv_rate




    save_nn_snapshots=[0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500]
    ctrnn_save_directory="jason/data/ctrnn_snapshots_recovery/"



    convergence_epsilon=0.05
    stop_at_convergence=False
    gaussian_mode=True
    # All Tasks
    stepsize=0.01
    tc_min=1
    tc_max=1

    rl_nn = RL_CTRNN( nnsize, weight_range=weight_range, bias_range=bias_range, tc_min=tc_min, tc_max=tc_max,\
        init_flux_amp=init_flux_amp, max_flux_amp=max_flux_amp, flux_period_min=flux_period_min, flux_period_max=flux_period_max, flux_conv_rate=flux_conv_rate, learn_rate=learn_rate,\
        gaussian_mode=gaussian_mode, \
        bias_init_flux_amp=bias_init_flux_amp, bias_max_flux_amp=bias_max_flux_amp, \
        bias_flux_period_min=bias_flux_period_min,bias_flux_period_max=bias_flux_period_max,\
        bias_flux_conv_rate=bias_flux_conv_rate)

    rl_nn.set_normalized_parameters(norm_params)

    task = SimpleOscillatorTask( learning_duration, stepsize, stop_at_convergence, \
        convergence_epsilon=convergence_epsilon, \
        performance_update_rate=performance_update_rate, performance_bias=performance_bias,\
        running_window_mode=running_window_mode, running_window_size=running_window_size)

    save_data_filename=save_recover_data_filename

    nn, plot_info, converged = task.simulate(rl_nn, ignore_transients=ignore_transients, \
        show_plots=show_plots, show_subplots=show_subplots,  record_data=True, \
        record_array=record_array,\
        record_every_n_steps=record_every_n_steps,\
        save_data_filename=save_data_filename,\
        save_nn_snapshots=save_nn_snapshots,  ctrnn_save_directory=ctrnn_save_directory)
    ctrnn = CTRNN(nnsize, weight_range=weight_range, bias_range=bias_range, tc_min=tc_min, tc_max=tc_max )
    ctrnn.set_normalized_parameters( nn.get_normalized_parameters() )
    recovered_fitness = fitness_maximize_output_change( ctrnn) 
    return recovered_fitness, plot_info
   



if __name__ == "__main__":
    main()