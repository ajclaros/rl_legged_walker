import numpy as np
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

from multiprocessing import Pool

def main():
    run_sweep()
    #main_perturb_AND_recover()


def get_config():
    
    init_flux=3
    learning_duration=5000
    prefix="v1_plusminus8_by2"
    save_filename=f"jason/figure2/{prefix}_fig2_data_{learning_duration/1000}k_initflux-{init_flux}.csv"
    save_dat_filename=f"jason/figure2/{prefix}_fig2_{learning_duration/1000}k_initflux-{init_flux}.dat"
    performance_bias=0.01   #0.03
    performance_update_rate=0.001  #0.05   0.03
    flux_convergence= 1.5  #1.5
    performance_bias=0.05           #0.03
    performance_update_rate=0.001   #0.05   0.03
    running_window_mode=True
    running_window_size=2000   # 2000 = 20 seconds ~= 0.001
    ignore_transients=100  #20
    ###########################


    return init_flux, learning_duration, save_filename, save_dat_filename, performance_bias, performance_update_rate,\
         flux_convergence, performance_bias, performance_update_rate, running_window_mode, running_window_size, ignore_transients



def run_sweep():

    #plusminus 2
    w00s=[ 4,   5,   6, 7, 8 ]  #+/- 2   6
    w01s=[16,  15,  14]     #+/- 2 
    w10s=[-16,-15, -14 ]  #+/- 2
    w11s=[ 2,   3,   4, 5, 6  ]  #+/- 2  4

    #close range within
    adj=8
    adj_inc=2
    w00s= range(6-adj,6+adj+1,adj_inc)
    w01s= range(16,16-adj-1,-adj_inc)
    w10s= range(-16,-16+adj+1, adj_inc)
    w11s= range(4-adj,4+adj+1,adj_inc)

    #general sweep
    # w00s= range(-12, 13, 6)
    # w01s= range(-12, 13, 6)
    # w10s= range(-12, 13, 6)
    # w11s= range(-12, 13, 6)


    THREAD_COUNT=10
    p = Pool(THREAD_COUNT)

    sweep = get_sweep( w00s, w01s, w10s, w11s)
    init_flux, learning_duration, save_filename, save_dat_filename, performance_bias, performance_update_rate, \
        flux_convergence, performance_bias, performance_update_rate, running_window_mode, running_window_size, ignore_transients = get_config()
   
    
    
    line="init_fit,final_fit,init_est_dist,final_est_dist,"
    line+="init_w00,init_w01,init_w10,init_w11,"
    line+="final_w00,final_w01,final_w10,final_w11,"
    # for i in range(learning_duration+1):
    #     line+=f"ARP{i},"
    
    if not os.path.exists(save_filename):
        print("File does not exist, writing to new file")
        write_to_file( save_filename, line,'w' )
        write_to_file( save_dat_filename, "",'w' )   #make sure file is created
    print( len(sweep) )
    sweep = get_sweep(w00s, w01s, w10s, w11s)
    r = process_map(main_recover, sweep, max_workers=THREAD_COUNT, chunksize=1)
    #data = p.map(main_recover, sweep )


def get_sweep( w00s, w01s, w10s, w11s):
    params = []
    for w00 in w00s:
        for w01 in w01s:
            for w10 in w10s:
                for w11 in w11s:
                    params.append((w00, w01, w10, w11))
    
    return params



def main_recover( params ):

    init_flux, learning_duration, save_filename, save_dat_filename, performance_bias, performance_update_rate, \
        flux_convergence, performance_bias, performance_update_rate, running_window_mode, running_window_size, ignore_transients = get_config()
    

    w00s=[params[0]]
    w01s=[params[1]]
    w10s=[params[2]]
    w11s=[params[3]]

    weight_range=16
    bias_range=16
    tc_min=1
    tc_max=1


    show_plots=False
    show_subplots=False
    seed=1
    size=2
    nnsize=size
    sol_seed=6
    

    seeds=[0]  #range(10)
    sol_seeds=[1]  #4 is best in 10nnsize
    nnsizes=[2]
    test_duration=10   #?

    #Load a previously evolved example ctrnn
    best_nn = CTRNN( nnsize, weight_range=weight_range, bias_range=bias_range, tc_min=tc_min, tc_max=tc_max )
    best_evolved_filename=f"data/evolved_solutions/mga_pop-20_gen-1000/ALL/discovery_mga_best_nn{nnsize}_seed-{sol_seed}.json"
    best_nn.load_json( best_evolved_filename )

    new_ctrnn = CTRNN( nnsize, weight_range=weight_range, bias_range=bias_range, tc_min=tc_min, tc_max=tc_max )
    orig_params=new_ctrnn.get_normalized_parameters()

    #faster to simulate and easier to evolve
    orig_fit=fitness_maximize_output_change(best_nn, test_duration=test_duration)

    
    #try systematically perturbing the network 

    for w00 in w00s:
        for w01 in w01s:
            for w10 in w10s:
                for w11 in w11s:
                    new_ctrnn.load_json( best_evolved_filename )
                    new_ctrnn.inner_weights[0][0] = w00
                    new_ctrnn.inner_weights[0][1] = w01
                    new_ctrnn.inner_weights[1][0] = w10
                    new_ctrnn.inner_weights[1][1] = w11

                    #vector difference
                    diff_vec = (best_nn.inner_weights - new_ctrnn.inner_weights )            
                    #Cartesian distance
                    init_est_dist = np.sqrt( np.sum(diff_vec**2) ) 
                    init_fit=fitness_maximize_output_change(new_ctrnn, test_duration=test_duration)

                    #used to run recovery
                    norm_params = new_ctrnn.get_normalized_parameters()


                    final_fitness, final_ctrnn, arp_timeseries = run_recovery( norm_params, performance_bias=performance_bias, \
                        init_flux=init_flux,\
                        running_window_mode=running_window_mode, running_window_size=running_window_size,\
                        performance_update_rate=performance_update_rate, nnsize=nnsize,learning_duration=learning_duration,\
                        ignore_transients=ignore_transients   )
                    
                    diff_vec = (best_nn.inner_weights - final_ctrnn.inner_weights )      
                    final_est_dist = np.sqrt( np.sum(diff_vec**2) ) 

                    

                    final_w00 = final_ctrnn.inner_weights[0][0]
                    final_w01 = final_ctrnn.inner_weights[0][1]
                    final_w10 = final_ctrnn.inner_weights[1][0]
                    final_w11 = final_ctrnn.inner_weights[1][1]
                    
                    line2=f"{init_fit},{final_fitness},{init_est_dist},{final_est_dist},"
                    line2+=f"{w00},{w01},{w10},{w11},"
                    line2+=f"{final_w00},{final_w01},{final_w10},{final_w11}"  #",{arp_timeseries}"

                    write_to_file( save_filename, line2,'a' )
                    write_to_file( save_dat_filename, line2,'a' )
                    #print(line2)
                    #quit()
                    #print( new_ctrnn.inner_weights )
                    #print( f"fit: {init_fit:.4f}->{final_fitness:.4f}  dist: {init_est_dist:.4f}->{final_est_dist:.4f}" ) 


def run_recovery( norm_params, init_flux=1, nnsize=2, weight_range=16, bias_range=16,learning_duration=2000, performance_bias=0.005, \
    performance_update_rate=0.002, flux_convergence=1.0, show_plots=False, show_subplots=False, save_recover_data_filename=False,\
        running_window_mode=True, running_window_size=2000, \
        ignore_transients=0 ):
    
    # Parameters RL-CTRNN specific
    init_flux_amp=init_flux
    max_flux_amp=8
    flux_period_min=4
    flux_period_max=8
    flux_conv_rate=flux_convergence
    learn_rate=1.0

    # could be tuned
    bias_init_flux_amp=0
    bias_max_flux_amp=0
    bias_flux_period_min=0
    bias_flux_period_max=0
    bias_flux_conv_rate=0

   
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
        running_window_mode=running_window_mode, running_window_size=running_window_size, \
        convergence_epsilon=convergence_epsilon, performance_update_rate=performance_update_rate, performance_bias=performance_bias)



    nn, plot_info, converged = task.simulate(rl_nn, ignore_transients=ignore_transients, \
        show_plots=show_plots, show_subplots=show_subplots,  record_data=True )
    
    final_ctrnn = CTRNN(nnsize, weight_range=weight_range, bias_range=bias_range, tc_min=tc_min, tc_max=tc_max )
    final_ctrnn.set_normalized_parameters( nn.get_normalized_parameters() )
    final_fitness = fitness_maximize_output_change( final_ctrnn) 

    arp_timeseries=plot_info["running_average_performances"][::100]  #only 1 in eveyr 100
    arp_timeseries_string=""
    for val in arp_timeseries:
        arp_timeseries_string+=f"{val},"


    return final_fitness, final_ctrnn, arp_timeseries_string
    


def write_to_file(save_filename, line, flag='a'):
    with open( save_filename, flag) as filehandle:
        filehandle.write( line+"\n" )
    filehandle.close()



if __name__ == "__main__":
    main()