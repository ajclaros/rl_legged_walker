import numpy as np
import numpy as np
import random
from jason.ctrnn import CTRNN
from jason.rl_ctrnn import RL_CTRNN
from jason.mga import MicrobialGA
import matplotlib.pyplot as plt
import sys
import json
import os
from jason.simple_oscillator_task import SimpleOscillatorTask
from util.fitness_functions import fitness_maximize_output_change

def main():

    main_demo()
    #main_sweep()
    #main_time_series_sweep()


def main_demo():
    show_plots=True
    #  Using these makes it more viable to evolve from scratch
    nnsize=2
    wb_range=16
    learning_duration=5000
    init_flux_amp=1
    flux_period_min=4
    flux_period_max= 2 * flux_period_min
    max_flux_amp = 8
    flux_conv_rate=0.5  #003
    learn_rate=1.0
    performance_bias=0.005
    performance_update_rate=0.001

    stop_at_convergence=False

    seed=1
    rl_nn_fit, plot_info, converged = run_experiment(seed=seed, nnsize=nnsize, weight_range=wb_range, bias_range=wb_range, learning_duration=learning_duration,max_flux_amp=max_flux_amp,\
                                                init_flux_amp=init_flux_amp, flux_period_min=flux_period_min, flux_period_max=flux_period_max, flux_conv_rate=flux_conv_rate, learn_rate=learn_rate,\
                                                performance_bias=performance_bias, performance_update_rate=performance_update_rate, show_plots=show_plots,stop_at_convergence=stop_at_convergence )
    time_passed= plot_info["time_passed"]
    print( f"rl_nn_fit:{rl_nn_fit} time_passed:{time_passed}"  )


def main_time_series_sweep():
    show_plots=False
    #  Using these makes it more viable to evolve from scratch
    nnsize=2
    wb_range=16
    learning_duration=5000
    init_flux_amp=1
    flux_period_min=4
    flux_period_max= 2 * flux_period_min
    max_flux_amp = 8
    flux_conv_rate=0.02  #003
    learn_rate=1.0
    performance_bias=0.005
    performance_update_rate=0.001

    stop_at_convergence=False

    seed=2
    flux_amps={}
    times={}
    performances={}
    running_average_performances={}
    for flux_conv_rate in [0.25, 0.5, 1, 1.5, 2, 3, 4  ]:
        label=f"flux-conv-{flux_conv_rate}"
        rl_nn_fit, plot_info, converged = run_experiment(seed=seed, nnsize=nnsize, weight_range=wb_range, bias_range=wb_range, learning_duration=learning_duration,max_flux_amp=max_flux_amp,\
                                                init_flux_amp=init_flux_amp, flux_period_min=flux_period_min, flux_period_max=flux_period_max, flux_conv_rate=flux_conv_rate, learn_rate=learn_rate,\
                                                performance_bias=performance_bias, performance_update_rate=performance_update_rate, show_plots=show_plots,stop_at_convergence=stop_at_convergence )
        flux_amps[ label ] = plot_info["amps"]
        times[ label ] = plot_info["time"]
        performances[ label ] = plot_info["performances"]
        running_average_performances[ label ] = plot_info["running_average_performances"]
        time_passed= plot_info["time_passed"]
        print( f"rl_nn_fit:{rl_nn_fit} time_passed:{time_passed}"  )
    

    plt.xlabel("Time")
    plt.ylabel("Fluctuation Amplitude")
    for key in flux_amps.keys():
        time=times[key]
        amps=flux_amps[key]
        plt.plot(time,amps,label=f"{key}" )
        #plt.plot(time[0:stop_step],plot_info["amps"][0:stop_step]/highestAmp/100,label="flux amplitude/{:.2f} for scaling".format(highestAmp*100) )
 
    plt.legend()
    plt.title("Flux Amps by Convergence Rate")
    plt.show()
    ###########################  performance
    plt.xlabel("Time")
    plt.ylabel("Running Average Performance")
    for key in flux_amps.keys():
        time=times[key]
        ave_perf=running_average_performances[key]
        plt.plot(time,ave_perf,label=f"{key}" )
        #post processing to find running average
        window100=[]
        #for ind_ave_perf in running_average_performances[key]:


        #plt.plot(time[0:stop_step],plot_info["amps"][0:stop_step]/highestAmp/100,label="flux amplitude/{:.2f} for scaling".format(highestAmp*100) )
 
    plt.legend()
    plt.title("Running Average Performance by Convergence Rate")
    plt.show()





def main_sweep():

    learning_duration=20000
    nnsizes=[2,3,4,5,6,7,8,9,10]
    stop_at_convergence=False
    performance_biases=[0.005]

    # ####################################
    experiment_name="SCALE_PERF-BIAS_HUGE_[2-10]"
    max_flux_amps=[ 16 ]
    flux_period_mins=[ 4 ] 
    flux_conv_rates=[0.03, 0.05, 0.07, 0.09, 0.1 ]
    seeds=[0,1,2,3,4,5,6,7,8,9]
    performance_update_rates=[0.001 ]
    performance_biases=[0.005,  0.007,  0.009, 0.01,  0.012 ]
    # ####################################




    # #####################################
    # experiment_name="AMP_vs_CONVSTOP_"
    # nnsizes=[3,4,5,6,7,8,9,10]
    # max_flux_amps=[2,4,6,8,10,12,14,16]
    # flux_period_mins=[4] 
    # flux_conv_rates=[0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    # seeds=[1,2,3,4,5,6,7,8,9]
    # performance_update_rates=[0.001]
    # #####################################

    #####################################
    # experiment_name="PERIOD_vs_CONVSTOP_"
    # max_flux_amps=[16]
    # flux_period_mins=[1,2,3,4,5,6,7,8,9,10] 
    # flux_conv_rates=[0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    # seeds=[0]
    # performance_update_rates=[0.001]
    #####################################

    # ####################################
    # experiment_name="PERIOD_vs_AMPSTOP_"
    # max_flux_amps=[2,4,6,8,10,12,14,16]
    # flux_period_mins=[1,2,3,4,5,6,7,8,9,10] 
    # flux_conv_rates=[ 0.05 ]
    # seeds=[1,2,3,4,5,6,7,8,9]
    # performance_update_rates=[0.001]
    # ####################################

    # ####################################
    # nnsizes=[3,4,5,6,7,8,9,10]
    # experiment_name="SPREAD_10SEEDS_STOP_"
    # max_flux_amps=[8,16]
    # flux_period_mins=[4,8] 
    # flux_conv_rates=[ 0.025, 0.05 ]
    # seeds=[0,1,2,3,4,5,6,7,8,9]
    # performance_update_rates=[0.001]
    # ####################################


    # ####################################
    # experiment_name="UPDATE_vs_CONV_STOP_"
    # max_flux_amps=[ 16]
    # flux_period_mins=[ 4 ] 
    # flux_conv_rates=[0.2, 0.1, 0.075, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0075, 0.005, 0.0025]
    # seeds=[1,2,3,4,5,6,7,8,9]
    # performance_update_rates=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    # ####################################

    

    #seeds=[3]   #,5,6,7,8,9]
    #seeds=[0,1,2,3,4]
    #seeds=range(0,9)
    
    
    wb_ranges=[16]
    #max_flux_amps=[8,12,16,24,32]
    #max_flux_amps=[2,4,6,8,10,12,14,16]
    #max_flux_amps=[8]

    #nnsizes=[2,3,4,5]
    

    init_flux_amps=[1]
    #flux_period_mins=[2,4,8,12,16,24,32]   #max = min * 2  - this ratio should be sufficient
    #flux_period_mins=[3,4,5] 
    #flux_period_mins=[4] 
    

    # flux_period_max=[10]
    # flux_conv_rates=[0.08, 0.1, 0.12]
    # learn_rates=[0.8, 1.0, 1.2]
    # performance_biases=[0.001, 0.0025, 0.005, 0.006]
    # performance_update_rates=[0.0001, 0.00025, 0.0005]
    # flux_conv_rates=[0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005]
    #flux_conv_rates=[0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005]
    #flux_conv_rates=[0.03]

    save_filename=f"jason/data/rl-discovery__{experiment_name}.csv"

    learn_rates=[1.0]
      #  [0.005, 0.01]
       #0.0005, 0.001

    # performance_biases=[ 0.01]
    # performance_update_rates=[ 0.001 ]

    line= f"seed,nnsize,wbrange,rl_nn_b_fit,time_passed,max_flux_amp,init_flux_amp,flux_period_min,flux_period_max,flux_conv_rate,learn_rate,performance_bias,performance_update_rate,"
    print(line)
    if not os.path.exists(save_filename):
        print("File does not exist, writing to new file")
        write_to_file( save_filename, line,'w' )
    
    for nnsize in nnsizes:
        for init_flux_amp in init_flux_amps:
            for flux_period_min in flux_period_mins:
                #ratio 2 to 1
                flux_period_max = flux_period_min * 2
                for flux_conv_rate in flux_conv_rates:
                    for learn_rate in learn_rates:
                        for performance_bias in performance_biases:
                            for performance_update_rate in performance_update_rates:
                                for max_flux_amp in max_flux_amps:
                                    for wb_range in wb_ranges:
                                        for seed in seeds:
                                            rl_nn_fit, plot_info, converged = run_experiment(seed=seed, nnsize=nnsize, weight_range=wb_range, bias_range=wb_range, learning_duration=learning_duration,max_flux_amp=max_flux_amp,\
                                                init_flux_amp=init_flux_amp, flux_period_min=flux_period_min, flux_period_max=flux_period_max, flux_conv_rate=flux_conv_rate, learn_rate=learn_rate,\
                                                performance_bias=performance_bias, performance_update_rate=performance_update_rate, stop_at_convergence=stop_at_convergence )
                                            time_passed= plot_info["time_passed"]
                                            line= f"{seed},{nnsize},{wb_range},{rl_nn_fit},{time_passed},{max_flux_amp},{init_flux_amp},{flux_period_min},{flux_period_max},{flux_conv_rate},{learn_rate},{performance_bias},{performance_update_rate},"
                                            print(line)
                                            write_to_file( save_filename, line,'a' )




def run_experiment(seed=0, nnsize=2, weight_range=16, bias_range=16, learning_duration=5000, max_flux_amp=16, \
    init_flux_amp=1, flux_period_min=2, flux_period_max=10, flux_conv_rate=0.1, learn_rate=1.0,\
    performance_bias=0.01, performance_update_rate=0.001, show_plots=False, stop_at_convergence=False):
    
    convergence_epsilon=0.05 #for use when stopAtConvergence is true
    # parameters for all CTRNNs
    gaussian_mode=False   #using uniform instead in order to more clearly define boundaries
    tc_min=1
    tc_max=1

    # Bias specific fluctations set to match the same as the weights
    bias_init_flux_amp=init_flux_amp
    bias_max_flux_amp=max_flux_amp
    bias_flux_period_min=flux_period_min
    bias_flux_period_max=flux_period_max
    bias_flux_conv_rate=flux_conv_rate

    # All Tasks
    stepsize=0.01
    

    rl_nn = RL_CTRNN( nnsize, weight_range=weight_range, bias_range=bias_range, tc_min=tc_min, tc_max=tc_max,\
        init_flux_amp=init_flux_amp, max_flux_amp=max_flux_amp, flux_period_min=flux_period_min, flux_period_max=flux_period_max, flux_conv_rate=flux_conv_rate, learn_rate=learn_rate,\
        gaussian_mode=gaussian_mode, \
        bias_init_flux_amp=bias_init_flux_amp, bias_max_flux_amp=bias_max_flux_amp, \
        bias_flux_period_min=bias_flux_period_min,bias_flux_period_max=bias_flux_period_max,\
        bias_flux_conv_rate=bias_flux_conv_rate)
    
    task = SimpleOscillatorTask( learning_duration, stepsize, stop_at_convergence, \
        convergence_epsilon=convergence_epsilon, performance_update_rate=performance_update_rate, performance_bias=performance_bias )
    
   
    rl_nn.randomize_parameters_with_seed(seed)
    rl_nn_fit, plot_info, converged = rl_ctrnn_ff( rl_nn, task, show_plots=show_plots )
   
    return rl_nn_fit, plot_info, converged


def rl_ctrnn_ff( rl_ctrnn, task, show_plots=False ):
    """Simulates using the RL-CTRNN but then evaluates the final form using the more traditional fitness measure"""
    trained_rlctrnn, plot_info, converged = task.simulate(rl_ctrnn, show_plots=show_plots)
    normalized_params = trained_rlctrnn.get_normalized_parameters()
    ctrnn = CTRNN(rl_ctrnn.size , weight_range=rl_ctrnn.weight_range, bias_range=rl_ctrnn.bias_range, tc_min=rl_ctrnn.tc_min, tc_max=rl_ctrnn.tc_max )
    ctrnn.set_normalized_parameters( normalized_params )
    fitness = fitness_maximize_output_change( ctrnn) 
    return fitness, plot_info, converged


def write_to_file(save_filename, line, flag='a'):
    with open( save_filename, flag) as filehandle:
        filehandle.write( line+"\n" )
    filehandle.close()


main()




