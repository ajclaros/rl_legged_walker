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

def main():
    print("inside of xp_perturb_and_recovery.py")
    #metaparameter_sweep()
    #train_weights_from_starting_files()
    dir="data/perturbed_networks/nnsize-2_sol-seed-6/seed1/"
    filename="trial-seed1_fit-perc-0.00__afig1_set_D_4d.json"
    repeatedly_train_weights_from_starting_file(filename)
    #rl_discover_new_solutions_using_flux_biases()
    #main_perturb_AND_recover()


def rl_discover_new_solutions_using_flux_biases():
    size=9
    seed=0
    #############
    ignore_transients=100  #20
    show_plots=False
    show_subplots=True
    nnsize=size
    init_flux=4
    performance_bias=0.05           #0.03
    performance_update_rate=0.001   #0.05   0.03
    flux_convergence= 1.5           #1.5
    learning_duration=5000  #in seconds

    running_window_mode=True
    running_window_size=2000   # 2000 = 20 seconds ~= 0.001


    
    random_filename=f"jason/random_size-{size}_seed-{seed}.json"

    save_recover_data_filename=None
    
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
            flux_bias_mode=True)
    print( final_fitness )


def repeatedly_train_weights_from_starting_file(filename):

    record_array=["performances", "running_average_performances", "outputs", "weights"]

    record_every_n_steps=100

    ignore_transients=100  #20
    show_plots=False
    show_subplots=False
    size=2
    nnsize=size
    sol_seed=6
    init_flux=4
    performance_bias=0.05           #0.03
    performance_update_rate=0.001   #0.05   0.03
    flux_convergence= 1.5           #1.5
    learning_duration=2000  #in seconds

    running_window_mode=True
    running_window_size=2000   # 2000 = 20 seconds ~= 0.001

    directory=f"data/perturbed_networks/nnsize-{size}_sol-seed-{sol_seed}/seed1/"

    seeds = range(0,100)
    
    for seed in seeds:
        np.random.seed(seed)
        print(seed)
        save_recover_data_directory=f"data/recovered_run_data/nnsize-{size}_sol-seed-{sol_seed}/repeated_4d_init-flux-{init_flux}/"
        save_recover_data_filename=f"{save_recover_data_directory}recover_seed-{seed}.csv"
        load_filename=f"{directory}{filename}"
        ctrnn = CTRNN(size)
        ctrnn.load_json(load_filename)

        recovered_fitness, plot_info = run_recovery( ctrnn.get_normalized_parameters(), \
            init_flux=init_flux,running_window_mode=running_window_mode, running_window_size=running_window_size, \
            performance_bias=performance_bias, performance_update_rate=performance_update_rate, \
            nnsize=nnsize,learning_duration=learning_duration, flux_convergence=flux_convergence, \
            show_plots=show_plots, show_subplots=show_subplots, save_recover_data_filename=save_recover_data_filename, \
            ignore_transients=ignore_transients,
            record_array=record_array,
            record_every_n_steps=record_every_n_steps)
        print( f"{recovered_fitness}     ..    {(plot_info['running_average_performances'][-1]+performance_bias)*100}" )


  

def train_weights_from_starting_files():
    ignore_transients=100  #20
    show_plots=False
    show_subplots=True
    seed=1
    size=2
    nnsize=size
    sol_seed=6
    init_flux=2
    performance_bias=0.05           #0.03
    performance_update_rate=0.001   #0.05   0.03
    flux_convergence= 1.5           #1.5
    learning_duration=5000  #in seconds

    running_window_mode=True
    running_window_size=2000   # 2000 = 20 seconds ~= 0.001


    directory=f"data/perturbed_networks/nnsize-{size}_sol-seed-{sol_seed}/seed{seed}/"
    filenames = os.listdir(directory)
    np.random.seed(seed)
    for filename in filenames:
        print(filename)
        save_recover_data_directory=f"data/recovered_run_data/nnsize-{size}_sol-seed-{sol_seed}/seed{seed}/"
        
        save_recover_data_filename=f"{save_recover_data_directory}recover_{filename}.csv"
        load_filename=f"{directory}{filename}"
        ctrnn = CTRNN(size)
        ctrnn.load_json(load_filename)
       

        recovered_fitness, plot_info = run_recovery( ctrnn.get_normalized_parameters(), \
            init_flux=init_flux,running_window_mode=running_window_mode, running_window_size=running_window_size, \
            performance_bias=performance_bias, performance_update_rate=performance_update_rate, \
            nnsize=nnsize,learning_duration=learning_duration, flux_convergence=flux_convergence, \
            show_plots=show_plots, show_subplots=show_subplots, save_recover_data_filename=save_recover_data_filename, \
                ignore_transients=ignore_transients)



        print(recovered_fitness)
        quit() #done after 1




def metaparameter_sweep( ):
    ignore_transients=100  #20
    start=ignore_transients*100
    show_plots=False
    show_subplots=False
    seed=1
    size=2
    nnsize=size
    sol_seed=6
    init_flux=4
    flux_period_min=2
    learning_rate=1
    performance_bias=0.05           #0.03
    performance_update_rate=0.001   #0.05   0.03
    flux_convergence= 1.5           #1.5
      #in seconds

    running_window_mode=True
    running_window_size=2000   # 2000 = 20 seconds ~= 0.001


    directory=f"data/perturbed_networks/nnsize-{size}_sol-seed-{sol_seed}/seed{seed}/"
    filenames = os.listdir(directory)
    np.random.seed(seed)


    pairs=[]
    param_name="learning_rate"
    param_vals=[0.25, 0.5, 1, 2, 3, 4, 5]
    pairs.append( [param_name, param_vals]  )
    #pairs.append( ["flux_period_min", [1,2,3,4,5] ]  )

    param_name="flux_period_min"
    param_vals=[1,2,3,4,5]
    #pairs.append( [param_name, param_vals]  )

    param_name="running_window_size"
    param_vals=np.arange(1000,5000,500)
    #pairs.append( [param_name, param_vals]  )

    param_name="performance_bias"
    param_vals=np.arange(1,11,1)/1000.0
    #pairs.append( [param_name, param_vals]  )

    param_name="init_flux"
    param_vals=np.arange(10,51,5)/10
    #pairs.append( [param_name, param_vals]  )

    param_name="flux_convergence"
    param_vals=np.arange(4,15,1)/10
    #pairs.append( [param_name, param_vals]  )

    filenames=[]
    #filenames.append("trial-seed1_fit-perc-0.00__fig1_set_A_short.json")
    #filenames.append("trial-seed1_fit-perc-0.00__fig1_set_B_2d_long.json")  #4000
    #filenames.append("trial-seed1_fit-perc-0.00__fig1_set_C_2d_med.json")  #3000
    filenames.append("trial-seed1_fit-perc-0.00__afig1_set_D_4D.json")  #5000

    learning_duration=500

    for filename in filenames:

        for param_name, param_vals in pairs:


            save_recover_data_directory=f"data/recovered_run_data/nnsize-{size}_sol-seed-{sol_seed}/seed{seed}/figure3/"
            
            load_filename=f"{directory}{filename}"
            ctrnn = CTRNN(size)
            ctrnn.load_json(load_filename)

            
            min=np.asarray(param_vals).min()
            max=np.asarray(param_vals).max()
            

            def map_param_to_color( val, alpha=0.5 ):
                cmap=plt.get_cmap('viridis')
                color = cmap( (val-min)/(max-min) )
                return [color[0], color[1], color[2], alpha]
            

            fig, axs = plt.subplots(2, 2)
            fig.suptitle(f"Comparing different {param_name}")
            for param in param_vals:
                if param_name=="learning_rate":
                    learning_rate=param  
                if param_name=="flux_period_min":
                    flux_period_min=param            
                if param_name=="running_window_size":
                    running_window_size=param
                if param_name=="performance_bias":
                    performance_bias=param
                if param_name=="init_flux":
                    init_flux=param
                if param_name=="flux_convergence":
                    flux_convergence=param

                save_recover_data_filename=f"{save_recover_data_directory}recover_{filename}__11072021__{param_name}-{param}.csv"

                if os.path.exists(save_recover_data_filename):
                    print("reading file")
                    plot_info={}
                    plot_info["time"]=[]
                    plot_info["amps"]=[]
                    plot_info["running_average_performances"]=[]

                    data = np.genfromtxt(save_recover_data_filename, delimiter=",", names=True, dtype=float)
                    plot_info["weights"]=np.zeros((size,size,len(data)))
                    for index in range(len(data)):
                        #0     1      2        3            4                          5        6     7     8   9    10   
                        #time,amps,rewards,performances,running_average_performances,output0,output1,w0_0,w0_1,w1_0,w1_1,flux_w0_0,flux_w0_1,flux_w1_0,flux_w1_1,
                        plot_info["time"].append( float(data[index][0]) )
                        plot_info["amps"].append( float(data[index][1]) )
                        plot_info["running_average_performances"].append( float(data[index][4]) )
                        plot_info["weights"][0][0][index] = data[index][7]
                        plot_info["weights"][0][1][index] = data[index][8]
                        plot_info["weights"][1][0][index] = data[index][9]
                        plot_info["weights"][1][1][index] = data[index][10]
                        recovered_fitness=0

                else:
                    recovered_fitness, plot_info = run_recovery( ctrnn.get_normalized_parameters(), \
                        flux_period_min=flux_period_min,\
                        init_flux=init_flux,running_window_mode=running_window_mode, running_window_size=running_window_size, \
                        learning_rate=learning_rate,\
                        performance_bias=performance_bias, performance_update_rate=performance_update_rate, \
                        nnsize=nnsize,learning_duration=learning_duration, flux_convergence=flux_convergence, \
                        show_plots=show_plots, show_subplots=show_subplots, save_recover_data_filename=save_recover_data_filename, \
                            ignore_transients=ignore_transients)
                print( param  )
                ax1 = plt.subplot(2, 2, 1)

                print(  plot_info["amps"] )

                ax1.plot(plot_info["time"][start:],plot_info["amps"][start:], label=f"{param_name}:{param:0.3f}" , color=map_param_to_color(param) )   #" fit={recovered_fitness:0.2f}"
                ax2 = plt.subplot(2, 2, 2)
                ax2.plot(plot_info["time"][start:], 100*(plot_info["running_average_performances"][start:]+performance_bias),label=f"{param_name}:{param:0.2f}", color=map_param_to_color(param) )
                ax1.set_title("Fluctuation Amplitude")
                ax2.set_title("Running Average Performance")
                ax1.legend()

                ########
                ax3 = plt.subplot(2, 2, 3)
    ###############
                
                flux_linewidth=1
                center_linewidth=1
                start_size=3
                endpoint_size=3
                end_alpha=1
                alpha=0.5

                i=0
                j=1

                ax3.scatter( plot_info["weights"][i][i][start], plot_info["weights"][j][j][start], color=map_param_to_color(param,alpha=end_alpha), linewidths=start_size )
                ax3.scatter( plot_info["weights"][i][i][-1], plot_info["weights"][j][j][-1], marker="x", color=map_param_to_color(param,alpha=end_alpha), linewidths=endpoint_size )
                #ax1.scatter( plot_info["weights"][i][i][int(stop_step/2)], plot_info["weights"][j][j][int(stop_step/2)],label="middle", color='orange' )
                ax3.set_xlabel("Weight {}->{}".format(i,i))
                ax3.set_ylabel("Weight {}->{}".format(j,j))

                ax3.plot(plot_info["weights"][i][i][start:], plot_info["weights"][j][j][start:], color=map_param_to_color(param), linewidth=center_linewidth  )




    ##############
                ax4 = plt.subplot(2, 2, 4)

                ax4.scatter( plot_info["weights"][i][j][start], plot_info["weights"][j][i][start], color=map_param_to_color(param), linewidths=start_size )
                ax4.scatter( plot_info["weights"][i][j][-1], plot_info["weights"][j][i][-1], marker="x", color=map_param_to_color(param), linewidths=endpoint_size )
                ax4.set_xlabel("Weight {}->{}".format(i,j))
                ax4.set_ylabel("Weight {}->{}".format(j,i))

                ax4.plot(plot_info["weights"][i][j][start:], plot_info["weights"][j][i][start:], color=map_param_to_color(param), linewidth=center_linewidth  )



                print(f"{param} : {recovered_fitness}")

            
            #plt.show()
            fig.set_size_inches(8, 6)
            fig.savefig(f"../rl_learning_plots/figure3/{param_name}_{filename}.png", dpi=300, \
                bbox_inches='tight')





            print(recovered_fitness)
        #quit() #done after 1


def main_perturb_AND_recover():
    save_filename="jason/data/recovery_sweep_perturbed-fit_update-rate.csv"
    seeds=[0]  #range(10)
    jump_sizes=np.arange(0.1, 5, 0.1)
    #jump_sizes=[1,2,3]
    sol_seeds=[0]  #4 is best in 10nnsize
    nnsizes=[10,9,8,7,6,5,4,3,2]  #range(2,11)
    nnsizes=[2]
    learning_durations=[1000]
    #try progressively perturbing the network farther and farther in the same direction...
    line="jumpsize,sol_seed,seed,nnsize,orig_fit,perturbed_fit,perturbed_fit-div-orig_fit,orig_beer_fit,perturbed_beer_fit,perturbed_beer_fit-div-orig_beer_fit,perf_bias,perf_upd_rate,recovered_fitness,timed_passed,learning_duration,recover-div-orig,"
    
    if not os.path.exists(save_filename):
        print("File does not exist, writing to new file")
        write_to_file( save_filename, line,'w' )
        print(line)

    performance_biases=[ 0.01,  0.015, 0.02, 0.05 ]
    performance_biases=[ 0.05 ]
    #performance_biases=[10, 5,2, 1, 0.5, 0.4, 0.3, 0.2, 0.1]
    performance_update_rates=[0.001, 0.025, 0.005, 0.01]

    performance_update_rates=[0.03]

    for learning_duration  in learning_durations:
        for nnsize in nnsizes:
            for sol_seed in sol_seeds:
                for jump_size in jump_sizes:
                    for performance_update_rate in performance_update_rates:
                        for performance_bias in performance_biases:
                            for seed in seeds:
                                line, norm_params, orig_fit = run_perturb_experiment( seed=seed, sol_seed=sol_seed, nnsize=nnsize, jump_size=jump_size)
                                
                                recovered_fitness, plot_info = run_recovery( norm_params, performance_bias=performance_bias, performance_update_rate=performance_update_rate, nnsize=nnsize,learning_duration=learning_duration)
                                timed_passed=plot_info["timed_passed"]
                                #print(recovered_fitness)
                                line2=f"{line}{performance_bias},{performance_update_rate},{recovered_fitness:.4f},{timed_passed:.1f},{learning_duration},{recovered_fitness/orig_fit:.4f}"
                                write_to_file( save_filename, line2,'a' )
                                print(line2)

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
    
def write_to_file(save_filename, line, flag='a'):
    with open( save_filename, flag) as filehandle:
        filehandle.write( line+"\n" )
    filehandle.close()

if __name__ == "__main__":
    main()
