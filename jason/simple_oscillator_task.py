from matplotlib import colors
from jason.rl_ctrnn import RL_CTRNN
from jason.ctrnn import CTRNN
import numpy as np
import random
import sys

from vis.plot_rl_figures import plot, subplots


# Goal is to produce constantly (and maximally) changing neuron outputs
# A default reward function is provided which is reliant on external information tracking the NNs performance
class SimpleOscillatorTask():

    def __init__(self, duration, stepsize=0.01, stop_at_convergence=True, \
        reward_func = None, performance_func = None,\
        convergence_epsilon = 0.05, performance_update_rate=0.005, performance_bias=0.007, \
            running_window_mode=False, running_window_size=2000  ):
        self.duration = duration                                    # how long to run at max
        self.stepsize = stepsize                                    # time to sim per iteration (i.e. 0.01)
        self.stop_at_convergence = stop_at_convergence              # stop simulating if the weights converge to a small enough value
        self.running_window_mode = running_window_mode
        if self.running_window_mode:
            self.running_window_size = running_window_size
            self.sliding_window = np.zeros(self.running_window_size)
            #self.sliding_window = np.ones(self.running_window_size)*-1*performance_bias
        else:
            self.performance_update_rate = performance_update_rate  # how quickly the running average performance is updated
        self.performance_bias = performance_bias                    # small negative offset to encourage exploration
        self.convergence_epsilon = convergence_epsilon              # how small of a value to consider converged (not tested much)
        
        if reward_func == None:                           # uses performance to determine reward
            self.reward_func = self.default_reward_func
        else:
            self.reward_func = reward_func
        if performance_func == None:                      # determines how performance is measured
            self.performance_func = self.default_performance_func
        else:
            self.performance_func = performance_func
    
    # Default reward function
    # Assumes a performances and running_average_performances are recorded
    # Reward is calculated based upon the difference between the current performance and running average
    # Positive when performance is increasing, negative when decreasing
    # Can replace this function with a different one as desired
    def default_reward_func(self, nn):
        performance = self.performance_func(nn)
        running_average_performance = self.running_average_performances[self.step-1]
        # Current instantaneous performance vs. the current running average (NOT the previous instantaneous performance)
        return performance - running_average_performance

    # Default reward function
    # Assumes network has been advanced to the next step, but performance has not been measured
    # Assumes previous outputs, performances and running_average_performances are recorded
    def default_performance_func(self, nn):
        lastOutput = self.outputs[self.step-1]
        #                             difference in output                   penalty for not increasing
        performance = (np.sum( abs(nn.outputs - lastOutput ) ) ) / nn.size - self.performance_bias
        self.performances[self.step] = performance

        if self.running_window_mode:
            #rotate everything forward
            self.sliding_window = np.roll(self.sliding_window, 1)
            #replace oldest value (which just rolled to the front)
            self.sliding_window[0] = performance
            #current running average
            self.running_average_performances[self.step] = np.mean(self.sliding_window)

        else:
            self.running_average_performances[self.step] = self.running_average_performances[self.step-1] * (1 - self.performance_update_rate ) \
                + self.performance_update_rate * performance
        return performance

    def simulate(self, nn, ignore_transients=50, show_plots = False, show_subplots=False, \
        record_data=True, save_data_filename=None, record_array=None, record_every_n_steps=None,\
            save_nn_snapshots=None, ctrnn_save_directory=None):
        if show_plots or show_subplots:
            record_data=True #cannot show unless recorded

        size = nn.size
        stepsize = self.stepsize

        #reset values each time - these are used in calculating
        time = np.arange(0.0, self.duration, stepsize)
        self.running_average_performances=np.zeros(len(time) )
        self.performances=np.zeros(len(time) )
        
        self.outputs=np.zeros((len(time),size))

        # Step is the current integer index from 0 up to len(time)
        self.step = 0
        nn.initializeState( np.zeros(size) )

        # Run simulation
        stop_step = -1        #if the experiment ends early due to convergence only plot up to that moment in time


        #track information to be used for plotting later
        plot_info={}
        
        if record_data:
            # time passed to use for plotting
            plot_info["time"] = time


            if record_array == None or "amps" in record_array:
                # amplitude of fluctuation at each timestep of sim
                plot_info["amps"] = np.zeros(len(time) )
            
            if record_array == None or "weights" in record_array:
                # underlying true centers of weights of each synaptic connection (A->B) at each timestep of sim
                plot_info["weights"] = np.zeros((len(time),size,size))

            if record_array == None or "flux_weights" in record_array:
                # the effective weights (including fluctuation) of each syn. conn. at each timestep of sim
                plot_info["flux_weights"] = np.zeros((len(time),size,size))
            
            if record_array == None or "flux_weights" in record_array:
                #rewards received at each timestep
                plot_info["rewards"]=np.zeros(len(time) )
            if record_array == None or "biases" in record_array:
                if nn.bias_flux_mode:
                    plot_info["biases"]=np.zeros((len(time),size) )
                    plot_info["flux_biases"]=np.zeros((len(time),size) )

        for t in time:
            #avoid inflating reward in first step (going form zero to 0.5 is a huge change!)
            #This steps the reward at 0 (no change in output)
            if self.step == 0:
                self.outputs[-1] = nn.outputs
            

            if not save_nn_snapshots == None:
                if t in save_nn_snapshots:
                    nn.save_json( f"{ctrnn_save_directory}time-{t}.json")


            nn.step(stepsize)

            reward = self.reward_func(nn)

            if t < ignore_transients:
                reward = 0
            
            #this should always happen to be consistent
            nn.update_weights_and_flux_amp_with_reward( reward )

            #record useful information
            self.outputs[self.step] = nn.outputs
            if self.running_window_mode:
                #rotate everything forward
                self.sliding_window = np.roll(self.sliding_window, 1)
                #replace oldest value (which just rolled to the front)
                self.sliding_window[0] = self.performances[self.step]
                #current running average
                self.running_average_performances[self.step] = np.mean(self.sliding_window)
            else:
                self.running_average_performances[self.step] = self.running_average_performances[self.step-1] * (1-self.performance_update_rate) + self.performance_update_rate * self.performances[self.step]
            
            if record_data:  #and (record_every_n_steps == None or self.step % record_every_n_steps == 0):
                # print(self.step)
                # print(  self.step % record_every_n_steps == 0  )
                if record_array == None or "weights" in record_array:
                    plot_info["weights"][self.step] = nn.inner_weights
                if record_array == None or "flux_weights" in record_array:
                    #have to reverse the transpose to get the same coordinates (gets multipled)
                    plot_info["flux_weights"][self.step] = nn.calc_inner_weights_with_flux().T
                if record_array == None or "rewards" in record_array:
                    plot_info["rewards"][self.step] = reward
                if record_array == None or "amps" in record_array:
                    plot_info["amps"][self.step] = nn.flux_amp
                    #print(nn.flux_amp)
                if record_array == None or "biases" in record_array:
                    if nn.bias_flux_mode:
                        plot_info["biases"][self.step] = nn.biases
                        plot_info["flux_biases"][self.step] = nn.calc_bias_with_flux() if hasattr(nn, 'calc_bias_with_flux') else nn.biases
            
            self.step += 1
            
            #if stopping at convergence then when FluxAmp gets to zero stop
            if nn.flux_amp < self.convergence_epsilon and self.stop_at_convergence:
                stop_step = self.step
                break
        #END OF SIMULATION

        if record_data:
            if record_every_n_steps == None:
                if record_array == None or "weights" in record_array:
                    plot_info["weights"] = plot_info["weights"].transpose(1,2,0)
                if record_array == None or "flux_weights" in record_array:
                    plot_info["flux_weights"] = plot_info["flux_weights"].transpose(1,2,0)

                if record_array == None or "biases" in record_array:
                    if nn.bias_flux_mode:
                        plot_info["biases"] = plot_info["biases"].transpose(1,0)
                        plot_info["flux_biases"] = plot_info["flux_biases"].transpose(1,0)
                if record_array == None or "outputs" in record_array:
                    #switch the dimensions for easier plotting
                    plot_info["outputs"] = self.outputs.transpose(1,0)
                if record_array == None or "running_average_performances" in record_array:
                    plot_info["running_average_performances"] = self.running_average_performances
                if record_array == None or "performances" in record_array:
                    plot_info["performances"] = self.performances
            else:
                if record_array == None or "amps" in record_array:
                    plot_info["amps"] = plot_info["amps"][::record_every_n_steps]
                
                if record_array == None or "weights" in record_array:
                    plot_info["weights"] = plot_info["weights"][::record_every_n_steps].transpose(1,2,0)
                if record_array == None or "flux_weights" in record_array:
                    plot_info["flux_weights"] = plot_info["flux_weights"][::record_every_n_steps].transpose(1,2,0)

                if record_array == None or "biases" in record_array:
                    if nn.bias_flux_mode:
                        plot_info["biases"] = plot_info["biases"][::record_every_n_steps].transpose(1,0)
                        plot_info["flux_biases"] = plot_info["flux_biases"][::record_every_n_steps].transpose(1,0)

                if record_array == None or "outputs" in record_array:
                    #switch the dimensions for easier plotting
                    plot_info["outputs"] = self.outputs[::record_every_n_steps].transpose(1,0)
                if record_array == None or "running_average_performances" in record_array:
                    plot_info["running_average_performances"] = self.running_average_performances[::record_every_n_steps]
                if record_array == None or "performances" in record_array:
                    plot_info["performances"] = self.performances[::record_every_n_steps]

            if show_plots:
                plot(self, plot_info, time, stop_step)
            if show_subplots:
                subplots(self, plot_info, time, stop_step)
        #end if record_data 
        
        converged = t < time[-1]

        plot_info["time_passed"]=t
        #redundant but useful for plotting later
        if record_every_n_steps == None:
            plot_info["time"]=time
        else:
            plot_info["time"]=time[::record_every_n_steps]
        
        # print( "time len")
        # print( len(plot_info["time"]) )

        if not save_data_filename == None:
            self.save_plot_info( plot_info, nn.size, nn.bias_flux_mode, save_data_filename )
        # print( "time len")
        # print( len(plot_info["time"]) )
        # quit()

        return nn, plot_info, converged
    
    def save_plot_info(self, plot_info, nn_size, nn_bias_flux_mode, filename ):
        keys=plot_info.keys()
        header="time,"
        if "amps" in plot_info:
            header+="amps,"
        if "rewards" in plot_info:
            header+="rewards,"
        if "performances" in plot_info:
            header+="performances,"
        if "running_average_performances" in plot_info:
            header+="running_average_performances,"
        if "outputs" in plot_info:
            for i in range(nn_size):
                header+=f"output{i},"
        if "weights" in plot_info:
            for i in range(nn_size):
                for j in range(nn_size):
                    header+=f"w{i}_{j},"
        if "flux_weights" in plot_info:
            for i in range(nn_size):
                for j in range(nn_size):
                    header+=f"flux_w{i}_{j},"
        if "biases" in plot_info:
            if nn_bias_flux_mode:
                for i in range(nn_size):
                    header+=f"bias{i},"
                for i in range(nn_size):
                    header+=f"flux_bias{i},"
        # for i in range(nn_size):
        #     header+=f"tc{i},"
       
        write_to_file(filename, header, 'w' )
        

        for t in range(len(plot_info["time"])):
            line=f'{plot_info["time"][t]:0.2f},'
            if "amps" in plot_info:
                line+=f'{plot_info["amps"][t]:0.2f},'
            if "rewards" in plot_info:
                line+=f'{plot_info["rewards"][t]:0.4f},'
            if "performances" in plot_info:
                line+=f'{plot_info["performances"][t]:0.4f},'
            if "running_average_performances" in plot_info:
                line+=f'{plot_info["running_average_performances"][t]:0.4f},'
     
                
                
            if "outputs" in plot_info:
                for i in range(nn_size):
                    line+=f'{plot_info["outputs"][i][t]:0.6f},'
            if "weights" in plot_info:
                for i in range(nn_size):
                    for j in range(nn_size):
                        line+=f'{plot_info["weights"][i][j][t]:0.2f},'
            if "flux_weights" in plot_info:
                for i in range(nn_size):
                    for j in range(nn_size):
                        line+=f'{plot_info["flux_weights"][i][j][t]:0.2f},'
            if "biases" in plot_info:
                if nn_bias_flux_mode:
                    for i in range(nn_size):
                        line+=f'{plot_info["biases"][i][t]:0.2f},'
                    for i in range(nn_size):
                        line+=f'{plot_info["flux_biases"][i][t]:0.2f},'
            # for i in range(nn_size):
            #     line+=f'{plot_info["outputs"][i][t]},'

            write_to_file(filename, line, 'a' )




def write_to_file(save_filename, line, flag='a'):
    with open( save_filename, flag) as filehandle:
        filehandle.write( line+"\n" )
    filehandle.close()
