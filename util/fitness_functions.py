import numpy as np
import matplotlib as plt

def fitness_maximize_output_change(ctrnn, stepsize=0.01, init_duration=0, test_duration=10):
    """This function simply provides an average change in output per neuron per time. Optionally can include initial duration to prevent transient changes at start of simulation."""

    init_time = np.arange(0.0, init_duration, stepsize)
    test_time = np.arange(0.0, test_duration, stepsize)

    #allow transients to clear
    ctrnn.initializeState( np.zeros( ctrnn.size ))
    for i in range(len(init_time)):
        ctrnn.step(stepsize)
    
    #evaluate after transient period
    change_in_output=0
    for i in range(len(test_time)):
        pastOutputs = ctrnn.outputs
        ctrnn.step(stepsize)
        currentOutputs = ctrnn.outputs
        change_in_output += np.sum(abs(currentOutputs - pastOutputs) ) 
        
    #average over time and per neuron
    return change_in_output / ctrnn.size / test_duration


def fitness_frequency_match(ctrnn, combo_func=None, show_plots = False, init_duration=10, test_duration=50, \
    clip_change_max=None, min_freq_score=0.001, stepsize=0.01, target_period=4):
    """ A fitness function that attempts to maximize change in output up to some limit as well as have neuron-0 match a target period."""
    time = np.arange(0.0, init_duration+test_duration, stepsize)

    #used to record peaks, only recorded when peaks are reached
    neuron_output_peaks={}
    for i in range(ctrnn.size):
        neuron_output_peaks[i]=[]

    # track the individual neuron outputs
    nn_outputs=np.zeros( (len(time) ,ctrnn.size))
    # track the individual neuron periods
    nn_periods=np.zeros( (len(time),ctrnn.size) )

    #allow transients to clear
    ctrnn.initializeState( np.zeros( ctrnn.size )) 
    i=-1
    testing_started=False
    #for i in range(len(init_time)):
    for t in time:
        i+=1
        ctrnn.step(stepsize)
        nn_outputs[i] = ctrnn.outputs
    
        if not testing_started and t >= init_duration:
            #evaluate after transient period
            change_in_output=0
            frequency_performances=np.arange(0.0, test_duration, stepsize)
            testing_started = True
            first_testing_iter=i
        
        if testing_started:
            change_in_output += np.sum(abs(nn_outputs[i] - nn_outputs[i-1]) ) 


        #TODO  use numpy to do this instead of a loop, filtering for the conditional inside
        # 1. get a list of peaks for each neuron
        for n in range(ctrnn.size):
            current_change = nn_outputs[i][n] - nn_outputs[i-1][n]
            previous_change = nn_outputs[i-1][n] - nn_outputs[i-2][n]
            previous_previous_change = nn_outputs[i-2][n] - nn_outputs[i-3][n]

            # if the previous delta was > 0  and the recent delta is less, then we came down from a peak
            # DO not onlyh check for >=  can have some decsending plateaus that give false positives
            #TODO - can do a boolean check on teh fly to make sure we have a peak (arbitrary depth)
            if previous_change > 0 and current_change < 0 or (previous_previous_change > 0 and previous_change == 0 and current_change < 0 ):
                #print(current_change )
                neuron_output_peaks[n].append( t )

        #TODO: neuron 0 only - must have 3 measures for a total of 2 periods for  reward (2 for perf.)
        if len(neuron_output_peaks[0]) >= 2:  #TODO reward: >=3  
            # 2. check the most recent period (period_cur) vs. the previous period (past_per)
            current_period = neuron_output_peaks[0][-1] - neuron_output_peaks[0][-2]
            #TODO reward:   prev_period = neuron_output_peaks[0][-2] - neuron_output_peaks[0][-3]
            
            # 3. calculate the period_error
            current_period_err = abs( target_period - current_period)
            #TODO reward:   previous_period_err = abs( target_period - prev_period)

            #plotting only -otherwise not required
            for n in range(ctrnn.size):
                if len(neuron_output_peaks[n]) >= 2:
                    nn_periods[i][n] = neuron_output_peaks[n][-1] - neuron_output_peaks[n][-2]
            #only record after init phase
            if testing_started:
                frequency_performance  = 1 / ( 1 + current_period_err)

        else:
            nn_periods[i] = np.zeros(ctrnn.size)
            
            #only record after init phase
            if testing_started:
                frequency_performance = min_freq_score   # minimum performance
        
        #only record after init phase
        if testing_started:
            frequency_performances[i-first_testing_iter]=frequency_performance
    
    freq_perf = np.average( frequency_performances )
    #average change in output over time
    change_perf=change_in_output/test_duration
    
    #clip as needed
    if not clip_change_max == None:
        change_perf=min(clip_change_max,change_perf)  #clip at 1

        #normalize clipping
        change_perf=change_perf/clip_change_max   #1.0 when at max and proportionally downgraded from there.


    if combo_func == None:
        combo_func= lambda freq, change: 0.5 * freq + 0.5 * change

    if show_plots:
        transposed_nn_outputs=nn_outputs.transpose(1,0)
        nn_periods_transposed=nn_periods.transpose(1,0)
        for i in range( ctrnn.size ):
            plt.plot(time,transposed_nn_outputs[i],label=i, alpha=0.5 )
            #Show the part that is calculated... in more opacity
            plt.plot(time[0:first_testing_iter],nn_periods_transposed[i][0:first_testing_iter],label=f"freq n{i}", alpha=0.1 )
            plt.plot(time[first_testing_iter:],nn_periods_transposed[i][first_testing_iter:],label=f"freq n{i}", alpha=0.5 )

        plt.xlabel("Time")
        plt.ylabel("Output")
        plt.legend()
        plt.title("Neural activity")
        print(f"freq_perf={freq_perf} * delta_out={change_in_output/test_duration} clipped:{change_perf} ")
        plt.show()
        print(f"freq-score:{freq_perf}  change-score:{change_perf}"  )

    return combo_func(freq_perf, change_perf), freq_perf, change_perf