import leggedwalker
import ctrnn
import numpy as np
from rl_ctrnn import RL_CTRNN
from fitnessFunction import fitnessFunction

class WalkingTask(RL_CTRNN):

    def __init__(self, size=2, duration=2000.0, stepsize=0.01,
                 reward_func=None, performance_func=None,
                 running_window_mode=True, running_window_size=400,
                 performance_update_rate=0.005, performance_bias=0.007,
                 init_flux_amp=2.75, max_flux_amp=10, flux_period_min=300,
                 flux_period_max=400, flux_conv_rate=0.004, learn_rate=0.008,
                 bias_init_flux_amp=2.75, bias_max_flux_amp=10,
                 bias_flux_period_min=300,
                 bias_flux_period_max=400, bias_flux_conv_rate=0.004,
                 WR= 16.0, BR=16.0, TR=5.0, TA=6.0
                 ):
        super().__init__(size,
                         init_flux_amp=init_flux_amp, max_flux_amp=max_flux_amp,
                         flux_period_min=flux_period_min,
                         flux_period_max=flux_period_max,
                         flux_conv_rate=flux_conv_rate, learn_rate=learn_rate,
                         bias_init_flux_amp=bias_init_flux_amp,
                         bias_max_flux_amp=bias_max_flux_amp,
                         bias_flux_period_min=bias_flux_period_min,
                         bias_flux_period_max=bias_flux_period_max,
                         bias_flux_conv_rate=bias_flux_conv_rate,
                         WR=WR, BR=BR, TR=TR, TA=TA
                         )
        self.size = size
        self.time_step = 0
        self.duration = duration
        self.stepsize = stepsize
        self.reward = 0
        self.time = np.arange(0,self.duration, self.stepsize)
        self.distance_hist = np.zeros((self.time.size))
        self.performance_hist = np.zeros((self.time.size))
        self.running_window_mode = running_window_mode
        self.running_average_performances = np.zeros(len(self.time) )
        if self.running_window_mode:
            self.running_window_size = int(running_window_size/self.stepsize)
            self.sliding_window = np.zeros(self.running_window_size)
        else:
            self.performance_update_rate = performance_update_rate  # how quickly the running average performance is updated
        # uses performance to determine reward
        if reward_func is None:
            self.reward_func = self.default_reward_func
        else:
            self.reward_func = reward_func
        # determines how performance is measured
        if performance_func is None:
            self.performance_func = self.default_performance_func
        else:
            self.performance_func = performance_func



    def save(self, filename):
        np.savez(filename, duration=self.duration, stepsize=self.stepsize,
                 time=self.time, weight_hist = self.weight_hist,
                 bias_hist=self.bias_hist, distanct_hist=self.distance_hist,
                 performance_hist=self.performance_hist, extended_weight_hist=self.extended_weight_hist,
                 extended_bias_hist=self.extended_bias_hist,
                 neural_outputs = self.neural_outputs,
                 distance_hist = self.distance_hist,
                 genotype = self.recoverParameters()
                 )
    def save2(self, filename):
        np.savez(filename,
                 genotype = self.recoverParameters())


    def default_reward_func(self,  distance, learning=True):
        performance = self.performance_func(distance)
        running_average_performance = self.running_average_performances[self.time_step-1]

        # Current instantaneous performance vs. the current running average (NOT the previous instantaneous performance)
        if not learning:
            return 0

        return  performance - running_average_performance

    def default_performance_func(self, body):
        self.distance_hist[self.time_step] = body.cx

        performance = self.distance_hist[self.time_step] - self.distance_hist[self.time_step-1]
        performance/=self.stepsize

        self.performance = performance
        self.performance_hist[self.time_step] = performance

        if self.running_window_mode:
            # rotate everything forward
            self.sliding_window = np.roll(self.sliding_window, 1)
            # replace oldest value (which just rolled to the front)
            self.sliding_window[0] = performance
            # current running average
            self.running_average_performances[self.time_step] = np.mean(self.sliding_window)

        else:
            self.running_average_performances[self.time_step] = self.running_average_performances[self.time_step-1] * (1 - self.performance_update_rate ) \
                + self.performance_update_rate * performance
        return performance

    def simulate(self, body, datalogger=None, track=False,learning_start = None, verbose=0.1, generator_type='RPG', configuration = [0], tolerance=0.0):

        if datalogger:
            datalogger.data['startgenome'] = self.recoverParameters()
            datalogger.data['track_fitness'] = np.zeros(self.time.size)
            datalogger.data['track_fitness'][self.time_step] = fitnessFunction(datalogger.data['startgenome'], N=self.size, generator_type= generator_type, configuration=configuration)
        for i,t in enumerate(self.time):

            #if track==Tree, runs fitnessfunction every given percentage:
            #verbose=0.1, runs fitnessFunction 10 at equal intervals
            if verbose>0.0 and i %(self.time.size*verbose) == 0 and verbose<1.0:
                print("{}% completed...".format(i/self.time.size *100))
                if track:
                    datalogger.data['track_fitness'][self.time_step] = fitnessFunction(self.recoverParameters(), N=self.size, generator_type= generator_type, configuration=configuration)
            if generator_type=='RPG':
                self.setInputs(np.array([body.anglefeedback()] * self.size))
            elif generator_type=='CPG':
                self.setInputs(np.array([0] * self.size))
            self.step(self.stepsize)
            body.stepN(self.stepsize, self.outputs, configuration)
            if self.time_step<learning_start:
                self.reward = self.default_reward_func(body, learning=False)
                #updating with 0 reward
                self.update_weights_and_flux_amp_with_reward(self.reward, tolerance=tolerance, learning=False)
            else:
                reward = self.default_reward_func(body, learning=True)
                self.reward = reward
                self.update_weights_and_flux_amp_with_reward(reward, tolerance=tolerance)
            if self.running_window_mode:
                self.sliding_window = np.roll(self.sliding_window, 1)
                self.sliding_window[0] = self.performance_hist[self.time_step]
                self.running_average_performances[self.time_step] = np.mean(self.sliding_window)
            else:
                self.running_average_performances[self.time_step] = self.running_average_performances[self.time_step-1] * (1-self.performance_update_rate) + self.performance_update_rate * self.performance_hist[self.time_step]
            if datalogger:
                for key in datalogger.data.keys():
                    if key in ["startgenome", 'track_fitness']:
                        if key =="track_fitness" and (self.time_step<self.time.size-1)/self.stepsize :
                            datalogger.data['track_fitness'][self.time_step+1] = datalogger.data['track_fitness'][self.time_step]
                        continue
                    elif "hist" in key or 'running' in key:
                        datalogger.data[key][self.time_step] = self.__dict__[key][self.time_step]
                    elif key in ['angle', 'omega','distance']:
                        datalogger.data['angle'][self.time_step] = body.angle
                        datalogger.data['omega'][self.time_step] = body.omega
                        datalogger.data['distance'][self.time_step] = body.cx
                    else:
                        datalogger.data[key][self.time_step] = self.__dict__[key]

            self.time_step += 1

        self.time_step = 0
        if datalogger:
            datalogger.data['track_fitness'][-1] = fitnessFunction(self.recoverParameters(), N=self.size, generator_type= generator_type, configuration=configuration)
            datalogger.data['learning_start'] = learning_start
            datalogger.data['tolerance'] = tolerance
            datalogger.data['size'] = self.size
            datalogger.data['duration'] = self.duration
            datalogger.data['stepsize'] = self.stepsize
            datalogger.data["performance_hist"] = self.performance_hist
            datalogger.data["running_average"] = self.running_average_performances
            if 'running_average_perforamnces' in datalogger.data.keys():
                datalogger.data['running_average'] = datalogger.data.pop('running_average_perforamnces')
