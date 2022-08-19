import leggedwalker
from jason import ctrnn
import numpy as np
from jason.rl_ctrnn import RL_CTRNN
from fitnessFunction import fitnessFunction

class WalkingTask(RL_CTRNN):

    def __init__(self, size, duration=1000.0, stepsize=0.01,
                 reward_func=None, performance_func=None,
                 running_window_mode=False, running_window_size=1000,
                 performance_update_rate=0.005, performance_bias=0.007,
                 init_flux_amp=1, max_flux_amp=10, flux_period_min=2,
                 flux_period_max=10, flux_conv_rate=0.001, learn_rate=1.0,
                 bias_init_flux_amp=1.0, bias_max_flux_amp=0,
                 bias_flux_period_min=0,
                 bias_flux_period_max=10, bias_flux_conv_rate=0.1,
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
        self.time = np.arange(0,self.duration, self.stepsize)
        self.distance_hist = np.zeros((self.time.size))
        self.performance_hist = np.zeros((self.time.size))
        self.running_window_mode = running_window_mode
        self.running_average_performances = np.zeros(len(self.time) )
        if self.running_window_mode:
            self.running_window_size = running_window_size
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

        #lastOutput = self.performance_hist[self.time_step-1] * self.stepsize
        #performance = (body.cx- lastOutput)/self.stepsize

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

    def simulate(self, body, datalogger=None, track=False, trackpercent = 1.00, learning_start = None, logfitness=False):

        if datalogger:
            datalogger.data['startgenome'] = self.recoverParameters()
            datalogger.data['size'] = self.size
            datalogger.data['duration'] = self.duration
            datalogger.data['stepsize'] = self.stepsize
            datalogger.data['weightHist'] = np.zeros((self.time.size, self.size, self.size))
            datalogger.data['biasHist'] = np.zeros((self.time.size, self.size))
            datalogger.data['distanceHist'] = np.zeros((self.time.size))
            datalogger.data['performanceHist'] = np.zeros((self.time.size))
            datalogger.data['biasesArr'] = np.zeros((self.time.size, self.size))
            datalogger.data['fluxAmpArr'] = np.zeros((self.time.size))
            datalogger.data['extendedWeightHist'] = np.zeros((self.time.size, self.size, self.size))
            datalogger.data['extendedBiasHist'] = np.zeros((self.time.size, self.size))
            datalogger.data['ampPeriodArr'] = np.zeros((self.time.size, self.size, self.size))
            datalogger.data['biasesPeriodArr'] = np.zeros((self.time.size, self.size, self.size))
            datalogger.data['feedback'] = np.zeros((self.time.size))
            datalogger.data['runningAveragePerformances'] = np.zeros(len(self.time))
            datalogger.data['neuralOutputs'] = np.zeros((len(self.time), self.size))
            datalogger.data['rewardHist']= np.zeros((self.time.size))
            if logfitness:
                datalogger.data['trackFitness'] = np.zeros((self.time.size))
        for i,t in enumerate(self.time):
            #if logfitness==Tree, runs fitnessfunction every given percentage:
            #trackpercent=0.1, runs fitnessFunction 10 at equal intervals
            if logfitness:
                datalogger.data['trackFitness'][self.time_step] = datalogger.data['trackFitness'][self.time_step-1]
            if i %(self.time.size*trackpercent) == 0 and trackpercent>=0:# and i!=0:
                print("{}% completed...".format(i/self.time.size *100))
                if logfitness:
                    datalogger.data['trackFitness'][self.time_step] = fitnessFunction(self.recoverParameters(), N=self.size)

            if datalogger:
                datalogger.data['weightHist'][self.time_step] = self.inner_weights
                datalogger.data['biasHist'][self.time_step] = self.biases

            self.setInputs(np.array([body.anglefeedback()] * self.size))
            self.step(self.stepsize)
            body.step1(self.stepsize, self.outputs)

            if self.time_step<learning_start:
                reward = self.default_reward_func(body, learning=False)
                self.update_weights_and_flux_amp_with_reward(reward)
            else:
                reward = self.default_reward_func(body, learning=True)
                self.update_weights_and_flux_amp_with_reward(reward)

            if datalogger:
                datalogger.data['rewardHist'][self.time_step] = reward
                datalogger.data['neuralOutputs'][self.time_step] = self.outputs
                datalogger.data['biasesArr'][self.time_step] = self.bias_flux_amp
                datalogger.data['fluxAmpArr'][self.time_step] = self.flux_amp
                datalogger.data['ampPeriodArr'][self.time_step] = self.inner_flux_periods
                datalogger.data['biasesPeriodArr'][self.time_step]= self.bias_inner_flux_periods
                datalogger.data['extendedWeightHist'][self.time_step] = self.extended_weights
                datalogger.data['extendedBiasHist'][self.time_step] = self.extended_biases
                datalogger.data['feedback'][self.time_step] = body.anglefeedback()


            if self.running_window_mode:
                # rotate everything forward
                self.sliding_window = np.roll(self.sliding_window, 1)
                # replace oldest value (which just rolled to the front)
                self.sliding_window[0] = self.performance_hist[self.time_step]
                # current running average
                self.running_average_performances[self.time_step] = np.mean(self.sliding_window)
            else:
                self.running_average_performances[self.time_step] = self.running_average_performances[self.time_step-1] * (1-self.performance_update_rate) + self.performance_update_rate * self.performance_hist[self.time_step]
            self.time_step += 1

        self.time_step = 0
        if datalogger:
            datalogger.data["performanceHist"] = self.performance_hist
            datalogger.data["distanceHist"] = self.distance_hist
            datalogger.data["runningAverage"] = self.running_average_performances
            datalogger.data['endGenome'] = self.recoverParameters()
            datalogger.data['duration'] = self.duration
            datalogger.data['learningStart'] = learning_start
