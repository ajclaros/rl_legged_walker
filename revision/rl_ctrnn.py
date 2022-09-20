import matplotlib.pyplot as plt
import numpy as np
import math
from ctrnn import CTRNN
from scipy.special import expit

""" This class is an extension of the CTRNN class in this folder.
It was designed to allow the code to be re-used but separated from the original CTRNN implmentation.
This helps to keep the differences as clear as possible.

Adapted from the paper: 
A Bio-inspired Reinforcement Learning Rule to Optimise Dynamical Neural Networks for Robot Control
Tianqi Wei and Barbara Webb (2018)
2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)

Important note from the paper:
"The periods of the fluctuations should be much longer than the periods of the learning objectives, such that
when the new weights cause an effect and the reward arrives later, the weights should still be near the region
that produced a reward."

The authors suggest that no explicit form of elgibility trace is required as long as the above condition is met.

"""


class RL_CTRNN(CTRNN):
    # Constructor including boundaries of acceptable range
    #### kwargs must include:
    def __init__(
        self,
        size,
        weight_range=16,
        bias_range=16,
        tc_min=1,
        tc_max=1,
        init_flux_amp=1,
        max_flux_amp=10,
        flux_period_min=2,
        flux_period_max=10,
        flux_conv_rate=0.1,
        learn_rate=1.0,
        gaussian_mode=False,
        square_oscillation_mode=False,
        bias_init_flux_amp=0,
        bias_max_flux_amp=0,
        bias_flux_period_min=0,
        bias_flux_period_max=0,
        bias_flux_conv_rate=0.1,
        WR=16.0,
        BR=16.0,
        TR=5.0,
        TA=6.0,
    ):
        super().__init__(
            size=size,
            weight_range=weight_range,
            bias_range=bias_range,
            tc_min=tc_min,
            tc_max=tc_max,
            WR=WR,
            BR=BR,
            TR=TR,
            TA=TA,
        )

        # Constants for simulation
        self.init_flux_amp = init_flux_amp
        self.max_flux_amp = max_flux_amp  # The largest allowed (+/1-) value of the DISPLACEMENT for ALL synaptic weight fluctuations
        self.flux_period_min = flux_period_min  # The shortest possible (randomly generated) period for an INDIVIDUAL synaptic weight
        self.flux_period_max = flux_period_max  # The longest possible (randomly generated) period for an INDIVIDUAL synaptic weight
        self.flux_conv_rate = flux_conv_rate  # The rate at which the current fluctation amplitude changes in response to reward (% of flux_period_max )
        self.learn_rate = learn_rate  # The rate at which the synaptic weights shift (up/down) in accordance with the reward signal & fluctation position
        self.gaussian_mode = gaussian_mode  # Whether to use a normal or uniform distribution for random samples in setting periods for each syn weight

        self.square_oscillation_mode = square_oscillation_mode  # experimental mode to look at difference in performance with square wave flux

        # Dynamically changing variables
        self.flux_amp = (
            self.init_flux_amp
        )  # The starting value of the amplitude for ALL synaptic weight fluctuations
        self.inner_flux_periods = np.zeros(
            (size, size)
        )  # current period for each synapse
        self.inner_flux_moments = np.zeros(
            (size, size)
        )  # current moment in time of the oscillation for each synapse

        # These are all ignored if the bias_max_flux_amp is <= 0
        self.bias_flux_amp = bias_init_flux_amp  # The starting value of the amplitude for ALL bias fluctuations
        self.bias_inner_flux_periods = np.zeros(size)  # current period for each synapse
        self.bias_inner_flux_moments = np.zeros(
            size
        )  # current moment in time of the oscillation for each synapse
        self.weight_flux_sign = (
            np.random.binomial(1, 0.5, size=(self.size, self.size)) * 2 - 1
        )
        self.bias_flux_sign = np.random.binomial(1, 0.5, size=(self.size)) * 2 - 1
        self.bias_init_flux_amp = bias_init_flux_amp
        self.bias_max_flux_amp = bias_max_flux_amp
        self.bias_flux_period_min = bias_flux_period_min
        self.bias_flux_period_max = bias_flux_period_max
        self.bias_flux_conv_rate = bias_flux_conv_rate
        self.extended_weights = np.zeros((self.size, self.size))
        self.extended_biases = np.zeros((self.size))
        self.inner_weights = np.zeros((self.size, self.size))
        self.reward = 0
        if bias_max_flux_amp <= 0:
            self.bias_flux_mode = False
        else:
            self.bias_flux_mode = True

    def reset(self):
        self.flux_amp = (
            self.init_flux_amp
        )  # The starting value of the amplitude for ALL synaptic weight fluctuations
        self.inner_flux_periods = np.zeros(
            (self.size, self.size)
        )  # current period for each synapse
        self.inner_flux_moments = np.zeros(
            (self.size, self.size)
        )  # current moment in time of the oscillation for each synapse

        if self.bias_flux_mode:
            self.bias_inner_flux_periods = np.zeros(
                self.size
            )  # current period for each synapse
            self.bias_inner_flux_moments = np.zeros(
                self.size
            )  # current moment in time of the oscillation for each synapse
        super().reset()

    def initializeState(self, v):
        # allow init flux to be used
        self.flux_amp = self.init_flux_amp
        if self.bias_flux_mode:
            self.bias_flux_amp = self.bias_init_flux_amp
        super().initializeState(v)

    def randomize_parameters(self):
        # Each synapse gets a random period to start, make sure to round to prevent weirdness with periods ending at uneven points
        # and resulting in consistent drifting of weights upward or downward based on the rounded error
        if self.gaussian_mode:
            center = (self.flux_period_max + self.flux_period_min) / 2
            dev = (self.flux_period_max - self.flux_period_min) / 4
            self.inner_flux_periods = np.clip(
                np.round(
                    np.random.normal(center, scale=dev, size=(self.size, self.size)), 3
                ),
                self.flux_period_min,
                self.flux_period_max,
            )
            if self.bias_flux_mode:
                self.bias_inner_flux_periods = np.clip(
                    np.round(np.random.normal(center, scale=dev, size=(self.size)), 3),
                    self.bias_flux_period_min,
                    self.bias_flux_period_max,
                )
        else:
            self.inner_flux_periods = np.round(
                np.random.uniform(
                    self.flux_period_min,
                    self.flux_period_max,
                    size=(self.size, self.size),
                ),
                3,
            )
            if self.bias_flux_mode:
                self.bias_inner_flux_periods = np.round(
                    np.random.uniform(
                        self.bias_flux_period_min,
                        self.bias_flux_period_max,
                        size=(self.size),
                    ),
                    3,
                )
        # randomize other parameters in the same way as the regular CTRNN class
        super().randomize_parameters()

    # Provided an external reward signal, update weights and fluctuations accordingly
    def update_weights_and_flux_amp_with_reward(
        self, reward, tolerance=0.0, learning=True
    ):
        # Change in amplitude is based on reward signal and convergence rate
        # Page 2, Equation 4:    dA  = -B * R(t)
        # Reward positive => amp decreases.    Reward negative => amp increases
        # Shift amplitude by percentage of the current self.max_flux_amp   multipled by the reward
        #                               0.1  *     10.0          *  generally small value
        #

        self.reward = reward
        if abs(reward) >= tolerance and learning:
            self.flux_amp -= self.flux_conv_rate * reward
            self.flux_amp = min(
                max(self.flux_amp, 0), self.max_flux_amp
            )  # Keep fluctation amplitude between 0 and max_flux_amp (10)
            # 0 at center, +1 above center, -1, below center
            if self.bias_flux_mode:
                self.bias_flux_amp -= self.bias_flux_conv_rate * reward
                self.bias_flux_amp = min(
                    max(self.bias_flux_amp, 0), self.bias_max_flux_amp
                )  # Keep fluctation amplitude between 0 and max_flux_amp (10)
        inner_flux_center_displacements = self.flux_amp * np.sin(
            self.inner_flux_moments / self.inner_flux_periods * 2 * math.pi
        )
        if self.bias_flux_mode:
            bias_inner_flux_center_displacements = self.bias_flux_amp * np.sin(
                self.bias_inner_flux_moments
                / self.bias_inner_flux_periods
                * 2
                * math.pi
            )
        # Page 2, Equation 3:    dC  = a ( W(t) - C) * R(t)
        #   NxN                  NxN                1.0              NxN                     -1              * small value
        # limiting to inter neuron weights, comment out and uncomment 129,130 for normal behavior
        self.extended_weights = self.inner_weights + inner_flux_center_displacements
        if abs(reward) <= tolerance and learning:
            self.inner_weights = np.clip(
                self.inner_weights
                + self.learn_rate * inner_flux_center_displacements * 0.1,
                -self.weight_range,
                self.weight_range,
            )
        else:
            self.inner_weights = np.clip(
                self.inner_weights
                + self.learn_rate * inner_flux_center_displacements * reward,
                -self.weight_range,
                self.weight_range,
            )
        if self.bias_flux_mode:
            self.extended_biases = self.biases + bias_inner_flux_center_displacements
            if abs(reward) <= tolerance and learning:
                self.biases = np.clip(
                    self.biases
                    + self.learn_rate * bias_inner_flux_center_displacements * 0.1,
                    -self.bias_range,
                    self.bias_range,
                )
            else:
                self.biases = np.clip(
                    self.biases
                    + self.learn_rate * bias_inner_flux_center_displacements * reward,
                    -self.bias_range,
                    self.bias_range,
                )

    #        for i in range(self.size):
    #            for j in range(self.size):
    #                if i!=j:
    #                    self.inner_weights[i,j] = np.clip( self.inner_weights[i,j] + self.learn_rate * inner_flux_center_displacements[i,j] * reward, -self.weight_range, self.weight_range)
    #        #if self.bias_flux_mode:
    #        #    self.biases = np.clip( self.biases + self.learn_rate * bias_inner_flux_center_displacements * reward, -self.bias_range, self.bias_range)
    #        self.extended_biases= self.biases + bias_inner_flux_center_displacements
    #        self.extended_biases= 1

    # This function is called in place of:   self.Weight.T in the classic CTRNN
    # In this code the individual instantaneous time within a period (moments) are scaled by the individual periods and current amplitude
    def calc_inner_weights_with_flux(self):
        #  NxN        NxN             1                      NxN                  NxN
        fluxxed_weights = self.inner_weights + self.flux_amp * np.sin(
            self.inner_flux_moments / self.inner_flux_periods * 2 * math.pi
        ) * np.sign(self.inner_flux_periods)
        return fluxxed_weights.T

    def calc_bias_with_flux(self):
        if self.bias_flux_mode:
            #  NxN        NxN             1                      NxN                  NxN
            fluxxed_bias = self.biases + self.bias_flux_amp * np.sin(
                self.bias_inner_flux_moments
                / self.bias_inner_flux_periods
                * 2
                * math.pi
            ) * np.sign(self.bias_inner_flux_periods)
        else:
            print("This should NOT be called when bias_flux_mode is false. Exiting...")
            quit()
        return fluxxed_bias.T

    # Replaces the step function from parent CTRNN class
    def step(self, dt):
        self.inner_flux_moments += dt
        if self.bias_flux_mode:
            self.bias_inner_flux_moments += dt
        for i in range(self.size):
            for j in range(self.size):
                # if period is reached/exceeded randomize new one...
                if self.inner_flux_moments[i][j] > abs(self.inner_flux_periods[i][j]):
                    # reset synaptic fluctation moment and pick new period
                    self.inner_flux_moments[i][j] = 0
                    # round to avoid the offsets causing the fluctuation to involvuntarily move up/down
                    if self.gaussian_mode:
                        center = (self.flux_period_max + self.flux_period_min) / 2
                        dev = (self.flux_period_max - self.flux_period_min) / 4
                        self.inner_flux_periods[i][j] = np.clip(
                            np.round(np.random.normal(center, scale=dev), 3),
                            self.flux_period_min,
                            self.flux_period_max,
                        )
                    else:
                        self.inner_flux_periods[i][j] = np.round(
                            np.random.uniform(
                                self.flux_period_min, self.flux_period_max
                            ),
                            3,
                        )
                    self.inner_flux_periods[i][j] *= self.weight_flux_sign[i][j]

            # Adjust biases when enabled
            if self.bias_flux_mode:
                if self.bias_inner_flux_moments[i] > self.bias_inner_flux_periods[i]:
                    # reset synaptic fluctation moment and pick new period
                    self.bias_inner_flux_moments[i] = 0
                    # round to avoid the offsets causing the fluctuation to involvuntarily move up/down
                    if self.gaussian_mode:
                        center = (
                            self.bias_flux_period_max + self.bias_flux_period_min
                        ) / 2
                        dev = (
                            self.bias_flux_period_max - self.bias_flux_period_min
                        ) / 4
                        self.bias_inner_flux_periods[i] = np.clip(
                            np.round(np.random.normal(center, scale=dev), 3),
                            self.bias_flux_period_min,
                            self.bias_flux_period_max,
                        )
                    else:
                        self.bias_inner_flux_periods[i] = round(
                            np.random.uniform(
                                self.bias_flux_period_min, self.bias_flux_period_max
                            ),
                            3,
                        )
                        self.bias_inner_flux_periods *= self.bias_flux_sign[i]

        netinput = self.inputs + np.dot(self.extended_weights.T, self.outputs)
        # netinput = self.inputs + np.dot( self.calc_inner_weights_with_flux(), self.outputs)
        self.voltages += dt * (self.inv_time_constants * (-self.voltages + netinput))

        if self.bias_flux_mode:
            self.outputs = sigmoid(self.voltages + self.extended_biases.T)
            # self.outputs = sigmoid( self.voltages + self.calc_bias_with_flux() )
        else:
            self.outputs = sigmoid(self.voltages + self.biases)

    def recoverParameters(self, inner=True):
        if inner:
            return np.append(
                self.inner_weights.reshape(self.inner_weights.size) / self.WR,
                [self.biases / self.BR, (self.time_constants - self.TA) / self.TR],
            )
        else:
            return np.append(
                self.extended_weights.reshape(self.extended_weights.size) / self.WR,
                [
                    self.extended_biases / self.BR,
                    (self.time_constants - self.TA) / self.TR,
                ],
            )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
