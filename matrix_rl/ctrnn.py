import numpy as np
from scipy.special import expit

# Parent class which can be extended to provide additional functionality
# To help keep track of matrix dimensions I use
# N for the size of the network (fully connected neurons, including self loops NxN )


class CTRNN:
    # Constructor including boundaries of acceptable range
    def __init__(
        self,
        size,
        weight_range=16,
        bias_range=16,
        tc_min=1,
        tc_max=1,
        WR=16.0,
        BR=16.0,
        TR=5.0,
        TA=6.0,
    ):
        self.size = size  # number of neurons in the network (N)
        self.voltages = np.zeros(size)  # neuron activation vector
        self.time_constants = np.ones(size)  # time-constant vector
        self.biases = np.zeros(size)  # bias vector
        self.weights_arr = np.zeros((size, size))  # inner weight matrix NxN
        self.outputs = np.zeros(size)  # neuron output vector
        self.inputs = np.zeros(size)
        self.inv_time_constants = 1 / self.time_constants
        self.WR = WR
        self.BR = BR
        self.TR = TR
        self.TA = TA
        # Parameter ranges - FIXED values
        self.weight_range = weight_range
        self.bias_range = bias_range
        self.tc_min = tc_min
        self.tc_max = tc_max

    def reset(self):
        self.voltages = np.zeros(self.size)
        self.time_constants = np.ones(self.size)
        self.biases = np.zeros(self.size)
        self.inner_weights = np.zeros((self.size, self.size))
        self.outputs = np.zeros(self.size)

    # allow runs to be easily reproduced while still assigning random initial starting states

    def randomize_parameters_with_seed(self, seed):
        np.random.seed(seed)
        self.randomize_parameters()

    def setSize(self, size):
        self.size = size

    def setVoltages(self, voltages):
        self.voltages = voltages

    def setTimeConstants(self, time_constants):
        self.time_constants = time_constants * self.TR + self.TA
        self.inv_time_constants = 1.0 / self.time_constants

    def setBiases(self, biases):
        self.biases = biases * self.BR

    def setWeights(self, inner_weights):
        self.inner_weights = inner_weights * self.WR

    def setInputs(self, inputs):
        self.inputs = inputs

    def randomize_parameters(self):
        self.inner_weights = np.random.uniform(
            -self.weight_range, self.weight_range, size=(self.size, self.size)
        )
        self.biases = np.random.uniform(
            -self.bias_range, self.bias_range, size=(self.size)
        )
        self.time_constants = np.random.uniform(
            self.tc_min, self.tc_max, size=(self.size)
        )
        self.inv_time_constants = 1.0 / self.time_constants

    def initializeState(self, v):
        self.voltages = v
        self.inv_time_constants = 1.0 / self.time_constants
        self.outputs = expit(self.voltages + self.biases)

    # step without input - used for oscillator task

    def step(self, dt):
        netinput = self.inputs + np.dot(self.inner_weights.T, self.outputs)
        self.voltages += dt * (self.inv_time_constants * (-self.voltages + netinput))
        self.outputs = expit(self.voltages + self.biases)

    def mapGenome(self, genome):
        self.setWeights(genome[0 : self.size * self.size].reshape(self.size, self.size))
        self.setBiases(
            genome[self.size * self.size : self.size * self.size + self.size]
        )
        self.setTimeConstants(genome[self.size * self.size + self.size :])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# import matplotlib.pyplot as plt

# from ctrnn import CTRNN

# t0 = ti.time()
# x = CTRNN(3)
testgenome = np.array(
    [
        0.04774359502446968,
        -0.683711607305986,
        0.45036338411104737,
        0.9721092700062304,
        0.7891519578423444,
        -0.00960243655211588,
        -0.9358149684117485,
        -0.8152701212733787,
        0.6207119728559448,
        0.28996795347325205,
        0.3639871362038097,
        -0.6154338363438252,
        0.4644851766806753,
        -0.4605993067803686,
        -0.4491022368326481,
    ]
)
# x.mapGenome(testgenome)
# x.initializeState(np.zeros(3))


def runCTRNN(duration, dt=0.1):
    time = np.arange(0, duration, dt)
    outputs = np.zeros((time.size, 3))
    for (
        i,
        t,
    ) in enumerate(time):
        x.step(0.1)
        outputs[i, :] = x.outputs
    return outputs


# outputs = runCTRNN(220)
# print(outputs[-1,-1])
# for i in range(100):
#     print(outputs[i][-1])

# duration = 1
# outputs = runCTRNN(duration)
# print(outputs[-1])
