import numpy as np
import numpy.typing as npt
from ctrnn import CTRNN
from learning_rule import LRule
from scipy.special import expit


class RL_CTRNN(CTRNN):
    def __init__(
        self,
        size: int,
        genome: npt.NDArray,
        duration: int = 2000,
        learn_rate: float = 0.0,
        conv_rate: float = 0.0,
        delay: int = 200,
        window_size: int = 220,
        p_min: npt.NDArray = np.zeros((2, 2)),
        p_max: npt.NDArray = np.zeros((2, 2)),
        period_min: npt.NDArray = np.zeros((2, 2)),
        period_max: npt.NDArray = np.zeros((2, 2)),
        init_flux: npt.NDArray = np.zeros((2, 2)),
        max_flux: npt.NDArray = np.zeros((2, 2)),
        min_flux: npt.NDArray = np.zeros((2, 2)),
        weight_range: int = 16,
        bias_range: int = 16,
        tc_min: int = 1,
        tc_max: int = 1,
        WR: float = 16.0,
        BR: float = 16.0,
        TR: float = 5.0,
        TA: float = 6.0,
        stepsize: float = 0.1,
        performance_bias: float = 0.0,
    ):

        super().__init__(
            size,
            tc_min=tc_min,
            tc_max=tc_max,
            WR=WR,
            BR=BR,
            TR=TR,
            TA=TA,
        )

        # map 1D genome to their correct parameter ranges for a CTRNN
        self.size = size
        self.index = 0
        self.stepsize = stepsize
        self.delay = int(delay / self.stepsize)

        # map genome to individual weights, biases and inv time constants
        weights = genome[0 : size * size].reshape(size, size)
        biases = genome[size * size : size * size + size]
        tc = genome[size * size + size :]
        self.genome_arr = np.vstack([weights, biases, tc])
        self.setWeights(self.genome_arr[:size])
        self.setBiases(self.genome_arr[size])
        self.setTimeConstants(self.genome_arr[size + 1])

        # create single matrix of weights, biases and inv time constants
        # to pass into LRule
        self.mapped_genome = np.vstack([self.inner_weights, self.biases])

        # initialize learning rule
        self.sim = LRule(
            duration=duration,
            learn_rate=learn_rate,
            conv_rate=conv_rate,
            delay=delay,
            center_mat=self.mapped_genome,
            p_min=p_min,
            p_max=p_max,
            period_min=period_min,
            period_max=period_max,
            init_flux=init_flux,
            max_flux=max_flux,
            min_flux=min_flux,
            performance_bias=performance_bias,
        )

        # remap LRule's extendedd matrix to RL_CTRNN parameters via pointers
        # this is done to avid copying the matrix slices every iteration
        self.inner_weights = self.sim.center_mat[:size]
        self.biases = self.sim.center_mat[size]
        self.extended_weights = self.sim.extended_mat[:size]
        self.extended_biases = self.sim.extended_mat[size]

        # set parameters
        self.window_size = int(window_size / self.stepsize)
        self.size = size
        self.duration = duration
        self.reward = 0
        self.distance_track = np.zeros(int(self.window_size + self.delay))
        self.window_a_track = np.zeros(int(self.window_size / 2))
        self.window_b_track = np.zeros(int(self.window_size / 2))

    def initializeState(self, v):
        super().initializeState(v)

    def stepRL(self, dt):
        self.inner_weights = self.sim.center_mat[: self.size].view()
        self.biases = (self.sim.center_mat[self.size]).view()
        self.extended_weights = self.sim.extended_mat[: self.size].view()
        self.extended_biases = self.sim.extended_mat[self.size].view()
        netinput = self.inputs + np.dot(self.extended_weights.T, self.outputs)
        self.voltages += dt * (self.inv_time_constants * (-self.voltages + netinput))

        self.outputs = expit(self.voltages + self.extended_biases)
        self.sim.iter_moment()

    def reward_func(self, distance, learning=True):
        self.update_performance(distance)
        self.update_windows(distance)
        if not learning:
            return 0
        return self.window_b_track.mean() - self.window_a_track.mean()

    def update_performance(self, distance):
        self.distance_track[self.delay] = distance
        self.distance_track = np.roll(self.distance_track, -1)

    def update_windows(self, distance):
        self.window_b_track[0] = (
            self.distance_track[0]
            - self.distance_track[-(self.window_b_track.size - 1)]
        ) / (self.window_b_track.size * self.stepsize)
        self.window_a_track[0] = (
            self.distance_track[-(self.window_a_track.size - 1)]
            - self.distance_track[-2 * (self.window_a_track.size - 1)]
        ) / (self.window_a_track.size * self.stepsize)

        self.window_b_track = np.roll(self.window_b_track, -1)
        self.window_a_track = np.roll(self.window_a_track, -1)

    def recoverParameters(self, inner=True):
        # maps values for weights, biases and taus to interval [-1, 1]
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

    # reward_func2, update_windows2, update_performance2 manually implement a ring buffer
    # this cuts runtime to ~1.9 of np.roll where a new array is created each time
    def reward_func2(self, distance, learning=True):

        self.delayed = self.index + self.delay
        self.windowed = self.index - self.window_b_track.size + 1
        self.windowed2 = self.index - 2 * self.window_b_track.size + 2

        self.window_index = self.index % self.window_b_track.size
        self.distance_index = self.index % self.distance_track.size
        self.update_performance2(distance)
        self.update_windows2(distance)
        self.index += 1
        if not learning:
            return 0
        return self.window_b_track.mean() - self.window_a_track.mean()

    def update_performance2(self, distance):
        self.distance_track[(self.delayed) % self.distance_track.size] = distance

    def update_windows2(self, distance):
        self.window_b_track[self.window_index] = (
            self.distance_track[self.distance_index]
            - self.distance_track[(self.windowed) % self.distance_track.size]
        ) / (self.window_b_track.size * self.stepsize)
        self.window_a_track[self.window_index] = (
            self.distance_track[(self.windowed) % self.distance_track.size]
            - self.distance_track[(self.windowed2) % self.distance_track.size]
        ) / (self.window_a_track.size * self.stepsize)
