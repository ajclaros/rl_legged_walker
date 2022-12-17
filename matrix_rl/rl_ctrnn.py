import matplotlib.pyplot as plt
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
        weight_range: int = 16,
        bias_range: int = 16,
        tc_min: int = 1,
        tc_max: int = 1,
        WR: float = 16.0,
        BR: float = 16.0,
        TR: float = 5.0,
        TA: float = 6.0,
        stepsize: float = 0.1,
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
        self.stepsize = stepsize
        self.delay = int(delay / self.stepsize)
        weights = genome[0 : size * size].reshape(size, size)
        biases = genome[size * size : size * size + size]
        tc = genome[size * size + size :]
        self.genome_arr = np.vstack([weights, biases, tc])
        self.setWeights(self.genome_arr[:size])
        self.setBiases(self.genome_arr[size])
        self.setTimeConstants(self.genome_arr[size + 1])
        self.mapped_genome = np.vstack(
            [self.inner_weights, self.biases, self.inv_time_constants]
        )

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
        )

        # map inner_centers and extended weights to CTRNN
        self.inner_weights = self.sim.center_mat[:size]
        self.biases = self.sim.center_mat[size]
        self.inv_time_constants = self.sim.center_mat[size + 1]
        self.extended_weights = self.sim.extended_mat[:size]
        self.extended_biases = self.sim.extended_mat[size]

        # set parameters
        self.window_size = int(window_size / self.stepsize)
        self.size = size
        self.duration = duration
        self.reward = 0
        self.distance_track = np.zeros(int(self.window_size + self.delay))
        self.performance_track = np.zeros(int(self.window_size / 2))
        self.window_a_track = np.zeros(int(self.window_size / 2))
        self.window_b_track = np.zeros(int(self.window_size / 2))

    def initializeState(self, v):
        super().initializeState(v)

    def stepRL(self, dt):
        netinput = self.inputs + np.dot(self.extended_weights.T, self.outputs)
        self.voltages += dt * (self.inv_time_constants * (-self.voltages + netinput))
        self.outputs = expit(self.voltages + self.extended_biases.T)
        self.sim.iter_moment(dt)

    def reward_func(self, distance, learning=True):
        self.update_performance(distance)
        self.update_windows(distance)
        if not learning:
            return 0
        return self.window_b_track.mean() - self.window_a_track.mean()

    def update_performance(self, distance):
        self.distance_track[self.delay] = distance
        self.performance_track[0] = (
            self.distance_track[self.delay - 1] - self.distance_track[-1]
        ) / (self.window_b_track.size * self.stepsize)
        self.distance_track = np.roll(self.distance_track, -1)
        self.performance_track = np.roll(self.performance_track, -1)

    def update_windows(self, distance):
        self.window_b_track[0] = (
            self.distance_track[0]
            - self.distance_track[-(self.window_b_track.size - 1)]
        ) / (self.window_b_track.size * self.stepsize)
        self.window_a_track[0] = (
            self.distance_track[-(self.window_a_track.size)]
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

    def reward_func2(self, distance, learning=True):
        self.performance_func2(distance)
        if not learning:
            return 0
        return (
            self.window_b_track[-1] - self.window_a_track[-1]
        )  # .mean() - self.window_a_track.mean()

    def performance_func2(self, distance):
        self.distance_track[self.delay] = distance
        self.window_b_track[0] = (
            self.distance_track[0]
            - self.distance_track[-(self.window_b_track.size - 1)]
        ) / (self.window_b_track.size * self.stepsize)
        self.window_a_track[0] = (
            self.distance_track[-(self.window_a_track.size)]
            - self.distance_track[-2 * (self.window_a_track.size - 1)]
        ) / (self.window_a_track.size * self.stepsize)

        self.distance_track = np.roll(self.distance_track, -1)
        self.window_b_track = np.roll(self.window_b_track, -1)
        self.window_a_track = np.roll(self.window_a_track, -1)
