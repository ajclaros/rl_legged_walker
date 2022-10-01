import leggedwalker
import ctrnn
import numpy as np
from rl_ctrnn import RL_CTRNN
from fitnessFunction import fitnessFunction
from scipy.special import expit


class WalkingTask(RL_CTRNN):
    def __init__(
        self,
        size=2,
        duration=2000.0,
        stepsize=0.01,
        reward_func=None,
        performance_func=None,
        running_window_mode=True,
        running_window_size=400,
        performance_update_rate=0.005,
        performance_bias=0.007,
        init_flux_amp=2.75,
        max_flux_amp=10,
        flux_period_min=300,
        flux_period_max=400,
        flux_conv_rate=0.004,
        learn_rate=0.008,
        bias_init_flux_amp=2.75,
        bias_max_flux_amp=10,
        bias_flux_period_min=300,
        bias_flux_period_max=400,
        bias_flux_conv_rate=0.004,
        WR=16.0,
        BR=16.0,
        TR=5.0,
        TA=6.0,
        record_every=1,
    ):
        super().__init__(
            size,
            init_flux_amp=init_flux_amp,
            max_flux_amp=max_flux_amp,
            flux_period_min=flux_period_min,
            flux_period_max=flux_period_max,
            flux_conv_rate=flux_conv_rate,
            learn_rate=learn_rate,
            bias_init_flux_amp=bias_init_flux_amp,
            bias_max_flux_amp=bias_max_flux_amp,
            bias_flux_period_min=bias_flux_period_min,
            bias_flux_period_max=bias_flux_period_max,
            bias_flux_conv_rate=bias_flux_conv_rate,
            WR=WR,
            BR=BR,
            TR=TR,
            TA=TA,
        )
        self.stepsize = stepsize
        self.running_window_mode = running_window_mode
        self.running_window_size = int(running_window_size / self.stepsize)
        self.size = size
        self.time_step = 0
        self.duration = duration
        self.reward = 0
        self.sample_rate = 1 / record_every
        self.time = np.arange(0, self.duration, self.stepsize)
        self.distance_track = np.zeros((self.running_window_size))
        self.performance_track = np.zeros((self.running_window_size))
        if self.running_window_mode:
            self.sliding_window = np.zeros(self.running_window_size)
            self.window_a = np.zeros(int(self.running_window_size / 2))
            self.window_b = np.zeros(int(self.running_window_size / 2))
        else:
            self.performance_update_rate = performance_update_rate  # how quickly the running average performance is updated
        # uses performance to determine reward
        if reward_func is None:
            self.reward_func = self.default_reward_func
        else:
            self.reward_func = self.secondary_reward_func
        # determines how performance is measured
        if performance_func is None:
            self.performance_func = self.default_performance_func
        else:
            self.performance_func = self.secondary_performance_func

    def save2(self, filename):
        np.savez(filename, genotype=self.recoverParameters())

    def default_reward_func(self, distance, learning=True):
        self.default_performance_func(distance)
        # Current instantaneous performance vs. the current running average (NOT the previous instantaneous performance)
        if not learning:
            return 0

        return (self.window_b.mean() - self.window_a.mean()) / (
            self.window_b.size / self.stepsize
        )

    def default_performance_func(self, body):
        self.distance_track[0] = body.cx
        self.window_b[0] = (
            self.distance_track[0] - self.distance_track[self.window_b.size - 1]
        )
        self.window_a[0] = (
            self.distance_track[self.window_a.size]
            - self.distance_track[self.running_window_size - 1]
        )
        self.performance_track[0] = self.window_a.mean() / (
            self.stepsize * self.window_a.size
        )

        self.distance_track = np.roll(self.distance_track, 1)
        self.window_a = np.roll(self.window_a, 1)
        self.window_b = np.roll(self.window_b, 1)
        self.performance_track = np.roll(self.performance_track, 1)

    def secondary_reward_func(self, body, learning=True):
        self.performance_func(body)
        # Current instantaneous performance vs. the current running average (NOT the previous instantaneous performance)
        if not learning:
            return 0
        return self.performance - self.performance_track.mean()

    def secondary_performance_func(self, body):
        self.distance_track[0] = body.cx
        performance = self.distance_track[0] - self.distance_track[1]
        performance /= self.stepsize
        self.performance = performance
        self.performance_track[0] = performance
        self.performance_track = np.roll(self.performance_track, 1)

    def simulate(
        self,
        body,
        datalogger=None,
        learning_start=None,
        verbose=0.1,
        generator_type="RPG",
        configuration=[0],
        tolerance=0.0,
    ):

        if datalogger:
            datalogger.data["startgenome"] = self.recoverParameters()
        for i, t in enumerate(self.time):

            # verbose=0.1, runs fitnessFunction 10 at equal intervals
            if verbose > 0.0 and i % (self.time.size * verbose) == 0 and verbose < 1.0:
                print("{}% completed...".format(i / self.time.size * 100))
            if generator_type == "RPG":
                self.setInputs(np.array([body.anglefeedback()] * self.size))
            elif generator_type == "CPG":
                self.setInputs(np.array([0] * self.size))

            # rl_ctrnn step
            self.step(self.stepsize)
            # body.step3(self.stepsize, self.outputs)
            body.stepN(self.stepsize, self.outputs, configuration)
            if self.time_step < learning_start:
                self.reward = self.reward_func(body, learning=False)
                # updating with 0 reward
                self.update_weights_and_flux_amp_with_reward(
                    self.reward, tolerance=tolerance, learning=False
                )
            else:
                reward = self.reward_func(body, learning=True)
                self.reward = reward
                self.update_weights_and_flux_amp_with_reward(
                    reward, tolerance=tolerance
                )
            if datalogger and self.time_step % (1 / self.sample_rate) == 0:
                position = int(self.time_step * self.sample_rate)
                for key in datalogger.data.keys():
                    if key in ["startgenome"]:
                        continue
                    elif "hist" in key:
                        key_name = key.split("_")[0] + "_track"
                        datalogger.data[key][position] = self.__dict__[key_name][1]
                    elif "average" in key:
                        key_name = key.split("_")[0] + "track"
                        datalogger.data[key][position] = self.__dict__[key_name].mean()

                    elif key in ["angle", "omega", "distance"]:
                        datalogger.data["angle"][position] = body.angle
                        datalogger.data["omega"][position] = body.omega
                        datalogger.data["distance"][position] = body.cx
                    else:
                        datalogger.data[key][position] = self.__dict__[key]

            self.time_step += 1

        self.time_step = 0
        if datalogger:
            #     self.recoverParameters(),
            #     N=self.size,
            #     generator_type=generator_type,
            #     configuration=configuration,
            #     verbose=verbose,
            # )
            datalogger.data["learning_start"] = learning_start
            datalogger.data["tolerance"] = tolerance
            datalogger.data["size"] = self.size
            datalogger.data["duration"] = self.duration
            datalogger.data["stepsize"] = self.stepsize
            datalogger.data["sample_rate"] = self.sample_rate
            datalogger.data["metric"] = self.performance_func.__name__.split("_")[0]
            datalogger.data["endgenome"] = self.recoverParameters()
            print(self.time_constants)
