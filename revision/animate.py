import leggedwalker
from ctrnn import CTRNN
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.widgets import Slider, Button, CheckButtons
from fitnessFunction import fitnessFunction

np.random.seed(1000)


class Rl_visualize:
    def __init__(
        self,
        window_size,
        duration,
        generator,
        neuron,
        config,
        conf_list,
        x,
        fit_range,
        index=0,
        ax=None,
        legend=False,
    ):
        self.flux_start = 800
        self.window_size = window_size
        self.duration = duration
        self.generator = generator
        self.neuron = neuron
        self.config = config
        self.conf_list = conf_list
        self.x = x
        self.legend = legend
        index = 0
        ax = (ax,)
        legend = legend
        folderName = f"./evolved/{self.generator}/{neuron}/{config}"
        genomes = []
        self.found_fit = []
        files = os.listdir(folderName)
        fitnesses = [float(name.split("-")[1].split(".")[0]) / 100000 for name in files]
        for k, fitness in enumerate(fitnesses):
            if fitness > fit_range[0] and fitness < fit_range[1]:
                genomes.append(files[k])
                self.found_fit.append(fitnesses[k])

        self.start_genome = np.load(f"{folderName}/{genomes[index]}")
        self.found_fit = self.found_fit[index]
        N = int(neuron)
        self.N = N
        size = N
        self.size = size
        size = N
        self.stepsize = 0.1
        # initialize agent and nervous system
        self.ns = CTRNN(N)
        # self.inner_weights = self.start_genome[0 : N * N].reshape((N, N))
        # self.inner_biases = self.start_genome[N * N : N * N + N]
        # self.extended_weights = self.inner_weights
        # self.extended_biases = self.inner_biases
        self.inner_weights = np.zeros((size, size))
        self.inner_biases = np.zeros((size))
        self.extended_weights = np.zeros((size, size))
        self.extended_biases = np.zeros((size))
        self.time_constants = self.start_genome[N * N + N :]
        self.ns.setWeights(self.start_genome[0 : self.N * self.N].reshape(size, size))
        self.ns.setBiases(self.start_genome[size * size : size * size + size])
        self.ns.setTimeConstants(self.time_constants)
        self.ns.initializeState(np.zeros(N))

        self.inner_weights = self.ns.inner_weights
        self.inner_biases = self.ns.biases
        self.extended_weights = self.ns.inner_weights
        self.extended_biases = self.ns.biases
        self.legged = leggedwalker.LeggedAgent()

        # initialize tracking windowsl
        self.x_axis = np.arange(0, self.window_size, 0.1)
        self.velocity = np.zeros(len(self.x_axis))
        self.dist_track = np.zeros(len(self.x_axis) * 2)
        self.window_a = np.zeros(len(self.x_axis))
        self.window_b = np.zeros(len(self.x_axis))
        self.avg_window_a = np.zeros(len(self.x_axis))
        self.avg_window_b = np.zeros(len(self.x_axis))
        self.difference_track = np.zeros(len(self.x_axis))
        self.time = np.arange(0, self.duration, self.stepsize)
        # initialize full tracked parameters during trial
        self.window_b_arr = np.zeros(self.time.size)
        self.window_a_arr = np.zeros(self.time.size)
        self.avg_window_b_arr = np.zeros(self.time.size)
        self.avg_window_a_arr = np.zeros(self.time.size)
        self.difference_arr = np.zeros(self.time.size)
        self.weight_centers_arr = np.zeros((self.time.size, N, N))
        self.weights_arr = np.zeros((self.time.size, N, N))
        self.bias_centers_arr = np.zeros((self.time.size, N))
        self.bias_arr = np.zeros((self.time.size, N))
        self.flux_sign = np.random.binomial(1, 0.5, size=(N, N)) * 2 - 1
        self.bias_flux_sign = np.random.binomial(1, 0.5, size=(N)) * 2 - 1
        self.inner_flux_periods = np.zeros((size, size))
        self.inner_flux_moments = np.zeros((size, size))
        self.bias_inner_flux_periods = np.zeros(size)
        self.bias_inner_flux_moments = np.zeros(size)
        self.starting_amp = 1
        self.period_range = (self.window_size, self.window_size * 10)
        # self.weight_centers_arr[0] = self.ns.inner_weights
        # self.weights_arr[0] = self.ns.inner_weights

    def simulate(self):
        for time_step, t in enumerate(self.time):
            if self.generator == "RPG":
                self.ns.setInputs(np.array([self.legged.anglefeedback()] * self.N))
            else:
                self.ns.setInputs(np.array([0] * self.N))
            self.ns.step(self.stepsize)
            self.legged.stepN(self.stepsize, self.ns.outputs, self.conf_list)
            # ns.inner_weights =
            self.velocity[0] = self.legged.vx
            self.dist_track[0] = self.legged.cx
            self.window_b[0] = (
                self.dist_track[0] - self.dist_track[-(self.window_b.size - 1)]
            ) / (self.window_size)
            self.window_a[0] = (
                self.dist_track[-(self.window_a.size - 1)]
                - self.dist_track[-2 * (self.window_a.size - 1)]
            ) / (self.window_size)
            self.avg_window_b[0] = self.window_b.mean()
            self.avg_window_a[0] = self.window_a.mean()
            self.difference_track[0] = self.avg_window_b[0] - self.avg_window_a[0]
            self.window_b_arr[time_step] = self.window_b[0]
            self.window_a_arr[time_step] = self.window_a[0]
            self.avg_window_a_arr[time_step] = self.avg_window_a[0]
            self.avg_window_b_arr[time_step] = self.avg_window_b[0]
            self.difference_arr[time_step] = self.difference_track[0]
            # rotate array for next iteration
            self.velocity = np.roll(self.velocity, -1)
            self.dist_track = np.roll(self.dist_track, -1)
            self.window_a = np.roll(self.window_a, -1)
            self.window_b = np.roll(self.window_b, -1)
            self.avg_window_a = np.roll(self.avg_window_a, -1)
            self.avg_window_a = np.roll(self.avg_window_b, -1)
            self.difference_track = np.roll(self.difference_track, -1)
        ax[0].plot(self.time, self.window_a_arr, label="Window_a")
        ax[1].plot(self.time, self.inner_weights, label="inner_weights")
        ax[1].plot(self.time, self.inner_biases, label="inner biases")
        ax[0].plot(self.time, self.window_b_arr, label="Window_b")
        ax[0].plot(self.time, self.avg_window_a_arr, label="avg_window_a")
        ax[0].plot(self.time, self.avg_window_b_arr, label="avg_window_b")
        ax[0].plot(self.time, self.difference_arr, label="diff(window_a, window_b)")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Performance")
        ax[0].legend()
        ax[1].legend()

    def update_moments(self):
        self.inner_flux_moments += self.stepsize
        self.bias_inner_flux_moments += self.stepsize
        for i in range(self.N):
            for j in range(self.N):
                if self.inner_flux_moments[i][j] > abs(self.inner_flux_periods[i][j]):
                    self.inner_flux_moments[i][j] = 0
                    self.inner_flux_periods[i][j] = np.round(
                        np.random.uniform(self.period_range[0], self.period_range[1]),
                        3,
                    )
                    self.inner_flux_periods[i][j] *= self.flux_sign[i][j]
            if self.bias_inner_flux_moments[i] > abs(self.bias_inner_flux_periods[i]):
                self.bias_inner_flux_moments[i] = 0
                self.bias_inner_flux_periods[i] = np.round(
                    np.random.uniform(self.period_range[0], self.period_range[1]),
                    3,
                )
                self.bias_inner_flux_periods *= self.bias_flux_sign[i]

    def update_weights_and_biases(self):
        inner_flux_center_displacements = self.starting_amp * np.sin(
            self.inner_flux_moments / self.inner_flux_periods * 2 * np.pi
        )
        bias_inner_flux_center_displacements = self.starting_amp * np.sin(
            self.bias_inner_flux_moments / self.bias_inner_flux_periods * 2 * np.pi
        )
        self.extended_weights = self.inner_weights + inner_flux_center_displacements
        self.extended_biases = self.inner_biases + bias_inner_flux_center_displacements

    def simulateFlux(self):
        for time_step, t in enumerate(self.time):
            self.time_step = time_step
            if self.generator == "RPG":
                self.ns.setInputs(np.array([self.legged.anglefeedback()] * self.N))
            else:
                self.ns.setInputs(np.array([0] * self.N))

            if t > self.flux_start:
                self.update_moments()
                self.update_weights_and_biases()

            self.weight_centers_arr[self.time_step] = self.inner_weights
            self.bias_centers_arr[self.time_step] = self.inner_biases
            self.weights_arr[self.time_step] = self.extended_weights
            self.bias_arr[self.time_step] = self.extended_biases
            self.ns.inner_weights = self.extended_weights
            self.ns.biases = self.extended_biases
            self.ns.step(self.stepsize)
            self.legged.stepN(self.stepsize, self.ns.outputs, self.conf_list)
            # ns.inner_weights =
            self.velocity[0] = self.legged.vx
            self.dist_track[0] = self.legged.cx
            self.window_b[0] = (
                self.dist_track[0] - self.dist_track[-(self.window_b.size - 1)]
            ) / (self.window_size)
            self.window_a[0] = (
                self.dist_track[-(self.window_a.size - 1)]
                - self.dist_track[-2 * (self.window_a.size - 1)]
            ) / (self.window_size)
            self.avg_window_b[0] = self.window_b.mean()
            self.avg_window_a[0] = self.window_a.mean()
            self.difference_track[0] = self.avg_window_b[0] - self.avg_window_a[0]
            self.window_b_arr[time_step] = self.window_b[0]
            self.window_a_arr[time_step] = self.window_a[0]
            self.avg_window_a_arr[time_step] = self.avg_window_a[0]
            self.avg_window_b_arr[time_step] = self.avg_window_b[0]
            self.difference_arr[time_step] = self.difference_track[0]
            # rotate array for next iteration
            self.velocity = np.roll(self.velocity, -1)
            self.dist_track = np.roll(self.dist_track, -1)
            self.window_a = np.roll(self.window_a, -1)
            self.window_b = np.roll(self.window_b, -1)
            self.avg_window_a = np.roll(self.avg_window_a, -1)
            self.avg_window_a = np.roll(self.avg_window_b, -1)
            self.difference_track = np.roll(self.difference_track, -1)


generators = ["RPG", "RPG"]
neurons = ["2", "3"]
config = ["0", "01"]
conf_list = [[int(elt) for elt in c] for c in config]
fit_range = [(0.3, 0.5), (0.3, 0.5)]
indices = [0, 0, 0, 0]
window_size = 220
duration = 4000
# for i, subfig in enumerate(subfigs):
fig, ax = plt.subplots(ncols=2, nrows=2)
# fig.suptitle(f"{}, N={neurons}, config={config}")
legend = True
x = 0
RL_arr = []
for i in range(1):
    g = generators[i]
    n = neurons[i]
    cf = config[i]
    cfl = conf_list[i]
    fr = fit_range[i]
    for j in range(1):
        if x == 3:
            legend = True
        RL_arr.append(
            Rl_visualize(
                window_size,
                duration,
                g,
                n,
                cf,
                cfl,
                0,
                fr,
                index=0,
                legend=legend,
            )
        )
        x += 1
        RL_arr[i].simulateFlux()


cmap = plt.get_cmap("tab20").colors

time = np.arange(0, 4000, 0.1)
(start, end) = (400, 1200)
for i, RL in enumerate(RL_arr):
    if i > 0:
        continue
    drop_indx = np.where(np.diff(RL.avg_window_b_arr[start * 10 : end * 10]))
    plt.plot(
        time[start * 10 : end * 10],
        RL.avg_window_b_arr[start * 10 : end * 10],
    )
plt.show()

fit_duration = 220
fitnesses = []
for i in range(int(end - start)):
    print(i)
    legged = leggedwalker.LeggedAgent()
    ns = CTRNN(RL.size)
    ns.inner_weights = RL.weights_arr[start * 10 + i * 10]
    ns.biases = RL.bias_arr[start * 10 + i * 10]
    ns.setTimeConstants(RL.time_constants)
    ns.initializeState(np.zeros(RL.N))
    time = np.arange(0, fit_duration, RL.stepsize)
    for j, t in enumerate(time):
        ns.setInputs([legged.anglefeedback()] * RL.N)
        ns.step(RL.stepsize)
        legged.stepN(RL.stepsize, ns.outputs, RL.conf_list)
    fitnesses.append(legged.cx / fit_duration)
plt.title(f"{RL.generator}, N:{RL.neuron}, Cong:{RL.config}, fitness:{RL.found_fit}")
plt.plot(fitnesses, color="b", lw=2)
plt.plot(RL.avg_window_b_arr[start * 10 : end * 10 : 10], lw=2, color="r")
plt.xlabel("Time")
plt.ylabel("Performance/Fitness")
plt.show()

# plt.plot(RL.avg_window_b_arr); plt.show()


# for k, RL in enumerate(RL_arr):
#     ax[0][k].title.set_text(
#         f"{RL.generator}, N:{RL.neuron}, Config:{RL.config}, fitness:{RL.found_fit}"
#     )
#     ax[0][k].plot(RL.time, RL.window_a_arr, label="Window_a")
#     ax[0][k].plot(RL.time, RL.window_b_arr, label="Window_b")
#     ax[0][k].plot(RL.time, RL.avg_window_a_arr, label="avg_window_a")
#     ax[0][k].plot(RL.time, RL.avg_window_b_arr, label="avg_window_b")
#     ax[0][k].plot(RL.time, RL.difference_arr, label="diff(window_a, window_b)")
#     for i in range(RL.size):
#         for j in range(RL.size):
#             if i == 0 and k == 1:
#                 wlabel = "w_center"
#                 eqlabel = "extended_W"
#                 blabel = "bias_center"
#                 eb = "extended_bias"

#             else:
#                 wlabel = ""
#                 eqlabel = ""
#                 blabel = ""
#                 eb = ""
#             ax[1][k].plot(
#                 RL.time,
#                 RL.weight_centers_arr.T[i][j],
#                 color=cmap[0],
#                 label=wlabel,
#                 lw=2,
#             )
#             ax[1][k].plot(
#                 RL.time,
#                 RL.weights_arr.T[i][j],
#                 color=cmap[1],
#                 ls="dotted",
#                 label=eqlabel,
#                 lw=3,
#             )
#             ax[1][k].plot(
#                 RL.time,
#                 RL.bias_centers_arr.T[i],
#                 color=cmap[2],
#                 label=blabel,
#                 ls="-",
#                 lw=2,
#             )
#             ax[1][k].plot(
#                 RL.time, RL.bias_arr.T[i], ls="dotted", lw=3, color=cmap[3], label=eb
#             )
#     ax[0][k].set_xlabel("Time")
#     ax[0][k].set_ylabel("Performance")
#     # if RL.legend:
#     ax[1][k].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.0)
# # fig.constrained_layout()

# plt.show()
