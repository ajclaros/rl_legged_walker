import leggedwalker
from ctrnn import CTRNN
from simulator import simulator2 as sim2
from rl_ctrnn import RL_CTRNN2
import matplotlib.pyplot as plt
import os
import numpy as np
import numpy.typing as npt
from matplotlib.widgets import Slider, Button, CheckButtons

params = {
    "duration": 4000,
    "size": 2,
    "delay": 220,
    "generator": "RPG",
    "config": "0",
    "window_size": 440,
    "learn_rate": 0.00001,
    "conv_rate": 0.0007,
    "init_flux": 0.001,  # RPG 0
    "max_flux": 0.001,
    # "learn_rate": 0.000001,
    # "conv_rate": 0.000001,   #CPG 012 fit 20288
    # "init_flux": 0.00001,
    # "max_flux": 0.0001,
    "period_min": 440,
    "period_max": 4400,
    "learning_start": 800,
    "record_every": 100,
    "tolerance": 0.00,
    "stepsize": 0.1,
    "fit_range": (0.2, 0.70),
    "index": 0,
}


duration = params["duration"]
learn_rate = params["learn_rate"]
conv_rate = params["learn_rate"]
delay = params["delay"]
pathname = f"./evolved/{params['generator']}/{params['size']}/{params['config']}"
files = os.listdir(pathname)
genome_list = []
genome_fitness = []
fitnesses = [float(name.split("-")[1].split(".")[0]) / 100000 for name in files]
for i, fitness in enumerate(fitnesses):
    if fitness > params["fit_range"][0] and fitness < params["fit_range"][1]:
        genome_list.append(files[i])
        genome_fitness.append(fitness)
genome_list = sorted(genome_list)
genome_fitness = sorted(genome_fitness)

size = params["size"]
print(genome_list[params["index"]])
genome = np.load(f"{pathname}/{genome_list[params['index']]}")
weights = genome[0 : params["size"] * params["size"]].reshape(size, size)
biases = genome[size * size : size * size + size]
tc = genome[size * size + size :]
genome_arr = np.vstack([weights, biases, tc])

p_min = np.ones((size + 2, size)) * -16
p_max = np.ones((size + 2, size)) * 16
period_min = np.ones((size + 2, size)) * params["period_min"]
period_max = np.ones((size + 2, size)) * params["period_max"]

init_flux = np.ones((size + 2, size)) * params["init_flux"]
max_flux = np.ones((size + 2, size)) * params["max_flux"]
init_flux[-1] = 0
max_flux[-1] = 0
conf_list = [int(a) for a in params["config"]]
period_min[-1] = duration

period_max[-1] = duration
agent = RL_CTRNN2(
    size,
    genome,
    duration,
    learn_rate,
    conv_rate,
    delay,
    params["window_size"],
    p_min,
    p_max,
    period_min,
    period_max,
    init_flux,
    max_flux,
)
agent.initializeState(np.zeros(size))
body = leggedwalker.LeggedAgent()
time = np.arange(0, int(duration), params["stepsize"])
distance = np.zeros(time.size)
distance2 = np.zeros(time.size)
performance = np.zeros(time.size)
performance2 = np.zeros(time.size)
window_b = np.zeros(time.size)

avg_window_b = np.zeros(time.size)

window_a = np.zeros(time.size)
avg_window_a = np.zeros(time.size)
reward_track = np.zeros(time.size)
difference_track = np.zeros(time.size)
weight_track = np.zeros((time.size, size, size))
for i, t in enumerate(time):
    if params["generator"] == "RPG":
        agent.setInputs(np.array([body.anglefeedback()] * agent.size))
    else:
        agent.setInputs([0] * agent.size)
    if t < params["learning_start"]:
        agent.step(params["stepsize"])
        reward = agent.reward_func(body.cx, learning=False)
    else:
        agent.stepRL(params["stepsize"])
        reward = agent.reward_func(body.cx, learning=True)
        reward_track[i] = agent.sim.flux_mat[0, 0]
        agent.sim.update_weights_with_reward(reward)
    body.stepN(agent.stepsize, agent.outputs, conf_list)
    distance[i] = body.cx
    performance[i] = (distance[i] - distance[i - (2200 - 1)]) / 220
    distance2[i] = agent.distance_track[0]
    performance2[i] = (distance2[i] - distance2[i - (2200 - 1)]) / 220
    weight_track[i] = agent.extended_weights
    window_b[i] = agent.window_b[-1]
    window_a[i] = agent.window_a[-1]
    avg_window_b[i] = agent.window_b.mean()
    avg_window_a[i] = agent.window_a.mean()
    difference_track[i] = avg_window_b[i] - avg_window_a[i]
    agent.inner_weights = agent.sim.center_mat[:size].copy()
    agent.biases = agent.sim.center_mat[size].copy()
    agent.extended_weights = agent.sim.extended_mat[:size].copy()
    agent.extended_biases = agent.sim.extended_mat[size].copy()
fig, ax = plt.subplots(nrows=2)
# ax[0].plot(time, performance, label="current performacne1")
# ax[0].plot(time, performance2, label="delayed performance2")
# ax[0].plot(time, window_b, label="window_b")
# ax[0].plot(time, window_a, label="window_a")
ax[1].plot(time, reward_track)
ax[0].plot(time, difference_track, label="difference")
ax[0].plot(time, avg_window_b, label="avg_window_b")
ax[0].plot(time, avg_window_a, label="avg_window_a")
ax[0].legend()
plt.show()
