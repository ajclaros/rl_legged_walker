import leggedwalker
from rl_ctrnn import RL_CTRNN
import matplotlib.pyplot as plt
import os
import numpy as np

params = {
    "duration": 8000,
    "size": 3,
    "delay": 220,  # changes in performance are observed in the averaging windows ~200 time units after the real time performance
    # compare end performance using delay==220 with delay=0 to test learning efficacy
    "generator": "RPG",
    "config": "0",
    "window_size": 440,  # size of the two averaging windows. Fitness function runs for 220 time units
    # ------------------------------------------------------------------------------------------------
    "learn_rate": 0.000001,
    "conv_rate": 0.000001,
    "init_flux": 0.0001,  # Params identified for generator:RPG, size:3, config:0, fitness ~0.234
    "max_flux": 0.001,  # May be robust to different starting positions
    # ------------------------------------------------------------------------------------------------
    # "learn_rate": 0.000001, #
    # "conv_rate": 0.000001,  # Params identified for generator:CPG, size:3, config:012 and fitness ~0.2
    # "init_flux": 0.00001,   # Likely not  robust to different genomes/starting fitnesses
    # "max_flux": 0.0001,     #
    # ------------------------------------------------------------------------------------------------
    "period_min": 440,  # integer time units
    "period_max": 4400,  # integer time units
    "learning_start": 800,  # integer time units, determines how long the agent will gather data from static weights.
    "tolerance": 0.000,  # ignore abs(reward) below tolerance and only update moment
    "fit_range": (0.2, 0.70),  # select genomes within (min, max) fitness range
    "index": 0,  # given all genomes matching "$generator/$size/$configuration"
    # choose the file at specified index position
    "stepsize": 0.1,
    "record_every": 100,  # not currently implemented. When logging data, records every n time steps
}

# convert configuration string into a list: "012" -> [0,1,2]
conf_list = [int(a) for a in params["config"]]

duration = params["duration"]
learn_rate = params["learn_rate"]
conv_rate = params["learn_rate"]
delay = params["delay"]

# Find genome with specified parameters
pathname = f"./evolved/{params['generator']}/{params['size']}/{params['config']}"
files = os.listdir(pathname)
genome_list = []
genome_fitness = []
print(pathname)
fitnesses = [float(name.split("-")[1].split(".")[0]) / 100000 for name in files]
for i, fitness in enumerate(fitnesses):
    if fitness > params["fit_range"][0] and fitness < params["fit_range"][1]:
        genome_list.append(files[i])
        genome_fitness.append(fitness)
genome_list = sorted(genome_list)
genome_fitness = sorted(genome_fitness)
print(genome_list[params["index"]])


# load genome
size = params["size"]
genome = np.load(f"{pathname}/{genome_list[params['index']]}")


# parameter minimum and maxima
p_min = np.ones((size + 2, size)) * -16
p_max = np.ones((size + 2, size)) * 16
# time_constants are not fluctuating
period_min[-1] = duration
period_max[-1] = duration

# period range
period_min = np.ones((size + 2, size)) * params["period_min"]
period_max = np.ones((size + 2, size)) * params["period_max"]

# flux
init_flux = np.ones((size + 2, size)) * params["init_flux"]
max_flux = np.ones((size + 2, size)) * params["max_flux"]

agent = RL_CTRNN(
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


# initialize parameters to record
distance = np.zeros(time.size)  # real time distance
agent_distance = np.zeros(time.size)  # agent recorded distance with delay
performance = np.zeros(time.size)
agent_performance = np.zeros(time.size)

window_b = np.zeros(time.size)
avg_window_b = np.zeros(time.size)
window_a = np.zeros(time.size)
avg_window_a = np.zeros(time.size)
flux_track = np.zeros(time.size)
difference_track = np.zeros(time.size)

weight_track = np.zeros((time.size, size, size))


for i, t in enumerate(time):
    if params["generator"] == "RPG":
        agent.setInputs(np.array([body.anglefeedback()] * agent.size))
    else:
        # CPG
        agent.setInputs([0] * agent.size)

    if t < params["learning_start"]:
        # gather performance data before learning
        agent.step(params["stepsize"])
        reward = agent.reward_func(body.cx, learning=False)
    else:
        # learning phase
        agent.stepRL(params["stepsize"])
        reward = agent.reward_func(body.cx, learning=True)
        flux_track[i] = agent.sim.flux_mat[0, 0]
        agent.sim.update_weights_with_reward(reward)

    body.stepN(agent.stepsize, agent.outputs, conf_list)

    # update center and fluctuating weights
    agent.inner_weights = agent.sim.center_mat[:size].copy()
    agent.biases = agent.sim.center_mat[size].copy()
    agent.extended_weights = agent.sim.extended_mat[:size].copy()
    agent.extended_biases = agent.sim.extended_mat[size].copy()

    # record data to plot
    distance[i] = body.cx
    performance[i] = (distance[i] - distance[i - (2200 - 1)]) / 220
    agent_distance[i] = agent.distance_track[0]
    agent_performance[i] = (agent_distance[i] - agent_distance[i - (2200 - 1)]) / 220
    weight_track[i] = agent.extended_weights
    window_b[i] = agent.window_b[-1]
    window_a[i] = agent.window_a[-1]
    avg_window_b[i] = agent.window_b.mean()
    avg_window_a[i] = agent.window_a.mean()
    difference_track[i] = avg_window_b[i] - avg_window_a[i]

fig, ax = plt.subplots(nrows=2)
ax[0].plot(time, performance, label="Real time performance")
# ax[0].plot(time, window_b, label="window_b")  # current performance as observed by agent
# ax[0].plot(time, window_a, label="window_a")  #past performance as observed by agent
ax[1].plot(time, flux_track, label="Fluctuation size")
ax[0].plot(
    time, difference_track, label="difference"
)  # difference in averaged performance windows, the neuromodulatory single
ax[0].plot(time, avg_window_b, label="avg_window_b")  # average current performance
ax[0].plot(time, avg_window_a, label="avg_window_a")  # average past performance
ax[0].legend()
ax[1].legend()
plt.show()
