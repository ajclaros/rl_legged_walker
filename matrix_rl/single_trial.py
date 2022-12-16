import leggedwalker
from rl_ctrnn import RL_CTRNN
import matplotlib.pyplot as plt
import os
import numpy as np
import concurrent.futures

num_processes = 10
num_trials = 3
random_genomes_in_range = False
indices = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16]
num_genomes = 8
params = {
    "duration": 20000,
    "size": 3,
    "delay": 600,  # changes in performance are observed in the averaging windows ~200 time units after the real time performance
    # compare end performance using delay==220 with delay=0 to test learning efficacy
    "generator": "CPG",
    "config": "0",
    "window_size": 440,  # size of the two averaging windows. Fitness function runs for 220 time units
    # ------------------------------------------------------------------------------------------------
    # "learn_rate": 0.000001,
    # "conv_rate": 0.000001,
    # "init_flux": 0.0001,  # Params identified for generator:RPG, size:3, config:0, fitness ~0.234
    # "max_flux": 0.005,  # May be robust to different starting positions
    # ------------------------------------------------------------------------------------------------
    "learn_rate": 0.000001,  #
    "conv_rate": 0.000001,  # Params identified for generator:CPG, size:3, config:0 and fitness ~0.2
    "init_flux": 0.00015,  # Likely not robust to different genomes/starting fitnesses
    "max_flux": 0.0001,  #
    # ------------------------------------------------------------------------------------------------
    # "learn_rate": 0.000005,
    # "conv_rate": 0.000000,
    # "init_flux": 0.0001,  # experimental with exponential growth rate
    # "max_flux": 0.005,
    "period_min": 440,  # integer time units
    "period_max": 4400,  # integer time units
    "learning_start": 2000,  # integer time units, determines how long the agent will gather data from static weights.
    "tolerance": 0.001,  # ignore abs(reward) below tolerance and only update moment
    "fit_range": (0.2, 0.7),  # select genomes within (min, max) fitness range
    "index": 0,  # given all genomes matching "$generator/$size/$configuration"
    # choose the file at specified index position
    "stepsize": 0.1,
    "record_every": 1000,  # not currently implemented. When logging data, records every n time steps
}

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
genomes = []
fit_list = []
size = params["size"]

if random_genomes_in_range == False:
    for index in indices:
        genomes.append(np.load(f"{pathname}/{genome_list[index]}"))
        fit_list.append(genome_fitness[index])

# load genome
# genome = np.load(f"{pathname}/{genome_list[params['index']]}")


def learn(
    genome,
    verbose=0,
    params=None,
    output=[],
    ix=0,
    record_every=1,
):

    np.random.seed()
    sample_rate = 1 / record_every
    # convert configuration string into a list: "012" -> [0,1,2]
    conf_list = [int(a) for a in params["config"]]

    duration = params["duration"]
    learn_rate = params["learn_rate"]
    conv_rate = params["learn_rate"]
    delay = params["delay"]
    size = params["size"]
    # Find genome with specified parameters

    # parameter minimum and maxima
    p_min = np.ones((size + 2, size)) * -16
    p_max = np.ones((size + 2, size)) * 16

    # period range
    period_min = np.ones((size + 2, size)) * params["period_min"]
    period_max = np.ones((size + 2, size)) * params["period_max"]

    # time_constants are not fluctuating, therefore set their fluctuations to zero
    # period_min[-1] = duration
    # period_max[-1] = duration

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
    time = np.arange(0, int(duration), params["stepsize"] * sample_rate)

    data = {}
    # initialize parameters to record
    data["distance"] = np.zeros(time.size)  # real time distance
    data["agent_distance"] = np.zeros(time.size)  # agent recorded distance with delay
    data["performance"] = np.zeros(time.size)
    data["inst_perf"] = np.zeros(time.size)
    data["agent_performance"] = np.zeros(time.size)
    data["window_b"] = np.zeros(time.size)
    data["avg_window_b"] = np.zeros(time.size)
    data["window_a"] = np.zeros(time.size)
    data["avg_window_a"] = np.zeros(time.size)
    data["flux_track"] = np.zeros(time.size)
    data["difference_track"] = np.zeros(time.size)
    data["weight_track"] = np.zeros((time.size, size, size))

    if verbose > 0:
        for i, t in enumerate(time):
            if verbose > 0.0 and i % (time.size * verbose) == 0 and verbose < 1.0:
                print("{}% completed...".format(i / time.size * 100))
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
                data["flux_track"][i] = agent.sim.flux_mat[0, 0]
                agent.sim.update_weights_with_reward(reward)
            body.stepN(agent.stepsize, agent.outputs, conf_list)

            # update center and fluctuating weights
            agent.inner_weights = agent.sim.center_mat[:size].copy()
            agent.biases = agent.sim.center_mat[size].copy()
            agent.extended_weights = agent.sim.extended_mat[:size].copy()
            agent.extended_biases = agent.sim.extended_mat[size].copy()
            # record data to plot
            if i % record_every == 0:
                data["distance"][i] = body.cx
                data["performance"][i] = (
                    data["distance"][i] - data["distance"][i - (2200 - 1)]
                ) / 220
                data["inst_perf"][i] = (
                    data["distance"][i] - data["distance"][i - 1]
                ) / 0.1
                data["agent_distance"][i] = agent.distance_track[0]
                data["agent_performance"][i] = (
                    data["agent_distance"][i] - data["agent_distance"][i - (2200 - 1)]
                ) / 220
                data["weight_track"][i] = agent.extended_weights
                data["window_b"][i] = agent.window_b[-1]
                data["window_a"][i] = agent.window_a[-1]
                data["avg_window_b"][i] = agent.window_b.mean()
                data["avg_window_a"][i] = agent.window_a.mean()
                data["difference_track"][i] = (
                    data["avg_window_b"][i] - data["avg_window_a"][i]
                )
    else:
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
                data["flux_track"][i] = agent.sim.flux_mat[0, 0]
                agent.sim.update_weights_with_reward(reward)
            body.stepN(agent.stepsize, agent.outputs, conf_list)
            # update center and fluctuating weights
            agent.inner_weights = agent.sim.center_mat[:size].copy()
            agent.biases = agent.sim.center_mat[size].copy()
            agent.extended_weights = agent.sim.extended_mat[:size].copy()
            agent.extended_biases = agent.sim.extended_mat[size].copy()
            # record data to plot
            if i % record_every == 0:
                data["distance"][i] = body.cx
                data["performance"][i] = (
                    data["distance"][i] - data["distance"][i - (2200 - 1)]
                ) / 220
                data["inst_perf"][i] = (
                    data["distance"][i] - data["distance"][i - 1]
                ) / 0.1
                data["agent_distance"][i] = agent.distance_track[0]
                data["agent_performance"][i] = (
                    data["agent_distance"][i] - data["agent_distance"][i - (2200 - 1)]
                ) / 220
                data["weight_track"][i] = agent.extended_weights
                data["window_b"][i] = agent.window_b[-1]
                data["window_a"][i] = agent.window_a[-1]
                data["avg_window_b"][i] = agent.window_b.mean()
                data["avg_window_a"][i] = agent.window_a.mean()
                data["difference_track"][i] = (
                    data["avg_window_b"][i] - data["avg_window_a"][i]
                )
    data["ix"] = ix
    # ax[0].plot(time, performance, label="Real time performance", color=cmap[c * 4 + 0])
    # ax[1].plot(time, flux_track, label="Fluctuation size", color=cmap[c * 4 + 0])
    # ax[0].plot(
    #     time, difference_track, label="difference"
    # )  # difference in averaged performance windows, the neuromodulatory single
    # ax[0].plot(
    #     time, avg_window_b, label="avg_window_b", color=cmap[c * 4 + 1], ls="dashed"
    # )  # average current performance
    # ax[0].plot(
    #     time, avg_window_a, label="avg_window_a", color=cmap[c * 4 + 1], ls="dashdot"
    # )  # average past performance
    # ax[0].plot(time, window_b, label="window_b")  # current performance as observed by agent
    # ax[0].plot(time, window_a, label="window_a")  #past performance as observed by agent
    # ax[0].plot(time, inst_perf, label="instant perf")
    x = {key: item for key, item in data.items() if key in output}
    x["ix"] = ix
    return x


cmap = plt.get_cmap("tab10").colors
fig, ax = plt.subplots(nrows=2)
results = []
verbose = 0.1
plots = []
output = ["performance", "flux_track"]
with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    for ix, g in enumerate(genomes):
        for trial in range(num_trials):
            np.random.seed(np.random.randint(10000000))
            if len(results) == 0:
                print_verbose = verbose
            else:
                print_verbose = -1
            results.append(
                executor.submit(
                    learn,
                    g,
                    print_verbose,
                    params,
                    output=[
                        "avg_window_b",
                        "performance",
                        "flux_track",
                    ],
                    ix=ix,
                )
            )
        if len(results) == num_processes:
            for future in concurrent.futures.as_completed(results):
                plots.append(future.result())
            results = []
for future in concurrent.futures.as_completed(results):
    plots.append(future.result())
time = np.arange(0, params["duration"], params["stepsize"])

means = {ix: {} for ix in range(num_genomes)}

for param in output:
    for ix in range(num_genomes):
        means[ix][param] = []
        for p in plots:
            if p["ix"] == ix:
                means[ix][param].append(p[param])
for param in output:
    print(param)
    for key in means.keys():
        if param == "flux_track":
            ax[1].plot(time, np.mean(means[key][param], axis=0), color=cmap[key])
        else:
            ax[0].plot(time, np.mean(means[key][param], axis=0), color=cmap[key])

ax[0].legend()
ax[1].legend()
plt.show()
