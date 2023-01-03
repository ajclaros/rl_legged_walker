import leggedwalker
from rl_ctrnn import RL_CTRNN
import matplotlib.pyplot as plt
import os
import numpy as np
import concurrent.futures
import time as ti
from initial_run import checkdir
from aux import lighten


x = ti.time()
num_processes = 16
num_trials = 4
random_genomes_in_range = False
indices = [
    2,
    4,
    5,
    7,
]  # indices of the sorted genomes. Index 0 will have the lowest fitness in the group
num_genomes = len(indices)
plot_vars = ["flux_mat", "avg window_b_track"]
params = {
    "duration": 40000,
    "size": 3,
    "delay": 330,  # changes in performance are observed in the averaging windows ~200 time units after the real time performance
    # compare end performance using delay \in [250, 350] with delay=0 to test learning efficacy
    # effect prominant as task becomes more difficult (from 3N RPG, [0] -> 3N CPG [0,1,2] )
    # on the simplist task, agent can still learn with 0 delay.
    "generator": "CPG",
    "config": "012",
    "window_size": 1000,  # size of the two averaging windows. Fitness function runs for 220 time units
    # ------------------------------------------------------------------------------------------------
    # "learn_rate": 0.000007,
    # "conv_rate": 0.000007,
    # "init_flux": 0.00015,  # Params identified for generator:RPG, size:3, config:0, fitness ~0.234
    # "max_flux": 0.002,  # May be robust to different starting positions
    # ------------------------------------------------------------------------------------------------
    # "learn_rate": 0.000004,  #
    # "conv_rate": 0.000004,  # Params identified for generator:CPG
    # "init_flux": 0.00015,  # less likely to be robust to different genomes/starting fitnesses
    # "max_flux": 0.001,  #
    # ----------------------
    "learn_rate": 0.0000002,  #
    "conv_rate": 0.0000002,  # Params identified for generator:CPG size:3, config:012
    "init_flux": 0.000025,  # less likely to be robust to different genomes/starting fitnesses
    "max_flux": 0.00003,  #
    # -------------
    # "learn_rate": 0.000005,
    # "conv_rate": 0.000000,
    # "init_flux": 0.0001,  # experimental with exponential growth rate
    # "max_flux": 0.005,
    "period_min": 1000,  # integer time units
    "period_max": 4000,  # integer time units
    "learning_start": 2000,  # integer time units, determines how long the agent will gather data from static weights.
    # -------------------------------------------------------------------------------------------------
    # Parameters for reward below a value
    "tolerance": 0.0000,  # ignore abs(reward) below tolerance and only update moment (and subsequently update extended weights)
    "performance_bias": 0.000,  # penalize for reward below value -> increase amplitude if reward below value
    # -------------------------------------------------------------------------------------------------
    "fit_range": (0.25, 0.7),  # select genomes within (min, max) fitness range
    "index": 0,  # given all genomes matching "$generator/$size/$configuration"
    # choose the file at specified index position
    "stepsize": 0.1,
    "record_every": 100000,  # not currently implemented. When logging data, records every n time steps
}

pathname = f"./evolved/{params['generator']}/{params['size']}/{params['config']}"
if not os.path.exists(pathname):
    os.makedirs(pathname)
checkdir(
    pathname, fit_range=params["fit_range"], min_files=10, num_processes=num_processes
)
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
    plot_vars=[],
    ix=0,
    record_every=1,
):
    def custom_reward(body, agent, previous, param):
        if param in body.__dict__.keys():
            reward = abs(body.__dict__[param]) - abs(previous)
        else:
            reward = abs(agent.__dict__[param].sum()) - abs(previous)
        return reward

    # initial setup
    np.random.seed()
    # Purpose of
    sample_rate = 1 / record_every
    # convert configuration string into a list: "012" -> [0,1,2]
    conf_list = [int(a) for a in params["config"]]
    duration = params["duration"]
    learn_rate = params["learn_rate"]
    conv_rate = params["learn_rate"]
    delay = params["delay"]
    size = params["size"]
    # parameter minimum and maxima
    p_min = np.ones((size + 2, size)) * -16
    p_max = np.ones((size + 2, size)) * 16

    # period range
    period_min = np.ones((size + 2, size)) * params["period_min"]
    period_max = np.ones((size + 2, size)) * params["period_max"]

    # flux
    init_flux = np.ones((size + 2, size)) * params["init_flux"]
    max_flux = np.ones((size + 2, size)) * params["max_flux"]

    time = np.arange(0, int(duration), params["stepsize"])

    track = []
    mat = []
    avg = []
    no_delay = []
    other = []
    data = {}

    for name in plot_vars:
        if "avg" in name:
            avg.append(name)
        elif "no_delay" in name:
            no_delay.append(name)
        elif "track" in name:
            track.append(name)
        elif "mat" in name:
            mat.append(name)
        else:
            other.append(name)
        data[name] = np.zeros(int(time.size * sample_rate))
    data["distance"] = np.zeros(int(time.size * sample_rate))
    # initialize parameters to record

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
    # end inital setup

    previous = 0
    param = "omega"

    tmp = 0

    def run(previous, param, tmp):
        if params["generator"] == "RPG":
            agent.setInputs(np.array([body.anglefeedback()] * agent.size))
        else:
            # CPG
            agent.setInputs([0] * agent.size)

        if t < params["learning_start"]:
            # gather performance data before learning
            agent.step(params["stepsize"])
            reward = agent.reward_func2(body.cx, learning=False)
        else:
            # learning phase
            agent.stepRL(params["stepsize"])
            reward = agent.reward_func2(body.cx, learning=True)
            agent.sim.update_weights_with_reward(reward)
            if abs(reward) < 0.0005:

                # agent.sim.flux_mat *= 1 + np.log(tmp) * 0.0001
                # Gompertz curve performas best
                agent.sim.flux_mat += 0.00001 * np.exp(
                    -1.8 * np.exp(-0.01001 * tmp + 0.5)
                )
                # agent.sim.flux_mat = agent.sim.flux_mat * (np.log(tmp) * 0.001)
                tmp += 1
            else:
                tmp = 10

        body.stepN(agent.stepsize, agent.outputs, conf_list)
        return reward, t

    def record():
        for name in avg:
            t_name = name.split(" ")[1]
            data[name][int(i * sample_rate)] = agent.__dict__[t_name].mean()
        for name in no_delay:
            t_name = name.split(" ")[1]
            data[name][int(i * sample_rate)] = agent.__dict__[t_name][
                (agent.index + params["delay"] * 10) % (agent.__dict__[t_name].size)
            ]
        for name in track:
            if name == "performance_track":
                data["performance_track"][int(i * sample_rate)] = (
                    agent.distance_track[agent.delayed % agent.distance_track.size]
                    - agent.distance_track[
                        (agent.delayed - 2200) % agent.distance_track.size
                    ]
                ) / (agent.stepsize * 2200)
            else:
                data[name][int(i * sample_rate)] = agent.__dict__[name][
                    agent.index % agent.__dict__[name].size
                ]
        for name in mat:
            data[name][int(i * sample_rate)] = agent.sim.__dict__[name][0, 0]
        for name in other:
            data[name][int(i * sample_rate)] = agent.__dict__[name][
                agent.index % agent.__dict__[name].size
            ]

    # simulation start
    if verbose > 0:
        for i, t in enumerate(time):
            if verbose > 0.0 and i % (time.size * verbose) == 0 and verbose < 1.0:
                print("{}% completed...".format(i / time.size * 100))
            previous, tmp = run(previous, param, tmp)
            if i % record_every == 0:
                record()
    else:
        for i, t in enumerate(time):
            previous, tmp = run(previous, param, tmp)
            if i % record_every == 0:
                record()
    # simulation end
    data["ix"] = ix
    x = {key: item for key, item in data.items() if key in plot_vars}
    x["ix"] = ix
    return x


cmap = plt.get_cmap("tab10").colors
fig, ax = plt.subplots(nrows=2)
results = []
verbose = 0.1
plots = []
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
                    plot_vars=plot_vars,
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
print("Plotting")

means = {ix: {} for ix in range(num_genomes)}

for param in plot_vars:
    for ix in range(num_genomes):
        means[ix][param] = []
        for p in plots:
            if p["ix"] == ix:
                means[ix][param].append(p[param])

for param in plot_vars:
    print(param)
    for key in means.keys():
        color = None
        if param == "flux_mat":
            ax[1].plot(time, np.mean(means[key][param], axis=0), color=cmap[key])
        else:
            if param == "avg window_b_track":
                color = lighten(cmap[key], 1.0)
                lw = 2
                ls = "solid"
            elif param == "avg window_a_track":
                color = lighten(cmap[key], 1.2)
                lw = 3
                ls = "solid"
            else:
                color = cmap[key]
                lw = 1
                ls = "solid"
            for i in range(len(means[key][param])):
                ax[0].plot(time, means[key][param][i], color=color, lw=lw, ls=ls)
            # ax[0].plot(time, np.mean(means[key][param], axis=0), color=c)

print(ti.time() - x)
# ax[0].legend()
# ax[1].legend()
plt.show()
