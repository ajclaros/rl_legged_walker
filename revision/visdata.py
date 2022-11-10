import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# folders = os.listdir("data")
# generator_type = "RPG"
# folders = [name for name in folders if generator_type in name]
# files = [
#     name
#     for name in os.listdir(f"./data/{folders[0]}")
#     if ".npz" in name and generator_type in name
# ]
# data = ""
# for i, name in enumerate(files):
#     data = np.load(f"./data/{folders[0]}/{name}")


def plotWeightsBiases(
    data, show=False, legend=True, extended=False, linewidth=2, save=False, ax=None
):
    if not ax:
        fig, ax = plt.subplots()
    cmap = plt.get_cmap("tab20").colors
    time = np.arange(0, data["duration"], data["stepsize"] / data["sample_rate"])

    for i in range(data["size"]):
        for j in range(data["size"]):
            ax.plot(
                time,
                data["inner_weights"].T[i, j],
                color=cmap[i * data["size"] + j],
                label=f"w_{i}{j}",
                lw=linewidth,
            )
            if extended:
                ax.plot(
                    time,
                    data["extended_weights"].T[i, j],
                    color=cmap[i * data["size"] + j],
                    ls="dotted",
                    lw=linewidth - 1,
                )

        if "biases" in data.files:
            ax.plot(
                time,
                data["biases"].T[i],
                color=cmap[-i],
                label=f"bias_{i}",
                lw=linewidth,
            )
            if extended:
                ax.plot(
                    time,
                    data["extended_biases"].T[i],
                    ls="dotted",
                    color=cmap[-i],
                    lw=linewidth - 1,
                )
    ax.axvline(
        data["learning_start"],
        color="k",
        lw="1",
        ls="--",
    )
    ax.title.set_text(f"Weight and Bias change during Trial:{data['generator_type']}")
    plt.tight_layout()
    if show:
        if legend:
            plt.legend()
        plt.show()
    if save:
        plt.savefig("./data/images/weight-bias.png")


def plotBehavior(data, show=False, save=False):
    fig, ax = plt.subplots(nrows=2, ncols=2)
    time = np.arange(0, data["duration"], data["stepsize"] / data["sample_rate"])
    ax[0][0].plot(time, data["outputs"])
    ax[0][0].axvline(data["learning_start"], c="k", ls="--")
    ax[0][0].set_title("Neural outputs")
    ax[0][1].plot(time, data["distance"])
    ax[0][1].axvline(data["learning_start"], c="k", ls="--")
    ax[0][1].set_title("Distance")
    ax[1][0].plot(time, data["omega"])
    ax[1][0].axvline(data["learning_start"], c="k", ls="--")
    ax[1][0].set_title("Omega")
    ax[1][1].plot(time, data["angle"])
    ax[1][1].set_title("Angle")
    ax[1][1].axvline(data["learning_start"], c="k", ls="--")
    fig.suptitle(
        f"Duration:{data['duration']},\nStartFit:{np.round(data['start_fitness'],3)}\nEndFit:{np.round(data['end_fitness'],3)}\nSize:{data['size']}"
    )
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig("./data/images/behavior.png")


def plotChosenParam(
    filename, params, show=False, save=True, title=None, title_params=None
):
    numplots = int(np.ceil(np.sqrt(len(params))))
    print(params)
    fig, ax = plt.subplots(
        nrows=numplots,
        ncols=numplots,
        figsize=(
            numplots,
            numplots,
        ),
    )
    data = np.load(f"./{filename}")
    time = np.arange(0, data["duration"], data["stepsize"] / data["sample_rate"])
    if (len(params)) == 1:
        ax.plot(time, data[params[0]])
    else:
        for i, param in enumerate(params):
            row = i // numplots
            col = i % numplots

            if type(param) == tuple:
                for p in param:
                    ax[row][col].plot(time, data[p], label=p.replace("_", " "))
                    ax[row][col].legend()
                param = [name.replace("_", " ") for name in param]
                ax[row][col].title.set_text("\nvs\n".join(param))
            else:
                ax[row][col].plot(time, data[param])
                ax[row][col].title.set_text(f"{param}")
            ax[row][col].axvline(
                data["learning_start"],
                c="k",
                ls="--",
            )
    if not title:
        fig.suptitle(
            f"startFit:{np.round(data['start_fitness'],3)}\nEndFit:{np.round(data['end_fitness'], 3)}\nDuration: {data['duration']}"
        )
    if save:
        title = []
        for param in params:
            if type(param) == tuple:
                for p in param:
                    title.append(p)
            else:
                title.append(param)

        figname = "_".join(title)

        plt.savefig(f"./data/images/{figname}")
    if show:
        plt.show()


def plotAverageParam(
    param, show=False, save=True, b=60, pathname="./data", baseline=None
):
    files = os.listdir(pathname)
    averaged = []
    genome_list = []
    files = [name for name in files if ".npz" in name]
    data = np.load(f"{pathname}/{files[0]}")
    time = np.arange(0, data["duration"], data["stepsize"] / data["sample_rate"])
    genome = data["startgenome"]
    genome_list = genome.reshape((1, genome.size))
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("tab10").colors
    skip = 0
    for i, name in enumerate(files):
        try:
            if i == b:
                break
            data = np.load(f"{pathname}/{name}")
            genome = data["startgenome"]

            genome_list = np.vstack([genome_list, genome])
            genome_list = np.unique(genome_list, axis=0)
            colorindex = (genome_list == genome.reshape((1, genome.size))).nonzero()[0][
                0
            ]
            # .nonzero()[0][0]
            ax.plot(time, data[param], ls="--", c=cmap[colorindex])
            averaged.append(data[param])
        except:
            skip += 1
            continue
    ax.axvline(data["learning_start"], ls="--")
    if baseline:
        ax.axhline(baseline, ls="--", c="r")
    ax.title.set_text(
        f"Averaged {param} over duration {data['duration']}\n all {len(files)-skip} trials\nUsing {data['metric']} measurement"
    )
    ax.plot(time, np.median(averaged, axis=0), c="k")
    if show:
        plt.show()
    if save:
        plt.savefig(f"./data/images/all_files-{data['metric']}-{param}.png")


def plotDistributionParam(
    param, show=False, save=True, b=60, pathname="./data", bins=20
):
    files = os.listdir(pathname)
    files = [name for name in files if ".npz" in name]
    data = np.load(f"{pathname}/{files[0]}")
    time = np.arange(0, data["duration"], data["stepsize"] * data["sample_rate"])
    fig, ax = plt.subplots()
    param_data = []
    skip = 0
    for i, name in enumerate(files):
        try:
            if i == b:
                break
            data = np.load(f"{pathname}/{name}")
            param_data.append(data[param][-1])
        except:
            skip += 1

    ax.title.set_text(
        f"Histogram {param} over duration {data['duration']}\n all {len(files)-skip} trials\n using {data['metric']} measurement"
    )
    ax.hist(param_data, bins=bins, density=True)
    if show:
        plt.show()
    if save:
        plt.savefig(f"./data/images/{param}-{data['metric']}-distribution.png")


# plotWeightsBiases(data, show=True)
# plotBehavior(data, show=True)


def plot_NeuralOutputs(data, show=False, cmap="jet"):
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6, 3))
    outputs = data["outputs"].T
    time = np.arange(0, data["learning_start"], data["stepsize"] / data["sample_rate"])
    points = np.array(data["outputs"][: data["learning_start"]].T).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    time = np.arange(0, data["learning_start"], data["stepsize"] / data["sample_rate"])
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(time.min(), time.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=8)
    # Set the values used for colormapping
    lc.set_array(time)
    lc.set_linewidth(8)
    line = ax[0].add_collection(lc)
    fig.colorbar(line, ax=ax[0])
    points = np.array(data["outputs"][data["learning_start"] :].T).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    time = np.arange(
        data["learning_start"], data["duration"], data["stepsize"] / data["sample_rate"]
    )
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(time.min(), time.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=8)
    # Set the values used for colormapping
    lc.set_array(time)
    lc.set_linewidth(8)
    line = ax[1].add_collection(lc)
    fig.colorbar(line, ax=ax[1])
    ax[0].set_xlabel("Neuron 0")
    ax[0].set_ylabel("Neuron 1")
    ax[1].set_xlabel("Neuron 0")
    ax[1].set_ylabel("Neuron 1")
    plt.suptitle("Neural Outputs")
    if show == True:
        plt.legend()
        plt.show()
