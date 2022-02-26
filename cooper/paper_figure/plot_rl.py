import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from vis.colorline import colorline
import matplotlib.path as mpath
import os
from vis.plot_rl_figures import plot_from_file


def main():

    filename="cooper/paper_figure/rl_data/recover_seed-0.csv"
    plot_from_file(filename,reduce_plot_data=1, stepsize=1)
    
    directory="cooper/paper_figure/rl_data/"
    plot_agg_from_files(directory )

def plot_agg_from_files(directory, max_entries=None):

    save_folder="cooper/paper_figure/plots/"
    
    ignore_transients=100
    perf_bias = 0.05


    filenames = os.listdir(directory)
    agg_dict={}
    if max_entries != None:
        filenames = filenames[:max_entries]

    for filename in filenames:
        print(filename)
        label_to_data_map=plot_from_file(f"{directory}{filename}", show_subplots=False)
        for key in label_to_data_map.keys():
            if key not in agg_dict.keys():
                agg_dict[key]=[]
            else:
                agg_dict[key].append(label_to_data_map[key])
    #gotta deal with the weights and outputs across dimensions...
    for i in range(len(agg_dict["time"])):
        plt.plot(agg_dict["time"][i][ignore_transients:], agg_dict["performances"][i][ignore_transients:],\
            label="instantaenous performance", color=[0,0,0.5,0.1] )
    


    plt.xlabel("Time")
    plt.ylabel("Performance")
    plt.show()


    for perf in ["performances","running_average_performances"]:

        
        avg = (np.mean(agg_dict[perf], axis=0)+ perf_bias)*100

        #err= np.std(agg_dict["performances"], ddof=1, axis=0)  / np.sqrt(np.size(agg_dict[perf]))
        err= np.std(agg_dict[perf], axis=0)*100
        plt.plot(agg_dict["time"][0][ignore_transients:], avg[ignore_transients:], label=perf ) 
        plt.fill_between(agg_dict["time"][0][ignore_transients:], (avg-err)[ignore_transients:], (avg+err)[ignore_transients:], alpha=0.2)
        plt.xlabel("Time")
        plt.ylabel(perf)
        plt.savefig(f"{save_folder}{perf}.png")
        plt.show()


    size=2
    if "w0_0" in agg_dict:
        for i in range(size):
            for j in range(size):

                avg = np.mean(agg_dict[f"w{i}_{j}"], axis=0)
                err= np.std(agg_dict[f"w{i}_{j}"], axis=0)

                plt.plot(agg_dict["time"][0], avg, label=f"avg w{i}_{j}" )
                plt.fill_between(agg_dict["time"][0], avg-err, avg+err, alpha=0.2)
        plt.show()

if __name__ == "__main__":
    main()