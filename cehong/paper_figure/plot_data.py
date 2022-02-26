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
    
    #This can be used to plot in-depth details of the performance of a single run
    #The data collected in this folder has much more detail than is normally recorded or needed.
    #filename="cehong/paper_figure/data/duration-2000_perfbias-0.05/size2/learn_data_random_size-2_seed-0.csv"
    #plot_from_file(filename, size=2, perf_bias = 0.05, reduce_plot_data=1, stepsize=1)
    
    #takes a while to read (270 megabytes)
    filename="cehong/paper_figure/data/duration-2000_perfbias-0.05/size10/learn_data_random_size-10_seed-0.csv"
    plot_from_file(filename, size=10, perf_bias = 0.05, reduce_plot_data=1, stepsize=1)
    max_entries=None



    # This can be used to compare the performance of different runs as aggregated by directory
    # with multiple runs of the same size in each directory
    directories=[]
    # These two parameters are set in the learn_solutions script
    # These are used to name the directory
    duration=10000
    perf_bias = 0.1
    labels=[]
    for i in range (2,11):
        directory=f"cehong/paper_figure/data/duration-{duration}_perfbias-{perf_bias}/size{i}/"
        #plot_agg_from_files(directory, max_entries=20, size=i)
        directories.append(directory)
        labels.append(f"size{i}")
    save_filename=f"perf-ALL_entries-{max_entries}_duration-{duration}.png"
    plot_scalability( directories, labels, save_filename, max_entries=None, perf_bias=perf_bias, stepsize=1)

def plot_scalability(directories, labels, save_filename, max_entries=None, perf_bias = 0.05, stepsize=100):
    plot_directory="cehong/paper_figure/plots"
    #for each size
    for i in range(len(directories)):
        ignore_transients=100*stepsize
        
        directory=directories[i]

        filenames = os.listdir(directory)
        agg_dict={}
        if max_entries != None:
            filenames = filenames[:max_entries]
        #get all runs
        for filename in filenames:
            print(filename)
            label_to_data_map=plot_from_file(f"{directory}{filename}", show_subplots=False)
            for key in label_to_data_map.keys():
                if key not in agg_dict.keys():
                    agg_dict[key]=[]
                else:
                    agg_dict[key].append(label_to_data_map[key])
        
                   
        avg = (np.mean(agg_dict["running_average_performances"], axis=0)+ perf_bias)*100
        #err= np.std(agg_dict["performances"], ddof=1, axis=0)  / np.sqrt(np.size(agg_dict[perf]))
        err= np.std(agg_dict["running_average_performances"], axis=0)*100
        plt.plot(agg_dict["time"][0][ignore_transients:], avg[ignore_transients:], label=labels[i] ) 
        plt.fill_between(agg_dict["time"][0][ignore_transients:], (avg-err)[ignore_transients:], (avg+err)[ignore_transients:], alpha=0.2)
        plt.xlabel("Time")
        plt.ylabel("running_average_performances by size")
    plt.legend(loc="upper left")
    plt.savefig(f"{plot_directory}/{save_filename}")
    plt.show()




def plot_agg_from_files(directory, max_entries=None, size=2, stepsize=100):

    plot_directory="cehong/paper_figure/plots"
    show_figures=["instantaenous_performance", "running_average_performances" "aggregated_weights" ]
    #show_figures=[2]
    
    ignore_transients=100*stepsize
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

    if "instantaenous_performance" in show_figures:
        #gotta deal with the weights and outputs across dimensions...
        for i in range(len(agg_dict["time"])):
            plt.plot(agg_dict["time"][i][ignore_transients:], agg_dict["performances"][i][ignore_transients:],\
                label="instantaenous performance", color=[0,0,0.5,0.1] )
        


        plt.xlabel("Time")
        plt.ylabel("Performance")
        plt.show()

    if "running_average_performances" in show_figures:
        #for perf in ["performances","running_average_performances"]:
        for perf in ["running_average_performances"]:

            
            avg = (np.mean(agg_dict[perf], axis=0)+ perf_bias)*100

            #err= np.std(agg_dict["performances"], ddof=1, axis=0)  / np.sqrt(np.size(agg_dict[perf]))
            err= np.std(agg_dict[perf], axis=0)*100
            plt.plot(agg_dict["time"][0][ignore_transients:], avg[ignore_transients:], label=perf ) 
            plt.fill_between(agg_dict["time"][0][ignore_transients:], (avg-err)[ignore_transients:], (avg+err)[ignore_transients:], alpha=0.2)
            plt.xlabel("Time")
            plt.ylabel(perf)
            #plt.show()
            plt.savefig(f"{plot_directory}/perf-{size}_entries-{max_entries}.png")

    if "aggregated_weights" in show_figures:
        #size=2
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