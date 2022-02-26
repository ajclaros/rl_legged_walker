import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from vis.colorline import colorline
import matplotlib.path as mpath
import os


def main():

    filename="data/recovered_run_data/nnsize-2_sol-seed-6/seed1/recover_trial-seed1_fit-perc-0.00__afig1_set_D_4d.json.csv"
    #plot_from_file(filename)
    directory="data/recovered_run_data/nnsize-2_sol-seed-6/repeated_4d_init-flux-4/"
    #directory="data/recovered_run_data/nnsize-2_sol-seed-6/repeated_abb_close/"
    directory="cooper/figure2/data/"
    plot_agg_from_files(directory, max_entries=-1)

def plot_agg_from_files(directory, max_entries=-1):
    
    stepsize=0
    ignore_transients=100

    filenames = os.listdir(directory)
    agg_dict={}
    #print(  filenames )

    for filename in filenames[:max_entries]:
        print(filename)
        label_to_data_map=plot_from_file(f"{directory}{filename}", show_subplots=False)
        for key in label_to_data_map.keys():
            if key not in agg_dict.keys():
                agg_dict[key]=[]
            else:
                agg_dict[key].append(label_to_data_map[key])
    #gotta deal with the weights and outputs across dimensions...
    for i in range(len(agg_dict["time"])):
        plt.plot(agg_dict["time"][i][ignore_transients:], agg_dict["performances"][i][ignore_transients:],label="instantaenous performance" )
    


    plt.xlabel("Time")
    plt.ylabel("Performance")
    plt.show()


    for perf in ["performances","running_average_performances"]:

        perf_bias = 0.05
        avg = (np.mean(agg_dict[perf], axis=0)+ perf_bias)*100

        #err= np.std(agg_dict["performances"], ddof=1, axis=0)  / np.sqrt(np.size(agg_dict[perf]))
        err= np.std(agg_dict[perf], axis=0)*100
        plt.plot(agg_dict["time"][0][ignore_transients:], avg[ignore_transients:], label=perf ) 
        plt.fill_between(agg_dict["time"][0][ignore_transients:], (avg-err)[ignore_transients:], (avg+err)[ignore_transients:], alpha=0.2)
        plt.xlabel("Time")
        plt.ylabel(perf)
        plt.show()

    avg=[]
    std=[]

    # print ( np.shape(agg_dict["w0_0"]) )
    # agg_dict["performances"] = np.transpose(agg_dict["performances"], 1, 0)
    #print ( np.shape(agg_dict["w0_0"]) )
    size=2
    if "w0_0" in agg_dict:
        for i in range(size):
            for j in range(size):

                avg = np.mean(agg_dict[f"w{i}_{j}"], axis=0)
                err= np.std(agg_dict[f"w{i}_{j}"], axis=0)

                plt.plot(agg_dict["time"][0], avg, label=f"avg w{i}_{j}" )
                plt.fill_between(agg_dict["time"][0], avg-err, avg+err, alpha=0.2)
        plt.show()

    





def plot_from_file(filename, size=2, perf_bias = 0.05, reduce_plot_data=10, stepsize=100, show_subplots=True):
    
    data = np.genfromtxt(filename,delimiter=",", dtype=float, names=True)
    length=len(data)
    header_length=len(data.dtype.names)
    plot_data=np.zeros( (header_length,length ) )
    
    label_to_data_map={}
    #print(data.dtype.names)

    labels=[]
    for i in range(header_length):
        labels.append(data.dtype.names[i])

    for t in range(len(data)):

        for i in range(header_length):
            plot_data[i][t] = data[t][i]
       
    for i in range(header_length):
        label_to_data_map[data.dtype.names[i]]=plot_data[i]
    #label_to_data_map["time"] = data series for time...
    #post-processing outputs, weights, etc

    
    if "outputs" in labels:
        label_to_data_map["outputs"]=np.zeros( (size,length ) )
        for i in range(size):
            label_to_data_map["outputs"][i]=label_to_data_map["output"+str(i)]
    
    if "w0_0" in labels:
        label_to_data_map["weights"]=np.zeros( (size,size,length ) )
        for i in range(size):
            for j in range(size):
                label_to_data_map["weights"][i][j]=label_to_data_map[f"w{i}_{j}"]
    
    if "fluxw0_0" in labels:
        label_to_data_map["flux_weights"]=np.zeros( (size,size,length ) )
        for i in range(size):
            for j in range(size):
                label_to_data_map["flux_weights"][i][j]=label_to_data_map[f"flux_w{i}_{j}"]
    
    if "bias0" in labels:
        label_to_data_map["biases"]=np.zeros( (size,length ) )
        for i in range(size):
            label_to_data_map["biases"][i]=label_to_data_map[f"bias{i}"]
    
    if "flux_bias0" in labels:
        label_to_data_map["flux_biases"]=np.zeros( (size,length ) )
        for i in range(size):
            label_to_data_map["flux_biases"][i]=label_to_data_map[f"flux_bias{i}"]
    
    
    #add biases

    # plt.scatter(wAs, wBs, c=colors )
    #plt.plot( label_to_data_map["time"], label_to_data_map["performances"] )

    #plt.show()
    if show_subplots:
        subplots(None, label_to_data_map, label_to_data_map["time"], -1, reduce_plot_data=reduce_plot_data,\
            stepsize=stepsize, perf_bias = perf_bias, size=size)

    return label_to_data_map
    #plot(None, label_to_data_map, label_to_data_map["time"], -1, None)



# stopStep provided because we only want to plot as long as the experiment ran
def plot(osc_exp, plot_info, time, stop_step, reduce_plot_data=100):
    nnsize = len(plot_info["outputs"])


    plt.xlabel("Time")
    plt.ylabel("Performance")
    if reduce_plot_data == None:
        plt.plot(time[0:stop_step],plot_info["rewards"][0:stop_step],label="instantaenous reward" )
        plt.plot(time[0:stop_step],plot_info["performances"][0:stop_step],label="instantaenous performance" )
        plt.plot(time[0:stop_step],plot_info["running_average_performances"][0:stop_step],label="running average performance" )
        highestAmp = np.max(plot_info["amps"] )
        plt.plot(time[0:stop_step],plot_info["amps"][0:stop_step]/highestAmp/100,label="flux amplitude/{:.2f} for scaling".format(highestAmp*100) )
    else:
        plt.plot(time[0:stop_step][0::reduce_plot_data],plot_info["rewards"][0:stop_step][0::reduce_plot_data],label="instantaenous reward" )
        plt.plot(time[0:stop_step][0::reduce_plot_data],plot_info["performances"][0:stop_step][0::reduce_plot_data],label="instantaenous performance" )
        plt.plot(time[0:stop_step][0::reduce_plot_data],plot_info["running_average_performances"][0:stop_step][0::reduce_plot_data],label="running average performance" )
        highestAmp = np.max(plot_info["amps"] )
        plt.plot(time[0:stop_step][0::reduce_plot_data],plot_info["amps"][0:stop_step][0::reduce_plot_data]/(highestAmp*100),label="flux amplitude/{:.2f} for scaling".format(highestAmp*100) )



    plt.legend()
    plt.title("Fluctuation Amplitude, Instantaneous and Running Average Performance over Time")
    plt.show()




    
    #plot output of each neuron
    for i in range( nnsize ):
        #if reduce_plot_data == None:
        plt.plot(time[0:stop_step],plot_info["outputs"][i][0:stop_step],label=i, alpha=0.5 )
        #else:
        #    plt.plot(time[0:stop_step][0::reduce_plot_data],plot_info["outputs"][i][0:stop_step][0::reduce_plot_data],label=i, alpha=0.5 )
        
    plt.xlabel("Time")
    plt.ylabel("Output")
    plt.legend()
    plt.title("Neural activity")
    plt.show()
    if reduce_plot_data == None:
        plt.plot(time[0:stop_step],plot_info["rewards"][0:stop_step],label="instantaenous reward" )
    else:
        plt.plot(time[0:stop_step][0::reduce_plot_data],plot_info["rewards"][0:stop_step][0::reduce_plot_data],label="instantaenous reward" )
    
    #divided by (10 normally) so as to keep the vertical scale compressed (otherwise hard to see the other factors)
    highestAmp = np.max(plot_info["amps"] )
    plt.plot(time[0:stop_step],plot_info["amps"][0:stop_step]/highestAmp,label="flux amplitude/{:.2f} for scaling".format(highestAmp) )
    plt.plot(time[0:stop_step],plot_info["running_average_performances"][0:stop_step],label="running average (100 steps) rewards" )

    plt.xlabel("Time")
    plt.ylabel("Reward & Flux")
    plt.legend()
    plt.title("Reward")
    plt.show()

    # Plot Synaptic Weights over time
    for i in range(nnsize):
        for j in range(nnsize):
            if reduce_plot_data == None:
                plt.plot(time[0:stop_step],plot_info["weights"][i][j][0:stop_step],label="{}->{}".format(i,j) )
            else:
                plt.plot(time[0:stop_step][0::reduce_plot_data],plot_info["weights"][i][j][0:stop_step][0::reduce_plot_data],label="{}->{}".format(i,j) )


    plt.xlabel("Time")
    plt.ylabel("Weight Centers")
    plt.legend()
    plt.title("Synaptic Centers Strength over time")
    plt.show()

    # Plot Synaptic Weights over time
    for i in range(nnsize):
        for j in range(nnsize):
            if reduce_plot_data == None:
                plt.plot(time[0:stop_step],plot_info["flux_weights"][i][j][0:stop_step],label="{}->{}".format(i,j) )
            else:
                plt.plot(time[0:stop_step][0::reduce_plot_data],plot_info["flux_weights"][i][j][0:stop_step][0::reduce_plot_data],label="{}->{}".format(i,j) )

    plt.xlabel("Time")
    plt.ylabel("Fluctuating Weights")
    plt.legend()
    plt.title("Fluctating Weights over time")
    plt.show()

    if "biases" in plot_info.keys():
        # Plot Biases over time
        for i in range(nnsize):
            plt.plot(time[0:stop_step],plot_info["biases"][i][0:stop_step],label="bias{}".format(i) )


        plt.xlabel("Time")
        plt.ylabel("Bias Centers")
        plt.legend()
        plt.title("Bias Centers  over time")
        plt.show()
    if "flux_biases" in plot_info.keys():
        # Plot Synaptic Weights over time
        for i in range(nnsize):
            plt.plot(time[0:stop_step],plot_info["flux_biases"][i][0:stop_step],label="bias{}".format(i) )


        plt.xlabel("Time")
        plt.ylabel("Fluctating Biases")
        plt.legend()
        plt.title("Fluctating Biases over time")
        plt.show()


def subplots(osc_exp, plot_info, time, stop_step, reduce_plot_data=1, stepsize=100, perf_bias = None, size=None):

    #reduce_plot_data=10

    loctext="upper left"

    start_time=100
    if reduce_plot_data == None:
        start_step= int(start_time*stepsize )
    else:
        start_step= int(start_time*stepsize / reduce_plot_data)

    if not "outputs" in plot_info:
        nnsize=2
    else:
        nnsize = len(plot_info["outputs"])
    if size!=None:
        nnsize=size

    rows=3
    cols=2
    if "biases" in plot_info.keys():
        cols+=1

    fig, axs = plt.subplots(rows, cols)
    subplot_index=1


##############################  1
    ax1 = plt.subplot(rows, cols, subplot_index)

    #plt.xlabel("Time")
    #plt.ylabel("Performance")
    # if reduce_plot_data == None:
    #     ax2=ax1.twinx()
    #     ax1.plot(time[start_step:stop_step],plot_info["amps"][start_step:stop_step],label="flux amplitude", color='r' )
    #     #ax1.set_ylim(0,8)
    #     ax2.plot(time[start_step:stop_step],plot_info["performances"][start_step:stop_step],label="instantaneous", color='c' )
    #     ax2.plot(time[start_step:stop_step],plot_info["running_average_performances"][start_step:stop_step],label="running average", color='b' )
    ampcolor=(1,0,0, 0.5)
    instcolor=(0,0,1, 0.1)
    runavecolor=(0,0,1, 0.25)

    # else:
    ax2=ax1.twinx()
    if "amps" in plot_info.keys():
        ax1.plot(time[start_step:stop_step:reduce_plot_data],plot_info["amps"][start_step:stop_step:reduce_plot_data],label="flux amplitude",color=ampcolor )
    #ax1.set_ylim(0,8)

    perf_data=plot_info["performances"][start_step:stop_step:reduce_plot_data]
    run_ave_perf_data=plot_info["running_average_performances"][start_step:stop_step:reduce_plot_data]
    if perf_bias != None:
        perf_data = 100*(perf_bias+perf_data)
        run_ave_perf_data=100*(perf_bias+run_ave_perf_data)

    ax2.plot(time[start_step:stop_step:reduce_plot_data], perf_data ,label="instantaneous", color=instcolor )
    ax2.plot(time[start_step:stop_step:reduce_plot_data], run_ave_perf_data,label="running average", color=runavecolor )

    ax1.set_ylabel("Flux Size",color='r',labelpad=-40)
    ax2.set_ylabel("Performance", color='b',labelpad=-60)

    cmap = plt.cm.turbo
    custom_lines = [Line2D([0], [0], color=ampcolor, lw=2),
                    Line2D([0], [0], color=instcolor, lw=2),
                    Line2D([0], [0], color=runavecolor, lw=2)]
    legend_array=['flux', 'inst. perf.', 'run. ave.']

    ax1.legend(custom_lines, legend_array, loc=loctext)
    
    ax1.set_title("Fluctuation and Performance")
################################### 2
    subplot_index+=1
    ax1 = plt.subplot(rows, cols, subplot_index)


    # Plot Synaptic Weights over time
    for i in range(nnsize):
        for j in range(nnsize):
            # if reduce_plot_data == None:
            #     ax1.plot(time[start_step:stop_step:reduce_plot_data],plot_info["weights"][i][j][start_step:stop_step],label="{}->{}".format(i,j) )
            # else:
            ax1.plot(time[start_step:stop_step:reduce_plot_data],plot_info["weights"][i][j][start_step:stop_step:reduce_plot_data],label="{}->{}".format(i,j) )


    #plt.xlabel("Time")
    #plt.ylabel("Weight Centers")
    if nnsize < 4:
        ax1.legend(loc=loctext)
    ax1.set_title("Synaptic Weight Centers over time")
    #plt.show()
################################### 3
    if "biases" in plot_info.keys():
        subplot_index+=1
        ax1 = plt.subplot(rows, cols, subplot_index)


    if "biases" in plot_info.keys():
        # Plot Biases over time
        for i in range(nnsize):
            plt.plot(time[start_step:stop_step:reduce_plot_data],plot_info["biases"][i][start_step:stop_step:reduce_plot_data],label="bias{}".format(i) )


        plt.xlabel("Time")
        plt.ylabel("Bias Centers")
        plt.legend()
        plt.title("Bias Centers  over time")
        #plt.show()
###########################   4
    subplot_index+=1
    ax1 = plt.subplot(rows, cols, subplot_index)

    
    #plot output of each neuron
    for i in range( nnsize ):
        #if reduce_plot_data == None:
        if not "outputs" in plot_info.keys():
            if "output0" in plot_info.keys():
                ax1.plot(time[start_step:stop_step:reduce_plot_data],plot_info[f"output{i}"][start_step:stop_step:reduce_plot_data],label=i, alpha=0.5 )
        else:
            ax1.plot(time[start_step:stop_step:reduce_plot_data],plot_info["outputs"][i][start_step:stop_step:reduce_plot_data],label=i, alpha=0.5 )
        #else:
        #    plt.plot(time[start_step:stop_step][start_step::reduce_plot_data],plot_info["outputs"][i][start_step:stop_step][start_step::reduce_plot_data],label=i, alpha=0.5 )
        
    #plt.xlabel("Time")
    #plt.ylabel("Output")
    ax1.legend(loc=loctext)
    ax1.set_title("Neural Outputs")      
################################### 5

    subplot_index+=1
    ax1 = plt.subplot(rows, cols, subplot_index)

    # Plot Synaptic Weights over time
    for i in range(nnsize):
        for j in range(nnsize):
            if "flux_weights" in plot_info.keys():
                if reduce_plot_data == None:
                    ax1.plot(time[start_step:stop_step],plot_info["flux_weights"][i][j][start_step:stop_step],label="{}->{}".format(i,j) )
                else:
                    ax1.plot(time[start_step:stop_step:reduce_plot_data],plot_info["flux_weights"][i][j][start_step:stop_step:reduce_plot_data],label="{}->{}".format(i,j) )
            else:
                if "flux_w0_0" in plot_info.keys():
                    if reduce_plot_data == None:
                        ax1.plot(time[start_step:stop_step],plot_info[f"flux_w{i}_{j}"][start_step:stop_step],label="{}->{}".format(i,j) )
                    else:
                        ax1.plot(time[start_step:stop_step:reduce_plot_data],plot_info[f"flux_w{i}_{j}"][start_step:stop_step:reduce_plot_data],label="{}->{}".format(i,j) )
    


    #plt.xlabel("Time")
    #plt.ylabel("Fluctating Weights")
    if nnsize < 4:
        ax1.legend(loc=loctext)
    ax1.set_title("Fluctuating Weights over time")
################################### 6
    if "flux_biases" in plot_info.keys():
        subplot_index+=1
        ax1 = plt.subplot(rows, cols, subplot_index)
        # Plot Synaptic Weights over time
        for i in range(nnsize):
            plt.plot(time[start_step:stop_step:reduce_plot_data],plot_info["flux_biases"][i][start_step:stop_step:reduce_plot_data],label="bias{}".format(i) )


        plt.xlabel("Time")
        plt.ylabel("Fluctating Biases")
        plt.legend()
        plt.title("Fluctating Biases over time")
        #plt.show()
###################################  7
    subplot_index+=1
    ax1 = plt.subplot(rows, cols, subplot_index)

    cmapname = 'viridis'
    cmap=plt.get_cmap(cmapname)
    flux_linewidth=1
    center_linewidth=2
    flux_alpha=0.1
    center_alpha=1
    endpoint_size=5

    start_color=cmap(0.0)
    stop_color=cmap(0.99)

    r,g,b = cmap(0.5)[0], cmap(0.5)[1], cmap(0.5)[2]

    custom_lines = [Line2D([0], [0], color=cmap(0), lw=2),
                    Line2D([0], [0], color=(r,g,b,center_alpha) , lw=2),
                    Line2D([0], [0], color=(r,g,b,flux_alpha), lw=1),
                    Line2D([0], [0], color=cmap(0.99), lw=2)]
    legend_array=['start', 'center', 'flux', 'stop']

    i=0
    j=1

    #ax1.scatter( plot_info["weights"][i][i][int(stop_step/2)], plot_info["weights"][j][j][int(stop_step/2)],label="middle", color='orange' )
    ax1.set_xlabel("Weight {}->{}".format(i,i))
    ax1.set_ylabel("Weight {}->{}".format(j,j))

    

    if "weights" in plot_info.keys():
        wii=plot_info["weights"][i][i][start_step:stop_step:reduce_plot_data]
        wjj=plot_info["weights"][j][j][start_step:stop_step:reduce_plot_data]
    elif "w0_0" in plot_info.keys():
        wii=plot_info[f"w{i}_{i}"][start_step:stop_step:reduce_plot_data]
        wjj=plot_info[f"w{j}_{j}"][start_step:stop_step:reduce_plot_data]

    if "flux_weights" in plot_info.keys():
        fwii=plot_info["flux_weights"][i][i][start_step:stop_step:reduce_plot_data]
        fwjj=plot_info["flux_weights"][j][j][start_step:stop_step:reduce_plot_data]
    elif "flux_w0_0" in plot_info.keys():
        fwii=plot_info[f"flux_w{i}_{i}"][start_step:stop_step:reduce_plot_data]
        fwjj=plot_info[f"flux_w{j}_{j}"][start_step:stop_step:reduce_plot_data]

    if "flux_w0_0" in plot_info.keys():
        ax1.set_ylim(np.min(fwjj), np.max(fwjj))
        ax1.set_xlim(np.min(fwii), np.max(fwii))
        lcfw = create_colored_lc( fwii, fwjj, alpha=flux_alpha, linewidth=flux_linewidth,cmapname=cmapname)
        ax1.add_collection(lcfw)
        lc = create_colored_lc( wii, wjj, alpha=center_alpha, linewidth=center_linewidth,cmapname=cmapname)
        ax1.add_collection(lc)

    if "weights" in plot_info.keys():
        ax1.scatter( plot_info["weights"][i][i][start_step], plot_info["weights"][j][j][start_step],label="start", color=start_color, linewidths=endpoint_size )
        ax1.scatter( plot_info["weights"][i][i][stop_step], plot_info["weights"][j][j][stop_step],label="end", color=stop_color, linewidths=endpoint_size )
    elif "w0_0" in plot_info.keys():
        ax1.scatter( plot_info[f"w{i}_{i}"][start_step], plot_info[f"w{j}_{j}"][start_step],label="start", color=start_color, linewidths=endpoint_size )
        ax1.scatter( plot_info[f"w{i}_{i}"][stop_step], plot_info[f"w{j}_{j}"][stop_step],label="end", color=stop_color, linewidths=endpoint_size )


    # ax1.plot(plot_info["weights"][i][i][start_step:stop_step:reduce_plot_data],plot_info["weights"][j][j][start_step:stop_step:reduce_plot_data],color=[0,1,0,0.5] )
    # ax1.plot(plot_info["flux_weights"][i][i][start_step:stop_step:reduce_plot_data],plot_info["flux_weights"][j][j][start_step:stop_step:reduce_plot_data], color=[0,1,0,0.2] )
    ax1.set_title("Fluctuating Weights")
    ax1.legend(custom_lines, legend_array, loc=loctext)
################################### 8
    subplot_index+=1

    ax1 = plt.subplot(rows, cols, subplot_index)

    #ax1.scatter( plot_info["weights"][i][j][int(stop_step/2)], plot_info["weights"][j][i][int(stop_step/2)],label="middle", color='orange' )

    #ax1.plot(plot_info["weights"][i][j][start_step:stop_step:reduce_plot_data],plot_info["weights"][j][i][start_step:stop_step:reduce_plot_data],color=[0,1,0,0.5] )
    #ax1.plot(plot_info["flux_weights"][i][j][start_step:stop_step:reduce_plot_data],plot_info["flux_weights"][j][i][start_step:stop_step:reduce_plot_data], color=[0,1,0,0.2] )
    if "weights" in plot_info.keys():
        wij=plot_info["weights"][i][j][start_step:stop_step:reduce_plot_data]
        wji=plot_info["weights"][j][i][start_step:stop_step:reduce_plot_data]
    elif "w0_0" in plot_info.keys():
        wij=plot_info[f"w{i}_{j}"][start_step:stop_step:reduce_plot_data]
        wji=plot_info[f"w{j}_{i}"][start_step:stop_step:reduce_plot_data]
    if "flux_weights" in plot_info.keys():
        fwij=plot_info["flux_weights"][i][j][start_step:stop_step:reduce_plot_data]
        fwji=plot_info["flux_weights"][j][i][start_step:stop_step:reduce_plot_data]
    elif "flux_w0_0" in plot_info.keys():
        fwij=plot_info[f"flux_w{i}_{j}"][start_step:stop_step:reduce_plot_data]
        fwji=plot_info[f"flux_w{j}_{i}"][start_step:stop_step:reduce_plot_data]
    
    if "flux_w0_0" in plot_info.keys():
        ax1.set_ylim(np.min(fwji), np.max(fwji))
        ax1.set_xlim(np.min(fwij), np.max(fwij))
        lcfw = create_colored_lc( fwij, fwji, alpha=flux_alpha, linewidth=flux_linewidth,cmapname=cmapname)
        ax1.add_collection(lcfw)
        lc = create_colored_lc( wij, wji, alpha=center_alpha, linewidth=center_linewidth,cmapname=cmapname)
        ax1.add_collection(lc)
    
    if "weights" in plot_info.keys():
        ax1.scatter( plot_info["weights"][i][j][start_step], plot_info["weights"][j][i][start_step],label="start", color=start_color )
        ax1.scatter( plot_info["weights"][i][j][stop_step], plot_info["weights"][j][i][stop_step],label="end", color=stop_color )
    elif "w0_0" in plot_info.keys():
        ax1.scatter( plot_info[f"w{i}_{j}"][start_step], plot_info[f"w{j}_{i}"][start_step],label="start", color=start_color )
        ax1.scatter( plot_info[f"w{i}_{j}"][stop_step], plot_info[f"w{j}_{i}"][stop_step],label="end", color=stop_color )

    ax1.set_title("Fluctuating Weights")
    ax1.set_xlabel("Weight {}->{}".format(i,j))
    ax1.set_ylabel("Weight {}->{}".format(j,i))

    ax1.legend(custom_lines, legend_array, loc=loctext)
################################### 9
    if "flux_biases" in plot_info.keys():
        subplot_index+=1
        ax1 = plt.subplot(rows, cols, subplot_index)
        ax1.scatter( plot_info["biases"][0][start_step], plot_info["biases"][1][start_step],label="start", color=start_color )
        ax1.scatter( plot_info["biases"][0][stop_step], plot_info["biases"][1][stop_step],label="end", color=stop_color )
        #ax1.scatter( plot_info["weights"][i][j][int(stop_step/2)], plot_info["weights"][j][i][int(stop_step/2)],label="middle", color='orange' )

        #ax1.plot(plot_info["weights"][i][j][start_step:stop_step:reduce_plot_data],plot_info["weights"][j][i][start_step:stop_step:reduce_plot_data],color=[0,1,0,0.5] )
        #ax1.plot(plot_info["flux_weights"][i][j][start_step:stop_step:reduce_plot_data],plot_info["flux_weights"][j][i][start_step:stop_step:reduce_plot_data], color=[0,1,0,0.2] )
        
        b0=plot_info["biases"][0][start_step:stop_step:reduce_plot_data]
        b1=plot_info["biases"][1][start_step:stop_step:reduce_plot_data]
        fb0=plot_info["flux_biases"][0][start_step:stop_step:reduce_plot_data]
        fb1=plot_info["flux_biases"][1][start_step:stop_step:reduce_plot_data]
        ax1.set_xlim(np.min(fb0), np.max(fb0))
        ax1.set_ylim(np.min(fb1), np.max(fb1))
        lcfw = create_colored_lc( fb0, fb1, alpha=flux_alpha, linewidth=flux_linewidth,cmapname=cmapname)
        ax1.add_collection(lcfw)
        lc = create_colored_lc( b0, b1, alpha=center_alpha, linewidth=center_linewidth,cmapname=cmapname)
        ax1.add_collection(lc)
                    
        ax1.set_title("Fluctuating Biases over time")
        ax1.set_xlabel("Bias0")
        ax1.set_ylabel("Bias1")
        legend_array=['start', 'bias center', 'fluctuating bias', 'stop']
        ax1.legend(custom_lines, legend_array, loc=loctext)
###################################
    
    


    plt.show()

def create_colored_lc(x,y, alpha=1.0, linewidth=2, cmapname='jet'):

    path = mpath.Path(np.column_stack([x, y]))
    verts = path.interpolated(steps=3).vertices
    x, y = verts[:, 0], verts[:, 1]
    z = np.linspace(0, 1, len(x))
    lc = colorline(x, y, z, cmap=plt.get_cmap(cmapname), linewidth=linewidth, alpha=alpha)
    return lc


if __name__ == "__main__":
    main()