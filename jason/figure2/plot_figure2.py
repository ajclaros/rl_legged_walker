import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def alpha_adj(color, alpha=0.25):
    """
    Adjust alpha of color.
    """
    return [color[0], color[1], color[2], alpha]

def improvement_adj(init_fit, final_fit):
    """
    Calculate improvement of y over x.
    """
    #return x
    # if init_fit > final_fit:
    #     print("it got worse")
    

    maxxed= max(0,  0.5 + (final_fit - init_fit)  )
    # minned = min(1, max(0,  0.5 + (final_fit - init_fit)  ))
    # if not maxxed == minned:
    #     print(maxxed, minned)


    return maxxed
    #return min(1, max(0,  0.5 + (final_fit - init_fit)  ))



#####################################################

legend_array=['much worse', 'worse', 'no change', 'better', 'much better']

#     500, 2000
# learning_duration=5000
# initflux=2
init_flux=4
learning_duration=5000
prefix="v1_plusminus8_by2"
#prefix="v1_12x12x6"
arrow_alpha=0.25
filename_prefix="10sec_"

save_dat_filenames=[\
    "v1_12x12x4_fig2_1.0k_initflux-1.dat",\
    "v1_12x12x6_fig2_5.0k_initflux-1.dat",\
    "v1_12x12x6_fig2_5.0k_initflux-2.dat",\
    "v1_12x12x6_fig2_5.0k_initflux-4.dat",\
    "v1_12x12x6_trial2_fig2_5.0k_initflux-1.dat",\
    "v1_plusminus2_fig2_1.0k_initflux-1.dat",\
    "v1_plusminus2_fig2_1.0k_initflux-2.dat",\
    "v1_plusminus2_fig2_1.0k_initflux-4.dat",\
    "v1_plusminus3_fig2_1.0k_initflux-1.dat",\
    "v1_plusminus8_by2_fig2_5.0k_initflux-3.dat",\
    "v1_plusminus8_by2_fig2_5.0k_initflux-4.dat"]

#save_dat_filenames.append( f"{prefix}_fig2_{learning_duration/1000}k_initflux-{init_flux}.dat" )

write_figures=[0,1,2,3,4,5,6,7,8]
write_figures=[7,8]
loctext="lower left"
img_dim=(3.5,2)

for save_dat_filename in save_dat_filenames:

    plot_save_filename=f"plots/{save_dat_filename}".replace(".dat", "")
    #plot_save_filename=f"plots/{prefix}_fig2_{learning_duration/1000}k_initflux-{init_flux}"

    show_fitness_background=True

    data = np.genfromtxt(save_dat_filename,delimiter=",", dtype=float)

    cmap = plt.cm.gnuplot
    cmap = plt.cm.PiYG
    cmap = plt.cm.Spectral
    #cmap = plt.cm.turbo
    custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                    Line2D([0], [0], color=cmap(.25), lw=4), 
                    Line2D([0], [0], color=cmap(.5), lw=4),
                    Line2D([0], [0], color=cmap(.75), lw=4),
                    Line2D([0], [0], color=cmap(1.), lw=4)]




    first_time_index=13

    threshold=0.5
    success=0
    for index in range(len(data)):
        final_fit=data[index][1]
        if final_fit>threshold:
            alpha=1
            success+=1
###################################  1
    if 1 in write_figures:
        for index in range(len(data)):
            #init_fit,final_fit,init_est_dist,final_est_dist
            init_fit=data[index][0]
            final_fit=data[index][1]
            init_dist=data[index][2]
            final_dist=data[index][3]
            improvement=improvement_adj(init_fit, final_fit)

            #plt.scatter( init_fit, final_fit, color=[improvement,0,0,alpha] )
            plt.scatter( init_fit, final_fit, color=alpha_adj(cmap(improvement) ) )
        plt.plot( [0,100], [0,100], color=[.5,.5,.5,0.25] )
        plt.title(f"success: {100*success/len(data):0.3f}%")
        plt.xlabel("init_fit")
        plt.ylabel("final_fit")



        plt.legend(custom_lines, legend_array,loc="upper right")
        plt.xlim(0,.8)
        plt.ylim(0,.8)
        plt.rcParams["figure.figsize"] = img_dim
        plt.savefig(f"{plot_save_filename}_init_fit_X_final_fit.png", dpi=300, \
                        bbox_inches='tight' )
        plt.clf()
        #plt.show()
###################################   2
    if 2 in write_figures:
        for index in range(len(data)):
            #init_fit,final_fit,init_est_dist,final_est_dist
            init_fit=data[index][0]
            final_fit=data[index][1]
            init_dist=data[index][2]
            final_dist=data[index][3]
            improvement=improvement_adj(init_fit, final_fit)



            #plt.scatter( init_dist, final_dist, color=[final_fit,0,0,0.25] )
            plt.scatter( init_dist, final_dist, color=alpha_adj(cmap(improvement) ) )

        x_min = plt.xlim()
        y_min = plt.ylim()
        plt.xlim(0,x_min[1])
        plt.ylim(0,y_min[1])

        plt.plot( [0,100], [0,100], color=[.5,.5,.5,0.25] )

        plt.title(f"success: {100*success/len(data):0.3f}%")
        plt.xlabel("init_dist")
        plt.ylabel("final_dist")
        plt.legend(custom_lines, legend_array,loc="upper right")

        plt.rcParams["figure.figsize"] = img_dim
        plt.savefig(f"{plot_save_filename}_init_dist_X_final_dist.png", dpi=300, \
                        bbox_inches='tight' )
        plt.clf()
###################################  3
    if 3 in write_figures:
        #plt.show()
        #

        for index in range(len(data)):
            #init_fit,final_fit,init_est_dist,final_est_dist
            init_fit=data[index][0]
            final_fit=data[index][1]
            init_dist=data[index][2]
            final_dist=data[index][3]
            improvement=improvement_adj(init_fit, final_fit)

            #plt.scatter( init_dist, final_fit, color=[improvement,0,0,0.25] )
            #plt.scatter( init_dist, final_fit, color=cmap(improvement) )
            plt.arrow( init_dist, init_fit, 0,final_fit-init_fit, color=alpha_adj(cmap(improvement),alpha=arrow_alpha ), head_width=0.04, head_length=0.01 )

            
            plt.xlabel("init_dist")
            plt.ylabel("fitness")
            
        plt.legend(custom_lines, legend_array,loc=loctext)
        plt.rcParams["figure.figsize"] = img_dim
        plt.savefig(f"{plot_save_filename}_init_dist_X_fit_change.png", dpi=300, \
                        bbox_inches='tight' )
        plt.clf()
        #plt.show()
###################################   4
    if 4 in write_figures:

        #ADD PLOT START AND STOP
        ################################
        if show_fitness_background:
            #draw fitness plot and overlay...
            load="fitness_0_0__1_1.csv"
            fit_data = np.genfromtxt(load,delimiter=",", dtype=float, names=True)

            w_a_label=fit_data.dtype.names[0]
            w_b_label=fit_data.dtype.names[1]

            alpha=1
            length=len(fit_data)
            plot_fit_data=np.zeros( (2,length ) )
            colors=[]

            for i in range(len(fit_data)):
                plot_fit_data[0][i] = w_a = fit_data[i][0]
                plot_fit_data[1][i] = w_a = fit_data[i][1]
                fit=fit_data[i][2]
                if fit > 1.0:
                    fit = 1.0
                colors.append( [fit, fit, fit, alpha] )
            wAs=plot_fit_data[0]
            wBs=plot_fit_data[1]
            plt.scatter(wAs, wBs, c=colors )


        #######################################
        for index in range(len(data)):
            #init_fit,final_fit,init_est_dist,final_est_dist
            init_fit=data[index][0]
            final_fit=data[index][1]
            init_dist=data[index][2]
            final_dist=data[index][3]
            init_w00=data[index][4]
            init_w01=data[index][5]
            init_w10=data[index][6]
            init_w11=data[index][7]

            final_w00=data[index][8]
            final_w01=data[index][9]
            final_w10=data[index][10]
            final_w11=data[index][11]
            improvement=improvement_adj(init_fit, final_fit)

            #plt.scatter( init_dist, final_fit, color=[improvement,0,0,0.25] )
            #plt.scatter( init_dist, final_fit, color=cmap(improvement) )

            head_width=0.4
            head_length=0.1

            plt.arrow( init_w00, init_w11, final_w00-init_w00, final_w11-init_w11, color=alpha_adj(cmap(improvement),alpha=arrow_alpha ),\
                head_width=head_width, head_length=head_length )

            
            plt.xlabel("w00")
            plt.ylabel("w11")
            
        plt.legend(custom_lines, legend_array,loc=loctext)
        plt.rcParams["figure.figsize"] = img_dim
        plt.savefig(f"{plot_save_filename}_w00_X_w11.png", dpi=300, \
                        bbox_inches='tight' )
        plt.clf()
        #plt.show()

###################################  5
    if 5 in write_figures:

        for index in range(len(data)):
            #init_fit,final_fit,init_est_dist,final_est_dist
            init_fit=data[index][0]
            final_fit=data[index][1]
            init_dist=data[index][2]
            final_dist=data[index][3]
            improvement=improvement_adj(init_fit, final_fit)

            #plt.scatter( init_dist, final_fit, color=[improvement,0,0,0.25] )
            plt.scatter( init_dist, init_fit, color=alpha_adj(cmap(improvement) ) )
            
            plt.xlabel("init_dist")
            plt.ylabel("init_fit")
            
        plt.legend(custom_lines, legend_array,loc=loctext)
        plt.rcParams["figure.figsize"] = img_dim
        plt.savefig(f"{plot_save_filename}_init_dist_X_init_fit.png", dpi=300, \
                        bbox_inches='tight' )
        #plt.show()
        plt.clf()
###################################  6
    if 6 in write_figures:

        for index in range(len(data)):
            #init_fit,final_fit,init_est_dist,final_est_dist
            init_fit=data[index][0]
            final_fit=data[index][1]
            init_dist=data[index][2]
            final_dist=data[index][3]
            improvement=improvement_adj(init_fit, final_fit)

            #plt.scatter( init_dist, final_fit, color=[improvement,0,0,0.25] )
            plt.scatter( init_dist, final_fit, color=alpha_adj(cmap(improvement) ) )
            
            plt.xlabel("init_dist")
            plt.ylabel("final_fit")
            
        plt.legend(custom_lines, legend_array,loc=loctext)
        plt.rcParams["figure.figsize"] = img_dim
        plt.savefig(f"{plot_save_filename}_init_dist_X_final_fit.png", dpi=300, \
                        bbox_inches='tight' )
        #plt.show()
        plt.clf()

###################################  7
    if 7 in write_figures:
        cmap=plt.get_cmap("autumn")

        #ADD PLOT START AND STOP
        ################################
        if show_fitness_background:
            #draw fitness plot and overlay...
            load=f"{filename_prefix}fitness_0_1__1_0.csv"
            fit_data = np.genfromtxt(load,delimiter=",", dtype=float, names=True)

            w_a_label=fit_data.dtype.names[0]
            w_b_label=fit_data.dtype.names[1]

            alpha=1
            length=len(fit_data)
            plot_fit_data=np.zeros( (2,length ) )
            colors=[]

            for i in range(len(fit_data)):
                plot_fit_data[0][i] = w_a = fit_data[i][0]
                plot_fit_data[1][i] = w_a = fit_data[i][1]
                fit=fit_data[i][2]
                if fit > 1.0:
                    fit = 1.0
                colors.append( [fit, fit, fit, alpha] )
            wAs=plot_fit_data[0]
            wBs=plot_fit_data[1]
            plt.scatter(wAs, wBs, c=colors )


        #######################################
        for index in range(len(data)):
            #init_fit,final_fit,init_est_dist,final_est_dist
            init_fit=data[index][0]
            final_fit=data[index][1]
            init_dist=data[index][2]
            final_dist=data[index][3]
            init_w00=data[index][4]
            init_w01=data[index][5]
            init_w10=data[index][6]
            init_w11=data[index][7]

            final_w00=data[index][8]
            final_w01=data[index][9]
            final_w10=data[index][10]
            final_w11=data[index][11]
            fitness_adj= min(1.0, final_fit/.7 )   #improvement_adj(init_fit, final_fit)

            #plt.scatter( init_dist, final_fit, color=[improvement,0,0,0.25] )
            #plt.scatter( init_dist, final_fit, color=cmap(improvement) )

            head_width=0.4
            head_length=0.1

            #if improvement> threshold:

            plt.arrow( init_w00, init_w11, final_w00-init_w00, final_w11-init_w11, color=alpha_adj(cmap(fitness_adj),alpha=arrow_alpha ),\
                head_width=head_width, head_length=head_length )

            
            plt.xlabel("w00")
            plt.ylabel("w11")
        legend_array = ['zero', 'poor', 'high']
        custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                Line2D([0], [0], color=cmap(.5), lw=4),
                Line2D([0], [0], color=cmap(1.), lw=4)]
        plt.legend(custom_lines, legend_array,loc=loctext)
        plt.rcParams["figure.figsize"] = img_dim
        plt.savefig(f"{plot_save_filename}_w00_X_w11_color-finalfitness{filename_prefix}.png", dpi=300, \
                        bbox_inches='tight' )
        plt.clf()
    
    if 8 in write_figures:

        #ADD PLOT START AND STOP
        ################################
        if show_fitness_background:
            #draw fitness plot and overlay...
            
            load=f"{filename_prefix}fitness_0_1__1_0.csv"
            fit_data = np.genfromtxt(load,delimiter=",", dtype=float, names=True)

            w_a_label=fit_data.dtype.names[0]
            w_b_label=fit_data.dtype.names[1]

            alpha=1
            length=len(fit_data)
            plot_fit_data=np.zeros( (2,length ) )
            colors=[]

            for i in range(len(fit_data)):
                plot_fit_data[0][i] = w_a = fit_data[i][0]
                plot_fit_data[1][i] = w_a = fit_data[i][1]
                fit=fit_data[i][2]
                if fit > 1.0:
                    fit = 1.0
                colors.append( [fit, fit, fit, alpha] )
            wAs=plot_fit_data[0]
            wBs=plot_fit_data[1]
            plt.scatter(wAs, wBs, c=colors )


        #######################################
        for index in range(len(data)):
            #init_fit,final_fit,init_est_dist,final_est_dist
            init_fit=data[index][0]
            final_fit=data[index][1]
            init_dist=data[index][2]
            final_dist=data[index][3]
            init_w00=data[index][4]
            init_w01=data[index][5]
            init_w10=data[index][6]
            init_w11=data[index][7]

            final_w00=data[index][8]
            final_w01=data[index][9]
            final_w10=data[index][10]
            final_w11=data[index][11]
            fitness_adj= min(1.0, final_fit/.7 )   #improvement_adj(init_fit, final_fit)

            #plt.scatter( init_dist, final_fit, color=[improvement,0,0,0.25] )
            #plt.scatter( init_dist, final_fit, color=cmap(improvement) )

            head_width=0.4
            head_length=0.1

            #if improvement> threshold:

            plt.arrow( init_w01, init_w10, final_w01-init_w01, final_w10-init_w10, color=alpha_adj(cmap(fitness_adj),alpha=arrow_alpha ),\
                head_width=head_width, head_length=head_length )

            
            plt.xlabel("w01")
            plt.ylabel("w10")

        plt.legend(custom_lines, legend_array,loc=loctext)
        plt.rcParams["figure.figsize"] = img_dim
        plt.savefig(f"{plot_save_filename}_w01_X_w10_color-finalfitness{filename_prefix}.png", dpi=300, \
                        bbox_inches='tight' )
        plt.clf()



    
