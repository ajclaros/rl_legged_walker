import matplotlib.pyplot as plt
import csv


# Installation instructions for matplotlib
# https://matplotlib.org/stable/users/installing.html

# Longer tutorial here
# https://docs.python.org/3/library/csv.html

# load_file="jason/data/rl-discovery__SPREAD_10SEEDS__nnsizes-[2]__maxamp-[8, 16]__periodmin-[4, 8]__duration-10k.csv"
# x_value_key="max_flux_amp"
# y_value_key="rl_nn_b_fit"

load_file="jason/data/rl-discovery__AMP_vs_CONV__nnsizes-[2]__maxamp-[2, 4, 6, 8, 10, 12, 14, 16]__periodmin-[4]__duration-10k.csv"
x_value_key="flux_conv_rate"
y_value_key="rl_nn_b_fit"

load_file="jason/data/rl-discovery__AMP_vs_CONV__nnsizes-[2]__maxamp-[2, 4, 6, 8, 10, 12, 14, 16]__periodmin-[4]__duration-10k.csv"
x_value_key="flux_conv_rate"
y_value_key="rl_nn_b_fit"



series_key="max_flux_amp" #used to create series that match each other
#series_key=""



if series_key != "":
    x_values={}
    y_values={}
else:
    x_values=[]
    y_values=[]


with open(load_file, newline='') as csvfile:
    # Using the csv reader automatically places all values 
    # in columns within a row in a dictionary with a 
    # key based on the header (top line of the file)
    reader = csv.DictReader(csvfile)
    for row in reader:
        str_series_key=str(row[series_key])
        if series_key != "":
            if not str_series_key in x_values.keys():
                x_values[str_series_key] = []
            x_values[str_series_key].append( float(row[x_value_key]) )
            if not str_series_key in y_values.keys():
                y_values[str_series_key] = []
            y_values[str_series_key].append( float(row[y_value_key]) )

        else:
            x_values.append( float(row[x_value_key]) )
            y_values.append( float(row[y_value_key]) )
        

#TODO add in error checking for the file path since it does not always work with PyCharm

if series_key != "":
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)
    for str_series_key in x_values.keys():
        ax1.scatter(x_values[str_series_key], y_values[str_series_key], label=str_series_key)
    #plt.show()

    data=[]
    labels=[]

    for str_series_key in x_values.keys():
        data.append( y_values[str_series_key] )    
        labels.append( str_series_key )   
    ax2.violinplot(data )
    #print(labels)
    ax2.set_xticklabels(labels)
    ax1.set_ylabel( y_value_key )
    ax2.set_xlabel( series_key )
    ax1.set_xlabel( x_value_key )
    #plt.show()


else:
    #plt.plot(x_values,y_values,label="y_value_key")
    plt.scatter(x_values,y_values)
    #plt.violinplot( [x_values,y_values] )
    plt.xlabel(x_value_key)
    

plt.legend()

plt.ylabel(y_value_key)


#plt.savefig('demo_plot.png')
plt.show()

